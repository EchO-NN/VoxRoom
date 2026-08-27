from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from voxroom_online.isaac_runtime.baselines.data_contract import resolve_map_info
from voxroom_online.isaac_runtime.baselines.mask_io import enforce_room_mask_contract
from voxroom_online.isaac_runtime.baselines.ros_bridge_payload import load_input_arrays, load_request, write_result
from voxroom_online.isaac_runtime.baselines.ros_grid_io import snapshot_to_ipa_image
from voxroom_online.isaac_runtime.baselines.offline.fallback_morphological import IPA_UNASSIGNED_FREE_LABEL
from voxroom_online.isaac_runtime.baselines.offline.fallback_voronoi import build_voronoi_ipa_input_image
from voxroom_online.isaac_runtime.baselines.offline.ipa_runner import (
    IPA_ALGORITHM_IDS,
    ORIGINAL_ACTION,
    ORIGINAL_PACKAGE,
    ORIGINAL_REPO,
    ORIGINAL_REPO_COMMIT,
    _algorithm_display_name,
    _input_free_definition,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one original ipa_room_segmentation ROS action request.")
    parser.add_argument("--request", required=True)
    args = parser.parse_args(argv)
    request = load_request(Path(args.request))
    arrays = load_input_arrays(request)
    params = dict(request.get("params") or {})
    result = run_ipa_snapshot(request=request, arrays=arrays, params=params)
    write_result(request, label_map=result["label_map"], metadata=result["metadata"], debug_arrays=result["debug_arrays"])
    print(json.dumps(result["metadata"], ensure_ascii=False, sort_keys=True))
    return 0


def run_ipa_snapshot(
    *,
    request: Mapping[str, Any],
    arrays: Mapping[str, Any],
    params: Mapping[str, Any],
) -> dict[str, Any]:
    import actionlib
    import rospy
    from geometry_msgs.msg import Pose
    from ipa_building_msgs.msg import MapSegmentationAction, MapSegmentationGoal
    from sensor_msgs.msg import Image

    algorithm = str(params["algorithm"])
    if algorithm not in IPA_ALGORITHM_IDS:
        raise ValueError("unsupported IPA algorithm %s" % algorithm)
    action_name = str(params.get("action_name") or "room_segmentation_server")
    timeout_s = float(params.get("timeout_s") or 120.0)
    if not rospy.core.is_initialized():
        rospy.init_node("voxroom_ipa_roomseg_replay", anonymous=True, disable_signals=True)

    map_info = resolve_map_info(snapshot_arrays=arrays, default_resolution_m=_optional_float(params.get("map_resolution_m")))
    if algorithm == "voronoi":
        ipa_image, _, voronoi_debug = build_voronoi_ipa_input_image(arrays)
    else:
        ipa_image = snapshot_to_ipa_image(arrays)
        voronoi_debug = {}
    image_msg = _mono8_image_msg(np.asarray(ipa_image, dtype=np.uint8), Image)

    goal = MapSegmentationGoal()
    goal.input_map = image_msg
    goal.map_resolution = float(map_info.resolution_m)
    goal.map_origin = Pose()
    goal.map_origin.position.x = float(map_info.min_x)
    goal.map_origin.position.y = float(map_info.min_y)
    goal.return_format_in_pixel = True
    goal.return_format_in_meter = False
    goal.robot_radius = float(params.get("robot_radius_m") or 0.05)
    goal.room_segmentation_algorithm = int(IPA_ALGORITHM_IDS[algorithm])

    client = actionlib.SimpleActionClient(action_name, MapSegmentationAction)
    if not client.wait_for_server(rospy.Duration(timeout_s)):
        raise TimeoutError("timed out waiting for IPA action server %r" % action_name)
    client.send_goal(goal)
    if not client.wait_for_result(rospy.Duration(timeout_s)):
        client.cancel_goal()
        raise TimeoutError("timed out waiting for IPA %s result" % algorithm)
    action_result = client.get_result()
    if action_result is None:
        raise RuntimeError("IPA %s returned no action result" % algorithm)
    segmented_map = _image_msg_to_int32(action_result.segmented_map)
    segmented_map[segmented_map >= IPA_UNASSIGNED_FREE_LABEL] = 0
    label_map = enforce_room_mask_contract(segmented_map, arrays, clip_to_eval_domain=True)
    metadata = {
        "method": algorithm,
        "runner_type": "original_ros_action",
        "main_experiment_allowed": True,
        "source_snapshot": str(request.get("snapshot_path")),
        "scene_id": request.get("scene_id"),
        "input_free_definition": _input_free_definition(arrays),
        "unknown_treated_as": "occupied/inaccessible",
        "unknown_treated_as_free": False,
        "map_resolution_m": float(map_info.resolution_m),
        "map_origin_xy_m": [float(map_info.min_x), float(map_info.min_y)],
        "uses_rgb": False,
        "uses_depth": False,
        "uses_oracle_semantics": False,
        "original_repo": ORIGINAL_REPO,
        "original_repo_commit": params.get("original_repo_commit"),
        "original_repo_reference_commit": ORIGINAL_REPO_COMMIT,
        "original_package": ORIGINAL_PACKAGE,
        "original_action": ORIGINAL_ACTION,
        "action_type": "ipa_building_msgs/MapSegmentationAction",
        "action_server": action_name,
        "original_action_server": action_name,
        "input_image_encoding": "mono8",
        "input_free_value": 255,
        "input_occupied_value": 0,
        "room_segmentation_algorithm": int(IPA_ALGORITHM_IDS[algorithm]),
        "algorithm_id_verified": True,
        "original_algorithm_id": int(IPA_ALGORITHM_IDS[algorithm]),
        "original_algorithm_name": _algorithm_display_name(algorithm),
        "parameters": {
            "room_segmentation_algorithm": int(IPA_ALGORITHM_IDS[algorithm]),
            "robot_radius_m": float(params.get("robot_radius_m") or 0.05),
        },
        "rooms_after_contract": int(len([v for v in np.unique(label_map) if int(v) > 0])),
    }
    debug = {f"{algorithm}_ipa_input_image": ipa_image}
    debug.update({str(k): np.asarray(v) for k, v in voronoi_debug.items()})
    return {"label_map": label_map, "metadata": metadata, "debug_arrays": debug}


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _mono8_image_msg(image: np.ndarray, image_cls: Any) -> Any:
    arr = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    if arr.ndim != 2:
        raise ValueError("IPA input image must be 2D mono8, got shape %s" % (arr.shape,))
    msg = image_cls()
    msg.height = int(arr.shape[0])
    msg.width = int(arr.shape[1])
    msg.encoding = "mono8"
    msg.is_bigendian = 0
    msg.step = int(arr.shape[1])
    msg.data = arr.tobytes()
    return msg


def _image_msg_to_int32(msg: Any) -> np.ndarray:
    encoding = str(getattr(msg, "encoding", "") or "").lower()
    height = int(getattr(msg, "height"))
    width = int(getattr(msg, "width"))
    data = bytes(getattr(msg, "data"))
    if encoding in {"32sc1", "32sc"}:
        arr = np.frombuffer(data, dtype=np.int32).copy()
    elif encoding in {"mono8", "8uc1", "8uc"}:
        arr = np.frombuffer(data, dtype=np.uint8).astype(np.int32, copy=True)
    else:
        raise ValueError("unsupported IPA segmented_map encoding %r" % getattr(msg, "encoding", ""))
    expected = height * width
    if arr.size < expected:
        raise ValueError("IPA segmented_map data too short: got %d values, expected %d" % (arr.size, expected))
    return arr[:expected].reshape((height, width)).astype(np.int32, copy=False)


if __name__ == "__main__":
    raise SystemExit(main())
