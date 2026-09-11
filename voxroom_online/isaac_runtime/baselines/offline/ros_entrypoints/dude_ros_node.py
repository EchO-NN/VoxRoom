from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from voxroom_online.isaac_runtime.baselines.data_contract import resolve_map_info
from voxroom_online.isaac_runtime.baselines.mask_io import (
    build_segmentation_domain_from_source,
)
from voxroom_online.isaac_runtime.baselines.offline.dude_image_io import (
    DUDE_OUTPUT_COORDINATE_CONTRACT,
    labels_from_tagged_image as _labels_from_tagged_image,
    tagged_image_to_source_labels,
)
from voxroom_online.isaac_runtime.baselines.ros_bridge_payload import load_input_arrays, load_request, write_result
from voxroom_online.isaac_runtime.baselines.ros_grid_io import snapshot_to_ros_occupancy_grid


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish one VoxRoom snapshot to original Incremental_DuDe_ROS.")
    parser.add_argument("--request", required=True)
    args = parser.parse_args(argv)
    request = load_request(Path(args.request))
    arrays = load_input_arrays(request)
    params = dict(request.get("params") or {})
    result = run_dude_snapshot(request=request, arrays=arrays, params=params)
    write_result(request, label_map=result["label_map"], metadata=result["metadata"], debug_arrays=result["debug_arrays"])
    print(json.dumps(result["metadata"], ensure_ascii=False, sort_keys=True))
    return 0


def run_dude_snapshot(
    *,
    request: Mapping[str, Any],
    arrays: Mapping[str, Any],
    params: Mapping[str, Any],
) -> dict[str, Any]:
    import rospy
    from cv_bridge import CvBridge
    from geometry_msgs.msg import Point, Pose, Quaternion
    from nav_msgs.msg import OccupancyGrid
    from sensor_msgs.msg import Image
    from std_msgs.msg import Header

    input_topic = str(params.get("input_topic") or "/map")
    output_topic = str(params.get("output_topic") or "/tagged_image")
    timeout_s = float(params.get("timeout_s") or 180.0)
    node_name = "voxroom_dude_replay_%s" % str(params.get("scene_id") or request.get("scene_id") or "scene")
    if not rospy.core.is_initialized():
        rospy.init_node(node_name, anonymous=True, disable_signals=True)

    map_info = resolve_map_info(snapshot_arrays=arrays, default_resolution_m=_optional_float(params.get("map_resolution_m")))
    _segmentation_free, segmentation_source = build_segmentation_domain_from_source(arrays)
    occupancy_values = snapshot_to_ros_occupancy_grid(arrays)
    grid = OccupancyGrid()
    grid.header = Header()
    grid.header.frame_id = "map"
    publish_stamp = rospy.Time.now()
    grid.header.stamp = publish_stamp
    grid.info.resolution = float(map_info.resolution_m)
    grid.info.height = int(occupancy_values.shape[0])
    grid.info.width = int(occupancy_values.shape[1])
    grid.info.origin = Pose()
    grid.info.origin.position = Point(float(map_info.min_x), float(map_info.min_y), 0.0)
    grid.info.origin.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
    grid.data = [int(v) for v in np.asarray(occupancy_values, dtype=np.int8).reshape(-1)]

    publisher = rospy.Publisher(input_topic, OccupancyGrid, queue_size=1, latch=True)
    deadline = rospy.Time.now() + rospy.Duration(float(params.get("subscriber_timeout_s") or 15.0))
    while publisher.get_num_connections() < 1 and rospy.Time.now() < deadline and not rospy.is_shutdown():
        rospy.sleep(0.05)
    if publisher.get_num_connections() < 1:
        raise TimeoutError("Incremental_DuDe_ROS has no subscriber on %s" % input_topic)
    publisher.publish(grid)
    tagged = _wait_for_fresh_message(rospy, output_topic, Image, publish_stamp=publish_stamp, timeout_s=timeout_s)
    tagged_image = CvBridge().imgmsg_to_cv2(tagged, desired_encoding="passthrough")
    label_map = _labels_from_tagged_image(np.asarray(tagged_image), arrays)
    method = str(request.get("method") or "dude_incremental")
    use_incremental = bool(params.get("use_incremental", method == "dude_incremental"))
    metadata = {
        "method": method,
        "baseline_name": method,
        "runner_type": "original_ros",
        "main_experiment_allowed": True,
        "source_snapshot": str(request.get("snapshot_path")),
        "scene_id": request.get("scene_id"),
        "input_topic": input_topic,
        "output_topic": output_topic,
        "input_message_type": "nav_msgs/OccupancyGrid",
        "output_message_type": "sensor_msgs/Image",
        "input_occupancy_values": {"free": 0, "occupied": 100, "unknown": -1},
        "segmentation_source": str(segmentation_source),
        "binary_free_value": 255,
        "binary_occupied_value": 0,
        "unknown_treated_as": "occupied/inaccessible",
        "uses_rgb": False,
        "uses_depth": False,
        "uses_oracle_semantics": False,
        "incremental_state_reset_per_scene": bool(use_incremental),
        "incremental_order_enforced": bool(use_incremental),
        "offline_state_reset_per_snapshot": bool(not use_incremental),
        "state_reset_scope": str(
            params.get("state_reset_scope") or ("scene" if use_incremental else "snapshot")
        ),
        "ros_bridge": "subprocess_rospy_publish_wait",
        "original_repo": "lfermin77/Incremental_DuDe_ROS",
        "original_repo_url": "https://github.com/lfermin77/Incremental_DuDe_ROS",
        "original_repo_commit": params.get("original_repo_commit"),
        "original_repo_reference_commit": params.get("original_repo_reference_commit"),
        "map_resolution_m": float(map_info.resolution_m),
        "map_info": map_info.to_metadata(),
        "parameters": {
            "concavity_threshold_m": _optional_float(params.get("concavity_threshold_m")),
            "use_incremental": bool(use_incremental),
        },
        "tagged_image_encoding": str(getattr(tagged, "encoding", "")),
        "output_coordinate_contract": DUDE_OUTPUT_COORDINATE_CONTRACT,
        "output_vertical_flip_applied": True,
        "freshness_guard": "header_stamp_at_or_after_published_map",
        "rooms_after_contract": int(len([v for v in np.unique(label_map) if int(v) > 0])),
    }
    return {"label_map": label_map, "metadata": metadata, "debug_arrays": {
        "dude_ros_occupancy_grid": occupancy_values,
        "dude_tagged_image_native": np.asarray(tagged_image),
        "dude_labels_source_frame": tagged_image_to_source_labels(tagged_image),
    }}


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _wait_for_fresh_message(rospy: Any, topic: str, msg_type: Any, *, publish_stamp: Any, timeout_s: float) -> Any:
    deadline = rospy.Time.now() + rospy.Duration(float(timeout_s))
    while rospy.Time.now() < deadline and not rospy.is_shutdown():
        remaining = max(0.05, (deadline - rospy.Time.now()).to_sec())
        msg = rospy.wait_for_message(topic, msg_type, timeout=remaining)
        stamp = getattr(getattr(msg, "header", None), "stamp", None)
        if stamp is None or stamp.to_sec() == 0.0 or stamp >= publish_stamp:
            return msg
    raise TimeoutError("timed out waiting for fresh message on %s" % topic)


if __name__ == "__main__":
    raise SystemExit(main())
