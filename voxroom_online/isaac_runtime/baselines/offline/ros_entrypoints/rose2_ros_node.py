from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from voxroom_online.isaac_runtime.baselines.ros_bridge_payload import load_input_arrays, load_request, write_result
from voxroom_online.isaac_runtime.baselines.offline.rose2_runner import (
    ORIGINAL_ROSE2_COMMIT_INSPECTED,
    ORIGINAL_ROSE2_CONFIRMED_CONTRACT,
    ORIGINAL_ROSE2_PACKAGE,
    ORIGINAL_ROSE2_REPO_URL,
    ROSE2_INPUT_OCCUPANCY_VALUES,
    build_occupancy_grid_values,
    build_output_domain,
    enforce_room_mask_contract,
    infer_grid_resolution_origin,
    polygon_array_to_label_map,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish one VoxRoom snapshot to original ROSE2 ROS topics.")
    parser.add_argument("--request", required=True)
    args = parser.parse_args(argv)
    request = load_request(Path(args.request))
    arrays = load_input_arrays(request)
    params = dict(request.get("params") or {})
    result = run_rose2_snapshot(request=request, arrays=arrays, params=params)
    write_result(request, label_map=result["label_map"], metadata=result["metadata"], debug_arrays=result["debug_arrays"])
    print(json.dumps(result["metadata"], ensure_ascii=False, sort_keys=True))
    return 0


def run_rose2_snapshot(
    *,
    request: Mapping[str, Any],
    arrays: Mapping[str, Any],
    params: Mapping[str, Any],
) -> dict[str, Any]:
    import rospy
    from geometry_msgs.msg import Point, Pose, Quaternion
    from jsk_recognition_msgs.msg import PolygonArray
    from nav_msgs.msg import OccupancyGrid
    from std_msgs.msg import Header

    input_topic = str(params.get("input_topic") or "/map")
    output_topic = str(params.get("output_topic") or "/rooms")
    result_timeout_s = float(params.get("result_timeout_s") or 180.0)
    startup_timeout_s = float(params.get("startup_timeout_s") or 30.0)
    if not rospy.core.is_initialized():
        rospy.init_node("voxroom_rose2_replay", anonymous=True, disable_signals=True)

    occupancy_values, occupancy_meta = build_occupancy_grid_values(arrays)
    resolution_m, origin_xy, grid_meta = infer_grid_resolution_origin(
        arrays,
        default_resolution_m=float(params.get("default_resolution_m") or 1.0),
        default_origin_xy=tuple(params.get("default_origin_xy") or (0.0, 0.0)),  # type: ignore[arg-type]
    )
    grid = OccupancyGrid()
    grid.header = Header()
    grid.header.frame_id = "map"
    publish_stamp = rospy.Time.now()
    grid.header.stamp = publish_stamp
    grid.info.resolution = float(resolution_m)
    grid.info.height = int(occupancy_values.shape[0])
    grid.info.width = int(occupancy_values.shape[1])
    grid.info.origin = Pose()
    grid.info.origin.position = Point(float(origin_xy[0]), float(origin_xy[1]), 0.0)
    grid.info.origin.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
    grid.data = [int(v) for v in np.asarray(occupancy_values, dtype=np.int16).reshape(-1)]

    rose_precheck = _precheck_original_rose_stage(grid)
    if rose_precheck.get("skip_reason"):
        labels = enforce_room_mask_contract(np.zeros(occupancy_values.shape, dtype=np.int32), arrays)
        metadata = _base_metadata(
            request=request,
            params=params,
            occupancy_meta=occupancy_meta,
            grid_meta=grid_meta,
            resolution_m=resolution_m,
            origin_xy=origin_xy,
            input_topic=input_topic,
            output_topic=output_topic,
            rooms_polygons_received=0,
            rooms_after_contract=0,
            output_source="original_rose_stage_no_features",
        )
        metadata.update(
            {
                "valid_zero_room_output": True,
                "zero_room_reason": str(rose_precheck["skip_reason"]),
                "original_rose_precheck": rose_precheck,
            }
        )
        return {"label_map": labels, "metadata": metadata, "debug_arrays": {"rose2_ros_occupancy_grid": occupancy_values}}

    publisher = rospy.Publisher(input_topic, OccupancyGrid, queue_size=1, latch=True)
    deadline = rospy.Time.now() + rospy.Duration(startup_timeout_s)
    while publisher.get_num_connections() < 1 and rospy.Time.now() < deadline and not rospy.is_shutdown():
        rospy.sleep(0.05)
    if publisher.get_num_connections() < 1:
        raise TimeoutError("ROSE2 has no subscriber on %s" % input_topic)
    publisher.publish(grid)
    rooms_msg = _wait_for_fresh_message(rospy, output_topic, PolygonArray, publish_stamp=publish_stamp, timeout_s=result_timeout_s)
    domain = build_output_domain(arrays, occupancy_values.shape)
    labels = polygon_array_to_label_map(
        rooms_msg,
        shape=occupancy_values.shape,
        resolution_m=resolution_m,
        origin_xy=origin_xy,
        domain=domain,
    )
    labels = enforce_room_mask_contract(labels, arrays)
    metadata = _base_metadata(
        request=request,
        params=params,
        occupancy_meta=occupancy_meta,
        grid_meta=grid_meta,
        resolution_m=resolution_m,
        origin_xy=origin_xy,
        input_topic=input_topic,
        output_topic=output_topic,
        rooms_polygons_received=int(len(getattr(rooms_msg, "polygons", []) or [])),
        rooms_after_contract=int(len([v for v in np.unique(labels) if int(v) > 0])),
        output_source=output_topic,
    )
    metadata["original_rose_precheck"] = rose_precheck
    return {"label_map": labels, "metadata": metadata, "debug_arrays": {"rose2_ros_occupancy_grid": occupancy_values}}


def _base_metadata(
    *,
    request: Mapping[str, Any],
    params: Mapping[str, Any],
    occupancy_meta: Mapping[str, Any],
    grid_meta: Mapping[str, Any],
    resolution_m: float,
    origin_xy: tuple[float, float],
    input_topic: str,
    output_topic: str,
    rooms_polygons_received: int,
    rooms_after_contract: int,
    output_source: str,
) -> dict[str, Any]:
    metadata = {
        "method": "rose2",
        "runner_type": "original_ros",
        "main_experiment_allowed": True,
        "source_snapshot": str(request.get("snapshot_path")),
        "scene_id": request.get("scene_id"),
        "original_repo": "aislabunimi/ROSE2",
        "original_repo_url": ORIGINAL_ROSE2_REPO_URL,
        "original_repo_commit": params.get("original_repo_commit"),
        "original_repo_reference_commit": ORIGINAL_ROSE2_COMMIT_INSPECTED,
        "original_commit_inspected": ORIGINAL_ROSE2_COMMIT_INSPECTED,
        "ros_package": params.get("rose_package") or ORIGINAL_ROSE2_PACKAGE,
        "launch_file": params.get("launch_file"),
        "input_topic": input_topic,
        "output_topic": output_topic,
        "output_source": output_topic,
        "output_topic_confirmed_from_source": True,
        "features_topic": "/features_ROSE2",
        "services": {"/ROSESrv": "rose2/ROSE", "/ROSE2Srv": "rose2/ROSE2"},
        "input_message_type": "nav_msgs/OccupancyGrid",
        "output_message_type": "jsk_recognition_msgs/PolygonArray",
        "input_occupancy_values": dict(ROSE2_INPUT_OCCUPANCY_VALUES),
        "input_grid": {**occupancy_meta, **grid_meta},
        "map_resolution_m": float(resolution_m),
        "map_origin_xy_m": [float(origin_xy[0]), float(origin_xy[1])],
        "rooms_message_type": "jsk_recognition_msgs/PolygonArray",
        "rooms_polygons_received": int(rooms_polygons_received),
        "freshness_guard": "header_stamp_at_or_after_published_map",
        "rooms_after_contract": int(rooms_after_contract),
        "confirmed_contract": ORIGINAL_ROSE2_CONFIRMED_CONTRACT,
        "output_source": str(output_source),
    }
    return metadata


def _precheck_original_rose_stage(grid: Any) -> dict[str, Any]:
    """Mirror ROSE's own early-return gates so skipped maps become zero-room outputs."""
    try:
        _ensure_original_rose_src_on_path()
        from PIL import Image as ImagePIL
        from skimage.util import img_as_ubyte
        from util import MsgUtils as mu
        from rose_v1_repo.fft_structure_extraction import FFTStructureExtraction as structure_extraction

        img_map = mu.fromOccupancyGridToImg(grid)
        grid_map = img_as_ubyte(ImagePIL.fromarray(img_map))
        rose = structure_extraction(grid_map, peak_height=0.2, par=50)
        rose.process_map()
        main_directions = list(getattr(rose, "main_directions", []) or [])
        result: dict[str, Any] = {
            "main_directions_count": int(len(main_directions)),
            "skip_reason": None,
        }
        if len(main_directions) <= 2:
            result["skip_reason"] = "original_rose_not_enough_directions"
            return result
        rose.simple_filter_map(0.18)
        rose.generate_initial_hypothesis_simple()
        try:
            rose.find_walls_flood_filing()
        except ValueError:
            result["skip_reason"] = "original_rose_cannot_find_walls"
            return result
        return result
    except Exception as exc:
        return {
            "main_directions_count": None,
            "skip_reason": None,
            "precheck_error": "%s: %s" % (type(exc).__name__, exc),
        }


def _ensure_original_rose_src_on_path() -> None:
    candidates: list[Path] = []
    root_env = os.environ.get("ROSE2_ROOT")
    if root_env:
        root = Path(root_env).expanduser()
        candidates.extend([root / "src", root])
        workspace = root.parent.parent
        for package_dir in sorted((workspace / "devel" / "lib").glob("python*/site-packages")):
            package_path = str(package_dir)
            if package_path not in sys.path:
                sys.path.insert(0, package_path)
    for item in os.environ.get("ROS_PACKAGE_PATH", "").split(os.pathsep):
        if not item:
            continue
        base = Path(item).expanduser()
        candidates.extend([base / "ROSE2" / "src", base / "src"])
        workspace = base.parent
        for package_dir in sorted((workspace / "devel" / "lib").glob("python*/site-packages")):
            package_path = str(package_dir)
            if package_path not in sys.path:
                sys.path.insert(0, package_path)
    for candidate in candidates:
        if (candidate / "util" / "MsgUtils.py").exists():
            path = str(candidate)
            if path not in sys.path:
                sys.path.insert(0, path)
            return


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
