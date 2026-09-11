#!/usr/bin/env python3
"""Replay one saved checkpoint through CMU SysNav room segmentation.

The SysNav implementation is kept unmodified.  This adapter converts the saved
3-D occupancy state and 2-D navigation-free mask into all three point-cloud
streams consumed by the official ROS 2 node, subscribes to its integer room
mask and accepted watershed-door cloud, and projects both 0.1 m outputs back
to the checkpoint grid.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header


SYSNAV_RESOLUTION_M = 0.1
SYSNAV_AUXILIARY_RESOLUTION_M = 0.2
SYSNAV_COMMIT = "0fa15cc8bc18be7409272fb65fa514a9a5bca6b0"
VOXEL_OCCUPIED = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--sysnav-root", type=Path, default=Path("/home/joey/SysNav")
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--subscriber-timeout",
        type=float,
        default=15.0,
        help="Seconds allowed for ROS/DDS subscription discovery.",
    )
    parser.add_argument("--dilation-iteration", type=int, default=6)
    parser.add_argument("--ceiling-height", type=float, default=2.7)
    parser.add_argument(
        "--region-growing-radius",
        type=float,
        default=0.0,
        help="Plane-fit radius in metres; <=0 covers the entire snapshot map.",
    )
    parser.add_argument("--normal-search-num", type=int, default=20)
    parser.add_argument("--normal-search-radius", type=float, default=0.3)
    return parser.parse_args()


def _scalar(arrays: np.lib.npyio.NpzFile, key: str, default):
    return arrays[key].item() if key in arrays.files else default


def load_input_clouds(snapshot: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(snapshot, allow_pickle=False) as arrays:
        required = {
            "voxel_occupancy_state_zyx",
            "voxel_occupancy_z_centers_m",
            "voxel_nav_free_xy",
            "map_resolution_m",
        }
        missing = sorted(required.difference(arrays.files))
        if missing:
            raise KeyError(f"snapshot lacks full SysNav input: {missing}")

        state = np.asarray(arrays["voxel_occupancy_state_zyx"], dtype=np.uint8)
        z_centers = np.asarray(arrays["voxel_occupancy_z_centers_m"], dtype=np.float32)
        nav_free = np.asarray(arrays["voxel_nav_free_xy"], dtype=bool)
        resolution = float(_scalar(arrays, "map_resolution_m", 0.05))
        if "map_bounds_xyxy_m" in arrays.files:
            bounds = np.asarray(arrays["map_bounds_xyxy_m"], dtype=np.float64)
        elif all(
            key in arrays.files
            for key in ("map_min_x_m", "map_min_y_m", "map_max_x_m", "map_max_y_m")
        ):
            bounds = np.asarray(
                [
                    _scalar(arrays, "map_min_x_m", 0.0),
                    _scalar(arrays, "map_min_y_m", 0.0),
                    _scalar(arrays, "map_max_x_m", 0.0),
                    _scalar(arrays, "map_max_y_m", 0.0),
                ],
                dtype=np.float64,
            )
        elif "map_origin_xy_m" in arrays.files:
            origin = np.asarray(arrays["map_origin_xy_m"], dtype=np.float64)
            bounds = np.asarray(
                [
                    origin[0],
                    origin[1],
                    origin[0] + nav_free.shape[1] * resolution,
                    origin[1] + nav_free.shape[0] * resolution,
                ],
                dtype=np.float64,
            )
        else:
            raise KeyError("snapshot lacks map bounds/origin metadata")
        pose = np.asarray(
            arrays["base_pose_world_xyzyaw"]
            if "base_pose_world_xyzyaw" in arrays.files
            else arrays["demo_pose_world"]
            if "demo_pose_world" in arrays.files
            else np.zeros(4),
            dtype=np.float64,
        )
        event_id = str(_scalar(arrays, "roomseg_eval_event_id", "unknown"))
        coverage = float(_scalar(arrays, "roomseg_eval_coverage_ratio", float("nan")))

    if state.ndim != 3 or state.shape[1:] != nav_free.shape:
        raise ValueError(
            f"incompatible voxel/nav shapes: {state.shape} versus {nav_free.shape}"
        )
    if z_centers.shape != (state.shape[0],):
        raise ValueError("voxel z centers do not match voxel state")
    if bounds.shape != (4,):
        raise ValueError("map_bounds_xyxy_m must contain xmin,ymin,xmax,ymax")

    xmin, ymin, xmax, ymax = map(float, bounds)
    height, width = nav_free.shape
    expected_xmax = xmin + width * resolution
    expected_ymax = ymin + height * resolution
    if abs(expected_xmax - xmax) > resolution or abs(expected_ymax - ymax) > resolution:
        raise ValueError("map bounds and array shape are inconsistent")

    # Convert source cells to world coordinates, then quantize exactly as the
    # official node's 0.1 m VoxelGrid.  Keeping one point per target voxel makes
    # replay much smaller without changing the surface given to SysNav.
    zz, rr, cc = np.nonzero(state == VOXEL_OCCUPIED)
    occ_x = xmin + (cc.astype(np.float32) + 0.5) * resolution
    occ_y = ymin + (rr.astype(np.float32) + 0.5) * resolution
    occ_z = z_centers[zz]

    # The official simulation configuration rejects points above robot+2.7 m.
    robot_z = float(pose[2]) if pose.size >= 3 else 0.0
    keep_z = occ_z <= robot_z + 2.7
    occ_x, occ_y, occ_z = occ_x[keep_z], occ_y[keep_z], occ_z[keep_z]

    qx = np.floor(occ_x / SYSNAV_RESOLUTION_M).astype(np.int32)
    qy = np.floor(occ_y / SYSNAV_RESOLUTION_M).astype(np.int32)
    qz = np.floor(occ_z / SYSNAV_RESOLUTION_M).astype(np.int32)
    quantized = np.stack((qx, qy, qz), axis=1)
    quantized = np.unique(quantized, axis=0)
    occupied_points = np.empty((quantized.shape[0], 4), dtype=np.float32)
    occupied_points[:, :3] = (
        quantized.astype(np.float32) + np.float32(0.5)
    ) * np.float32(SYSNAV_RESOLUTION_M)
    occupied_points[:, 3] = 1.0

    free_r, free_c = np.nonzero(nav_free)
    free_x = xmin + (free_c.astype(np.float32) + 0.5) * resolution
    free_y = ymin + (free_r.astype(np.float32) + 0.5) * resolution
    free_qx = np.floor(free_x / SYSNAV_RESOLUTION_M).astype(np.int32)
    free_qy = np.floor(free_y / SYSNAV_RESOLUTION_M).astype(np.int32)
    free_quantized = np.unique(np.stack((free_qx, free_qy), axis=1), axis=0)
    floor_points = np.empty((free_quantized.shape[0], 4), dtype=np.float32)
    floor_points[:, 0:2] = (
        free_quantized.astype(np.float32) + np.float32(0.5)
    ) * np.float32(SYSNAV_RESOLUTION_M)
    floor_points[:, 2] = np.float32(robot_z - 0.05)
    floor_points[:, 3] = 1.0

    registered_points = np.concatenate((occupied_points, floor_points), axis=0)

    # SysNav receives these as independent streams in the official pipeline.
    # The freespace stream is a 2-D viewpoint cloud at robot height, while the
    # occupied stream contains occupancy-grid cells around the robot height and
    # uses intensity 0 for occupied cells.
    # The official viewpoint manager and rolling occupancy grid both operate at
    # 0.2 m, not at the room segmenter's 0.1 m surface resolution.  A free
    # auxiliary cell is emitted only when every source cell covered by that
    # 0.2 m cell is navigation-free; this avoids turning wall-straddling cells
    # into false freespace updates.
    all_r, all_c = np.indices(nav_free.shape, dtype=np.int32)
    all_x = xmin + (all_c.reshape(-1).astype(np.float32) + 0.5) * resolution
    all_y = ymin + (all_r.reshape(-1).astype(np.float32) + 0.5) * resolution
    all_qx = np.floor(all_x / SYSNAV_AUXILIARY_RESOLUTION_M).astype(np.int32)
    all_qy = np.floor(all_y / SYSNAV_AUXILIARY_RESOLUTION_M).astype(np.int32)
    all_q = np.stack((all_qx, all_qy), axis=1)
    unique_all_q, all_inverse, all_counts = np.unique(
        all_q, axis=0, return_inverse=True, return_counts=True
    )
    free_counts = np.bincount(
        all_inverse,
        weights=nav_free.reshape(-1).astype(np.int32),
        minlength=all_counts.size,
    )
    free_aux_q = unique_all_q[free_counts == all_counts]
    freespace_points = np.empty((free_aux_q.shape[0], 4), dtype=np.float32)
    freespace_points[:, 0:2] = (
        free_aux_q.astype(np.float32) + np.float32(0.5)
    ) * np.float32(SYSNAV_AUXILIARY_RESOLUTION_M)
    freespace_points[:, 2] = np.float32(robot_z)
    freespace_points[:, 3] = 1.0

    occupied_state_source = occupied_points[
        (occupied_points[:, 2] >= robot_z - 0.5)
        & (occupied_points[:, 2] <= robot_z + 0.5)
    ]
    occupied_aux_q = np.floor(
        occupied_state_source[:, :3] / SYSNAV_AUXILIARY_RESOLUTION_M
    ).astype(np.int32)
    occupied_aux_q = np.unique(occupied_aux_q, axis=0)
    occupied_state_points = np.empty((occupied_aux_q.shape[0], 4), dtype=np.float32)
    occupied_state_points[:, :3] = (
        occupied_aux_q.astype(np.float32) + np.float32(0.5)
    ) * np.float32(SYSNAV_AUXILIARY_RESOLUTION_M)
    occupied_state_points[:, 3] = 0.0

    if registered_points.size:
        max_radius = float(
            np.sqrt(
                np.max(
                    (registered_points[:, 0] - float(pose[0])) ** 2
                    + (registered_points[:, 1] - float(pose[1])) ** 2
                )
            )
        )
    else:
        max_radius = 15.0
    max_abs_xy = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax))
    dimension_xy = int(math.ceil((2.0 * max_abs_xy + 4.0) / SYSNAV_RESOLUTION_M))
    if dimension_xy % 2:
        dimension_xy += 1
    dimension_z = max(80, int(math.ceil(2.0 * (float(np.max(np.abs(z_centers))) + 1.0) / SYSNAV_RESOLUTION_M)))
    if dimension_z % 2:
        dimension_z += 1

    metadata = {
        "snapshot": str(snapshot.resolve()),
        "source_shape_yx": [height, width],
        "source_resolution_m": resolution,
        "source_bounds_xyxy_m": bounds.tolist(),
        "base_pose_world_xyzyaw": pose.tolist(),
        "event_id": event_id,
        "coverage_ratio": coverage,
        "sysnav_resolution_m": SYSNAV_RESOLUTION_M,
        "sysnav_auxiliary_resolution_m": SYSNAV_AUXILIARY_RESOLUTION_M,
        "sysnav_room_xy": dimension_xy,
        "sysnav_room_z": dimension_z,
        "occupied_surface_point_count": int(occupied_points.shape[0]),
        "navigation_floor_point_count": int(floor_points.shape[0]),
        "registered_scan_point_count": int(registered_points.shape[0]),
        "freespace_cloud_point_count": int(freespace_points.shape[0]),
        "occupied_cloud_point_count": int(occupied_state_points.shape[0]),
        "full_snapshot_region_radius_m": max(15.0, max_radius + 1.0),
    }
    return {
        "registered_scan": registered_points,
        "freespace_cloud": freespace_points,
        "occupied_cloud": occupied_state_points,
    }, metadata


def make_cloud(node: Node, points: np.ndarray) -> PointCloud2:
    header = Header()
    header.stamp = node.get_clock().now().to_msg()
    header.frame_id = "map"
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    return point_cloud2.create_cloud(header, fields, np.ascontiguousarray(points))


class ReplayNode(Node):
    def __init__(self) -> None:
        super().__init__("voxroom_sysnav_checkpoint_replay")
        self.registered_scan_publisher = self.create_publisher(PointCloud2, "/registered_scan", 20)
        self.freespace_publisher = self.create_publisher(PointCloud2, "/freespace_cloud", 20)
        self.occupied_publisher = self.create_publisher(PointCloud2, "/occupied_cloud", 20)
        self.odom_publisher = self.create_publisher(Odometry, "/state_estimation", 20)
        self.room_mask: np.ndarray | None = None
        self.door_cloud: np.ndarray | None = None
        self.room_mask_subscription = self.create_subscription(
            Image, "/room_mask", self._mask_cb, 5
        )
        self.door_cloud_subscription = self.create_subscription(
            PointCloud2, "/door_cloud", self._door_cloud_cb, 5
        )

    def _mask_cb(self, msg: Image) -> None:
        if msg.encoding != "32SC1":
            raise RuntimeError(f"unexpected SysNav room-mask encoding {msg.encoding!r}")
        values = np.frombuffer(bytes(msg.data), dtype=np.int32)
        self.room_mask = values.reshape(int(msg.height), int(msg.width)).copy()

    def _door_cloud_cb(self, msg: PointCloud2) -> None:
        """Capture SysNav's post-filter watershed door points verbatim.

        `door_cloud_` is populated from `door_mask = (markers == -1)` after
        the official code discards watershed ridges that do not border exactly
        two rooms.  It is consequently the algorithm's actual room-separation
        output, unlike an arbitrary band inferred from final labels.
        """
        points = point_cloud2.read_points(
            msg,
            field_names=("x", "y", "z"),
            skip_nans=True,
        )
        # ROS Jazzy returns a structured NumPy array here (earlier releases
        # returned an iterator of tuples), so select named fields explicitly.
        if isinstance(points, np.ndarray) and points.dtype.names is not None:
            self.door_cloud = np.column_stack(
                (points["x"], points["y"], points["z"])
            ).astype(np.float32, copy=False)
        else:
            point_rows = list(points)
            self.door_cloud = (
                np.asarray(point_rows, dtype=np.float32).reshape((-1, 3))
                if point_rows
                else np.empty((0, 3), dtype=np.float32)
            )


def project_room_mask(room_mask_xy: np.ndarray, metadata: dict) -> np.ndarray:
    height, width = metadata["source_shape_yx"]
    resolution = float(metadata["source_resolution_m"])
    xmin, ymin, _, _ = metadata["source_bounds_xyxy_m"]
    dim_x, dim_y = room_mask_xy.shape
    cols = np.arange(width, dtype=np.float64)
    rows = np.arange(height, dtype=np.float64)
    world_x = xmin + (cols + 0.5) * resolution
    world_y = ymin + (rows + 0.5) * resolution
    ix = np.floor(world_x / SYSNAV_RESOLUTION_M + dim_x / 2.0).astype(np.int64)
    iy = np.floor(world_y / SYSNAV_RESOLUTION_M + dim_y / 2.0).astype(np.int64)
    valid_x = (ix >= 0) & (ix < dim_x)
    valid_y = (iy >= 0) & (iy < dim_y)
    output = np.zeros((height, width), dtype=np.int32)
    output[np.ix_(valid_y, valid_x)] = room_mask_xy[np.ix_(ix[valid_x], iy[valid_y])].T
    return output


def project_door_cloud(door_cloud_xyz: np.ndarray, metadata: dict) -> np.ndarray:
    """Rasterize official 0.1 m accepted watershed ridges on the source grid."""
    height, width = metadata["source_shape_yx"]
    resolution = float(metadata["source_resolution_m"])
    xmin, ymin, _, _ = metadata["source_bounds_xyxy_m"]
    output = np.zeros((height, width), dtype=bool)
    if door_cloud_xyz.size == 0:
        return output

    # SysNav's voxel_to_point_cropped reports the lower-left corner of each
    # 0.1 m cell.  Mark every source cell that lies inside that same cell, so
    # the projected ridge preserves its original raster geometry instead of
    # becoming a sparse collection of point samples.
    start_cols = np.floor((door_cloud_xyz[:, 0] - xmin) / resolution).astype(np.int64)
    start_rows = np.floor((door_cloud_xyz[:, 1] - ymin) / resolution).astype(np.int64)
    cells_per_sysnav_pixel = max(1, int(round(SYSNAV_RESOLUTION_M / resolution)))
    for row, col in zip(start_rows, start_cols, strict=False):
        r0, r1 = max(0, int(row)), min(height, int(row) + cells_per_sysnav_pixel)
        c0, c1 = max(0, int(col)), min(width, int(col) + cells_per_sysnav_pixel)
        if r0 < r1 and c0 < c1:
            output[r0:r1, c0:c1] = True
    return output


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    input_clouds, metadata = load_input_clouds(args.snapshot)
    executable = args.sysnav_root / "build/tare_planner/room_segmentation"
    if not executable.is_file():
        raise FileNotFoundError(f"compiled official SysNav node not found: {executable}")

    region_growing_radius = (
        float(args.region_growing_radius)
        if float(args.region_growing_radius) > 0
        else float(metadata["full_snapshot_region_radius_m"])
    )
    metadata.update(
        {
            "method": "cmu_sysnav_official_room_segmentation_full_snapshot_input",
            "sysnav_repository": "https://github.com/zwandering/SysNav",
            "sysnav_commit": SYSNAV_COMMIT,
            "adapter": str(Path(__file__).resolve()),
            "dilation_iteration": args.dilation_iteration,
            "ceiling_height_m": args.ceiling_height,
            "input_streams": [
                "/registered_scan",
                "/freespace_cloud",
                "/occupied_cloud",
                "/state_estimation",
            ],
            "region_growing_radius_m": region_growing_radius,
            "region_growing_scope": "full_snapshot_map",
            "normal_search_num": args.normal_search_num,
            "normal_search_radius_m": args.normal_search_radius,
        }
    )
    log_path = args.output.with_suffix(".sysnav.log")
    log_handle = log_path.open("w", encoding="utf-8")
    command = [
        str(executable),
        "--ros-args",
        "-p", f"room_resolution:={SYSNAV_RESOLUTION_M}",
        "-p", f"room_x:={metadata['sysnav_room_xy']}",
        "-p", f"room_y:={metadata['sysnav_room_xy']}",
        "-p", f"room_z:={metadata['sysnav_room_z']}",
        "-p", f"dilation_iteration:={args.dilation_iteration}",
        "-p", f"ceilingHeight_:={args.ceiling_height}",
        "-p", f"region_growing_radius:={region_growing_radius}",
        "-p", f"normal_search_num:={args.normal_search_num}",
        "-p", f"normal_search_radius:={args.normal_search_radius}",
        "-p", "isDebug:=false",
    ]
    process = subprocess.Popen(
        command,
        cwd=args.output.parent,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    node: ReplayNode | None = None
    try:
        rclpy.init(args=None)
        node = ReplayNode()
        subscriber_deadline = time.monotonic() + args.subscriber_timeout
        while not all(
            publisher.get_subscription_count() > 0
            for publisher in (
                node.registered_scan_publisher,
                node.freespace_publisher,
                node.occupied_publisher,
            )
        ):
            if process.poll() is not None:
                raise RuntimeError(f"SysNav exited before subscribing; see {log_path}")
            if time.monotonic() >= subscriber_deadline:
                raise TimeoutError("timed out waiting for SysNav subscriber")
            rclpy.spin_once(node, timeout_sec=0.1)

        deadline = time.monotonic() + args.timeout

        pose = metadata["base_pose_world_xyzyaw"]
        odom = Odometry()
        odom.header.stamp = node.get_clock().now().to_msg()
        odom.header.frame_id = "map"
        odom.pose.pose.position.x = float(pose[0])
        odom.pose.pose.position.y = float(pose[1])
        odom.pose.pose.position.z = float(pose[2])
        for _ in range(3):
            node.odom_publisher.publish(odom)
            rclpy.spin_once(node, timeout_sec=0.05)

        # Supply the two occupancy-state streams before the scan pass.  This
        # lets the unmodified node populate freespace_indices_ and state_map_
        # before it extracts walls from the registered surface cloud.
        node.freespace_publisher.publish(make_cloud(node, input_clouds["freespace_cloud"]))
        rclpy.spin_once(node, timeout_sec=0.25)
        node.occupied_publisher.publish(make_cloud(node, input_clouds["occupied_cloud"]))
        rclpy.spin_once(node, timeout_sec=0.25)

        # Exactly five scan callbacks trigger one segmentation pass. Splitting
        # rather than duplicating preserves the complete snapshot surface.
        for chunk in np.array_split(input_clouds["registered_scan"], 5):
            node.registered_scan_publisher.publish(make_cloud(node, chunk))
            rclpy.spin_once(node, timeout_sec=0.15)

        while node.room_mask is None:
            if process.poll() is not None:
                raise RuntimeError(f"SysNav exited before publishing; see {log_path}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"SysNav did not publish a mask; see {log_path}")
            rclpy.spin_once(node, timeout_sec=0.2)

        # SysNav publishes /room_mask at the end of roomSegmentation(), then
        # immediately publishes /door_cloud from the same accepted watershed
        # mask.  Keep the adapter alive long enough to receive that second
        # message rather than replacing it with a derived separator heuristic.
        door_deadline = min(deadline, time.monotonic() + 5.0)
        while node.door_cloud is None:
            if process.poll() is not None:
                raise RuntimeError(
                    f"SysNav exited before publishing its door cloud; see {log_path}"
                )
            if time.monotonic() >= door_deadline:
                raise TimeoutError(
                    f"SysNav did not publish its door cloud; see {log_path}"
                )
            rclpy.spin_once(node, timeout_sec=0.2)

        projected = project_room_mask(node.room_mask, metadata)
        projected_door_lines = project_door_cloud(node.door_cloud, metadata)
        with np.load(args.snapshot, allow_pickle=False) as arrays:
            nav_free = np.asarray(arrays["voxel_nav_free_xy"], dtype=bool)
        projected[~nav_free] = 0
        metadata["raw_sysnav_room_count"] = int(
            np.unique(node.room_mask[node.room_mask > 0]).size
        )
        metadata["projected_room_count"] = int(np.unique(projected[projected > 0]).size)
        metadata["projected_labeled_cell_count"] = int(np.count_nonzero(projected))
        metadata["official_watershed_door_point_count"] = int(node.door_cloud.shape[0])
        metadata["projected_official_watershed_door_cell_count"] = int(
            np.count_nonzero(projected_door_lines)
        )
        np.savez_compressed(
            args.output,
            sysnav_room_label_map=projected,
            sysnav_door_line_map=projected_door_lines,
            source_snapshot=np.asarray(str(args.snapshot.resolve())),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        args.output.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(metadata, sort_keys=True))
        return 0
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        log_handle.close()


if __name__ == "__main__":
    sys.exit(main())
