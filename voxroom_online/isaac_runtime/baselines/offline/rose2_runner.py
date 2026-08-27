from __future__ import annotations

import argparse
import json
import os
import pickle
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from voxroom_online.isaac_runtime.baselines.offline.base import BaselineResult, MissingOriginalImplementationError
    from voxroom_online.isaac_runtime.baselines.ros_subprocess import RosSubprocessConfig, run_ros_module
except Exception:  # pragma: no cover - keeps this module importable before the shared baseline package exists.
    @dataclass
    class BaselineResult:  # type: ignore[no-redef]
        label_map: np.ndarray
        metadata: dict[str, Any]
        debug_arrays: dict[str, np.ndarray] | None = None

    class MissingOriginalImplementationError(RuntimeError):  # type: ignore[no-redef]
        pass

    RosSubprocessConfig = None  # type: ignore[assignment]
    run_ros_module = None  # type: ignore[assignment]


ORIGINAL_ROSE2_REPO_URL = "https://github.com/aislabunimi/ROSE2.git"
ORIGINAL_ROSE2_COMMIT_INSPECTED = "3a010b9e6bb2477de3b5b46208ebfccd71dfafbf"
ORIGINAL_ROSE2_PACKAGE = "rose2"
ORIGINAL_ROSE2_LAUNCH = "ROSE.launch"

ROSE2_INPUT_OCCUPANCY_VALUES = {"free": 0, "occupied": 100, "unknown": -1}

ORIGINAL_ROSE2_CONFIRMED_CONTRACT: dict[str, Any] = {
    "launch_file": "ROSE.launch",
    "nodes": {
        "ROSE": "FeatureExtractorROSE.py",
        "ROSE2": "FeatureExtractorROSE2.py",
        "rviz_player": "rviz",
    },
    "input_topic": "/map",
    "input_message_type": "nav_msgs/OccupancyGrid",
    "intermediate_topics": {
        "/features_ROSE": "rose2/ROSEFeatures",
        "/clean_map": "nav_msgs/OccupancyGrid",
        "/direction_markers": "visualization_msgs/MarkerArray",
    },
    "output_topics": {
        "/features_ROSE2": "rose2/ROSE2Features",
        "/rooms": "jsk_recognition_msgs/PolygonArray",
        "/extended_lines": "visualization_msgs/Marker",
        "/edges": "visualization_msgs/MarkerArray",
    },
    "services": {
        "/ROSESrv": "rose2/ROSE",
        "/ROSE2Srv": "rose2/ROSE2",
    },
    "message_files": {
        "ROSEFeatures.msg": ["nav_msgs/OccupancyGrid originalMap", "nav_msgs/OccupancyGrid cleanMap", "float32[] directions"],
        "ROSE2Features.msg": ["ExtendedLine[] lines", "Edge[] edges", "Room[] rooms", "Contour contour"],
        "Room.msg": ["int8[] bytes"],
        "ROSE.srv": ["nav_msgs/OccupancyGrid map", "rose2/ROSEFeatures features"],
        "ROSE2.srv": ["rose2/ROSEFeatures features1", "rose2/ROSE2Features features2"],
    },
}


class Rose2DependencyError(MissingOriginalImplementationError):
    pass


@dataclass(frozen=True)
class Rose2Result:
    final_room_label_map: np.ndarray
    metadata: dict[str, Any]


def inspect_ros_dependency_status() -> dict[str, Any]:
    commands = {
        "roslaunch": shutil.which("roslaunch"),
        "roscore": shutil.which("roscore"),
        "catkin_make": shutil.which("catkin_make"),
        "catkin": shutil.which("catkin"),
    }
    modules = {name: _probe_import(name) for name in ("rospy", "nav_msgs.msg", "geometry_msgs.msg", "jsk_recognition_msgs.msg")}
    missing_commands = [name for name in ("roslaunch", "roscore") if commands.get(name) is None]
    missing_modules = [name for name, available in modules.items() if not available]
    return {
        "commands": commands,
        "python_modules": modules,
        "missing_commands": missing_commands,
        "missing_python_modules": missing_modules,
        "original_repo_url": ORIGINAL_ROSE2_REPO_URL,
        "original_commit_inspected": ORIGINAL_ROSE2_COMMIT_INSPECTED,
        "confirmed_contract": ORIGINAL_ROSE2_CONFIRMED_CONTRACT,
        "package_dependency_gap": "ROSE2 imports jsk_recognition_msgs but package.xml inspected at the commit above does not declare it.",
    }


def build_occupancy_grid_values(arrays: Mapping[str, np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    shape = infer_snapshot_shape(arrays)
    occupied = _first_bool_array(arrays, ("obstacle_mask", "occupancy_map", "voxel_nav_occupied_xy", "voxel_wall_xy"), shape)
    free, free_key = _first_nonempty_bool_array(
        arrays,
        ("navigation_free_room_domain", "observed_free_mask", "voxel_nav_free_xy", "vertical_free_room_domain"),
        shape,
    )
    unknown = _first_bool_array(arrays, ("unknown_mask", "voxel_nav_unknown_xy"), shape)
    if unknown is None:
        unknown = ~(free | occupied)

    data = np.full(shape, ROSE2_INPUT_OCCUPANCY_VALUES["unknown"], dtype=np.int16)
    data[free] = ROSE2_INPUT_OCCUPANCY_VALUES["free"]
    data[occupied] = ROSE2_INPUT_OCCUPANCY_VALUES["occupied"]
    data[unknown & ~occupied] = ROSE2_INPUT_OCCUPANCY_VALUES["unknown"]

    metadata = {
        "shape_hw": [int(shape[0]), int(shape[1])],
        "free_source_key": free_key,
        "occupied_count": int(np.count_nonzero(data == ROSE2_INPUT_OCCUPANCY_VALUES["occupied"])),
        "free_count": int(np.count_nonzero(data == ROSE2_INPUT_OCCUPANCY_VALUES["free"])),
        "unknown_count": int(np.count_nonzero(data == ROSE2_INPUT_OCCUPANCY_VALUES["unknown"])),
    }
    return data, metadata


def infer_snapshot_shape(arrays: Mapping[str, np.ndarray]) -> tuple[int, int]:
    for key in (
        "final_room_label_map",
        "navigation_free_room_domain",
        "observed_free_mask",
        "occupancy_map",
        "obstacle_mask",
        "unknown_mask",
        "voxel_nav_free_xy",
    ):
        value = arrays.get(key)
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.ndim == 2:
            return (int(arr.shape[0]), int(arr.shape[1]))
    raise ValueError("cannot infer 2D snapshot shape for ROSE2")


def infer_grid_resolution_origin(
    arrays: Mapping[str, np.ndarray],
    *,
    default_resolution_m: float = 1.0,
    default_origin_xy: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, tuple[float, float], dict[str, Any]]:
    resolution_keys = (
        "map_resolution_m",
        "resolution_m",
        "online_map_resolution_m",
        "online_resolution_m",
        "occupancy_grid_resolution_m",
        "voxel_occupancy_xy_resolution_m",
    )
    origin_keys = (
        "map_origin_xy_m",
        "map_origin_xy",
        "origin_xy",
        "occupancy_grid_origin_xy",
        "voxel_occupancy_origin_xy",
    )
    resolution = None
    resolution_key = None
    for key in resolution_keys:
        value = arrays.get(key)
        if value is None:
            continue
        try:
            resolution = float(np.asarray(value).reshape(-1)[0])
            resolution_key = key
            break
        except Exception:
            continue
    if resolution is None or not np.isfinite(resolution) or resolution <= 0.0:
        resolution = float(default_resolution_m)
        resolution_key = "runner_default"

    origin = None
    origin_key = None
    for key in origin_keys:
        value = arrays.get(key)
        if value is None:
            continue
        flat = np.asarray(value, dtype=float).reshape(-1)
        if flat.size >= 2 and np.all(np.isfinite(flat[:2])):
            origin = (float(flat[0]), float(flat[1]))
            origin_key = key
            break
    if origin is None:
        x_value = arrays.get("map_origin_x_m")
        y_value = arrays.get("map_origin_y_m")
        if x_value is not None and y_value is not None:
            try:
                x = float(np.asarray(x_value).reshape(-1)[0])
                y = float(np.asarray(y_value).reshape(-1)[0])
                if np.isfinite(x) and np.isfinite(y):
                    origin = (x, y)
                    origin_key = "map_origin_x_m/map_origin_y_m"
            except Exception:
                origin = None
                origin_key = None
    if origin is None:
        origin = (float(default_origin_xy[0]), float(default_origin_xy[1]))
        origin_key = "runner_default"

    return float(resolution), origin, {
        "resolution_m": float(resolution),
        "resolution_source_key": str(resolution_key),
        "origin_xy": [float(origin[0]), float(origin[1])],
        "origin_source_key": str(origin_key),
    }


def polygon_array_to_label_map(
    polygon_array_msg: Any,
    *,
    shape: tuple[int, int],
    resolution_m: float,
    origin_xy: tuple[float, float],
    domain: np.ndarray | None = None,
) -> np.ndarray:
    labels = np.zeros(shape, dtype=np.int32)
    polygons = list(getattr(polygon_array_msg, "polygons", []) or [])
    for idx, polygon_stamped in enumerate(polygons, start=1):
        polygon = getattr(polygon_stamped, "polygon", polygon_stamped)
        points = getattr(polygon, "points", [])
        world_xy = [(float(point.x), float(point.y)) for point in points]
        if len(world_xy) < 3:
            continue
        mask = rasterize_world_polygon(world_xy, shape=shape, resolution_m=resolution_m, origin_xy=origin_xy)
        labels[mask] = int(idx)
    if domain is not None:
        labels[~np.asarray(domain, dtype=bool)] = 0
    return labels


def rose2_features_to_label_map(
    features_msg: Any,
    *,
    shape: tuple[int, int],
    resolution_m: float,
    origin_xy: tuple[float, float],
    domain: np.ndarray | None = None,
) -> np.ndarray:
    labels = np.zeros(shape, dtype=np.int32)
    rooms = list(getattr(features_msg, "rooms", []) or [])
    for idx, room_msg in enumerate(rooms, start=1):
        room = pickle.loads(_int8_sequence_to_bytes(getattr(room_msg, "bytes", [])))
        exterior = getattr(getattr(room, "exterior", None), "coords", None)
        if exterior is None:
            continue
        world_xy = _cell_coords_to_world(list(exterior), resolution_m=resolution_m, origin_xy=origin_xy)
        mask = rasterize_world_polygon(world_xy, shape=shape, resolution_m=resolution_m, origin_xy=origin_xy)
        for interior in getattr(room, "interiors", []) or []:
            hole_xy = _cell_coords_to_world(list(interior.coords), resolution_m=resolution_m, origin_xy=origin_xy)
            mask &= ~rasterize_world_polygon(hole_xy, shape=shape, resolution_m=resolution_m, origin_xy=origin_xy)
        labels[mask] = int(idx)
    if domain is not None:
        labels[~np.asarray(domain, dtype=bool)] = 0
    return labels


def rasterize_world_polygon(
    world_xy: Sequence[Sequence[float]],
    *,
    shape: tuple[int, int],
    resolution_m: float,
    origin_xy: tuple[float, float],
) -> np.ndarray:
    points = np.asarray(world_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
        return np.zeros(shape, dtype=bool)
    resolution = max(float(resolution_m), 1.0e-9)
    cols = (points[:, 0] - float(origin_xy[0])) / resolution
    rows = (points[:, 1] - float(origin_xy[1])) / resolution
    min_r = max(0, int(np.floor(np.min(rows))) - 1)
    max_r = min(int(shape[0]), int(np.ceil(np.max(rows))) + 2)
    min_c = max(0, int(np.floor(np.min(cols))) - 1)
    max_c = min(int(shape[1]), int(np.ceil(np.max(cols))) + 2)
    mask = np.zeros(shape, dtype=bool)
    if min_r >= max_r or min_c >= max_c:
        return mask
    rr, cc = np.mgrid[min_r:max_r, min_c:max_c]
    samples = np.column_stack((cc.reshape(-1) + 0.5, rr.reshape(-1) + 0.5))
    inside = _points_in_polygon(samples[:, 0], samples[:, 1], np.column_stack((cols, rows))).reshape(
        (max_r - min_r, max_c - min_c)
    )
    mask[min_r:max_r, min_c:max_c] = inside
    return mask


def _points_in_polygon(x: np.ndarray, y: np.ndarray, polygon_xy: np.ndarray) -> np.ndarray:
    polygon = np.asarray(polygon_xy, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] < 2:
        return np.zeros_like(np.asarray(x, dtype=np.float64), dtype=bool)
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    inside = np.zeros(x_values.shape, dtype=bool)
    x0 = polygon[:, 0]
    y0 = polygon[:, 1]
    x1 = np.roll(x0, 1)
    y1 = np.roll(y0, 1)
    for xa, ya, xb, yb in zip(x0, y0, x1, y1, strict=False):
        crosses = (ya > y_values) != (yb > y_values)
        x_intersections = (xb - xa) * (y_values - ya) / ((yb - ya) + 1.0e-12) + xa
        inside ^= crosses & (x_values < x_intersections)
    return inside


def enforce_room_mask_contract(label_map: np.ndarray, arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    shape = infer_snapshot_shape(arrays)
    labels = np.asarray(label_map, dtype=np.int32)
    if labels.shape != shape:
        raise ValueError("ROSE2 label map shape %s does not match snapshot shape %s" % (labels.shape, shape))
    domain = build_output_domain(arrays, shape)
    out = np.where(domain & (labels > 0), labels, 0).astype(np.int32)
    unique = [int(v) for v in np.unique(out) if int(v) > 0]
    relabeled = np.zeros_like(out, dtype=np.int32)
    for new_id, old_id in enumerate(unique, start=1):
        relabeled[out == old_id] = int(new_id)
    return relabeled


def build_output_domain(arrays: Mapping[str, np.ndarray], shape: tuple[int, int] | None = None) -> np.ndarray:
    if shape is None:
        shape = infer_snapshot_shape(arrays)
    for key in ("navigation_free_room_domain", "observed_free_mask", "voxel_nav_free_xy", "vertical_free_room_domain"):
        value = arrays.get(key)
        if value is None:
            continue
        arr = np.asarray(value, dtype=bool)
        if arr.shape == shape and np.any(arr):
            domain = arr.copy()
            break
    else:
        domain = np.ones(shape, dtype=bool)
    obstacle = _first_bool_array(arrays, ("obstacle_mask", "occupancy_map", "voxel_nav_occupied_xy"), shape)
    unknown = _first_bool_array(arrays, ("unknown_mask", "voxel_nav_unknown_xy"), shape)
    domain &= ~obstacle
    if unknown is not None:
        domain &= ~unknown
    return domain


def load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]).copy() for name in data.files}


def save_rose2_baseline_snapshot_npz(
    *,
    source_snapshot_path: Path,
    output_path: Path,
    result: Rose2Result,
) -> None:
    arrays = load_npz_arrays(Path(source_snapshot_path))
    arrays["final_room_label_map"] = enforce_room_mask_contract(result.final_room_label_map, arrays)
    arrays["baseline_name"] = np.asarray("rose2")
    arrays["baseline_metadata_json"] = np.asarray(json.dumps(result.metadata, ensure_ascii=False, sort_keys=True))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)


class Rose2Runner:
    baseline_name = "rose2"

    def __init__(
        self,
        *,
        ros_workspace: Path | str | None = None,
        launch_file: str = ORIGINAL_ROSE2_LAUNCH,
        rose_package: str = ORIGINAL_ROSE2_PACKAGE,
        map_topic: str = "/map",
        output_topic: str = "/rooms",
        manage_launch: bool = True,
        startup_timeout_s: float = 30.0,
        result_timeout_s: float = 180.0,
        default_resolution_m: float = 1.0,
        default_origin_xy: tuple[float, float] = (0.0, 0.0),
        use_fallback_smoke: bool = False,
        fallback_python: bool | None = None,
        map_resolution_m: float | None = None,
        ros_setup: str | None = None,
        ros_python: str | None = None,
    ) -> None:
        self.ros_workspace = None if ros_workspace is None else Path(ros_workspace).expanduser()
        self.launch_file = str(launch_file)
        self.rose_package = str(rose_package)
        self.map_topic = str(map_topic)
        self.output_topic = str(output_topic)
        self.manage_launch = bool(manage_launch)
        self.startup_timeout_s = float(startup_timeout_s)
        self.result_timeout_s = float(result_timeout_s)
        self.default_resolution_m = float(map_resolution_m if map_resolution_m is not None else default_resolution_m)
        self.default_origin_xy = (float(default_origin_xy[0]), float(default_origin_xy[1]))
        self.use_fallback_smoke = bool(use_fallback_smoke if fallback_python is None else fallback_python)
        self._launch_process: subprocess.Popen[bytes] | None = None
        self._ros_modules: dict[str, Any] | None = None
        self._scene_id: str | None = None
        self.last_metadata: dict[str, Any] = {}
        self.last_debug_arrays: dict[str, np.ndarray] = {}
        if RosSubprocessConfig is None:
            self.ros_config = None
        else:
            self.ros_config = RosSubprocessConfig.from_env(
                ros_setup=ros_setup,
                ros_python=ros_python,
                workspace_roots=(self.ros_workspace,),
                timeout_s=max(self.result_timeout_s, self.startup_timeout_s) + 30.0,
            )

    def start_scene(self, scene_id: str) -> None:
        self._scene_id = str(scene_id)
        if self.use_fallback_smoke:
            return
        if not self.manage_launch:
            return
        self.close()
        env = self._ros_env()
        roslaunch = shutil.which("roslaunch", path=env.get("PATH"))
        if roslaunch is None:
            raise Rose2DependencyError(_format_dependency_error(inspect_ros_dependency_status()))
        cmd = [roslaunch, self.rose_package, self.launch_file]
        self._launch_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        time.sleep(min(2.0, self.startup_timeout_s))
        if self._launch_process.poll() is not None:
            output = self._read_launch_output()
            raise Rose2DependencyError("ROSE2 roslaunch exited early: %s\n%s" % (" ".join(cmd), output))

    def close(self) -> None:
        process = self._launch_process
        self._launch_process = None
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5.0)

    def end_scene(self) -> None:
        self.close()

    def segment_snapshot(self, snapshot_path: Path | str | None, arrays: Mapping[str, np.ndarray]) -> BaselineResult:
        result = self.run_snapshot(snapshot_path=snapshot_path, arrays=arrays)
        return BaselineResult(label_map=result.final_room_label_map, metadata=result.metadata, debug_arrays=self.last_debug_arrays)

    def run_snapshot(
        self,
        snapshot_path: Path | str | None = None,
        arrays: Mapping[str, np.ndarray] | None = None,
        *,
        scene_id: str | None = None,
    ) -> Rose2Result:
        if arrays is None:
            if snapshot_path is None:
                raise ValueError("snapshot_path or arrays is required")
            arrays = load_npz_arrays(Path(snapshot_path))
        if scene_id is not None and scene_id != self._scene_id:
            self.start_scene(str(scene_id))

        if self.use_fallback_smoke:
            from voxroom_online.isaac_runtime.baselines.offline.fallback_rose2 import smoke_segment_free_components

            labels, metadata = smoke_segment_free_components(arrays)
            metadata.update(self._base_metadata(arrays, snapshot_path=snapshot_path, runner_type="python_smoke_fallback"))
            labels = enforce_room_mask_contract(labels, arrays)
            metadata["rooms_after_contract"] = int(len([v for v in np.unique(labels) if int(v) > 0]))
            self.last_metadata = metadata
            return Rose2Result(final_room_label_map=labels, metadata=metadata)

        if self._scene_id is None:
            self.start_scene(scene_id or "default")
        if self.ros_config is None or run_ros_module is None:
            raise Rose2DependencyError("ROSE2 ROS subprocess bridge is unavailable")
        bridge_result = run_ros_module(
            "voxroom_online.isaac_runtime.baselines.offline.ros_entrypoints.rose2_ros_node",
            method="rose2",
            snapshot_path=str(snapshot_path),
            arrays=arrays,
            scene_id=self._scene_id,
            params={
                "input_topic": self.map_topic,
                "output_topic": self.output_topic,
                "launch_file": self.launch_file,
                "rose_package": self.rose_package,
                "startup_timeout_s": self.startup_timeout_s,
                "result_timeout_s": self.result_timeout_s,
                "default_resolution_m": self.default_resolution_m,
                "default_origin_xy": list(self.default_origin_xy),
                "original_repo_commit": _discover_original_commit(self.ros_workspace),
            },
            config=self.ros_config,
        )
        self.last_metadata = dict(bridge_result.metadata)
        self.last_debug_arrays = dict(bridge_result.debug_arrays)
        return Rose2Result(final_room_label_map=np.asarray(bridge_result.label_map, dtype=np.int32), metadata=self.last_metadata)

    def _base_metadata(
        self,
        arrays: Mapping[str, np.ndarray],
        *,
        snapshot_path: Path | str | None,
        runner_type: str,
    ) -> dict[str, Any]:
        shape = infer_snapshot_shape(arrays)
        return {
            "method": "rose2",
            "runner_type": str(runner_type),
            "original_repo_url": ORIGINAL_ROSE2_REPO_URL,
            "original_commit_inspected": ORIGINAL_ROSE2_COMMIT_INSPECTED,
            "launch_file": self.launch_file,
            "ros_package": self.rose_package,
            "map_topic": self.map_topic,
            "output_topic": self.output_topic,
            "output_topic_confirmed_from_source": True,
            "output_topic_note": "ROSE2 publishes rooms on /rooms as jsk_recognition_msgs/PolygonArray; no segmented OccupancyGrid topic exists in the inspected source.",
            "features_topic": "/features_ROSE2",
            "input_occupancy_values": dict(ROSE2_INPUT_OCCUPANCY_VALUES),
            "source_snapshot": None if snapshot_path is None else str(snapshot_path),
            "shape_hw": [int(shape[0]), int(shape[1])],
            "confirmed_contract": ORIGINAL_ROSE2_CONFIRMED_CONTRACT,
        }

    def _ros_env(self) -> dict[str, str]:
        env = dict(os.environ)
        setup = self._find_setup_script()
        if setup is None:
            return env
        command = "source %s >/dev/null 2>&1 && env -0" % shlex.quote(str(setup))
        proc = subprocess.run(["bash", "-lc", command], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise Rose2DependencyError("failed to source ROS workspace setup: %s\n%s" % (setup, proc.stderr.decode("utf-8", "replace")))
        loaded: dict[str, str] = {}
        for item in proc.stdout.split(b"\0"):
            if not item or b"=" not in item:
                continue
            key, value = item.split(b"=", 1)
            loaded[key.decode("utf-8", "replace")] = value.decode("utf-8", "replace")
        return loaded

    def _find_setup_script(self) -> Path | None:
        if self.ros_workspace is None:
            return None
        for relative in ("devel/setup.bash", "install/setup.bash", "setup.bash"):
            candidate = self.ros_workspace / relative
            if candidate.exists():
                return candidate
        return None

    def _init_ros_client(self) -> dict[str, Any]:
        if self._ros_modules is not None:
            return self._ros_modules
        self._apply_ros_env_to_current_process()
        try:
            import rospy
            from geometry_msgs.msg import Point, Pose, Quaternion
            from jsk_recognition_msgs.msg import PolygonArray
            from nav_msgs.msg import OccupancyGrid
            from std_msgs.msg import Header
        except Exception as exc:  # pragma: no cover - exercised only in ROS environments here.
            raise Rose2DependencyError(_format_dependency_error(inspect_ros_dependency_status(), import_error=exc)) from exc
        if not rospy.core.is_initialized():
            rospy.init_node("voxroom_rose2_runner_%d" % os.getpid(), anonymous=True, disable_signals=True)
        self._ros_modules = {
            "rospy": rospy,
            "Header": Header,
            "Pose": Pose,
            "Point": Point,
            "Quaternion": Quaternion,
            "OccupancyGrid": OccupancyGrid,
            "PolygonArray": PolygonArray,
        }
        return self._ros_modules

    def _apply_ros_env_to_current_process(self) -> None:
        env = self._ros_env()
        for key in ("ROS_MASTER_URI", "ROS_IP", "ROS_HOSTNAME", "ROS_PACKAGE_PATH", "ROS_ROOT", "ROS_ETC_DIR", "ROS_DISTRO"):
            value = env.get(key)
            if value:
                os.environ[key] = value
        for path in reversed([p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]):
            if path not in sys.path:
                sys.path.insert(0, path)

    def _make_occupancy_grid_message(
        self,
        values: np.ndarray,
        *,
        resolution_m: float,
        origin_xy: tuple[float, float],
        modules: Mapping[str, Any],
    ) -> Any:
        rospy = modules["rospy"]
        grid = modules["OccupancyGrid"]()
        grid.header = modules["Header"]()
        grid.header.frame_id = "map"
        grid.header.stamp = rospy.Time.now()
        grid.info.resolution = float(resolution_m)
        grid.info.height = int(values.shape[0])
        grid.info.width = int(values.shape[1])
        grid.info.origin = modules["Pose"]()
        grid.info.origin.position = modules["Point"](float(origin_xy[0]), float(origin_xy[1]), 0.0)
        grid.info.origin.orientation = modules["Quaternion"](0.0, 0.0, 0.0, 1.0)
        grid.data = [int(v) for v in np.asarray(values, dtype=np.int16).reshape(-1)]
        return grid

    def _publish_map_and_wait_for_rooms(self, grid: Any, *, modules: Mapping[str, Any]) -> Any:
        rospy = modules["rospy"]
        publisher = rospy.Publisher(self.map_topic, modules["OccupancyGrid"], queue_size=1, latch=True)
        deadline = time.monotonic() + max(0.0, self.startup_timeout_s)
        while publisher.get_num_connections() < 1 and time.monotonic() < deadline and not rospy.is_shutdown():
            rospy.sleep(0.05)
        if publisher.get_num_connections() < 1:
            raise TimeoutError("ROSE2 runner timed out waiting for a subscriber on %s" % self.map_topic)
        publisher.publish(grid)
        return rospy.wait_for_message(self.output_topic, modules["PolygonArray"], timeout=self.result_timeout_s)

    def _read_launch_output(self) -> str:
        process = self._launch_process
        if process is None or process.stdout is None:
            return ""
        try:
            return process.stdout.read(20000).decode("utf-8", "replace")
        except Exception:
            return ""

    def __enter__(self) -> "Rose2Runner":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def _probe_import(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def _discover_original_commit(path: Path | None) -> str | None:
    if path is None:
        return None
    candidates = [path]
    if (path / "src" / "ROSE2").exists():
        candidates.insert(0, path / "src" / "ROSE2")
    if (path / ".git").exists():
        candidates.insert(0, path)
    for candidate in candidates:
        try:
            head = subprocess.check_output(
                ["git", "-C", str(candidate), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            continue
        if head:
            return head
    return None


def _first_bool_array(arrays: Mapping[str, np.ndarray], keys: Sequence[str], shape: tuple[int, int]) -> np.ndarray:
    for key in keys:
        value = arrays.get(key)
        if value is None:
            continue
        arr = np.asarray(value, dtype=bool)
        if arr.shape == shape:
            return arr.copy()
    return np.zeros(shape, dtype=bool)


def _first_nonempty_bool_array(
    arrays: Mapping[str, np.ndarray],
    keys: Sequence[str],
    shape: tuple[int, int],
) -> tuple[np.ndarray, str]:
    fallback: tuple[np.ndarray, str] | None = None
    for key in keys:
        value = arrays.get(key)
        if value is None:
            continue
        arr = np.asarray(value, dtype=bool)
        if arr.shape != shape:
            continue
        if fallback is None:
            fallback = (arr.copy(), key)
        if np.any(arr):
            return arr.copy(), key
    if fallback is not None:
        return fallback
    return np.zeros(shape, dtype=bool), "none"


def _cell_coords_to_world(
    cell_xy: Sequence[Sequence[float]],
    *,
    resolution_m: float,
    origin_xy: tuple[float, float],
) -> list[tuple[float, float]]:
    return [
        (float(x) * float(resolution_m) + float(origin_xy[0]), float(y) * float(resolution_m) + float(origin_xy[1]))
        for x, y, *_ in cell_xy
    ]


def _int8_sequence_to_bytes(values: Sequence[int]) -> bytes:
    return bytes((int(v) + 256) % 256 for v in values)


def _format_dependency_error(status: Mapping[str, Any], import_error: BaseException | None = None) -> str:
    parts = ["ROSE2 original ROS runner dependencies are not available."]
    missing_commands = list(status.get("missing_commands", []) or [])
    missing_modules = list(status.get("missing_python_modules", []) or [])
    if missing_commands:
        parts.append("missing commands: %s" % ", ".join(str(v) for v in missing_commands))
    if missing_modules:
        parts.append("missing python modules: %s" % ", ".join(str(v) for v in missing_modules))
    if import_error is not None:
        parts.append("import error: %s: %s" % (type(import_error).__name__, import_error))
    parts.append("Build/source aislabunimi/ROSE2 in a catkin workspace with nav_msgs, geometry_msgs, visualization_msgs, jsk_recognition_msgs, rospy, and the Python requirements from ROSE2/requirements.txt.")
    return "\n".join(parts)


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the ROSE2 offline baseline on one VoxRoom roomseg snapshot.")
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--output", "--out", type=Path, default=None)
    parser.add_argument("--scene-id", default="default")
    parser.add_argument("--ros-workspace", type=Path, default=None)
    parser.add_argument("--launch-file", default=ORIGINAL_ROSE2_LAUNCH)
    parser.add_argument("--dependency-report", action="store_true")
    parser.add_argument("--use-fallback-smoke", action="store_true", help="Run contract smoke fallback only; not valid for main ROSE2 experiments.")
    args = parser.parse_args(argv)

    if args.dependency_report:
        print(json.dumps(inspect_ros_dependency_status(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.snapshot is None:
        parser.error("--snapshot is required unless --dependency-report is used")
    with Rose2Runner(
        ros_workspace=args.ros_workspace,
        launch_file=args.launch_file,
        use_fallback_smoke=bool(args.use_fallback_smoke),
    ) as runner:
        try:
            result = runner.run_snapshot(snapshot_path=args.snapshot, scene_id=str(args.scene_id))
        except Rose2DependencyError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        if args.output is not None:
            save_rose2_baseline_snapshot_npz(source_snapshot_path=args.snapshot, output_path=args.output, result=result)
        print(json.dumps(result.metadata, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
