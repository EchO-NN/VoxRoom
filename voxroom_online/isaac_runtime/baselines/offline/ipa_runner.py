from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

from voxroom_online.isaac_runtime.baselines.data_contract import MapInfo, resolve_map_info
from voxroom_online.isaac_runtime.baselines.mask_io import enforce_room_mask_contract
from voxroom_online.isaac_runtime.baselines.ros_grid_io import snapshot_to_ipa_image
from voxroom_online.isaac_runtime.baselines.ros_subprocess import RosSubprocessConfig, run_ros_module
from voxroom_online.isaac_runtime.baselines.offline.base import BaselineResult, MissingOriginalImplementationError
from voxroom_online.isaac_runtime.baselines.offline.fallback_distance import (
    DISTANCE_ALGORITHM_ID,
    ORIGINAL_ACTION,
    ORIGINAL_PACKAGE,
    ORIGINAL_REPO,
    ORIGINAL_REPO_COMMIT,
    segment_snapshot_arrays as segment_distance_python_fallback,
)
from voxroom_online.isaac_runtime.baselines.offline.fallback_morphological import (
    IPA_MORPHOLOGICAL_ALGORITHM_ID,
    IPA_MORPHOLOGICAL_LOWER_LIMIT_M2,
    IPA_MORPHOLOGICAL_UPPER_LIMIT_M2,
    IPA_UNASSIGNED_FREE_LABEL,
    MorphologicalFallbackConfig,
    segment_snapshot_morphological,
)
from voxroom_online.isaac_runtime.baselines.offline.fallback_voronoi import (
    VORONOI_ALGORITHM_ID,
    build_voronoi_ipa_input_image,
    segment_snapshot_arrays as segment_voronoi_python_fallback,
)


BormannIpaAlgorithm = Literal["morphological", "distance_transform", "voronoi"]

IPA_ALGORITHM_IDS: dict[str, int] = {
    "morphological": IPA_MORPHOLOGICAL_ALGORITHM_ID,
    "distance_transform": DISTANCE_ALGORITHM_ID,
    "voronoi": VORONOI_ALGORITHM_ID,
}


@dataclass(frozen=True)
class IpaRunnerParameters:
    robot_radius_m: float = 0.05
    return_format_in_pixel: bool = True
    return_format_in_meter: bool = False
    room_area_factor_lower_limit_morphological: float = IPA_MORPHOLOGICAL_LOWER_LIMIT_M2
    room_area_factor_upper_limit_morphological: float = IPA_MORPHOLOGICAL_UPPER_LIMIT_M2
    room_area_factor_lower_limit_distance: float = 0.35
    room_area_factor_upper_limit_distance: float = 163.0
    room_area_factor_lower_limit_voronoi: float = 0.1
    room_area_factor_upper_limit_voronoi: float = 1000000.0
    voronoi_neighborhood_index: int = 280
    max_iterations: int = 150
    min_critical_point_distance_factor: float = 0.5
    max_area_for_merging: float = 12.5


class IpaRoomSegmentationRunner:
    """Adapter for Bormann et al. methods in ipa320/ipa_coverage_planning.

    Main experiments use the original ipa_room_segmentation ROS action server.
    Python fallbacks are opt-in smoke paths and are metadata-marked as such.
    """

    def __init__(
        self,
        *,
        algorithm: BormannIpaAlgorithm = "distance_transform",
        ros_workspace: Path | str | None = None,
        action_name: str = "room_segmentation/room_segmentation_server",
        default_resolution_m: float | None = None,
        map_resolution_m: float | None = None,
        robot_radius_m: float = 0.05,
        timeout_s: float = 120.0,
        fallback_python: bool = False,
        allow_python_fallback_for_smoke: bool | None = None,
        ros_setup: str | None = None,
        ros_python: str | None = None,
        manage_action_server: bool = True,
    ) -> None:
        if algorithm not in IPA_ALGORITHM_IDS:
            raise NotImplementedError(f"unsupported IPA room segmentation algorithm: {algorithm}")
        self.algorithm = str(algorithm)
        self.baseline_name = self.algorithm
        self.ros_workspace = None if ros_workspace is None else Path(ros_workspace)
        self.action_name = str(action_name)
        self.default_resolution_m = (
            float(map_resolution_m)
            if map_resolution_m is not None
            else (None if default_resolution_m is None else float(default_resolution_m))
        )
        self.timeout_s = float(timeout_s)
        self.parameters = IpaRunnerParameters(robot_radius_m=float(robot_radius_m))
        self.fallback_python = bool(fallback_python if allow_python_fallback_for_smoke is None else allow_python_fallback_for_smoke)
        self.manage_action_server = bool(manage_action_server)
        self.ros_config = RosSubprocessConfig.from_env(
            ros_setup=ros_setup,
            ros_python=ros_python,
            workspace_roots=(self.ros_workspace,),
            timeout_s=float(timeout_s) + 30.0,
        )
        self._scene_id: str | None = None
        self._action_server_process: subprocess.Popen[bytes] | None = None

    def start_scene(self, scene_id: str) -> None:
        self._scene_id = str(scene_id)
        if self.ros_workspace is not None:
            _validate_ipa_workspace(self.ros_workspace, algorithm=self.algorithm)
        if not self.fallback_python and self.manage_action_server:
            self._start_original_action_server()

    def end_scene(self) -> None:
        self._stop_original_action_server()
        self._scene_id = None

    def segment_snapshot(
        self,
        snapshot_path: Path | str,
        arrays: Mapping[str, Any],
    ) -> BaselineResult:
        snapshot_path = Path(snapshot_path)
        if self.fallback_python and self.algorithm == "morphological":
            return self._segment_snapshot_morphological_fallback(
                snapshot_path,
                arrays,
                fallback_reason=RuntimeError("explicit --fallback-python smoke path requested"),
            )
        try:
            return self._segment_snapshot_original_ros(snapshot_path=snapshot_path, arrays=arrays)
        except Exception as exc:
            if not self.fallback_python:
                raise MissingOriginalImplementationError(
                    "%s main experiment requires the original ipa320/ipa_coverage_planning "
                    "ipa_room_segmentation ROS action server. Python fallback is disabled by "
                    "default and may only be enabled for smoke tests." % self.algorithm
                ) from exc
            if self.algorithm == "morphological":
                return self._segment_snapshot_morphological_fallback(snapshot_path, arrays, fallback_reason=exc)
            if self.algorithm == "distance_transform":
                labels, metadata, debug = segment_distance_python_fallback(
                    snapshot_path,
                    arrays,
                    default_resolution_m=self.default_resolution_m or 0.05,
                )
                metadata["fallback_reason"] = str(exc)
                metadata["fallback_requested_by_runner"] = True
                return BaselineResult(label_map=np.asarray(labels, dtype=np.int32), metadata=metadata, debug_arrays=debug)
            if self.algorithm == "voronoi":
                labels, metadata, debug = segment_voronoi_python_fallback(
                    snapshot_path,
                    arrays,
                    default_resolution_m=self.default_resolution_m or 0.05,
                )
                metadata["fallback_reason"] = str(exc)
                metadata["fallback_requested_by_runner"] = True
                return BaselineResult(label_map=np.asarray(labels, dtype=np.int32), metadata=metadata, debug_arrays=debug)
            raise

    def _segment_snapshot_original_ros(
        self,
        *,
        snapshot_path: Path,
        arrays: Mapping[str, Any],
    ) -> BaselineResult:
        return run_ros_module(
            "voxroom_online.isaac_runtime.baselines.offline.ros_entrypoints.ipa_ros_node",
            method=self.algorithm,
            snapshot_path=snapshot_path,
            arrays=arrays,
            scene_id=self._scene_id,
            params={
                "algorithm": self.algorithm,
                "action_name": self.action_name,
                "timeout_s": self.timeout_s,
                "map_resolution_m": self.default_resolution_m,
                "robot_radius_m": float(self.parameters.robot_radius_m),
                "original_repo_commit": _discover_original_commit(self.ros_workspace),
            },
            config=self.ros_config,
        )

    def _start_original_action_server(self) -> None:
        self._stop_original_action_server()
        cmd = build_ipa_roslaunch_shell(self.ros_config)
        self._action_server_process = subprocess.Popen(
            ["bash", "-lc", cmd],
            cwd=str(self.ros_config.repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + min(max(float(self.timeout_s), 15.0), 60.0)
        while time.monotonic() < deadline:
            process = self._action_server_process
            if process is None:
                break
            if process.poll() is not None:
                output = ""
                if process.stdout is not None:
                    try:
                        output = process.stdout.read(20000).decode("utf-8", "replace")
                    except Exception:
                        output = ""
                self._action_server_process = None
                raise MissingOriginalImplementationError("IPA room_segmentation action server exited during startup:\n%s" % output)
            if ipa_action_topics_available(self.ros_config, action_name=self.action_name):
                return
            time.sleep(0.5)
        raise MissingOriginalImplementationError(
            "timed out waiting for IPA action server topics for %r after roslaunch startup" % self.action_name
        )

    def _stop_original_action_server(self) -> None:
        process = self._action_server_process
        self._action_server_process = None
        _terminate_process_group(process)

    def _segment_snapshot_morphological_fallback(
        self,
        snapshot_path: Path,
        arrays: Mapping[str, Any],
        *,
        fallback_reason: Exception,
    ) -> BaselineResult:
        map_info = resolve_map_info(snapshot_arrays=arrays, default_resolution_m=self.default_resolution_m or 0.05)
        result = segment_snapshot_morphological(
            arrays,
            MorphologicalFallbackConfig(map_resolution_m=float(map_info.resolution_m)),
        )
        metadata = dict(result.metadata)
        metadata.update(
            {
                "source_snapshot": str(snapshot_path),
                "fallback_reason": str(fallback_reason),
                "fallback_requested_by_runner": True,
                "map_info": map_info.to_metadata(),
                "input_free_definition": _input_free_definition(arrays),
            }
        )
        return BaselineResult(
            label_map=np.asarray(result.room_label_map, dtype=np.int32),
            metadata=metadata,
            debug_arrays=result.debug_arrays,
        )

    def _metadata(
        self,
        *,
        snapshot_path: Path,
        arrays: Mapping[str, Any],
        map_info: MapInfo,
        runner_type: str,
    ) -> dict[str, Any]:
        return {
            "method": self.algorithm,
            "source_snapshot": str(snapshot_path),
            "input_free_definition": _input_free_definition(arrays),
            "unknown_treated_as": "occupied/inaccessible",
            "unknown_treated_as_free": False,
            "map_resolution_m": float(map_info.resolution_m),
            "map_origin_xy_m": [float(map_info.min_x), float(map_info.min_y)],
            "uses_rgb": False,
            "uses_depth": False,
            "uses_oracle_semantics": False,
            "runner_type": str(runner_type),
            "main_experiment_allowed": str(runner_type) == "original_ros_action",
            "original_repo": ORIGINAL_REPO,
            "original_repo_commit": _discover_original_commit(self.ros_workspace),
            "original_repo_reference_commit": ORIGINAL_REPO_COMMIT,
            "original_package": ORIGINAL_PACKAGE,
            "original_action": ORIGINAL_ACTION,
            "original_action_server": self.action_name,
            "original_algorithm_id": int(IPA_ALGORITHM_IDS[self.algorithm]),
            "original_algorithm_name": _algorithm_display_name(self.algorithm),
            "scene_id": self._scene_id,
            "parameters": _parameters_for_algorithm(self.algorithm, self.parameters),
        }


DistanceTransformIpaRunner = IpaRoomSegmentationRunner
MorphologicalIpaRunner = IpaRoomSegmentationRunner


def _import_ros_modules() -> dict[str, Any]:
    try:
        import actionlib
        import rospy
        from cv_bridge import CvBridge
        from geometry_msgs.msg import Pose
        from ipa_building_msgs.msg import MapSegmentationAction, MapSegmentationGoal
    except Exception as exc:
        raise RuntimeError(
            "ROS/IPA dependencies are unavailable. Build and source "
            "ipa320/ipa_coverage_planning with ipa_room_segmentation and ipa_building_msgs."
        ) from exc
    return {
        "actionlib": actionlib,
        "rospy": rospy,
        "CvBridge": CvBridge,
        "Pose": Pose,
        "MapSegmentationAction": MapSegmentationAction,
        "MapSegmentationGoal": MapSegmentationGoal,
    }


def _validate_ipa_workspace(path: Path, *, algorithm: str) -> None:
    package_root = _ipa_package_root(path)
    if package_root is None:
        raise FileNotFoundError(
            f"could not find {ORIGINAL_PACKAGE} under {path}; expected ipa320/ipa_coverage_planning checkout"
        )
    action_path = package_root.parent / "ipa_building_msgs" / "action" / "MapSegmentation.action"
    cfg_path = package_root / "cfg" / "RoomSegmentation.cfg"
    source_name = {
        "morphological": "morphological_segmentation.cpp",
        "distance_transform": "distance_segmentation.cpp",
        "voronoi": "voronoi_segmentation.cpp",
    }[str(algorithm)]
    algorithm_src = package_root / "common" / "src" / source_name
    missing = [str(p) for p in (action_path, cfg_path, algorithm_src) if not p.exists()]
    if missing:
        raise FileNotFoundError("incomplete ipa_coverage_planning workspace; missing: " + ", ".join(missing))


def _ipa_package_root(path: Path) -> Path | None:
    candidates = (
        path / "ipa_room_segmentation",
        path / "src" / "ipa_coverage_planning" / "ipa_room_segmentation",
        path / "src" / "ipa_room_segmentation",
    )
    for candidate in candidates:
        if (candidate / "cfg" / "RoomSegmentation.cfg").exists():
            return candidate
    return None


def _discover_original_commit(path: Path | None) -> str | None:
    if path is None:
        return None
    package_root = _ipa_package_root(path)
    git_root = package_root.parent if package_root is not None else path
    try:
        return (
            subprocess.check_output(["git", "-C", str(git_root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL)
            .strip()
            or None
        )
    except Exception:
        return None


def _input_free_definition(arrays: Mapping[str, Any]) -> str:
    nav = arrays.get("navigation_free_room_domain")
    if nav is not None:
        arr = np.asarray(nav, dtype=bool)
        if arr.ndim == 2 and bool(arr.any()):
            return "navigation_free_room_domain"
    return "observed_free_minus_obstacle_unknown"


def _algorithm_display_name(algorithm: str) -> str:
    return {
        "morphological": "MorphologicalSegmentation",
        "distance_transform": "DistanceSegmentation",
        "voronoi": "VoronoiSegmentation",
    }[str(algorithm)]


def _parameters_for_algorithm(algorithm: str, params: IpaRunnerParameters) -> dict[str, Any]:
    base: dict[str, Any] = {
        "room_segmentation_algorithm": int(IPA_ALGORITHM_IDS[str(algorithm)]),
        "robot_radius_m": float(params.robot_radius_m),
    }
    if algorithm == "morphological":
        base.update(
            {
                "room_area_factor_lower_limit_morphological": float(
                    params.room_area_factor_lower_limit_morphological
                ),
                "room_area_factor_upper_limit_morphological": float(
                    params.room_area_factor_upper_limit_morphological
                ),
            }
        )
    elif algorithm == "distance_transform":
        base.update(
            {
                "room_area_factor_lower_limit_distance": float(params.room_area_factor_lower_limit_distance),
                "room_area_factor_upper_limit_distance": float(params.room_area_factor_upper_limit_distance),
            }
        )
    elif algorithm == "voronoi":
        base.update(
            {
                "room_area_factor_lower_limit_voronoi": float(params.room_area_factor_lower_limit_voronoi),
                "room_area_factor_upper_limit_voronoi": float(params.room_area_factor_upper_limit_voronoi),
                "voronoi_neighborhood_index": int(params.voronoi_neighborhood_index),
                "max_iterations": int(params.max_iterations),
                "min_critical_point_distance_factor": float(params.min_critical_point_distance_factor),
                "max_area_for_merging": float(params.max_area_for_merging),
            }
        )
    return base


def build_ipa_roslaunch_shell(config: RosSubprocessConfig) -> str:
    lines = ["set -euo pipefail"]
    if config.ros_setup:
        setup = shlex.quote(str(config.ros_setup))
        lines.append(f"test -f {setup}")
        lines.append("set +u")
        lines.append(f"source {setup}")
        lines.append("set -u")
    for setup in config.workspace_setups:
        quoted = shlex.quote(str(setup))
        lines.append(f"if [ -f {quoted} ]; then set +u; source {quoted}; set -u; fi")
    lines.append("exec roslaunch ipa_room_segmentation room_segmentation_action_server.launch")
    return "\n".join(lines)


def build_ipa_action_probe_shell(config: RosSubprocessConfig, *, action_name: str) -> str:
    lines = ["set -euo pipefail"]
    if config.ros_setup:
        setup = shlex.quote(str(config.ros_setup))
        lines.append(f"test -f {setup}")
        lines.append("set +u")
        lines.append(f"source {setup}")
        lines.append("set -u")
    topic_prefix = "/" + str(action_name).strip("/")
    goal_topic = shlex.quote(topic_prefix + "/goal")
    result_topic = shlex.quote(topic_prefix + "/result")
    lines.append(f"topics=$(rostopic list)")
    lines.append(f"grep -Fx -- {goal_topic} <<< \"$topics\" >/dev/null")
    lines.append(f"grep -Fx -- {result_topic} <<< \"$topics\" >/dev/null")
    return "\n".join(lines)


def ipa_action_topics_available(
    config: RosSubprocessConfig,
    *,
    action_name: str,
    timeout_s: float = 3.0,
) -> bool:
    try:
        proc = subprocess.run(
            ["bash", "-lc", build_ipa_action_probe_shell(config, action_name=action_name)],
            cwd=str(config.repo_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=float(timeout_s),
            check=False,
        )
    except Exception:
        return False
    return proc.returncode == 0


def _terminate_process_group(process: subprocess.Popen[bytes] | None, *, timeout_s: float = 8.0) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=float(timeout_s))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=float(timeout_s))
