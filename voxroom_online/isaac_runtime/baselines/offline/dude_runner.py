from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy import ndimage

from voxroom_online.isaac_runtime.evaluation.online_roomseg.common import (
    parse_snapshot_step,
    relabel_positive_labels_sequentially,
)

from ..data_contract import resolve_map_info
from ..mask_io import build_metric_domain_from_source, enforce_room_mask_contract
from ..ros_subprocess import RosSubprocessConfig, run_ros_module
from .base import BaselineResult, MissingOriginalImplementationError


BASELINE_NAME = "dude_incremental"
UPSTREAM_REPO = "lfermin77/Incremental_DuDe_ROS"
UPSTREAM_REPO_URL = "https://github.com/lfermin77/Incremental_DuDe_ROS"
UPSTREAM_HEAD_AT_IMPLEMENTATION = "9500cab2e94a873935c6464703cfb9bd370dfa95"


@dataclass(frozen=True)
class OriginalDudeRepoStatus:
    repo_root: str
    exists: bool
    has_package_xml: bool
    has_cmakelists: bool
    has_inc_dude_source: bool
    has_wrapper_source: bool
    has_third_party_dude: bool
    executable: str | None
    git_head: str | None
    source_tree_present: bool
    catkin_executable_present: bool
    notes: tuple[str, ...]


class DudeIncrementalRunner:
    baseline_name = BASELINE_NAME

    def __init__(
        self,
        *,
        repo_root: Path | str | None = None,
        concavity_threshold_m: float = 3.0,
        use_incremental: bool = True,
        fallback_python: bool = False,
        allow_python_fallback: bool | None = None,
        map_resolution_m: float | None = None,
        ros_setup: str | None = None,
        ros_python: str | None = None,
        dude_ws: Path | str | None = None,
        timeout_s: float = 180.0,
    ) -> None:
        self.repo_root = Path(repo_root) if repo_root else _env_path("INCREMENTAL_DUDE_ROS_ROOT")
        self.dude_ws = Path(dude_ws) if dude_ws else _env_path("DUDE_WS")
        self.concavity_threshold_m = float(concavity_threshold_m)
        self.use_incremental = bool(use_incremental)
        self.fallback_python = bool(fallback_python if allow_python_fallback is None else allow_python_fallback)
        self.map_resolution_m = None if map_resolution_m is None else float(map_resolution_m)
        self.ros_config = RosSubprocessConfig.from_env(
            ros_setup=ros_setup,
            ros_python=ros_python,
            workspace_roots=(self.dude_ws, None if self.repo_root is None else self.repo_root.parent.parent),
            timeout_s=float(timeout_s),
        )
        self.scene_id: str | None = None
        self._last_step: int | None = None
        self._repo_status: OriginalDudeRepoStatus | None = None
        self._roscore_process: subprocess.Popen[bytes] | None = None
        self._node_process: subprocess.Popen[bytes] | None = None

    @property
    def repo_status(self) -> OriginalDudeRepoStatus | None:
        return self._repo_status

    def start_scene(self, scene_id: str) -> None:
        self.scene_id = str(scene_id)
        self._last_step = None
        self._repo_status = inspect_original_dude_repo(self.repo_root)
        if self.fallback_python:
            return
        if self.repo_root is None:
            raise MissingOriginalImplementationError(
                "Incremental_DuDe_ROS original implementation is required for main experiment. "
                "Set --dude-repo-root or INCREMENTAL_DUDE_ROS_ROOT, or pass --fallback-python for smoke only."
            )
        if not self._repo_status.source_tree_present or not self._repo_status.catkin_executable_present:
            raise MissingOriginalImplementationError(
                "Incremental_DuDe_ROS is not cloned/built enough for native DUDE. "
                f"repo_status={asdict(self._repo_status)}"
            )
        self._start_original_node()

    def segment_snapshot(self, snapshot_path: Path, arrays: Mapping[str, Any]) -> BaselineResult:
        step = _snapshot_step(Path(snapshot_path), arrays)
        if self.use_incremental and self._last_step is not None and step <= self._last_step:
            raise ValueError(
                "Incremental DUDE requires strictly increasing snapshot steps within one scene: "
                f"previous={self._last_step}, current={step}, snapshot={snapshot_path}"
            )
        self._last_step = int(step)

        if not self.fallback_python:
            if self._repo_status is None:
                self._repo_status = inspect_original_dude_repo(self.repo_root)
            return run_ros_module(
                "voxroom_online.isaac_runtime.baselines.offline.ros_entrypoints.dude_ros_node",
                method=self.baseline_name,
                snapshot_path=snapshot_path,
                arrays=arrays,
                scene_id=self.scene_id,
                params={
                    "scene_id": self.scene_id,
                    "input_topic": "/map",
                    "output_topic": "/tagged_image",
                    "timeout_s": float(self.ros_config.timeout_s),
                    "map_resolution_m": self.map_resolution_m,
                    "concavity_threshold_m": float(self.concavity_threshold_m),
                    "original_repo_commit": self._repo_status.git_head,
                    "original_repo_reference_commit": UPSTREAM_HEAD_AT_IMPLEMENTATION,
                },
                config=self.ros_config,
            )

        free = build_metric_domain_from_source(arrays)
        binary = source_snapshot_to_dude_binary(arrays)
        labels = python_fallback_segment(binary, free)
        labels = enforce_room_mask_contract(labels, arrays, clip_to_eval_domain=True)
        map_info_metadata, resolution_m = _map_info_metadata(arrays, default_resolution_m=self.map_resolution_m)
        concavity_threshold_cells = (
            None
            if resolution_m is None
            else int(round(self.concavity_threshold_m / max(float(resolution_m), 1.0e-9)))
        )
        metadata = {
            "method": self.baseline_name,
            "baseline_name": self.baseline_name,
            "source_snapshot": str(snapshot_path),
            "scene_id": self.scene_id,
            "step": int(step),
            "input_free_definition": _input_free_definition(arrays),
            "unknown_treated_as": "occupied/inaccessible",
            "binary_free_value": 255,
            "binary_occupied_value": 0,
            "uses_rgb": False,
            "uses_depth": False,
            "uses_oracle_semantics": False,
            "runner_type": "python_fallback",
            "fallback_only": True,
            "fallback_algorithm": "scipy.ndimage 4-connected components over metric free domain",
            "main_experiment_allowed": False,
            "not_main_experiment": True,
            "approximation_note": "Smoke fallback only; not DUDE Dual Space Decomposition or Incremental DUDE.",
            "map_info": map_info_metadata,
            "parameters": {
                "concavity_threshold_m": float(self.concavity_threshold_m),
                "concavity_threshold_cells": concavity_threshold_cells,
                "use_incremental": bool(self.use_incremental),
            },
            "incremental_order_enforced": bool(self.use_incremental),
            "original_repo": UPSTREAM_REPO,
            "original_repo_url": UPSTREAM_REPO_URL,
            "original_repo_commit": None if self._repo_status is None else self._repo_status.git_head,
            "original_repo_reference_commit": UPSTREAM_HEAD_AT_IMPLEMENTATION,
            "original_repo_status": None if self._repo_status is None else asdict(self._repo_status),
        }
        debug = {
            "dude_binary_map": binary,
        }
        return BaselineResult(label_map=np.asarray(labels, dtype=np.int32), metadata=metadata, debug_arrays=debug)

    def end_scene(self) -> None:
        self._stop_original_node()
        self.scene_id = None
        self._last_step = None

    def _start_original_node(self) -> None:
        self._stop_original_node()
        self._start_roscore()
        cmd = build_dude_rosrun_shell(self.ros_config, concavity_threshold_m=self.concavity_threshold_m)
        self._node_process = subprocess.Popen(
            ["bash", "-lc", cmd],
            cwd=str(self.ros_config.repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        time.sleep(2.0)
        if self._node_process.poll() is not None:
            output = ""
            if self._node_process.stdout is not None:
                try:
                    output = self._node_process.stdout.read(20000).decode("utf-8", "replace")
                except Exception:
                    output = ""
            raise MissingOriginalImplementationError("Incremental_DuDe_ROS node exited during startup:\n%s" % output)

    def _stop_original_node(self) -> None:
        process = self._node_process
        self._node_process = None
        _terminate_process_group(process)
        roscore_process = self._roscore_process
        self._roscore_process = None
        _terminate_process_group(roscore_process)

    def _start_roscore(self) -> None:
        roscore_process = self._roscore_process
        if roscore_process is not None and roscore_process.poll() is None:
            return
        if ros_master_reachable(self.ros_config):
            self._roscore_process = None
            return
        cmd = build_roscore_shell(self.ros_config)
        self._roscore_process = subprocess.Popen(
            ["bash", "-lc", cmd],
            cwd=str(self.ros_config.repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline:
            process = self._roscore_process
            if process is None:
                break
            if process.poll() is not None:
                output = ""
                if process.stdout is not None:
                    try:
                        output = process.stdout.read(20000).decode("utf-8", "replace")
                    except Exception:
                        output = ""
                self._roscore_process = None
                raise MissingOriginalImplementationError("roscore exited during DUDE startup:\n%s" % output)
            if ros_master_reachable(self.ros_config):
                return
            time.sleep(0.25)
        raise MissingOriginalImplementationError("timed out waiting for roscore before DUDE startup")


def inspect_original_dude_repo(repo_root: Path | str | None) -> OriginalDudeRepoStatus:
    if repo_root is None:
        return OriginalDudeRepoStatus(
            repo_root="",
            exists=False,
            has_package_xml=False,
            has_cmakelists=False,
            has_inc_dude_source=False,
            has_wrapper_source=False,
            has_third_party_dude=False,
            executable=None,
            git_head=None,
            source_tree_present=False,
            catkin_executable_present=False,
            notes=("repo_root_not_configured",),
        )

    root = Path(repo_root).expanduser()
    executable = _find_inc_dude_executable(root)
    has_package_xml = (root / "package.xml").exists()
    has_cmakelists = (root / "CMakeLists.txt").exists()
    has_inc_dude_source = (root / "src" / "inc_dude.cpp").exists()
    has_wrapper_source = (root / "include" / "wrapper.cpp").exists()
    has_third_party_dude = (root / "Third_Party" / "dude_final").exists()
    source_tree_present = bool(
        root.exists()
        and has_package_xml
        and has_cmakelists
        and has_inc_dude_source
        and has_wrapper_source
        and has_third_party_dude
    )
    notes: list[str] = []
    if not root.exists():
        notes.append("repo_missing")
    for name, present in (
        ("package_xml", has_package_xml),
        ("cmakelists", has_cmakelists),
        ("src_inc_dude_cpp", has_inc_dude_source),
        ("include_wrapper_cpp", has_wrapper_source),
        ("third_party_dude_final", has_third_party_dude),
    ):
        if not present:
            notes.append(f"{name}_missing")
    if executable is None:
        notes.append("catkin_executable_devel_lib_inc_dude_missing")
    if source_tree_present and executable is not None:
        notes.append("native_ros_replay_bridge_not_implemented")
    return OriginalDudeRepoStatus(
        repo_root=str(root),
        exists=root.exists(),
        has_package_xml=has_package_xml,
        has_cmakelists=has_cmakelists,
        has_inc_dude_source=has_inc_dude_source,
        has_wrapper_source=has_wrapper_source,
        has_third_party_dude=has_third_party_dude,
        executable=str(executable) if executable is not None else None,
        git_head=_git_head(root) if root.exists() else None,
        source_tree_present=source_tree_present,
        catkin_executable_present=executable is not None,
        notes=tuple(notes) if notes else ("source_tree_present",),
    )


def source_snapshot_to_dude_binary(arrays: Mapping[str, Any]) -> np.ndarray:
    free = build_metric_domain_from_source(arrays)
    binary = np.zeros(free.shape, dtype=np.uint8)
    binary[free] = np.uint8(255)
    return binary


def python_fallback_segment(binary: np.ndarray, free_domain: np.ndarray | None = None) -> np.ndarray:
    free = np.asarray(binary, dtype=np.uint8) == np.uint8(255)
    if free_domain is not None:
        free &= np.asarray(free_domain, dtype=bool)
    structure = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    labels, _ = ndimage.label(free, structure=structure)
    return relabel_positive_labels_sequentially(labels.astype(np.int32))


def labels_from_tagged_image(tagged_image: np.ndarray, source_arrays: Mapping[str, Any]) -> np.ndarray:
    image = np.asarray(tagged_image)
    if image.ndim == 2:
        labels = np.asarray(image, dtype=np.int32).copy()
        labels[labels < 0] = 0
        return enforce_room_mask_contract(labels, source_arrays, clip_to_eval_domain=True)
    if image.ndim == 3 and image.shape[2] in (3, 4):
        rgb = np.asarray(image[..., :3], dtype=np.uint8)
        flat = rgb.reshape(-1, 3)
        labels = np.zeros(flat.shape[0], dtype=np.int32)
        next_label = 1
        for color in sorted({tuple(int(v) for v in row) for row in flat.tolist()}):
            if color in {(0, 0, 0), (208, 208, 208)}:
                continue
            labels[np.all(flat == np.asarray(color, dtype=np.uint8), axis=1)] = next_label
            next_label += 1
        return enforce_room_mask_contract(labels.reshape(rgb.shape[:2]), source_arrays, clip_to_eval_domain=True)
    raise ValueError(f"unsupported DUDE tagged image shape: {image.shape}")


def _snapshot_step(path: Path, arrays: Mapping[str, Any]) -> int:
    metadata_step = None
    if "step" in arrays:
        try:
            metadata_step = int(np.asarray(arrays["step"]).reshape(-1)[0])
        except Exception:
            metadata_step = None
    step, _ = parse_snapshot_step(path, path.with_suffix(".summary.json"), metadata_step=metadata_step)
    if step is None:
        raise ValueError(f"could not parse snapshot step: {path}")
    return int(step)


def _input_free_definition(arrays: Mapping[str, Any]) -> str:
    nav = arrays.get("navigation_free_room_domain")
    if nav is not None:
        arr = np.asarray(nav, dtype=bool)
        if arr.ndim == 2 and bool(arr.any()):
            return "navigation_free_room_domain"
    return "observed_free_mask_minus_obstacle_unknown"


def _map_info_metadata(
    arrays: Mapping[str, Any],
    *,
    default_resolution_m: float | None,
) -> tuple[dict[str, Any], float | None]:
    try:
        map_info = resolve_map_info(snapshot_arrays=arrays, default_resolution_m=default_resolution_m)
        return map_info.to_metadata(), float(map_info.resolution_m)
    except Exception as exc:
        return {
            "source": "missing",
            "error": str(exc),
            "note": "native DUDE threshold-in-cells cannot be reproduced without map resolution",
        }, None


def _find_inc_dude_executable(root: Path) -> Path | None:
    candidates = (
        root / "devel" / "lib" / "inc_dude" / "inc_dude",
        root.parent / "devel" / "lib" / "inc_dude" / "inc_dude",
        root.parent.parent / "devel" / "lib" / "inc_dude" / "inc_dude",
        root / "build" / "inc_dude",
    )
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def _git_head(root: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    head = proc.stdout.strip()
    return head or None


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    return Path(value)


def build_dude_rosrun_shell(config: RosSubprocessConfig, *, concavity_threshold_m: float) -> str:
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
    lines.append(
        "exec rosrun inc_dude inc_dude %s"
        % shlex.quote(str(int(round(float(concavity_threshold_m)))))
    )
    return "\n".join(lines)


def build_roscore_shell(config: RosSubprocessConfig) -> str:
    lines = ["set -euo pipefail"]
    if config.ros_setup:
        setup = shlex.quote(str(config.ros_setup))
        lines.append(f"test -f {setup}")
        lines.append("set +u")
        lines.append(f"source {setup}")
        lines.append("set -u")
    lines.append("exec roscore")
    return "\n".join(lines)


def build_ros_master_probe_shell(config: RosSubprocessConfig) -> str:
    lines = ["set -euo pipefail"]
    if config.ros_setup:
        setup = shlex.quote(str(config.ros_setup))
        lines.append(f"test -f {setup}")
        lines.append("set +u")
        lines.append(f"source {setup}")
        lines.append("set -u")
    lines.append("rostopic list >/dev/null")
    return "\n".join(lines)


def ros_master_reachable(config: RosSubprocessConfig, *, timeout_s: float = 3.0) -> bool:
    try:
        proc = subprocess.run(
            ["bash", "-lc", build_ros_master_probe_shell(config)],
            cwd=str(config.repo_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=float(timeout_s),
            check=False,
        )
    except Exception:
        return False
    return proc.returncode == 0


def _terminate_process_group(process: subprocess.Popen[bytes] | None, *, timeout_s: float = 5.0) -> None:
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
