from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .ros_bridge_payload import load_result, save_input_arrays, write_request
from .offline.base import BaselineResult, MissingOriginalImplementationError


@dataclass(frozen=True)
class RosSubprocessConfig:
    ros_setup: str | None = None
    ros_python: str = "python3"
    workspace_setups: tuple[str, ...] = ()
    repo_root: Path = Path(__file__).resolve().parents[3]
    timeout_s: float = 180.0

    @classmethod
    def from_env(
        cls,
        *,
        ros_setup: str | None = None,
        ros_python: str | None = None,
        workspace_roots: Sequence[Path | str | None] = (),
        timeout_s: float = 180.0,
    ) -> "RosSubprocessConfig":
        setups: list[str] = []
        for root in workspace_roots:
            if root is None:
                continue
            path = Path(root).expanduser()
            for relative in ("devel/setup.bash", "install/setup.bash", "setup.bash"):
                candidate = path / relative
                if candidate.exists():
                    setups.append(str(candidate))
                    break
        return cls(
            ros_setup=ros_setup or os.environ.get("ROS_BASELINE_SETUP") or "/opt/ros/noetic/setup.bash",
            ros_python=ros_python or os.environ.get("ROS_BASELINE_PYTHON") or "python3",
            workspace_setups=tuple(setups),
            repo_root=Path(os.environ.get("VOXROOM_REPO_ROOT", Path(__file__).resolve().parents[3])).resolve(),
            timeout_s=float(timeout_s),
        )


def run_ros_module(
    module: str,
    *,
    method: str,
    snapshot_path: Path | str,
    arrays: Mapping[str, Any],
    scene_id: str | None,
    params: Mapping[str, Any] | None,
    config: RosSubprocessConfig,
) -> BaselineResult:
    with tempfile.TemporaryDirectory(prefix="voxroom_ros_bridge_") as tmp:
        tmp_path = Path(tmp)
        input_npz = tmp_path / "input_arrays.npz"
        output_npz = tmp_path / "result.npz"
        request_json = tmp_path / "request.json"
        save_input_arrays(input_npz, arrays)
        write_request(
            request_json,
            method=method,
            snapshot_path=snapshot_path,
            input_npz=input_npz,
            output_npz=output_npz,
            scene_id=scene_id,
            params=params or {},
        )
        shell = _shell_command(module=module, request_json=request_json, config=config)
        env = dict(os.environ)
        proc = subprocess.run(
            ["bash", "-lc", shell],
            cwd=str(config.repo_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=float(config.timeout_s),
            check=False,
        )
        if proc.returncode != 0:
            raise MissingOriginalImplementationError(
                "ROS subprocess failed for %s via %s (returncode=%s)\nstdout:\n%s\nstderr:\n%s"
                % (method, module, proc.returncode, proc.stdout[-4000:], proc.stderr[-4000:])
            )
        if not output_npz.exists():
            raise MissingOriginalImplementationError(
                "ROS subprocess %s for %s exited without writing %s\nstdout:\n%s\nstderr:\n%s"
                % (module, method, output_npz, proc.stdout[-4000:], proc.stderr[-4000:])
            )
        label_map, metadata, debug = load_result(output_npz)
        return BaselineResult(label_map=label_map, metadata=metadata, debug_arrays=debug)


def build_ros_shell_command(module: str, request_json: Path | str, config: RosSubprocessConfig) -> str:
    return _shell_command(module=module, request_json=Path(request_json), config=config)


def _shell_command(*, module: str, request_json: Path, config: RosSubprocessConfig) -> str:
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
    repo = shlex.quote(str(config.repo_root))
    lines.append(f"export PYTHONPATH={repo}:\"${{PYTHONPATH:-}}\"")
    lines.append(
        "exec %s -m %s --request %s"
        % (shlex.quote(str(config.ros_python)), shlex.quote(str(module)), shlex.quote(str(request_json)))
    )
    return "\n".join(lines)
