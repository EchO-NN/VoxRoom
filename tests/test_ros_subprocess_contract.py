from __future__ import annotations

from pathlib import Path

import pytest

from voxroom_online.isaac_runtime.baselines.ros_subprocess import RosSubprocessConfig, build_ros_shell_command
from voxroom_online.isaac_runtime.baselines.offline.dude_runner import build_dude_rosrun_shell
from voxroom_online.isaac_runtime.baselines.offline.run_saved_snapshots import run


def test_ros_subprocess_shell_sources_ros_and_workspace(tmp_path: Path) -> None:
    cfg = RosSubprocessConfig(
        ros_setup="/opt/ros/noetic/setup.bash",
        ros_python="/usr/bin/python3",
        workspace_setups=("/tmp/ws/devel/setup.bash",),
        repo_root=tmp_path,
        timeout_s=10,
    )
    shell = build_ros_shell_command("pkg.entrypoint", tmp_path / "request.json", cfg)
    assert "test -f /opt/ros/noetic/setup.bash" in shell
    assert "source /opt/ros/noetic/setup.bash" in shell
    assert "source /tmp/ws/devel/setup.bash" in shell
    assert "PYTHONPATH=" in shell
    assert "/usr/bin/python3 -m pkg.entrypoint --request" in shell


def test_dude_scene_shell_starts_original_inc_dude_node(tmp_path: Path) -> None:
    cfg = RosSubprocessConfig(
        ros_setup="/opt/ros/noetic/setup.bash",
        ros_python="/usr/bin/python3",
        workspace_setups=("/tmp/dude_ws/devel/setup.bash",),
        repo_root=tmp_path,
        timeout_s=10,
    )
    shell = build_dude_rosrun_shell(cfg, concavity_threshold_m=3.0)
    assert "source /opt/ros/noetic/setup.bash" in shell
    assert "source /tmp/dude_ws/devel/setup.bash" in shell
    assert "exec rosrun inc_dude inc_dude 3" in shell


def test_strict_native_rejects_fallback_python(tmp_path: Path) -> None:
    args = type(
        "Args",
        (),
        {
            "strict_native": True,
            "fallback_python": True,
            "allow_baseline_failure": False,
            "source_run_root": str(tmp_path),
            "output_run_root": str(tmp_path),
            "baselines": ["rose2"],
            "scene_glob": "kujiale_*",
            "snapshot_glob": "roomseg_step_*.npz",
            "max_snapshots_per_scene": None,
            "overwrite": True,
        },
    )()
    with pytest.raises(ValueError, match="strict-native"):
        run(args)
