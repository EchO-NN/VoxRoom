from __future__ import annotations

from voxroom_online.isaac_runtime.baselines.offline.dude_runner import (
    build_roscore_shell,
)
from voxroom_online.isaac_runtime.baselines.ros_subprocess import (
    RosSubprocessConfig,
    runtime_ros_env_export_lines,
    runtime_ros_master_port,
)


def test_per_job_ros_master_is_restored_after_login_shell(monkeypatch) -> None:
    monkeypatch.setenv("ROS_MASTER_URI", "http://127.0.0.1:11521")
    monkeypatch.setenv("ROS_LOG_DIR", "/tmp/voxroom-ros-job")

    exports = runtime_ros_env_export_lines()
    command = build_roscore_shell(RosSubprocessConfig(ros_setup="/opt/ros/setup.bash"))

    assert "export ROS_MASTER_URI=http://127.0.0.1:11521" in exports
    assert "export ROS_LOG_DIR=/tmp/voxroom-ros-job" in exports
    assert runtime_ros_master_port() == 11521
    assert command.endswith("exec roscore -p 11521")


def test_default_roscore_command_when_master_is_not_selected(monkeypatch) -> None:
    monkeypatch.delenv("ROS_MASTER_URI", raising=False)

    command = build_roscore_shell(RosSubprocessConfig(ros_setup=None))

    assert runtime_ros_master_port() is None
    assert command.endswith("exec roscore")
