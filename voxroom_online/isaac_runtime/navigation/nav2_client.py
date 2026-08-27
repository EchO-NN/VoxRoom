from __future__ import annotations


class Nav2UnavailableError(RuntimeError):
    pass


class Nav2NavigateToPoseClient:
    def __init__(self, *args, **kwargs):
        try:
            import rclpy  # noqa: F401
            from nav2_msgs.action import NavigateToPose  # noqa: F401
        except Exception as exc:
            raise Nav2UnavailableError(
                "ROS2/Nav2 is not installed in this environment. A* mode is the first-pass supported planner."
            ) from exc

