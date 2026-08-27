from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from voxroom_online.isaac_runtime.sensors.camera_geometry import CameraIntrinsics
from voxroom_online.real_runtime.geometry import (
    RigidTransform,
    camera_and_base_poses,
    matrix_to_planar_pose,
    rotation_matrix_to_rpy,
    transform_from_translation_quaternion,
)


def validate_floor_candidate(
    translation_xyz: np.ndarray,
    normal_xyz: np.ndarray,
    *,
    expected_height_m: float,
    height_tolerance_m: float,
    min_abs_normal_z: float,
) -> tuple[bool, str, float, np.ndarray]:
    translation = np.asarray(translation_xyz, dtype=np.float64).reshape(-1)
    normal = np.asarray(normal_xyz, dtype=np.float64).reshape(-1)
    empty_normal = np.zeros((3,), dtype=np.float64)
    if translation.size < 3 or normal.size < 3:
        return False, "candidate_shape_invalid", float("nan"), empty_normal
    if not np.all(np.isfinite(translation[:3])) or not np.all(np.isfinite(normal[:3])):
        return False, "candidate_non_finite", float("nan"), empty_normal
    height_m = float(translation[2])
    if height_m <= 0.0:
        return False, "camera_height_non_positive", height_m, empty_normal
    if abs(height_m - float(expected_height_m)) > float(height_tolerance_m):
        return False, "camera_height_out_of_range", height_m, empty_normal
    norm = float(np.linalg.norm(normal[:3]))
    if norm < 1.0e-8:
        return False, "floor_normal_zero", height_m, empty_normal
    normal_unit = normal[:3] / norm
    if abs(float(normal_unit[2])) < float(min_abs_normal_z):
        return False, "floor_normal_not_upward", height_m, normal_unit
    return True, "candidate_valid", height_m, normal_unit


def floor_candidates_are_consistent(
    candidates: list[tuple[float, np.ndarray]],
    *,
    required_count: int,
    max_height_spread_m: float,
    max_normal_deviation_deg: float,
) -> tuple[bool, str, float, float]:
    if len(candidates) < int(required_count):
        return False, "awaiting_consistent_candidates", float("inf"), float("inf")
    selected = candidates[-int(required_count) :]
    heights = np.asarray([item[0] for item in selected], dtype=np.float64)
    height_spread_m = float(np.max(heights) - np.min(heights))
    if height_spread_m > float(max_height_spread_m):
        return False, "candidate_height_unstable", height_spread_m, float("inf")
    normals = np.stack([np.asarray(item[1], dtype=np.float64) for item in selected], axis=0)
    reference = normals[0]
    normals = np.asarray([normal if float(np.dot(normal, reference)) >= 0.0 else -normal for normal in normals])
    mean_normal = np.mean(normals, axis=0)
    mean_norm = float(np.linalg.norm(mean_normal))
    if mean_norm < 1.0e-8:
        return False, "candidate_normal_unstable", height_spread_m, 180.0
    mean_normal /= mean_norm
    deviations = [
        math.degrees(math.acos(float(np.clip(np.dot(normal, mean_normal), -1.0, 1.0)))) for normal in normals
    ]
    max_deviation_deg = float(max(deviations, default=0.0))
    if max_deviation_deg > float(max_normal_deviation_deg):
        return False, "candidate_normal_unstable", height_spread_m, max_deviation_deg
    return True, "candidates_consistent", height_spread_m, max_deviation_deg


def validate_pose_motion(
    previous_transform: np.ndarray,
    current_transform: np.ndarray,
    *,
    dt_s: float,
    max_horizontal_speed_mps: float,
    max_vertical_speed_mps: float,
    max_angular_speed_deg_s: float,
    horizontal_slack_m: float,
    vertical_slack_m: float,
    angular_slack_deg: float,
) -> tuple[bool, str, dict[str, float]]:
    previous = np.asarray(previous_transform, dtype=np.float64)
    current = np.asarray(current_transform, dtype=np.float64)
    if previous.shape != (4, 4) or current.shape != (4, 4):
        return False, "pose_shape_invalid", {}
    if not np.all(np.isfinite(previous)) or not np.all(np.isfinite(current)):
        return False, "pose_non_finite", {}
    if not math.isfinite(float(dt_s)) or float(dt_s) <= 0.0:
        return False, "pose_timestamp_non_monotonic", {"dt_s": float(dt_s)}

    translation_delta = current[:3, 3] - previous[:3, 3]
    horizontal_m = float(np.linalg.norm(translation_delta[:2]))
    vertical_m = abs(float(translation_delta[2]))
    relative_rotation = previous[:3, :3].T @ current[:3, :3]
    rotation_cos = float(np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0))
    angular_deg = math.degrees(math.acos(rotation_cos))
    metrics = {
        "dt_s": float(dt_s),
        "horizontal_m": horizontal_m,
        "vertical_m": vertical_m,
        "angular_deg": angular_deg,
        "horizontal_limit_m": float(horizontal_slack_m) + float(max_horizontal_speed_mps) * float(dt_s),
        "vertical_limit_m": float(vertical_slack_m) + float(max_vertical_speed_mps) * float(dt_s),
        "angular_limit_deg": float(angular_slack_deg) + float(max_angular_speed_deg_s) * float(dt_s),
    }
    if horizontal_m > metrics["horizontal_limit_m"]:
        return False, "horizontal_jump", metrics
    if vertical_m > metrics["vertical_limit_m"]:
        return False, "vertical_jump", metrics
    if angular_deg > metrics["angular_limit_deg"]:
        return False, "angular_jump", metrics
    return True, "motion_valid", metrics


@dataclass
class ZedFrame:
    image_timestamp_ns: int
    host_timestamp_ns: int
    tracking_state: str
    camera_pose_world: tuple[float, float, float, float] | None
    base_pose_world: tuple[float, float, float, float] | None
    camera_transform_world: np.ndarray | None
    base_transform_world: np.ndarray | None
    camera_rpy_deg: tuple[float, float, float] | None
    base_rpy_deg: tuple[float, float, float] | None
    depth_m: np.ndarray | None
    intrinsics: CameraIntrinsics | None
    imu_linear_acceleration: tuple[float, float, float] | None
    imu_angular_velocity: tuple[float, float, float] | None
    pose_confidence: int | None = None
    odometry_status: str | None = None
    spatial_memory_status: str | None = None
    tracking_fusion_status: str | None = None
    global_camera_pose_world: tuple[float, float, float, float] | None = None
    global_camera_rpy_deg: tuple[float, float, float] | None = None


class Zed2iSource:
    """Direct PyZED source synchronized to the grabbed image timestamp."""

    def __init__(self, config: dict, extrinsics: RigidTransform):
        self.config = dict(config or {})
        self.extrinsics = extrinsics
        self._t_base_camera = extrinsics.matrix()
        self._t_world_zed: np.ndarray | None = None
        self._sl: Any = None
        self._camera: Any = None
        self._runtime_parameters: Any = None
        self._pose: Any = None
        self._delta_pose: Any = None
        self._t_odom_camera: np.ndarray | None = None
        self._depth: Any = None
        self._sensors: Any = None
        self._intrinsics_full: CameraIntrinsics | None = None
        self._downsample = max(1, int(self.config.get("depth_downsample", 1)))
        tracking_cfg = dict(self.config.get("tracking", {}) or {})
        self._enable_area_memory = bool(tracking_cfg.get("enable_area_memory", True))
        self._reset_odom_with_loop_closure = bool(
            tracking_cfg.get("reset_odom_with_loop_closure", False)
        )
        if self._reset_odom_with_loop_closure:
            raise ValueError(
                "handheld ZED mapping requires tracking.reset_odom_with_loop_closure=false"
            )
        self._enable_pose_smoothing = bool(tracking_cfg.get("enable_pose_smoothing", False))
        self._require_detailed_tracking_status = bool(tracking_cfg.get("require_detailed_tracking_status", True))
        continuity_cfg = dict(tracking_cfg.get("pose_continuity", {}) or {})
        self._pose_continuity_enabled = bool(continuity_cfg.get("enabled", True))
        self._max_horizontal_speed_mps = max(0.01, float(continuity_cfg.get("max_horizontal_speed_mps", 2.0)))
        self._max_vertical_speed_mps = max(0.01, float(continuity_cfg.get("max_vertical_speed_mps", 1.0)))
        self._max_angular_speed_deg_s = max(1.0, float(continuity_cfg.get("max_angular_speed_deg_s", 180.0)))
        self._horizontal_slack_m = max(0.0, float(continuity_cfg.get("horizontal_slack_m", 0.15)))
        self._vertical_slack_m = max(0.0, float(continuity_cfg.get("vertical_slack_m", 0.10)))
        self._angular_slack_deg = max(0.0, float(continuity_cfg.get("angular_slack_deg", 10.0)))
        self._pose_discontinuity_timeout_s = max(
            0.1, float(continuity_cfg.get("violation_timeout_s", 2.0))
        )
        self._last_accepted_camera_transform: np.ndarray | None = None
        self._last_accepted_image_timestamp_ns: int | None = None
        self._pose_discontinuity_started_at: float | None = None
        self._pose_discontinuity_reason = ""
        self._set_floor_as_origin = bool(tracking_cfg.get("set_floor_as_origin", False))
        floor_cfg = dict(tracking_cfg.get("floor_alignment", {}) or {})
        self._floor_alignment_enabled = bool(floor_cfg.get("enabled", False))
        if self._set_floor_as_origin and self._floor_alignment_enabled:
            raise ValueError("choose either set_floor_as_origin or validated floor_alignment, not both")
        self._preserve_zed_floor_z = bool(self._set_floor_as_origin or self._floor_alignment_enabled)
        self._floor_expected_height_m = float(floor_cfg.get("expected_camera_height_m", extrinsics.xyz_m[2]))
        self._floor_height_tolerance_m = max(0.01, float(floor_cfg.get("height_tolerance_m", 0.30)))
        self._floor_min_consistent = max(1, int(floor_cfg.get("min_consistent_detections", 5)))
        self._floor_max_height_spread_m = max(0.001, float(floor_cfg.get("max_height_spread_m", 0.05)))
        self._floor_min_abs_normal_z = float(np.clip(floor_cfg.get("min_abs_normal_z", 0.75), 0.0, 1.0))
        self._floor_max_normal_deviation_deg = max(0.1, float(floor_cfg.get("max_normal_deviation_deg", 4.0)))
        self._floor_timeout_s = max(1.0, float(floor_cfg.get("timeout_s", 15.0)))
        self._floor_aligned = not self._floor_alignment_enabled
        self._floor_candidates: list[tuple[float, np.ndarray]] = []
        self._floor_search_started_at: float | None = None
        self._floor_last_reason = "not_started"
        self._floor_plane: Any = None
        self._floor_reset_transform: Any = None
        self.finished = False
        self.camera_info: dict[str, object] = {}

    def open(self) -> None:
        try:
            import pyzed.sl as sl
        except Exception as exc:
            raise RuntimeError(
                "PyZED is not importable. Install the ZED SDK first, activate this Python environment, "
                "then run /usr/local/zed/get_python_api.py."
            ) from exc

        self._sl = sl
        camera = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = self._enum_value(sl.RESOLUTION, str(self.config.get("resolution", "HD720")))
        requested_fps = int(self.config.get("fps", 60))
        init.camera_fps = requested_fps
        init.depth_mode = self._enum_value(sl.DEPTH_MODE, str(self.config.get("depth_mode", "NEURAL_LIGHT")))
        init.coordinate_units = sl.UNIT.METER
        init.coordinate_system = self._zed_z_up_x_forward_coordinate_system(sl.COORDINATE_SYSTEM)
        if hasattr(init, "depth_minimum_distance"):
            init.depth_minimum_distance = float(self.config.get("depth_min_m", 0.2))
        if hasattr(init, "depth_maximum_distance"):
            init.depth_maximum_distance = float(self.config.get("depth_max_m", 3.0))
        if hasattr(init, "depth_stabilization"):
            init.depth_stabilization = int(self.config.get("depth_stabilization", 30))
        if hasattr(init, "sdk_verbose"):
            init.sdk_verbose = bool(self.config.get("sdk_verbose", False))

        svo_path = str(self.config.get("svo_path", "") or "").strip()
        if svo_path:
            path = Path(svo_path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"SVO file not found: {path}")
            init.set_from_svo_file(str(path))
            if hasattr(init, "svo_real_time_mode"):
                init.svo_real_time_mode = bool(self.config.get("svo_real_time_mode", False))
        else:
            serial_number = int(self.config.get("serial_number", 0) or 0)
            if serial_number > 0 and hasattr(init, "set_from_serial_number"):
                init.set_from_serial_number(serial_number)

        error = camera.open(init)
        if not self._is_success(error):
            raise RuntimeError(f"failed to open ZED camera: {error}")

        tracking_cfg = dict(self.config.get("tracking", {}) or {})
        tracking = sl.PositionalTrackingParameters()
        if hasattr(tracking, "mode") and hasattr(sl, "POSITIONAL_TRACKING_MODE"):
            tracking.mode = self._enum_value(
                sl.POSITIONAL_TRACKING_MODE,
                str(tracking_cfg.get("mode", "GEN_3")),
            )
        self._set_if_present(tracking, "enable_imu_fusion", bool(tracking_cfg.get("enable_imu_fusion", True)))
        self._set_if_present(tracking, "set_gravity_as_origin", bool(tracking_cfg.get("set_gravity_as_origin", True)))
        self._set_if_present(tracking, "enable_2d_ground_mode", bool(tracking_cfg.get("enable_2d_ground_mode", False)))
        self._set_if_present(tracking, "enable_area_memory", self._enable_area_memory)
        self._set_if_present(tracking, "enable_pose_smoothing", self._enable_pose_smoothing)
        if self._set_floor_as_origin and not hasattr(tracking, "set_floor_as_origin"):
            camera.close()
            raise RuntimeError("this PyZED build does not expose set_floor_as_origin")
        self._set_if_present(tracking, "set_floor_as_origin", self._set_floor_as_origin)
        area_file = str(tracking_cfg.get("area_file_path", "") or "").strip()
        if area_file and hasattr(tracking, "area_file_path"):
            tracking.area_file_path = str(Path(area_file).expanduser().resolve())

        error = camera.enable_positional_tracking(tracking)
        if not self._is_success(error):
            camera.close()
            raise RuntimeError(f"failed to enable ZED positional tracking: {error}")
        if self._require_detailed_tracking_status and not hasattr(camera, "get_positional_tracking_status"):
            camera.close()
            raise RuntimeError("this PyZED build does not expose detailed positional tracking status")

        runtime = sl.RuntimeParameters()
        if hasattr(runtime, "confidence_threshold"):
            runtime.confidence_threshold = int(self.config.get("confidence_threshold", 50))
        if hasattr(runtime, "texture_confidence_threshold"):
            runtime.texture_confidence_threshold = int(self.config.get("texture_confidence_threshold", 100))

        info = camera.get_camera_information()
        calibration = info.camera_configuration.calibration_parameters
        left = calibration.left_cam
        resolution = getattr(info.camera_configuration, "resolution", None)
        width, height = self._resolution_size(resolution)
        if width <= 0 or height <= 0:
            width, height = self._resolution_size(getattr(info, "camera_resolution", None))
        if width <= 0 or height <= 0:
            width, height = self._resolution_size(init.camera_resolution)
        if width <= 0 or height <= 0:
            raise RuntimeError("ZED SDK did not report a valid rectified image resolution")
        actual_fps = int(getattr(info.camera_configuration, "fps", 0) or 0)
        if actual_fps <= 0:
            actual_fps = requested_fps
        self._intrinsics_full = CameraIntrinsics(
            width=width,
            height=height,
            fx=float(left.fx),
            fy=float(left.fy),
            cx=float(left.cx),
            cy=float(left.cy),
        )
        self.camera_info = {
            "serial_number": int(getattr(info, "serial_number", 0) or 0),
            "camera_model": str(getattr(info, "camera_model", "unknown")),
            "firmware_version": int(getattr(info.camera_configuration, "firmware_version", 0) or 0),
            "resolution": [width, height],
            "requested_fps": requested_fps,
            "fps": actual_fps,
            "depth_mode": str(init.depth_mode),
            "coordinate_system": str(init.coordinate_system),
            "pose_mode": "se3",
            "mapping_pose_reference": "camera_relative_odometry",
            "diagnostic_pose_reference": "world",
            "tracking": {
                "mode": str(getattr(tracking, "mode", tracking_cfg.get("mode", "GEN_3"))),
                "enable_imu_fusion": bool(tracking_cfg.get("enable_imu_fusion", True)),
                "set_gravity_as_origin": bool(tracking_cfg.get("set_gravity_as_origin", True)),
                "enable_2d_ground_mode": bool(tracking_cfg.get("enable_2d_ground_mode", False)),
                "enable_area_memory": self._enable_area_memory,
                "reset_odom_with_loop_closure": self._reset_odom_with_loop_closure,
                "enable_pose_smoothing": self._enable_pose_smoothing,
                "require_detailed_tracking_status": self._require_detailed_tracking_status,
                "set_floor_as_origin": self._set_floor_as_origin,
                "validated_floor_alignment": {
                    "enabled": self._floor_alignment_enabled,
                    "expected_camera_height_m": self._floor_expected_height_m,
                    "height_tolerance_m": self._floor_height_tolerance_m,
                    "min_consistent_detections": self._floor_min_consistent,
                    "timeout_s": self._floor_timeout_s,
                },
                "pose_continuity": {
                    "enabled": self._pose_continuity_enabled,
                    "max_horizontal_speed_mps": self._max_horizontal_speed_mps,
                    "max_vertical_speed_mps": self._max_vertical_speed_mps,
                    "max_angular_speed_deg_s": self._max_angular_speed_deg_s,
                    "violation_timeout_s": self._pose_discontinuity_timeout_s,
                },
            },
            "intrinsics": {
                "fx": float(left.fx),
                "fy": float(left.fy),
                "cx": float(left.cx),
                "cy": float(left.cy),
            },
        }

        self._camera = camera
        self._runtime_parameters = runtime
        self._pose = sl.Pose()
        self._delta_pose = sl.Pose()
        self._depth = sl.Mat()
        self._sensors = sl.SensorsData()
        if self._floor_alignment_enabled:
            if not hasattr(camera, "find_floor_plane") or not hasattr(camera, "reset_positional_tracking"):
                self.close()
                raise RuntimeError("this PyZED build does not expose validated floor-plane alignment APIs")
            self._floor_plane = sl.Plane()
            self._floor_reset_transform = sl.Transform()
            self._floor_search_started_at = time.monotonic()

    def close(self) -> None:
        camera = self._camera
        self._camera = None
        self._delta_pose = None
        self._t_odom_camera = None
        self._floor_plane = None
        self._floor_reset_transform = None
        if camera is None:
            return
        try:
            if hasattr(camera, "disable_positional_tracking"):
                camera.disable_positional_tracking()
        finally:
            camera.close()

    def grab(self, *, retrieve_depth: bool) -> ZedFrame | None:
        if self._camera is None:
            raise RuntimeError("ZED camera is not open")
        sl = self._sl
        error = self._camera.grab(self._runtime_parameters)
        if not self._is_success(error):
            name = self._enum_name(error)
            if "END_OF_SVOFILE" in name.upper():
                self.finished = True
            return None

        host_ns = time.time_ns()
        image_ns = self._image_timestamp_ns()
        tracking_state_obj = self._camera.get_position(self._pose, sl.REFERENCE_FRAME.WORLD)
        tracking_state = self._enum_name(tracking_state_obj)
        odometry_status, spatial_memory_status, tracking_fusion_status = self._tracking_status_names()
        pose_confidence = self._pose_confidence_value()
        if self._floor_alignment_enabled and not self._floor_aligned:
            tracking_state = self._advance_validated_floor_alignment(tracking_state)

        if tracking_state.strip().upper() == "OK" and self._require_detailed_tracking_status:
            if odometry_status != "OK":
                tracking_state = f"ODOMETRY_{odometry_status}"
            elif self._enable_area_memory and spatial_memory_status in {
                "LOST",
                "NOT_ENOUGH_MEMORY_FOR_TRACKING",
                "OFF",
            }:
                tracking_state = f"SPATIAL_MEMORY_{spatial_memory_status}"
            elif not bool(getattr(self._pose, "valid", True)):
                tracking_state = "POSE_INVALID"

        camera_pose = None
        base_pose = None
        camera_transform = None
        base_transform = None
        camera_rpy_deg = None
        base_rpy_deg = None
        global_camera_pose = None
        global_camera_rpy_deg = None
        tracking_ok = tracking_state.strip().upper() == "OK"
        if tracking_ok:
            try:
                translation = self._pose_translation(self._pose)
                quaternion = self._pose_quaternion(self._pose)
                t_zed_camera = transform_from_translation_quaternion(translation, quaternion)
                if self._t_odom_camera is None:
                    t_odom_camera, _, self._t_world_zed = camera_and_base_poses(
                        t_zed_camera,
                        self._t_base_camera,
                        self._t_world_zed,
                        preserve_zed_floor_z=self._preserve_zed_floor_z,
                    )
                else:
                    delta_state_obj = self._camera.get_position(self._delta_pose, sl.REFERENCE_FRAME.CAMERA)
                    delta_state = self._enum_name(delta_state_obj).strip().upper()
                    if delta_state != "OK":
                        tracking_state = f"DELTA_ODOMETRY_{delta_state}"
                        t_odom_camera = None
                    else:
                        delta_translation = self._pose_translation(self._delta_pose)
                        delta_quaternion = self._pose_quaternion(self._delta_pose)
                        t_previous_current = transform_from_translation_quaternion(
                            delta_translation,
                            delta_quaternion,
                        )
                        t_odom_camera = self._t_odom_camera @ t_previous_current

                if self._t_world_zed is not None:
                    t_global_camera = self._t_world_zed @ t_zed_camera
                    global_camera_pose = matrix_to_planar_pose(t_global_camera)
                    global_camera_rpy_deg = tuple(
                        float(np.degrees(v)) for v in rotation_matrix_to_rpy(t_global_camera[:3, :3])
                    )

                if t_odom_camera is not None:
                    t_odom_base = t_odom_camera @ np.linalg.inv(self._t_base_camera)
                    camera_pose = matrix_to_planar_pose(t_odom_camera)
                    base_pose = matrix_to_planar_pose(t_odom_base)
                    camera_transform = np.asarray(t_odom_camera, dtype=np.float64).copy()
                    base_transform = np.asarray(t_odom_base, dtype=np.float64).copy()
                    camera_rpy_deg = tuple(
                        float(np.degrees(v)) for v in rotation_matrix_to_rpy(t_odom_camera[:3, :3])
                    )
                    base_rpy_deg = tuple(
                        float(np.degrees(v)) for v in rotation_matrix_to_rpy(t_odom_base[:3, :3])
                    )
            except Exception:
                camera_pose = None
                base_pose = None
                camera_transform = None
                base_transform = None

        # Keep continuity failures outside the pose-decoding guard: a persistent
        # physical impossibility is fatal and must not be converted into a
        # generic missing-pose frame.
        if tracking_state.strip().upper() == "OK" and camera_transform is not None:
            tracking_state = self._validate_pose_continuity(camera_transform, image_ns)
            if tracking_state == "OK":
                self._t_odom_camera = camera_transform.copy()

        depth = None
        intr = None
        if retrieve_depth:
            depth_error = self._camera.retrieve_measure(self._depth, sl.MEASURE.DEPTH)
            if self._is_success(depth_error):
                raw = np.asarray(self._depth.get_data(), dtype=np.float32)
                if raw.ndim == 3:
                    raw = raw[:, :, 0]
                if raw.ndim != 2:
                    raise RuntimeError(f"unexpected ZED depth shape: {raw.shape}")
                depth = np.ascontiguousarray(raw[:: self._downsample, :: self._downsample], dtype=np.float32)
                intr = self._scaled_intrinsics(depth.shape)

        accel, gyro = self._read_imu_at_image_time()
        return ZedFrame(
            image_timestamp_ns=int(image_ns),
            host_timestamp_ns=int(host_ns),
            tracking_state=tracking_state,
            camera_pose_world=camera_pose,
            base_pose_world=base_pose,
            camera_transform_world=camera_transform,
            base_transform_world=base_transform,
            camera_rpy_deg=camera_rpy_deg,
            base_rpy_deg=base_rpy_deg,
            depth_m=depth,
            intrinsics=intr,
            imu_linear_acceleration=accel,
            imu_angular_velocity=gyro,
            pose_confidence=pose_confidence,
            odometry_status=odometry_status,
            spatial_memory_status=spatial_memory_status,
            tracking_fusion_status=tracking_fusion_status,
            global_camera_pose_world=global_camera_pose,
            global_camera_rpy_deg=global_camera_rpy_deg,
        )

    def _advance_validated_floor_alignment(self, tracking_state: str) -> str:
        if self._floor_search_started_at is None:
            self._floor_search_started_at = time.monotonic()
        elapsed_s = time.monotonic() - self._floor_search_started_at
        if elapsed_s > self._floor_timeout_s:
            raise RuntimeError(
                "validated ZED floor alignment timed out after %.1fs; last rejection: %s"
                % (elapsed_s, self._floor_last_reason)
            )
        if str(tracking_state).strip().upper() != "OK":
            self._floor_candidates.clear()
            self._floor_last_reason = f"tracking_{str(tracking_state).strip().lower()}"
            return "SEARCHING_FLOOR_PLANE"

        assert self._camera is not None and self._floor_plane is not None and self._floor_reset_transform is not None
        status = self._camera.find_floor_plane(
            self._floor_plane,
            self._floor_reset_transform,
            float(self._floor_expected_height_m),
            None,
            float(self._floor_height_tolerance_m),
        )
        if not self._is_success(status):
            self._floor_candidates.clear()
            self._floor_last_reason = f"find_floor_plane_{self._enum_name(status).lower()}"
            return "SEARCHING_FLOOR_PLANE"

        translation = np.asarray(self._unwrap_vector(self._floor_reset_transform.get_translation()), dtype=np.float64).reshape(-1)
        normal = np.asarray(self._floor_plane.get_normal(), dtype=np.float64).reshape(-1)
        valid, reason, height_m, normal_unit = validate_floor_candidate(
            translation,
            normal,
            expected_height_m=self._floor_expected_height_m,
            height_tolerance_m=self._floor_height_tolerance_m,
            min_abs_normal_z=self._floor_min_abs_normal_z,
        )
        if not valid:
            self._floor_candidates.clear()
            self._floor_last_reason = reason
            return "SEARCHING_FLOOR_PLANE"

        self._floor_candidates.append((height_m, normal_unit))
        self._floor_candidates = self._floor_candidates[-self._floor_min_consistent :]
        consistent, reason, height_spread_m, max_normal_deviation_deg = floor_candidates_are_consistent(
            self._floor_candidates,
            required_count=self._floor_min_consistent,
            max_height_spread_m=self._floor_max_height_spread_m,
            max_normal_deviation_deg=self._floor_max_normal_deviation_deg,
        )
        self._floor_last_reason = reason
        if not consistent:
            return "SEARCHING_FLOOR_PLANE"

        reset_status = self._camera.reset_positional_tracking(self._floor_reset_transform)
        if not self._is_success(reset_status):
            raise RuntimeError(f"failed to reset ZED tracking to validated floor: {reset_status}")
        self._floor_aligned = True
        self._t_world_zed = None
        self._t_odom_camera = None
        self._last_accepted_camera_transform = None
        self._last_accepted_image_timestamp_ns = None
        self._pose_discontinuity_started_at = None
        heights = [candidate[0] for candidate in self._floor_candidates]
        print(
            "[zed-floor] aligned height_m=%.3f spread_m=%.4f normal_deviation_deg=%.2f samples=%d"
            % (
                float(np.mean(heights)),
                float(height_spread_m),
                float(max_normal_deviation_deg),
                len(self._floor_candidates),
            ),
            flush=True,
        )
        return "FLOOR_ALIGNMENT_RESET"

    def _tracking_status_names(self) -> tuple[str, str, str]:
        if not self._require_detailed_tracking_status:
            return "NOT_REQUIRED", "NOT_REQUIRED", "NOT_REQUIRED"
        try:
            status = self._camera.get_positional_tracking_status()
            odometry = self._enum_name(status.odometry_status).strip().upper()
            spatial_memory = self._enum_name(status.spatial_memory_status).strip().upper()
            fusion = self._enum_name(getattr(status, "tracking_fusion_status", "UNAVAILABLE")).strip().upper()
        except Exception as exc:
            raise RuntimeError("failed to read detailed ZED positional tracking status") from exc
        if not odometry or not spatial_memory:
            raise RuntimeError("ZED returned an empty detailed positional tracking status")
        return odometry, spatial_memory, fusion

    def _pose_confidence_value(self) -> int | None:
        try:
            value = int(self._pose.pose_confidence)
        except (AttributeError, TypeError, ValueError):
            return None
        return value if value >= 0 else None

    def _validate_pose_continuity(self, camera_transform: np.ndarray, image_timestamp_ns: int) -> str:
        current = np.asarray(camera_transform, dtype=np.float64)
        previous = self._last_accepted_camera_transform
        previous_ns = self._last_accepted_image_timestamp_ns
        if not self._pose_continuity_enabled or previous is None or previous_ns is None:
            self._last_accepted_camera_transform = current.copy()
            self._last_accepted_image_timestamp_ns = int(image_timestamp_ns)
            self._pose_discontinuity_started_at = None
            self._pose_discontinuity_reason = ""
            return "OK"
        if int(image_timestamp_ns) == int(previous_ns):
            return "DUPLICATE_IMAGE_TIMESTAMP"

        valid, reason, metrics = validate_pose_motion(
            previous,
            current,
            dt_s=(int(image_timestamp_ns) - int(previous_ns)) * 1.0e-9,
            max_horizontal_speed_mps=self._max_horizontal_speed_mps,
            max_vertical_speed_mps=self._max_vertical_speed_mps,
            max_angular_speed_deg_s=self._max_angular_speed_deg_s,
            horizontal_slack_m=self._horizontal_slack_m,
            vertical_slack_m=self._vertical_slack_m,
            angular_slack_deg=self._angular_slack_deg,
        )
        if valid:
            if self._pose_discontinuity_started_at is not None:
                print("[zed-tracking] pose continuity recovered", flush=True)
            self._last_accepted_camera_transform = current.copy()
            self._last_accepted_image_timestamp_ns = int(image_timestamp_ns)
            self._pose_discontinuity_started_at = None
            self._pose_discontinuity_reason = ""
            return "OK"

        now = time.monotonic()
        if self._pose_discontinuity_started_at is None:
            self._pose_discontinuity_started_at = now
            self._pose_discontinuity_reason = reason
            print(
                "[zed-tracking] rejected pose discontinuity reason=%s metrics=%s"
                % (reason, metrics),
                flush=True,
            )
        if now - self._pose_discontinuity_started_at > self._pose_discontinuity_timeout_s:
            raise RuntimeError(
                "ZED pose discontinuity persisted for %.1fs; mapping stopped before corrupting the voxel map; "
                "reason=%s metrics=%s"
                % (self._pose_discontinuity_timeout_s, self._pose_discontinuity_reason, metrics)
            )
        return "POSE_DISCONTINUITY"

    def _scaled_intrinsics(self, shape: tuple[int, int]) -> CameraIntrinsics:
        if self._intrinsics_full is None:
            raise RuntimeError("camera intrinsics are unavailable")
        h, w = int(shape[0]), int(shape[1])
        scale = float(self._downsample)
        return CameraIntrinsics(
            width=w,
            height=h,
            fx=float(self._intrinsics_full.fx) / scale,
            fy=float(self._intrinsics_full.fy) / scale,
            cx=float(self._intrinsics_full.cx) / scale,
            cy=float(self._intrinsics_full.cy) / scale,
        )

    def _read_imu_at_image_time(self) -> tuple[tuple[float, float, float] | None, tuple[float, float, float] | None]:
        try:
            error = self._camera.get_sensors_data(self._sensors, self._sl.TIME_REFERENCE.IMAGE)
            if not self._is_success(error):
                return None, None
            imu = self._sensors.get_imu_data()
            accel = self._vector3(imu.get_linear_acceleration())
            gyro = self._vector3(imu.get_angular_velocity())
            return accel, gyro
        except Exception:
            return None, None

    def _image_timestamp_ns(self) -> int:
        try:
            timestamp = self._camera.get_timestamp(self._sl.TIME_REFERENCE.IMAGE)
            if hasattr(timestamp, "get_nanoseconds"):
                return int(timestamp.get_nanoseconds())
            if hasattr(timestamp, "get_microseconds"):
                return int(timestamp.get_microseconds()) * 1000
        except Exception:
            pass
        return time.time_ns()

    def _pose_translation(self, pose: Any) -> np.ndarray:
        try:
            value = pose.get_translation()
        except TypeError:
            value = self._sl.Translation()
            pose.get_translation(value)
        return np.asarray(self._unwrap_vector(value), dtype=np.float64).reshape(3)

    def _pose_quaternion(self, pose: Any) -> np.ndarray:
        try:
            value = pose.get_orientation()
        except TypeError:
            value = self._sl.Orientation()
            pose.get_orientation(value)
        return np.asarray(self._unwrap_vector(value), dtype=np.float64).reshape(4)

    @staticmethod
    def _unwrap_vector(value: Any) -> Any:
        return value.get() if hasattr(value, "get") else value

    @classmethod
    def _vector3(cls, value: Any) -> tuple[float, float, float] | None:
        try:
            arr = np.asarray(cls._unwrap_vector(value), dtype=np.float64).reshape(-1)
            if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
                return None
            return float(arr[0]), float(arr[1]), float(arr[2])
        except Exception:
            return None

    def _is_success(self, value: Any) -> bool:
        try:
            return value == self._sl.ERROR_CODE.SUCCESS
        except Exception:
            return "SUCCESS" in self._enum_name(value).upper()

    @staticmethod
    def _enum_name(value: Any) -> str:
        name = getattr(value, "name", None)
        if name:
            return str(name)
        text = str(value)
        return text.rsplit(".", 1)[-1]

    @staticmethod
    def _set_if_present(obj: Any, name: str, value: Any) -> None:
        if hasattr(obj, name):
            setattr(obj, name, value)

    @staticmethod
    def _resolution_size(value: Any) -> tuple[int, int]:
        if value is None:
            return 0, 0
        try:
            return int(getattr(value, "width", 0) or 0), int(getattr(value, "height", 0) or 0)
        except (TypeError, ValueError):
            return 0, 0

    @staticmethod
    def _enum_value(namespace: Any, name: str) -> Any:
        key = str(name).strip().upper()
        if not hasattr(namespace, key):
            available = sorted(k for k in dir(namespace) if k.isupper() and not k.startswith("_"))
            raise ValueError(f"unsupported enum value {key!r}; available values include: {available[:20]}")
        return getattr(namespace, key)

    @staticmethod
    def _zed_z_up_x_forward_coordinate_system(namespace: Any) -> Any:
        # The SDK renamed this exact coordinate convention between releases.
        for name in ("RIGHT_HANDED_Z_UP_X_FWD", "RIGHT_HANDED_Z_UP_X_FORWARD"):
            if hasattr(namespace, name):
                return getattr(namespace, name)
        available = sorted(k for k in dir(namespace) if k.isupper() and not k.startswith("_"))
        raise ValueError(
            "ZED SDK lacks the required RIGHT_HANDED_Z_UP_X_FORWARD coordinate convention; "
            f"available values include: {available[:20]}"
        )
