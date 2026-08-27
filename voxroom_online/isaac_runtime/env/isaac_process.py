from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import List, Optional, Tuple

import numpy as np

from voxroom_online.isaac_runtime.config import str_to_bool
from voxroom_online.isaac_runtime.sensors.camera_geometry import CameraIntrinsics
from voxroom_online.isaac_runtime.sensors.depth_backproject import distance_to_camera_to_image_plane_depth


def yaw_to_quat_wxyz(yaw: float) -> np.ndarray:
    return np.asarray([math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5)], dtype=np.float32)


def _depth_value_to_array(value) -> Optional[np.ndarray]:
    if isinstance(value, dict):
        value = value.get("data")
    if value is None:
        return None
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        try:
            value = value.detach().cpu().numpy()
        except Exception:
            pass
    elif hasattr(value, "cpu") and hasattr(value, "numpy"):
        try:
            value = value.cpu().numpy()
        except Exception:
            pass
    elif hasattr(value, "numpy"):
        try:
            value = value.numpy()
        except Exception:
            pass
    try:
        array = np.asarray(value, dtype=np.float32)
    except Exception:
        return None
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[:, :, 0]
    if array.ndim != 2:
        return None
    return array


def _rgb_value_to_array(value) -> Optional[np.ndarray]:
    if isinstance(value, dict):
        value = value.get("data")
    if value is None:
        return None
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        try:
            value = value.detach().cpu().numpy()
        except Exception:
            pass
    elif hasattr(value, "cpu") and hasattr(value, "numpy"):
        try:
            value = value.cpu().numpy()
        except Exception:
            pass
    elif hasattr(value, "numpy"):
        try:
            value = value.numpy()
        except Exception:
            pass
    try:
        array = np.asarray(value)
    except Exception:
        return None
    if array.ndim != 3 or array.shape[2] < 3:
        return None
    return array[:, :, :3].astype(np.uint8, copy=False)


def _value_debug_description(value) -> str:
    if value is None:
        return "None"
    if isinstance(value, dict):
        keys = ",".join(str(key) for key in list(value.keys())[:8])
        return "dict(keys=[%s], data=%s)" % (keys, _value_debug_description(value.get("data")))
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    device = getattr(value, "device", None)
    pieces = [type(value).__module__ + "." + type(value).__name__]
    if shape is not None:
        pieces.append("shape=%s" % (tuple(shape) if not isinstance(shape, str) else shape,))
    if dtype is not None:
        pieces.append("dtype=%s" % dtype)
    if device is not None:
        pieces.append("device=%s" % device)
    return " ".join(pieces)


def _hashable_frame_value(value):
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.reshape(-1)[0].item()
        return tuple(value.reshape(-1).tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        return tuple(_hashable_frame_value(v) for v in value)
    return value


class _KinematicUsdPosePrim:
    """Small pose wrapper for a visual-only USD prim.

    Closed-loop Isaac benchmark runs integrate robot motion kinematically.  We
    still want Kaya visible in the GUI, but we do not need the Kaya wheel
    articulation to participate in PhysX.  This wrapper gives the rest of this
    module the one method it needs from the robot object: set_world_pose.
    """

    def __init__(self, prim_path: str) -> None:
        import omni.usd
        from pxr import UsdGeom

        self.prim_path = str(prim_path)
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Isaac stage is unavailable; cannot bind Kaya visual prim")
        prim = stage.GetPrimAtPath(self.prim_path)
        if prim is None or not prim.IsValid():
            raise RuntimeError("Kaya visual prim is unavailable at %s" % self.prim_path)
        self._prim = prim
        self._xform_api = UsdGeom.XformCommonAPI(prim)

    def set_world_pose(self, *, position, orientation) -> None:
        from pxr import Gf, UsdGeom

        pos = [float(v) for v in position]
        quat = [float(v) for v in orientation]
        if len(pos) != 3 or len(quat) != 4:
            raise ValueError("invalid kinematic USD pose")
        w, _qx, _qy, qz = quat
        yaw = math.atan2(2.0 * w * qz, 1.0 - 2.0 * qz * qz)
        self._xform_api.SetTranslate(Gf.Vec3d(pos[0], pos[1], pos[2]))
        self._xform_api.SetRotate(
            Gf.Vec3f(0.0, 0.0, math.degrees(yaw)),
            UsdGeom.XformCommonAPI.RotationOrderXYZ,
        )


class IsaacSimServer:
    def __init__(
        self,
        headless: bool = True,
        width: int = 640,
        height: int = 480,
        verbose: bool = True,
        camera_hfov_deg: float = 110.0,
        mast_height_m: float = 1.35,
        forward_offset_m: float = 0.0,
        camera_pitch_deg: float = 0.0,
        camera_near_m: float = 0.02,
        camera_far_m: float = 10.0,
        enable_depth: bool = False,
        camera_annotator_device: str = "cuda",
        enable_nearfield_depth: bool = False,
        nearfield_width: int = 192,
        nearfield_height: int = 192,
        nearfield_hfov_deg: float = 115.0,
        nearfield_height_m: float = 1.15,
        nearfield_near_m: float = 0.02,
        nearfield_far_m: float = 1.8,
        camera_fill_light_enabled: bool = True,
        camera_fill_light_intensity: float = 4000.0,
        camera_fill_light_radius_m: float = 0.35,
        camera_fill_light_exposure: float = 0.0,
        camera_fill_light_color_temperature_k: float = 5500.0,
        camera_fill_light_auto_distance_enabled: bool = True,
        camera_fill_light_reference_distance_m: float = 2.0,
        camera_fill_light_min_intensity: float = 500.0,
        camera_fill_light_max_intensity: float = 8000.0,
        camera_fill_light_auto_render_updates: int = 6,
    ):
        self.headless = bool(headless)
        self.width = int(width)
        self.height = int(height)
        self.verbose = bool(verbose)
        self.camera_hfov_deg = float(camera_hfov_deg)
        self.mast_height_m = float(mast_height_m)
        self.forward_offset_m = float(forward_offset_m)
        self.camera_pitch_deg = float(camera_pitch_deg)
        self.camera_near_m = float(camera_near_m)
        self.camera_far_m = float(camera_far_m)
        self.enable_depth = bool(enable_depth)
        self.enable_nearfield_depth = bool(enable_nearfield_depth)
        self.nearfield_width = int(nearfield_width)
        self.nearfield_height = int(nearfield_height)
        self.nearfield_hfov_deg = float(nearfield_hfov_deg)
        self.nearfield_height_m = float(nearfield_height_m)
        self.nearfield_near_m = float(nearfield_near_m)
        self.nearfield_far_m = float(nearfield_far_m)
        self.camera_fill_light_enabled = bool(camera_fill_light_enabled)
        self.camera_fill_light_intensity = float(camera_fill_light_intensity)
        self.camera_fill_light_current_intensity = float(camera_fill_light_intensity)
        self.camera_fill_light_radius_m = float(camera_fill_light_radius_m)
        self.camera_fill_light_exposure = float(camera_fill_light_exposure)
        self.camera_fill_light_color_temperature_k = float(camera_fill_light_color_temperature_k)
        self.camera_fill_light_auto_distance_enabled = bool(camera_fill_light_auto_distance_enabled)
        self.camera_fill_light_reference_distance_m = float(camera_fill_light_reference_distance_m)
        self.camera_fill_light_min_intensity = float(camera_fill_light_min_intensity)
        self.camera_fill_light_max_intensity = float(camera_fill_light_max_intensity)
        self.camera_fill_light_auto_render_updates = int(camera_fill_light_auto_render_updates)
        if not math.isfinite(self.camera_fill_light_intensity) or self.camera_fill_light_intensity < 0.0:
            raise ValueError("camera_fill_light_intensity must be finite and non-negative")
        if not math.isfinite(self.camera_fill_light_radius_m) or self.camera_fill_light_radius_m <= 0.0:
            raise ValueError("camera_fill_light_radius_m must be finite and positive")
        if not math.isfinite(self.camera_fill_light_exposure):
            raise ValueError("camera_fill_light_exposure must be finite")
        if (
            not math.isfinite(self.camera_fill_light_color_temperature_k)
            or self.camera_fill_light_color_temperature_k <= 0.0
        ):
            raise ValueError("camera_fill_light_color_temperature_k must be finite and positive")
        if (
            not math.isfinite(self.camera_fill_light_reference_distance_m)
            or self.camera_fill_light_reference_distance_m <= 0.0
        ):
            raise ValueError("camera_fill_light_reference_distance_m must be finite and positive")
        if not math.isfinite(self.camera_fill_light_min_intensity) or self.camera_fill_light_min_intensity < 0.0:
            raise ValueError("camera_fill_light_min_intensity must be finite and non-negative")
        if (
            not math.isfinite(self.camera_fill_light_max_intensity)
            or self.camera_fill_light_max_intensity < self.camera_fill_light_min_intensity
        ):
            raise ValueError("camera_fill_light_max_intensity must be finite and at least the minimum")
        if self.camera_fill_light_auto_render_updates < 1:
            raise ValueError("camera_fill_light_auto_render_updates must be positive")
        device = str(camera_annotator_device or "cpu").strip().lower()
        self.camera_annotator_device = device if device in {"cpu", "cuda"} else "cpu"
        self.app = None
        self.world = None
        self.robot = None
        self.controller = None
        self.camera = None
        self.nearfield_camera = None
        self.camera_prim_path = "/World/Kaya/camera_rgbd"
        self.camera_fill_light_prim_path = self.camera_prim_path + "/VoxRoomRgbFillLight"
        self.camera_fill_light_applied = False
        self.camera_fill_light_last_depth_median_m: Optional[float] = None
        self.camera_fill_light_adaptation_count = 0
        self.nearfield_camera_prim_path = "/World/Kaya/camera_nearfield_depth"
        self.kinematic_pose: Optional[Tuple[float, float, float, float]] = None
        self.last_rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.last_rgb_gpu = None
        self.last_rgb_device = "cpu"
        self.last_depth = None
        self.last_depth_source = "none"
        self.last_depth_semantics = "none"
        self.last_nearfield_depth = np.zeros((self.nearfield_height, self.nearfield_width), dtype=np.float32)
        self.last_nearfield_depth_source = "none"
        self.last_nearfield_depth_semantics = "none"
        self.last_camera_frame_token = None
        self.last_camera_frame_sync_updates = 0
        self.last_camera_wait_timing = {}
        self.last_observation_timing = {}
        self._logged_cuda_rgb_fallback = False
        self._logged_depth_read_debug = False
        self._kaya_physics_disabled_for_kinematic = False
        self.direct_replicator_enabled = str_to_bool(os.environ.get("VOXROOM_ISAAC_DIRECT_REPLICATOR_CAMERA", "0"))
        self.direct_replicator_primary = str_to_bool(os.environ.get("VOXROOM_ISAAC_DIRECT_REPLICATOR_PRIMARY", "1"))
        self.direct_replicator_source = "disabled"
        self._direct_rep = None
        self._direct_render_product = None
        self._direct_rgb_annotator = None
        self._direct_depth_annotator = None
        self._direct_render_product_path = None
        self._direct_replicator_logged = False
        self.robot_pose_sync_failures = 0
        self.camera_pose_sync_failures = 0
        self.nearfield_camera_pose_sync_failures = 0

    def log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    def _isaac_step_perf_path(self) -> Optional[Path]:
        raw_path = str(os.environ.get("VOXROOM_ISAAC_STEP_PERF_PATH", "")).strip()
        if not raw_path:
            return None
        return Path(raw_path).expanduser()

    def _write_isaac_step_perf(self, event: str, payload: dict) -> None:
        path = self._isaac_step_perf_path()
        if path is None:
            return
        row = {
            "event": str(event),
            "time_s": float(time.time()),
            "step": getattr(self, "perf_step_index", None),
            "phase": getattr(self, "perf_step_phase", None),
            **dict(payload),
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        except OSError:
            return

    def start(self) -> None:
        from isaacsim import SimulationApp

        self.log("[isaac] starting SimulationApp")
        # Isaac Sim 5.1 defaults multi_gpu=True even on a single-GPU host.
        # On the 24 GiB RTX 5090 D test machine that startup path can grow the
        # Kit process past 40 GiB RSS and trigger the system OOM killer before
        # a scene is opened.  The benchmark uses exactly one CUDA device, so
        # make the single-GPU contract explicit and size the startup viewport
        # to the actual sensor resolution.
        cpu_thread_limit = max(
            1, int(os.environ.get("VOXROOM_ISAAC_CPU_THREAD_LIMIT", "16"))
        )
        self.app = SimulationApp(
            {
                "headless": self.headless,
                # Headless sensor render products do not require the hidden
                # viewport to update.  Keeping it disabled saves another set
                # of render resources; the RGB/depth cameras still render.
                "disable_viewport_updates": self.headless,
                "limit_cpu_threads": cpu_thread_limit,
                "multi_gpu": False,
                "max_gpu_count": 1,
                "active_gpu": 0,
                "physics_gpu": 0,
                "width": self.width,
                "height": self.height,
            }
        )
        self.log("[isaac] SimulationApp ready")

    def load_scene(self, usd_path: str) -> None:
        if self.app is None:
            self.start()
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import is_stage_loading, open_stage

        effective_usd_path = Path(usd_path).expanduser().resolve(strict=True)
        if str_to_bool(os.environ.get("VOXROOM_ISAAC_SANITIZE_DISPLAY_PRIMVARS", "1")):
            from voxroom_online.isaac_runtime.env.usd_scene_sanitizer import (
                prepare_display_primvar_overlay,
            )

            cache_root = os.environ.get(
                "VOXROOM_ISAAC_USD_OVERLAY_CACHE",
                "outputs/isaac_usd_overlays",
            )
            result = prepare_display_primvar_overlay(effective_usd_path, cache_root)
            effective_usd_path = result.effective_path
            self.log(
                "[isaac] display primvar overlay corrections=%d covered_values=%d cache_hit=%s"
                % (result.correction_count, result.covered_values, result.cache_hit)
            )
        self.log("[isaac] opening stage %s" % effective_usd_path)
        open_stage(str(effective_usd_path))
        while is_stage_loading():
            self.app.update()
        self.disable_imported_scene_rigid_bodies()
        self.world = World(stage_units_in_meters=1.0)
        self.log("[isaac] stage loaded")

    def disable_imported_scene_rigid_bodies(self) -> None:
        try:
            import omni.usd
            from pxr import Sdf, UsdPhysics
        except Exception as exc:
            self.log("[isaac] scene rigid-body cleanup skipped: %s" % exc)
            return
        try:
            from pxr import PhysxSchema
        except Exception:
            PhysxSchema = None
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        if not str_to_bool(os.environ.get("VOXROOM_ISAAC_DEEP_SCENE_PHYSICS_CLEANUP", "0")):
            removed = 0
            for prim in stage.Traverse():
                path = str(prim.GetPath())
                if not path.startswith("/Root/Meshes"):
                    continue
                try:
                    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                        removed += 1
                except Exception:
                    continue
            if removed:
                self.log("[isaac] disabled %d imported scene rigid bodies" % removed)
            return
        physics_apis = [
            getattr(UsdPhysics, "RigidBodyAPI", None),
            getattr(UsdPhysics, "CollisionAPI", None),
            getattr(UsdPhysics, "MassAPI", None),
            getattr(UsdPhysics, "ArticulationRootAPI", None),
            getattr(UsdPhysics, "MeshCollisionAPI", None),
            getattr(PhysxSchema, "PhysxRigidBodyAPI", None) if PhysxSchema is not None else None,
            getattr(PhysxSchema, "PhysxCollisionAPI", None) if PhysxSchema is not None else None,
            getattr(PhysxSchema, "PhysxArticulationAPI", None) if PhysxSchema is not None else None,
            getattr(PhysxSchema, "PhysxMeshCollisionAPI", None) if PhysxSchema is not None else None,
        ]
        removed = 0
        disabled_attrs = 0
        deactivated = 0
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if not path.startswith("/Root"):
                continue
            type_name = str(prim.GetTypeName() or "")
            if ("Joint" in type_name and ("Physics" in type_name or "Physx" in type_name)) or type_name.startswith("Physics"):
                try:
                    prim.SetActive(False)
                    deactivated += 1
                    continue
                except Exception:
                    pass
            try:
                applied_schemas = list(prim.GetAppliedSchemas())
                kept_schemas = [
                    schema
                    for schema in applied_schemas
                    if "physics" not in str(schema).lower() and "physx" not in str(schema).lower()
                ]
                if len(kept_schemas) != len(applied_schemas):
                    prim.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(kept_schemas))
                    removed += len(applied_schemas) - len(kept_schemas)
            except Exception:
                pass
            try:
                for api in physics_apis:
                    if api is None:
                        continue
                    try:
                        if prim.HasAPI(api):
                            prim.RemoveAPI(api)
                            removed += 1
                    except Exception:
                        continue
                for attr in prim.GetAttributes():
                    name = attr.GetName().lower()
                    if name in {"physics:collisionenabled", "physics:rigidbodyenabled"} or (
                        (name.startswith("physics:") or name.startswith("physx")) and name.endswith(":enabled")
                    ):
                        try:
                            attr.Set(False)
                            disabled_attrs += 1
                        except Exception:
                            continue
            except Exception:
                continue
        if removed or disabled_attrs or deactivated:
            self.log(
                "[isaac] disabled imported scene physics: removed %d APIs, disabled %d attrs, deactivated %d prims"
                % (removed, disabled_attrs, deactivated)
            )

    def spawn_kaya(self, pose_world: Tuple[float, float, float, float]) -> None:
        self.log("[isaac] spawning Kaya kinematic visual proxy")
        self.create_kaya_visual_proxy("/World/Kaya")
        self.robot = _KinematicUsdPosePrim("/World/Kaya")
        self.controller = None
        self.set_pose_world(pose_world, sync_robot=True)

    def create_kaya_visual_proxy(self, prim_path: str) -> None:
        """Create a lightweight non-PhysX robot marker for kinematic runs.

        The closed-loop benchmark owns the robot pose directly and only needs a
        GUI-visible body aligned with the camera.  Referencing the full Kaya USD
        brings in wheel articulation and roller rigid bodies; in Isaac 5.1 those
        can produce invalid PhysX transforms when the benchmark also teleports
        the base kinematically.  A pure USD proxy avoids the physics subsystem
        entirely while preserving a visible robot pose in non-headless runs.
        """
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Isaac stage is unavailable; cannot create Kaya visual proxy")
        root = stage.DefinePrim(str(prim_path), "Xform")
        UsdGeom.XformCommonAPI(root).SetTranslate(Gf.Vec3d(0.0, 0.0, 0.0))

        base = UsdGeom.Cylinder.Define(stage, str(prim_path) + "/base")
        base.CreateRadiusAttr(0.24)
        base.CreateHeightAttr(0.11)
        base.CreateAxisAttr("Z")
        base.CreateDisplayColorAttr([Gf.Vec3f(0.05, 0.28, 0.95)])
        UsdGeom.XformCommonAPI(base.GetPrim()).SetTranslate(Gf.Vec3d(0.0, 0.0, 0.075))

        heading = UsdGeom.Cube.Define(stage, str(prim_path) + "/heading")
        heading.CreateSizeAttr(1.0)
        heading.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.15, 0.05)])
        heading_xform = UsdGeom.XformCommonAPI(heading.GetPrim())
        heading_xform.SetTranslate(Gf.Vec3d(0.18, 0.0, 0.15))
        heading_xform.SetScale(Gf.Vec3f(0.22, 0.035, 0.035))

        mast = UsdGeom.Cylinder.Define(stage, str(prim_path) + "/camera_mast")
        mast.CreateRadiusAttr(0.025)
        mast.CreateHeightAttr(0.65)
        mast.CreateAxisAttr("Z")
        mast.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.1, 0.1)])
        UsdGeom.XformCommonAPI(mast.GetPrim()).SetTranslate(Gf.Vec3d(0.0, 0.0, 0.43))

    def disable_kaya_physics_for_kinematic_pose(self) -> None:
        """Treat Kaya as a visual kinematic body during closed-loop benchmark runs.

        The benchmark integrates robot motion kinematically and reads RGB-D from
        cameras attached to that pose.  If the full Kaya articulation remains in
        PhysX while we also set the robot pose directly, Isaac can emit repeated
        "Invalid PhysX transform" warnings for roller child bodies.  Removing the
        physics APIs from the Kaya subtree keeps the visual robot and cameras
        synchronized without asking PhysX to solve the wheel articulation.
        """
        if self._kaya_physics_disabled_for_kinematic:
            return
        try:
            import omni.usd
            from pxr import Sdf, UsdPhysics
        except Exception as exc:
            self.log("[isaac] Kaya kinematic physics cleanup skipped: %s" % exc)
            return
        try:
            from pxr import PhysxSchema
        except Exception:
            PhysxSchema = None
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return

        physics_apis = [
            getattr(UsdPhysics, "RigidBodyAPI", None),
            getattr(UsdPhysics, "CollisionAPI", None),
            getattr(UsdPhysics, "MassAPI", None),
            getattr(UsdPhysics, "ArticulationRootAPI", None),
            getattr(UsdPhysics, "MeshCollisionAPI", None),
            getattr(PhysxSchema, "PhysxRigidBodyAPI", None) if PhysxSchema is not None else None,
            getattr(PhysxSchema, "PhysxCollisionAPI", None) if PhysxSchema is not None else None,
            getattr(PhysxSchema, "PhysxArticulationAPI", None) if PhysxSchema is not None else None,
            getattr(PhysxSchema, "PhysxMeshCollisionAPI", None) if PhysxSchema is not None else None,
        ]
        removed = 0
        disabled_attrs = 0
        deactivated = 0
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if not path.startswith("/World/Kaya"):
                continue
            type_name = str(prim.GetTypeName() or "")
            if ("Joint" in type_name and ("Physics" in type_name or "Physx" in type_name)) or type_name.startswith("Physics"):
                try:
                    prim.SetActive(False)
                    deactivated += 1
                    continue
                except Exception:
                    pass
            try:
                applied_schemas = list(prim.GetAppliedSchemas())
                kept_schemas = [
                    schema
                    for schema in applied_schemas
                    if "physics" not in str(schema).lower() and "physx" not in str(schema).lower()
                ]
                if len(kept_schemas) != len(applied_schemas):
                    prim.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(kept_schemas))
                    removed += len(applied_schemas) - len(kept_schemas)
            except Exception:
                pass
            for api in physics_apis:
                if api is None:
                    continue
                try:
                    if prim.HasAPI(api):
                        prim.RemoveAPI(api)
                        removed += 1
                except Exception:
                    continue
            for attr in prim.GetAttributes():
                name = attr.GetName().lower()
                if name in {"physics:collisionenabled", "physics:rigidbodyenabled"} or (
                    (name.startswith("physics:") or name.startswith("physx")) and name.endswith(":enabled")
                ):
                    try:
                        attr.Set(False)
                        disabled_attrs += 1
                    except Exception:
                        continue
        self._kaya_physics_disabled_for_kinematic = True
        self.log(
            "[isaac] Kaya visual-only sync: removed %d physics APIs, disabled %d attrs, deactivated %d physics prims"
            % (removed, disabled_attrs, deactivated)
        )

    def camera_pose_from_base(self, pose_world: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        x, y, z, yaw = [float(v) for v in pose_world]
        cam_x = x + math.cos(yaw) * self.forward_offset_m
        cam_y = y + math.sin(yaw) * self.forward_offset_m
        cam_z = z + self.mast_height_m
        return cam_x, cam_y, cam_z, yaw

    def nearfield_camera_pose_from_base(self, pose_world: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        x, y, z, yaw = [float(v) for v in pose_world]
        return x, y, z + self.nearfield_height_m, yaw

    def camera_orientation(self, yaw: float):
        from isaacsim.core.utils import rotations as rot_utils

        return rot_utils.euler_angles_to_quat(
            np.asarray([self.camera_pitch_deg, 0.0, math.degrees(float(yaw))], dtype=np.float32),
            degrees=True,
        )

    def nearfield_camera_orientation(self, pose_world: Tuple[float, float, float, float]):
        from isaacsim.core.utils import rotations as rot_utils
        from pxr import Gf

        cam_x, cam_y, cam_z, yaw = self.nearfield_camera_pose_from_base(pose_world)
        camera = Gf.Vec3f(float(cam_x), float(cam_y), float(cam_z))
        target = Gf.Vec3f(float(cam_x), float(cam_y), float(cam_z - 1.0))
        up = Gf.Vec3f(float(math.cos(yaw)), float(math.sin(yaw)), 0.0)
        quat = rot_utils.lookat_to_quatf(camera, target, up)
        return rot_utils.gf_quat_to_np_array(quat).astype(np.float32)

    def configure_camera_intrinsics(self) -> None:
        if self.camera is None:
            return
        hfov = min(max(float(self.camera_hfov_deg), 30.0), 150.0)
        aperture = 20.955
        focal_length = aperture / (2.0 * math.tan(math.radians(hfov) * 0.5))
        try:
            self.camera.set_horizontal_aperture(aperture)
            self.camera.set_focal_length(focal_length)
            self.log("[isaac] camera hfov %.1f deg, focal %.2f" % (hfov, focal_length))
        except Exception as exc:
            self.log("[isaac] camera intrinsics fallback: %s" % exc)
        self.configure_camera_clipping()

    def configure_camera_clipping(self) -> None:
        if self.camera is None:
            return
        near = max(0.001, float(self.camera_near_m))
        far = max(near + 0.1, float(self.camera_far_m))
        applied = False
        try:
            self.camera.set_clipping_range(near, far)
            applied = True
        except Exception:
            pass
        try:
            import omni.usd
            from pxr import Gf, UsdGeom

            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(self.camera_prim_path) if stage is not None else None
            if prim is not None and prim.IsValid():
                UsdGeom.Camera(prim).GetClippingRangeAttr().Set(Gf.Vec2f(near, far))
                applied = True
        except Exception as exc:
            if not applied:
                self.log("[isaac] camera clipping fallback: %s" % exc)
        if applied:
            self.log("[isaac] camera clipping %.3f..%.1f m" % (near, far))

    def configure_nearfield_camera_intrinsics(self) -> None:
        if self.nearfield_camera is None:
            return
        hfov = min(max(float(self.nearfield_hfov_deg), 30.0), 150.0)
        aperture = 20.955
        focal_length = aperture / (2.0 * math.tan(math.radians(hfov) * 0.5))
        try:
            self.nearfield_camera.set_horizontal_aperture(aperture)
            self.nearfield_camera.set_focal_length(focal_length)
            self.log("[isaac] nearfield camera hfov %.1f deg, focal %.2f" % (hfov, focal_length))
        except Exception as exc:
            self.log("[isaac] nearfield camera intrinsics fallback: %s" % exc)
        near = max(0.001, float(self.nearfield_near_m))
        far = max(near + 0.1, float(self.nearfield_far_m))
        applied = False
        try:
            self.nearfield_camera.set_clipping_range(near, far)
            applied = True
        except Exception:
            pass
        try:
            import omni.usd
            from pxr import Gf, UsdGeom

            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(self.nearfield_camera_prim_path) if stage is not None else None
            if prim is not None and prim.IsValid():
                UsdGeom.Camera(prim).GetClippingRangeAttr().Set(Gf.Vec2f(near, far))
                applied = True
        except Exception as exc:
            if not applied:
                self.log("[isaac] nearfield camera clipping fallback: %s" % exc)
        if applied:
            self.log("[isaac] nearfield camera clipping %.3f..%.1f m" % (near, far))

    def attach_camera(self, pose_world: Tuple[float, float, float, float]) -> None:
        from isaacsim.sensors.camera import Camera

        self.log("[isaac] attaching %s camera" % ("RGB-D" if self.enable_depth else "RGB"))
        self.log("[isaac] camera annotator device %s" % self.camera_annotator_device)
        cam_x, cam_y, cam_z, yaw = self.camera_pose_from_base(pose_world)
        camera_kwargs = {
            "prim_path": self.camera_prim_path,
            "position": np.asarray([cam_x, cam_y, cam_z], dtype=np.float32),
            "frequency": 20,
            "resolution": (self.width, self.height),
            "orientation": self.camera_orientation(yaw),
            "annotator_device": self.camera_annotator_device,
        }
        try:
            self.camera = Camera(**camera_kwargs)
        except TypeError:
            camera_kwargs.pop("annotator_device", None)
            self.camera = Camera(**camera_kwargs)
        self.configure_camera_fill_light()
        self.configure_camera_intrinsics()
        self.bind_viewport_to_robot_camera()
        if self.enable_nearfield_depth:
            self.attach_nearfield_depth_camera(pose_world)

    def camera_fill_light_metadata(self) -> dict:
        return {
            "enabled": bool(self.camera_fill_light_enabled),
            "applied": bool(self.camera_fill_light_applied),
            "type": "camera_mounted_sphere",
            "prim_path": str(self.camera_fill_light_prim_path),
            "intensity": float(self.camera_fill_light_current_intensity),
            "base_intensity": float(self.camera_fill_light_intensity),
            "radius_m": float(self.camera_fill_light_radius_m),
            "exposure": float(self.camera_fill_light_exposure),
            "color_temperature_k": float(self.camera_fill_light_color_temperature_k),
            "auto_distance_enabled": bool(self.camera_fill_light_auto_distance_enabled),
            "reference_distance_m": float(self.camera_fill_light_reference_distance_m),
            "min_intensity": float(self.camera_fill_light_min_intensity),
            "max_intensity": float(self.camera_fill_light_max_intensity),
            "auto_render_updates": int(self.camera_fill_light_auto_render_updates),
            "last_depth_median_m": self.camera_fill_light_last_depth_median_m,
            "adaptation_count": int(self.camera_fill_light_adaptation_count),
        }

    def configure_camera_fill_light(self) -> None:
        """Create a soft RGB fill light that follows the runtime camera in every scene."""
        self.camera_fill_light_applied = False
        if not self.camera_fill_light_enabled:
            self.log("[isaac] camera RGB fill light disabled")
            return
        try:
            import omni.usd
            from pxr import Gf, Sdf, UsdGeom, UsdLux

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                raise RuntimeError("USD stage is unavailable")
            camera_prim = stage.GetPrimAtPath(self.camera_prim_path)
            if camera_prim is None or not camera_prim.IsValid():
                raise RuntimeError("camera prim is unavailable")
            light = UsdLux.SphereLight.Define(stage, self.camera_fill_light_prim_path)
            self.camera_fill_light_current_intensity = float(self.camera_fill_light_intensity)
            light.CreateIntensityAttr().Set(float(self.camera_fill_light_current_intensity))
            light.CreateExposureAttr().Set(float(self.camera_fill_light_exposure))
            light.CreateRadiusAttr().Set(float(self.camera_fill_light_radius_m))
            light.CreateNormalizeAttr().Set(True)
            light.CreateColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))
            light.CreateEnableColorTemperatureAttr().Set(True)
            light.CreateColorTemperatureAttr().Set(float(self.camera_fill_light_color_temperature_k))
            light_prim = light.GetPrim()
            UsdGeom.XformCommonAPI(light_prim).SetTranslate(Gf.Vec3d(0.0, 0.0, 0.0))
            light_prim.CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(False)
            self.camera_fill_light_applied = True
            self.log(
                "[isaac] camera RGB fill light intensity=%.1f radius=%.3fm exposure=%.2f temperature=%.0fK"
                % (
                    self.camera_fill_light_current_intensity,
                    self.camera_fill_light_radius_m,
                    self.camera_fill_light_exposure,
                    self.camera_fill_light_color_temperature_k,
                )
            )
        except Exception as exc:
            raise RuntimeError("failed to configure camera RGB fill light: %s" % exc) from exc

    def set_camera_fill_light_intensity(self, intensity: float) -> None:
        """Update fill strength in-place for deterministic same-pose calibration."""
        value = float(intensity)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("camera fill light intensity must be finite and non-negative")
        self.camera_fill_light_current_intensity = value
        if not self.camera_fill_light_enabled:
            return
        try:
            import omni.usd
            from pxr import UsdLux

            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(self.camera_fill_light_prim_path) if stage is not None else None
            if prim is None or not prim.IsValid():
                raise RuntimeError("camera fill light prim is unavailable")
            UsdLux.SphereLight(prim).GetIntensityAttr().Set(value)
        except Exception as exc:
            raise RuntimeError("failed to update camera RGB fill light intensity: %s" % exc) from exc

    def camera_fill_light_intensity_from_depth(self, depth: np.ndarray) -> Optional[float]:
        values = np.asarray(depth, dtype=np.float32)
        valid = values[
            np.isfinite(values)
            & (values >= max(0.05, float(self.camera_near_m)))
            & (values <= float(self.camera_far_m))
        ]
        if valid.size == 0:
            return None
        median_depth = float(np.median(valid))
        self.camera_fill_light_last_depth_median_m = median_depth
        scaled = float(self.camera_fill_light_intensity) * (
            median_depth / float(self.camera_fill_light_reference_distance_m)
        ) ** 2
        return float(
            np.clip(
                scaled,
                float(self.camera_fill_light_min_intensity),
                float(self.camera_fill_light_max_intensity),
            )
        )

    def _adapt_camera_fill_light_from_observation(
        self,
        observation: dict,
        *,
        read_rgb: bool,
        read_depth: bool,
        rgb_device: Optional[str],
    ) -> dict:
        if not (
            self.camera_fill_light_enabled
            and self.camera_fill_light_auto_distance_enabled
            and read_rgb
            and read_depth
            and self.enable_depth
            and isinstance(observation.get("depth"), np.ndarray)
        ):
            return observation
        target = self.camera_fill_light_intensity_from_depth(observation["depth"])
        if target is None:
            return observation
        current = float(self.camera_fill_light_current_intensity)
        if abs(target - current) <= max(50.0, 0.05 * max(current, 1.0)):
            return observation
        previous_frame_token = self._camera_frame_token()
        self.set_camera_fill_light_intensity(target)
        self.camera_fill_light_adaptation_count += 1
        self.log(
            "[isaac] camera RGB fill auto depth=%.3fm intensity=%.1f->%.1f"
            % (float(self.camera_fill_light_last_depth_median_m), current, target)
        )
        direct_primary = bool(self.direct_replicator_is_primary())
        if direct_primary:
            self.step_direct_replicator_camera(self.camera_fill_light_auto_render_updates)
        elif self.app is not None:
            for _ in range(self.camera_fill_light_auto_render_updates):
                self.app.update()
            self._wait_for_fresh_camera_frame(
                previous_frame_token,
                read_depth=bool(read_depth),
                max_updates=max(4, int(self.camera_fill_light_auto_render_updates)),
                min_fresh_frames=2,
            )
        return self.get_observation(
            read_rgb=read_rgb,
            read_depth=read_depth,
            rgb_device=rgb_device,
            _allow_fill_adaptation=False,
        )

    def attach_nearfield_depth_camera(self, pose_world: Tuple[float, float, float, float]) -> None:
        from isaacsim.sensors.camera import Camera

        self.log("[isaac] attaching nearfield top-down depth camera")
        cam_x, cam_y, cam_z, _yaw = self.nearfield_camera_pose_from_base(pose_world)
        camera_kwargs = {
            "prim_path": self.nearfield_camera_prim_path,
            "position": np.asarray([cam_x, cam_y, cam_z], dtype=np.float32),
            "frequency": 20,
            "resolution": (self.nearfield_width, self.nearfield_height),
            "orientation": self.nearfield_camera_orientation(pose_world),
            "annotator_device": self.camera_annotator_device,
        }
        try:
            self.nearfield_camera = Camera(**camera_kwargs)
        except TypeError:
            camera_kwargs.pop("annotator_device", None)
            self.nearfield_camera = Camera(**camera_kwargs)
        self.configure_nearfield_camera_intrinsics()

    def bind_viewport_to_robot_camera(self) -> None:
        if self.headless:
            return
        try:
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
            if viewport is not None:
                viewport.camera_path = self.camera_prim_path
                self.log("[isaac] viewport camera -> %s" % self.camera_prim_path)
                return
        except Exception:
            pass
        try:
            from omni.kit.viewport.utility import get_active_viewport_window

            viewport_window = get_active_viewport_window()
            if viewport_window is not None:
                viewport_window.set_active_camera(self.camera_prim_path)
                self.log("[isaac] viewport camera -> %s" % self.camera_prim_path)
        except Exception as exc:
            self.log("[isaac] viewport camera binding skipped: %s" % exc)

    def _run_async_sensor_wait(self, awaitable, frames: int) -> bool:
        try:
            from omni.kit.async_engine import run_coroutine
        except Exception as exc:
            self.log("[isaac] sensor async engine unavailable: %s" % exc)
            try:
                awaitable.close()
            except Exception:
                pass
            return False
        try:
            task = run_coroutine(awaitable)
        except Exception as exc:
            self.log("[isaac] sensor render wait fallback: %s" % exc)
            try:
                awaitable.close()
            except Exception:
                pass
            return False
        max_updates = int(os.environ.get("VOXROOM_ISAAC_SENSOR_WAIT_MAX_UPDATES", str(max(120, int(frames) * 80))))
        for _ in range(max_updates):
            if task.done():
                try:
                    task.result()
                    return True
                except Exception as exc:
                    self.log("[isaac] sensor render wait failed: %s" % exc)
                    return False
            self.app.update()
        try:
            task.cancel()
        except Exception:
            pass
        self.log("[isaac] sensor render wait timed out after %d app updates" % max_updates)
        return False

    def render_sensor_frames(self, camera, frames: int) -> bool:
        if camera is None:
            return False
        if str_to_bool(os.environ.get("VOXROOM_ISAAC_DISABLE_ASYNC_SENSOR_WAIT", "0")):
            self.log("[isaac] async sensor render wait disabled; using app.update warmup")
            return False
        try:
            import omni.syntheticdata

            sd_sensors = getattr(omni.syntheticdata, "sensors", None)
            if sd_sensors is None:
                raise RuntimeError("omni.syntheticdata.sensors is unavailable")
            render_product_path = camera.get_render_product_path()
            self.log("[isaac] waiting for %d sensor render frames on %s" % (int(frames), render_product_path))
            return self._run_async_sensor_wait(
                sd_sensors.next_render_simulation_async(render_product_path, int(frames)),
                int(frames),
            )
        except Exception as exc:
            self.log("[isaac] sensor render wait unavailable: %s" % exc)
            return False

    def ensure_direct_replicator_camera(self) -> bool:
        if not self.direct_replicator_enabled:
            return False
        if self._direct_rgb_annotator is not None:
            return True
        if self.camera is None:
            return False
        try:
            import omni.replicator.core as rep

            render_product = rep.create.render_product(
                self.camera_prim_path,
                (int(self.width), int(self.height)),
            )
            rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
            rgb_annotator.attach([render_product])
            depth_annotator.attach([render_product])
            self._direct_rep = rep
            self._direct_render_product = render_product
            self._direct_rgb_annotator = rgb_annotator
            self._direct_depth_annotator = depth_annotator
            self._direct_render_product_path = str(render_product)
            self.direct_replicator_source = "replicator_render_product"
            self.log(
                "[isaac] direct Replicator RGB-D camera enabled on %s primary=%s"
                % (self.camera_prim_path, bool(self.direct_replicator_primary))
            )
            return True
        except Exception as exc:
            self.direct_replicator_source = "unavailable"
            self.log("[isaac] direct Replicator camera unavailable: %s" % exc)
            return False

    def step_direct_replicator_camera(self, frames: int = 1) -> None:
        if self._direct_rgb_annotator is None:
            return
        rep = self._direct_rep
        for _ in range(max(1, int(frames))):
            if self.app is not None:
                self.app.update()
            try:
                rep.orchestrator.step()
            except Exception:
                pass

    def read_direct_replicator_camera(self, *, read_rgb: bool, read_depth: bool) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self._direct_rgb_annotator is None:
            return None, None
        rgb = None
        depth = None
        if read_rgb:
            try:
                rgb = _rgb_value_to_array(self._direct_rgb_annotator.get_data(device="cpu"))
            except Exception as exc:
                if not self._direct_replicator_logged:
                    self.log("[isaac] direct Replicator RGB read failed: %s" % exc)
                    self._direct_replicator_logged = True
        if read_depth:
            try:
                depth = _depth_value_to_array(self._direct_depth_annotator.get_data(device="cpu"))
            except Exception as exc:
                if not self._direct_replicator_logged:
                    self.log("[isaac] direct Replicator depth read failed: %s" % exc)
                    self._direct_replicator_logged = True
        return rgb, depth

    def warmup_camera_frames(self, frames: int) -> None:
        frames = max(1, int(frames))
        rendered = False
        direct_render = self.ensure_direct_replicator_camera()
        if self.camera is not None:
            rendered = self.render_sensor_frames(self.camera, frames) or rendered
        if self.nearfield_camera is not None:
            rendered = self.render_sensor_frames(self.nearfield_camera, frames) or rendered
        if direct_render:
            self.step_direct_replicator_camera(max(1, min(frames, 4)))
        if rendered:
            return
        for _ in range(frames):
            self.app.update()

    def direct_replicator_is_primary(self) -> bool:
        return bool(
            self.direct_replicator_enabled
            and self.direct_replicator_primary
            and self._direct_rgb_annotator is not None
        )

    def set_pose_world(self, pose_world: Tuple[float, float, float, float], *, sync_robot: bool = True) -> None:
        x, y, z, yaw = [float(v) for v in pose_world]
        if not all(math.isfinite(v) for v in (x, y, z, yaw)):
            raise ValueError("Isaac kinematic pose contains non-finite values: %r" % (pose_world,))
        self.kinematic_pose = (x, y, z, yaw)
        quat = yaw_to_quat_wxyz(yaw)
        if bool(sync_robot) and self.robot is not None:
            try:
                self.robot.set_world_pose(
                    position=np.asarray([x, y, z], dtype=np.float32),
                    orientation=quat,
                )
            except Exception as exc:
                self.robot_pose_sync_failures += 1
                if self.robot_pose_sync_failures == 1:
                    self.log("[isaac] robot visual pose sync failed: %s" % exc)
        if self.camera is not None:
            cam_x, cam_y, cam_z, _ = self.camera_pose_from_base((x, y, z, yaw))
            try:
                self.camera.set_world_pose(
                    position=np.asarray([cam_x, cam_y, cam_z], dtype=np.float32),
                    orientation=self.camera_orientation(yaw),
                )
            except Exception as exc:
                self.camera_pose_sync_failures += 1
                if self.camera_pose_sync_failures == 1:
                    self.log("[isaac] camera pose sync failed: %s" % exc)
        if self.nearfield_camera is not None:
            near_x, near_y, near_z, _ = self.nearfield_camera_pose_from_base((x, y, z, yaw))
            try:
                self.nearfield_camera.set_world_pose(
                    position=np.asarray([near_x, near_y, near_z], dtype=np.float32),
                    orientation=self.nearfield_camera_orientation((x, y, z, yaw)),
                )
            except Exception as exc:
                self.nearfield_camera_pose_sync_failures += 1
                if self.nearfield_camera_pose_sync_failures == 1:
                    self.log("[isaac] nearfield camera pose sync failed: %s" % exc)

    def observe_at_pose_world(
        self,
        pose_world: Tuple[float, float, float, float],
        *,
        render_updates: int = 2,
        read_rgb: bool = True,
        read_depth: Optional[bool] = None,
        rgb_device: Optional[str] = None,
        sync_robot: bool = True,
    ) -> dict:
        requested_read_depth = self.enable_depth if read_depth is None else bool(read_depth)
        need_camera_frames = bool(read_rgb or requested_read_depth)
        direct_primary = bool(need_camera_frames and self.ensure_direct_replicator_camera() and self.direct_replicator_is_primary())
        previous_frame_token = None if direct_primary else (self._camera_frame_token() if need_camera_frames else None)
        self.set_pose_world(pose_world, sync_robot=sync_robot)
        if need_camera_frames:
            if direct_primary:
                self.step_direct_replicator_camera(max(1, int(render_updates)))
                self.last_camera_frame_sync_updates = 0
            else:
                for _ in range(max(0, int(render_updates))):
                    self.app.update()
                self._wait_for_fresh_camera_frame(previous_frame_token, read_depth=requested_read_depth)
        return self.get_observation(
            read_rgb=read_rgb,
            read_depth=requested_read_depth,
            rgb_device=rgb_device,
        )

    def reset_episode(
        self,
        usd_path: str,
        pose_world: Tuple[float, float, float, float],
        read_rgb: bool = True,
        rgb_device: Optional[str] = None,
    ) -> dict:
        self.load_scene(usd_path)
        self.spawn_kaya(pose_world)
        self.attach_camera(pose_world)
        if str_to_bool(os.environ.get("VOXROOM_ISAAC_SKIP_WORLD_RESET", "0")):
            self.log("[isaac] skipping world reset")
            if str_to_bool(os.environ.get("VOXROOM_ISAAC_PLAY_TIMELINE_FOR_SENSORS", "0")):
                self.log("[isaac] playing timeline for sensor capture")
                self.world.play()
        else:
            self.log("[isaac] resetting world")
            self.world.reset()
        self.set_pose_world(pose_world)
        need_camera_frames = bool(read_rgb or self.enable_depth or self.enable_nearfield_depth)
        if not need_camera_frames:
            self.log("[isaac] skipping camera sensor warmup")
            return self.get_observation(read_rgb=False, read_depth=False, rgb_device=rgb_device)
        self.log("[isaac] initializing camera")
        self.camera.initialize()
        self.configure_camera_intrinsics()
        if self.nearfield_camera is not None:
            self.log("[isaac] initializing nearfield depth camera")
            self.nearfield_camera.initialize()
            self.configure_nearfield_camera_intrinsics()
        self.bind_viewport_to_robot_camera()
        if self.enable_depth:
            try:
                self.camera.add_distance_to_camera_to_frame()
                self.camera.add_distance_to_image_plane_to_frame()
            except Exception as exc:
                self.log("[isaac] camera annotator fallback: %s" % exc)
        try:
            self.camera.resume()
        except Exception as exc:
            self.log("[isaac] camera resume fallback: %s" % exc)
        if self.enable_nearfield_depth and self.nearfield_camera is not None:
            try:
                self.nearfield_camera.add_distance_to_image_plane_to_frame()
            except Exception as exc:
                self.log("[isaac] nearfield camera annotator fallback: %s" % exc)
            try:
                self.nearfield_camera.resume()
            except Exception as exc:
                self.log("[isaac] nearfield camera resume fallback: %s" % exc)
        self.log("[isaac] rendering warmup frames")
        self.warmup_camera_frames(int(os.environ.get("VOXROOM_ISAAC_SENSOR_WARMUP_FRAMES", "10")))
        obs = self.get_observation(read_rgb=read_rgb, read_depth=self.enable_depth, rgb_device=rgb_device)
        if self.enable_depth and not obs.get("has_depth"):
            for _ in range(10):
                self.warmup_camera_frames(int(os.environ.get("VOXROOM_ISAAC_SENSOR_RETRY_FRAMES", "2")))
                obs = self.get_observation(read_rgb=read_rgb, read_depth=True, rgb_device=rgb_device)
                if obs.get("has_depth"):
                    break
            if not obs.get("has_depth"):
                self.log("[isaac] warning: depth unavailable after reset warmup")
        return obs

    def step_velocity(
        self,
        vx: float,
        vy: float,
        wz: float,
        frames: int = 3,
        read_rgb: bool = True,
        read_depth: Optional[bool] = None,
        rgb_device: Optional[str] = None,
    ) -> dict:
        requested_read_depth = self.enable_depth if read_depth is None else bool(read_depth)
        need_camera_frames = bool(read_rgb or requested_read_depth)
        direct_primary = bool(need_camera_frames and self.ensure_direct_replicator_camera() and self.direct_replicator_is_primary())
        previous_frame_token = None if direct_primary else (self._camera_frame_token() if need_camera_frames else None)
        if self.robot is not None and self.controller is not None:
            self.robot.apply_wheel_actions(self.controller.forward(command=[float(vx), float(vy), float(wz)]))
        for _ in range(int(frames)):
            self.world.step(render=need_camera_frames and not direct_primary)
        if need_camera_frames:
            if direct_primary:
                self.step_direct_replicator_camera(max(1, int(frames)))
                self.last_camera_frame_sync_updates = 0
            else:
                self._wait_for_fresh_camera_frame(previous_frame_token, read_depth=requested_read_depth)
        return self.get_observation(
            read_rgb=read_rgb,
            read_depth=requested_read_depth,
            rgb_device=rgb_device,
        )

    def step_kinematic_velocity(
        self,
        vx: float,
        vy: float,
        wz: float,
        dt: float = 0.2,
        render_updates: int = 2,
        read_rgb: bool = True,
        read_depth: Optional[bool] = None,
        rgb_device: Optional[str] = None,
    ) -> dict:
        total_started_at = time.perf_counter()
        timing: dict[str, object] = {}
        requested_read_depth = self.enable_depth if read_depth is None else bool(read_depth)
        need_camera_frames = bool(read_rgb or requested_read_depth)
        direct_primary = bool(need_camera_frames and self.ensure_direct_replicator_camera() and self.direct_replicator_is_primary())
        timing["direct_primary"] = bool(direct_primary)
        timing["need_camera_frames"] = bool(need_camera_frames)
        timing["read_rgb"] = bool(read_rgb)
        timing["read_depth"] = bool(requested_read_depth)
        timing["render_updates"] = int(render_updates)
        previous_frame_token = None if direct_primary else (self._camera_frame_token() if need_camera_frames else None)
        vx, vy, wz, dt = float(vx), float(vy), float(wz), float(dt)
        if not all(math.isfinite(v) for v in (vx, vy, wz, dt)):
            raise ValueError("Isaac kinematic command contains non-finite values: %r" % ((vx, vy, wz, dt),))
        if dt < 0.0:
            raise ValueError("Isaac kinematic dt must be non-negative, got %.6f" % dt)
        x, y, z, yaw = self.get_pose_world()
        if not all(math.isfinite(float(v)) for v in (x, y, z, yaw)):
            raise ValueError("Isaac kinematic pose contains non-finite values before step: %r" % ((x, y, z, yaw),))
        dx = math.cos(yaw) * vx - math.sin(yaw) * vy
        dy = math.sin(yaw) * vx + math.cos(yaw) * vy
        yaw = yaw + wz * dt
        while yaw > math.pi:
            yaw -= 2.0 * math.pi
        while yaw < -math.pi:
            yaw += 2.0 * math.pi
        # Keep the rendered Kaya body and the RGB-D camera on the same
        # kinematic pose.  The closed-loop controller still computes its pose
        # from this kinematic state; the visual robot must not lag behind the
        # sensor in non-headless debugging.
        stage_started_at = time.perf_counter()
        self.set_pose_world((x + dx * dt, y + dy * dt, z, yaw), sync_robot=True)
        timing["set_pose_world_ms"] = max(0.0, (time.perf_counter() - stage_started_at) * 1000.0)
        if need_camera_frames:
            if direct_primary:
                stage_started_at = time.perf_counter()
                self.step_direct_replicator_camera(max(1, int(render_updates)))
                timing["direct_replicator_step_ms"] = max(0.0, (time.perf_counter() - stage_started_at) * 1000.0)
                self.last_camera_frame_sync_updates = 0
            else:
                app_update_total_ms = 0.0
                for _ in range(int(render_updates)):
                    stage_started_at = time.perf_counter()
                    self.app.update()
                    app_update_total_ms += max(0.0, (time.perf_counter() - stage_started_at) * 1000.0)
                timing["app_update_total_ms"] = float(app_update_total_ms)
                timing["app_update_count"] = int(render_updates)
                stage_started_at = time.perf_counter()
                self._wait_for_fresh_camera_frame(previous_frame_token, read_depth=requested_read_depth)
                timing["wait_fresh_camera_frame_ms"] = max(0.0, (time.perf_counter() - stage_started_at) * 1000.0)
        stage_started_at = time.perf_counter()
        obs = self.get_observation(
            read_rgb=read_rgb,
            read_depth=requested_read_depth,
            rgb_device=rgb_device,
        )
        timing["get_observation_ms"] = max(0.0, (time.perf_counter() - stage_started_at) * 1000.0)
        timing["total_ms"] = max(0.0, (time.perf_counter() - total_started_at) * 1000.0)
        self._write_isaac_step_perf(
            "step_kinematic_velocity",
            {
                **timing,
                "camera_wait": dict(getattr(self, "last_camera_wait_timing", {}) or {}),
                "observation": dict(getattr(self, "last_observation_timing", {}) or {}),
                "camera_frame_sync_updates": int(getattr(self, "last_camera_frame_sync_updates", 0)),
                "has_rgb": bool(obs.get("has_rgb")),
                "has_depth": bool(obs.get("has_depth")),
                "rgb_device": obs.get("rgb_device"),
                "depth_source": obs.get("depth_source"),
            },
        )
        return obs

    def _camera_frame_token(self):
        if self.camera is None:
            return None
        try:
            frame = self.camera.get_current_frame() or {}
        except Exception:
            return None
        token = frame.get("rendering_frame")
        if isinstance(token, dict):
            return tuple(sorted((str(key), _hashable_frame_value(value)) for key, value in token.items()))
        if token is not None:
            return _hashable_frame_value(token)
        return _hashable_frame_value(frame.get("rendering_time"))

    def _wait_for_fresh_camera_frame(
        self,
        previous_token,
        read_depth: bool = True,
        max_updates: Optional[int] = None,
        min_fresh_frames: int = 2,
    ) -> None:
        """Avoid pairing a stale RGB-D frame with a newly-updated pose."""
        wait_started_at = time.perf_counter()
        token_total_ms = 0.0
        update_total_ms = 0.0
        updates_run = 0
        self.last_camera_frame_sync_updates = 0
        if self.app is None or self.camera is None or previous_token is None:
            self.last_camera_wait_timing = {
                "total_ms": 0.0,
                "reason": "skipped",
                "max_updates": 0,
                "updates_run": 0,
                "fresh_frames": 0,
            }
            return
        if max_updates is None:
            max_updates = int(os.environ.get("VOXROOM_ISAAC_SENSOR_WAIT_MAX_UPDATES", "10"))
        max_updates = max(0, int(max_updates))
        fresh_frames = 0
        last_token = previous_token
        for idx in range(int(max_updates) + 1):
            token_started_at = time.perf_counter()
            token = self._camera_frame_token()
            token_total_ms += max(0.0, (time.perf_counter() - token_started_at) * 1000.0)
            if token is not None and token != last_token:
                fresh_frames += 1
                last_token = token
            if fresh_frames >= max(1, int(min_fresh_frames)):
                self.last_camera_frame_token = last_token
                self.last_camera_frame_sync_updates = idx
                self.last_camera_wait_timing = {
                    "total_ms": max(0.0, (time.perf_counter() - wait_started_at) * 1000.0),
                    "reason": "fresh",
                    "max_updates": int(max_updates),
                    "updates_run": int(updates_run),
                    "fresh_frames": int(fresh_frames),
                    "token_total_ms": float(token_total_ms),
                    "app_update_total_ms": float(update_total_ms),
                }
                return
            update_started_at = time.perf_counter()
            self.app.update()
            update_total_ms += max(0.0, (time.perf_counter() - update_started_at) * 1000.0)
            updates_run += 1
        self.last_camera_frame_token = self._camera_frame_token()
        self.last_camera_frame_sync_updates = int(max_updates)
        self.last_camera_wait_timing = {
            "total_ms": max(0.0, (time.perf_counter() - wait_started_at) * 1000.0),
            "reason": "max_updates",
            "max_updates": int(max_updates),
            "updates_run": int(updates_run),
            "fresh_frames": int(fresh_frames),
            "token_total_ms": float(token_total_ms),
            "app_update_total_ms": float(update_total_ms),
        }

    def get_pose_world(self) -> Tuple[float, float, float, float]:
        if self.kinematic_pose is not None:
            return self.kinematic_pose
        if self.robot is None:
            return 0.0, 0.0, 0.0, 0.0
        pos, quat = self.robot.get_world_pose()
        # quat is wxyz; yaw only.
        w, _x, _y, z = [float(v) for v in quat]
        yaw = math.atan2(2.0 * w * z, 1.0 - 2.0 * z * z)
        return float(pos[0]), float(pos[1]), float(pos[2]), yaw

    def get_observation(
        self,
        read_rgb: bool = True,
        read_depth: bool = False,
        rgb_device: Optional[str] = None,
        *,
        _allow_fill_adaptation: bool = True,
    ) -> dict:
        observation_started_at = time.perf_counter()
        observation_timing: dict[str, object] = {}
        rgb = self.last_rgb
        rgb_gpu = self.last_rgb_gpu
        depth = None if bool(read_depth and self.enable_depth) else self.last_depth
        depth_source = "none"
        depth_semantics = "none"
        nearfield_depth = None if bool(read_depth and self.enable_nearfield_depth) else self.last_nearfield_depth
        nearfield_depth_source = "none"
        nearfield_depth_semantics = "none"
        frame = {}
        requested_rgb_device = str(rgb_device or self.camera_annotator_device).strip().lower()
        if requested_rgb_device not in {"cpu", "cuda"}:
            requested_rgb_device = "cpu"
        direct_primary = bool((read_rgb or (read_depth and self.enable_depth)) and self.direct_replicator_is_primary())
        if direct_primary:
            stage_started_at = time.perf_counter()
            self.step_direct_replicator_camera(1)
            observation_timing["direct_replicator_step_ms"] = max(0.0, (time.perf_counter() - stage_started_at) * 1000.0)
            stage_started_at = time.perf_counter()
            direct_rgb, direct_depth = self.read_direct_replicator_camera(
                read_rgb=bool(read_rgb),
                read_depth=bool(read_depth and self.enable_depth),
            )
            observation_timing["direct_replicator_read_ms"] = max(0.0, (time.perf_counter() - stage_started_at) * 1000.0)
            if read_rgb and direct_rgb is not None:
                rgb = direct_rgb
                self.last_rgb = rgb
                self.last_rgb_device = "cpu"
                rgb_gpu = None
                self.last_rgb_gpu = None
            if read_depth and self.enable_depth and direct_depth is not None:
                depth = direct_depth
                depth_source = "direct_replicator_distance_to_image_plane"
                depth_semantics = "image_plane_z"
                self.last_depth = depth
                self.last_depth_source = depth_source
                self.last_depth_semantics = depth_semantics
        normal_camera_needed = not (
            direct_primary
            and (not read_rgb or rgb is not None)
            and (not read_depth or not self.enable_depth or depth is not None)
        )
        if self.camera is not None:
            if normal_camera_needed and read_depth and self.enable_depth:
                stage_started_at = time.perf_counter()
                self.last_depth = None
                self.last_depth_source = "none"
                self.last_depth_semantics = "none"
                frame = self.camera.get_current_frame() or {}
                observation_timing["camera_get_current_frame_ms"] = max(0.0, (time.perf_counter() - stage_started_at) * 1000.0)
            if normal_camera_needed and read_rgb:
                stage_started_at = time.perf_counter()
                if requested_rgb_device == "cuda":
                    try:
                        rgb_gpu = self.camera.get_rgb(device="cuda")
                        self.last_rgb_gpu = rgb_gpu
                        self.last_rgb_device = "cuda"
                    except Exception as exc:
                        if not self._logged_cuda_rgb_fallback:
                            self.log("[isaac] CUDA RGB read failed; falling back to CPU: %s" % exc)
                            self._logged_cuda_rgb_fallback = True
                        requested_rgb_device = "cpu"
                if requested_rgb_device == "cpu":
                    try:
                        rgb_frame = np.asarray(self.camera.get_rgb(device="cpu"))
                    except Exception:
                        rgb_frame = np.asarray(self.camera.get_rgba())
                    if rgb_frame.ndim == 3 and rgb_frame.shape[2] >= 3:
                        rgb = rgb_frame[:, :, :3].astype(np.uint8)
                        self.last_rgb = rgb
                        self.last_rgb_device = "cpu"
                elif isinstance(frame.get("rgb"), np.ndarray):
                    rgb_frame = np.asarray(frame["rgb"])
                    if rgb_frame.ndim == 3 and rgb_frame.shape[2] >= 3:
                        rgb = rgb_frame[:, :, :3].astype(np.uint8)
                        self.last_rgb = rgb
                observation_timing["read_rgb_ms"] = max(0.0, (time.perf_counter() - stage_started_at) * 1000.0)
            if normal_camera_needed and read_depth and self.enable_depth:
                stage_started_at = time.perf_counter()
                depth_candidates = [
                    ("distance_to_image_plane", "image_plane_z", frame.get("distance_to_image_plane")),
                    ("distance_to_camera", "euclidean_distance_to_camera", frame.get("distance_to_camera")),
                ]
                try:
                    depth_candidates.append(("camera_get_depth", "image_plane_z", self.camera.get_depth(device="cpu")))
                except Exception:
                    depth_candidates.append(("camera_get_depth", "image_plane_z", None))
                for candidate_source, candidate_semantics, depth_value in depth_candidates:
                    depth_frame = _depth_value_to_array(depth_value)
                    if depth_frame is None:
                        continue
                    depth_source = candidate_source
                    depth_semantics = candidate_semantics
                    if depth_source == "distance_to_camera":
                        intr = CameraIntrinsics.from_hfov(
                            int(depth_frame.shape[1]),
                            int(depth_frame.shape[0]),
                            float(self.camera_hfov_deg),
                        )
                        depth_frame = distance_to_camera_to_image_plane_depth(depth_frame, intr)
                        depth_source = "distance_to_camera_converted_to_image_plane_z"
                        depth_semantics = "image_plane_z"
                    depth = depth_frame
                    self.last_depth = depth
                    self.last_depth_source = depth_source
                    self.last_depth_semantics = depth_semantics
                    break
                if depth is None and self._direct_depth_annotator is not None:
                    self.step_direct_replicator_camera(1)
                    direct_rgb, direct_depth = self.read_direct_replicator_camera(read_rgb=read_rgb, read_depth=True)
                    if direct_rgb is not None and read_rgb:
                        rgb = direct_rgb
                        self.last_rgb = rgb
                        self.last_rgb_device = "cpu"
                    if direct_depth is not None:
                        depth = direct_depth
                        self.last_depth = depth
                        self.last_depth_source = "direct_replicator_distance_to_image_plane"
                        self.last_depth_semantics = "image_plane_z"
                if depth is None and str_to_bool(os.environ.get("VOXROOM_ISAAC_DEPTH_DEBUG", "0")):
                    self.log_depth_read_debug(frame)
                observation_timing["read_depth_ms"] = max(0.0, (time.perf_counter() - stage_started_at) * 1000.0)
            elif read_rgb and self._direct_rgb_annotator is not None:
                stage_started_at = time.perf_counter()
                direct_rgb, _ = self.read_direct_replicator_camera(read_rgb=True, read_depth=False)
                if direct_rgb is not None:
                    rgb = direct_rgb
                    self.last_rgb = rgb
                    self.last_rgb_device = "cpu"
                observation_timing["direct_rgb_fallback_read_ms"] = max(0.0, (time.perf_counter() - stage_started_at) * 1000.0)
            if read_depth and self.enable_nearfield_depth and self.nearfield_camera is not None:
                stage_started_at = time.perf_counter()
                self.last_nearfield_depth = None
                self.last_nearfield_depth_source = "none"
                self.last_nearfield_depth_semantics = "none"
                nearfield_depth_candidates = []
                try:
                    nearfield_frame = self.nearfield_camera.get_current_frame() or {}
                    nearfield_depth_candidates.append(
                        (
                            "nearfield_distance_to_image_plane",
                            "image_plane_z",
                            nearfield_frame.get("distance_to_image_plane"),
                        )
                    )
                except Exception:
                    pass
                try:
                    nearfield_depth_candidates.append(
                        (
                            "nearfield_camera_get_depth",
                            "image_plane_z",
                            self.nearfield_camera.get_depth(device="cpu"),
                        )
                    )
                except Exception:
                    nearfield_depth_candidates.append(("nearfield_camera_get_depth", "image_plane_z", None))
                for candidate_source, candidate_semantics, nearfield_depth_value in nearfield_depth_candidates:
                    nearfield_depth_frame = _depth_value_to_array(nearfield_depth_value)
                    if nearfield_depth_frame is None:
                        continue
                    nearfield_depth = nearfield_depth_frame
                    self.last_nearfield_depth = nearfield_depth
                    self.last_nearfield_depth_source = candidate_source
                    self.last_nearfield_depth_semantics = candidate_semantics
                    break
                observation_timing["nearfield_depth_ms"] = max(0.0, (time.perf_counter() - stage_started_at) * 1000.0)
        pose = self.get_pose_world()
        stage_started_at = time.perf_counter()
        camera_frame_token = self._camera_frame_token()
        observation_timing["camera_frame_token_ms"] = max(0.0, (time.perf_counter() - stage_started_at) * 1000.0)
        if camera_frame_token is not None:
            self.last_camera_frame_token = camera_frame_token
        observation_timing["total_ms"] = max(0.0, (time.perf_counter() - observation_started_at) * 1000.0)
        observation_timing["normal_camera_needed"] = bool(normal_camera_needed)
        observation_timing["direct_primary"] = bool(direct_primary)
        observation_timing["requested_rgb_device"] = str(requested_rgb_device)
        self.last_observation_timing = observation_timing
        observation = {
            "rgb": rgb,
            "rgb_gpu": rgb_gpu,
            "rgb_device": self.last_rgb_device,
            "depth": depth,
            "depth_source": self.last_depth_source,
            "depth_semantics": self.last_depth_semantics,
            "nearfield_depth": nearfield_depth,
            "nearfield_depth_source": self.last_nearfield_depth_source,
            "nearfield_depth_semantics": self.last_nearfield_depth_semantics,
            "has_rgb": bool(read_rgb),
            "has_depth": bool(read_depth and self.enable_depth and isinstance(depth, np.ndarray)),
            "has_nearfield_depth": bool(read_depth and self.enable_nearfield_depth and isinstance(nearfield_depth, np.ndarray)),
            "pose_world": pose,
            "camera_pose_world": self.camera_pose_from_base(pose),
            "nearfield_camera_pose_world": self.nearfield_camera_pose_from_base(pose),
            "camera_rendering_frame": frame.get("rendering_frame") if isinstance(frame, dict) else None,
            "camera_rendering_time": frame.get("rendering_time") if isinstance(frame, dict) else None,
            "camera_frame_sync_updates": int(self.last_camera_frame_sync_updates),
            "visual_robot_proxy": True,
            "robot_pose_sync_failures": int(self.robot_pose_sync_failures),
            "camera_pose_sync_failures": int(self.camera_pose_sync_failures),
            "nearfield_camera_pose_sync_failures": int(self.nearfield_camera_pose_sync_failures),
            "camera_fill_light": self.camera_fill_light_metadata(),
            "sim_time": 0.0,
            "collided": False,
        }
        if _allow_fill_adaptation:
            return self._adapt_camera_fill_light_from_observation(
                observation,
                read_rgb=bool(read_rgb),
                read_depth=bool(read_depth),
                rgb_device=rgb_device,
            )
        return observation

    def log_depth_read_debug(self, frame: dict) -> None:
        if self._logged_depth_read_debug:
            return
        self._logged_depth_read_debug = True
        if not isinstance(frame, dict):
            self.log("[isaac-depth-debug] frame is %s" % _value_debug_description(frame))
            return
        frame_keys = ",".join(str(key) for key in sorted(frame.keys()))
        self.log("[isaac-depth-debug] frame keys: %s" % frame_keys)
        for key in ("distance_to_image_plane", "distance_to_camera", "rgb", "rendering_frame", "rendering_time"):
            self.log("[isaac-depth-debug] frame[%s]: %s" % (key, _value_debug_description(frame.get(key))))
        annotators = getattr(self.camera, "_custom_annotators", {}) if self.camera is not None else {}
        self.log("[isaac-depth-debug] custom annotators: %s" % ",".join(str(key) for key in sorted(annotators.keys())))
        for key in ("distance_to_image_plane", "distance_to_camera"):
            annotator = annotators.get(key)
            if annotator is None:
                continue
            try:
                value = annotator.get_data(device="cpu")
                self.log("[isaac-depth-debug] annotator[%s].cpu: %s" % (key, _value_debug_description(value)))
            except Exception as exc:
                self.log("[isaac-depth-debug] annotator[%s].cpu failed: %s" % (key, exc))
            try:
                value = annotator.get_data()
                self.log("[isaac-depth-debug] annotator[%s].default: %s" % (key, _value_debug_description(value)))
            except Exception as exc:
                self.log("[isaac-depth-debug] annotator[%s].default failed: %s" % (key, exc))

    def close(self) -> None:
        if self.app is not None:
            self.app.close()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-usd", required=True)
    parser.add_argument("--spawn", nargs=4, type=float, default=[0.0, 0.0, 0.05, 0.0])
    parser.add_argument("--headless", nargs="?", const=True, default=True, type=str_to_bool)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--save-frame", default=None)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--camera-hfov-deg", type=float, default=110.0)
    parser.add_argument("--camera-mast-height-m", type=float, default=1.35)
    parser.add_argument("--camera-forward-offset-m", type=float, default=0.0)
    parser.add_argument("--camera-pitch-deg", type=float, default=0.0)
    parser.add_argument("--camera-near-m", type=float, default=0.02)
    parser.add_argument("--camera-far-m", type=float, default=10.0)
    parser.add_argument(
        "--camera-fill-light-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--camera-fill-light-intensity", type=float, default=4000.0)
    parser.add_argument("--camera-fill-light-radius-m", type=float, default=0.35)
    parser.add_argument("--camera-fill-light-exposure", type=float, default=0.0)
    parser.add_argument("--camera-fill-light-color-temperature-k", type=float, default=5500.0)
    parser.add_argument(
        "--camera-fill-light-auto-distance-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--camera-fill-light-reference-distance-m", type=float, default=2.0)
    parser.add_argument("--camera-fill-light-min-intensity", type=float, default=500.0)
    parser.add_argument("--camera-fill-light-max-intensity", type=float, default=8000.0)
    parser.add_argument("--camera-fill-light-auto-render-updates", type=int, default=6)
    parser.add_argument("--camera-annotator-device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--read-depth", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--nearfield-depth", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--nearfield-width", type=int, default=192)
    parser.add_argument("--nearfield-height", type=int, default=192)
    parser.add_argument("--nearfield-hfov-deg", type=float, default=115.0)
    parser.add_argument("--nearfield-height-m", type=float, default=1.15)
    args = parser.parse_args(argv)
    server = IsaacSimServer(
        headless=args.headless,
        verbose=args.verbose,
        camera_hfov_deg=args.camera_hfov_deg,
        mast_height_m=args.camera_mast_height_m,
        forward_offset_m=args.camera_forward_offset_m,
        camera_pitch_deg=args.camera_pitch_deg,
        camera_near_m=args.camera_near_m,
        camera_far_m=args.camera_far_m,
        enable_depth=args.read_depth,
        camera_annotator_device=args.camera_annotator_device,
        camera_fill_light_enabled=args.camera_fill_light_enabled,
        camera_fill_light_intensity=args.camera_fill_light_intensity,
        camera_fill_light_radius_m=args.camera_fill_light_radius_m,
        camera_fill_light_exposure=args.camera_fill_light_exposure,
        camera_fill_light_color_temperature_k=args.camera_fill_light_color_temperature_k,
        camera_fill_light_auto_distance_enabled=args.camera_fill_light_auto_distance_enabled,
        camera_fill_light_reference_distance_m=args.camera_fill_light_reference_distance_m,
        camera_fill_light_min_intensity=args.camera_fill_light_min_intensity,
        camera_fill_light_max_intensity=args.camera_fill_light_max_intensity,
        camera_fill_light_auto_render_updates=args.camera_fill_light_auto_render_updates,
        enable_nearfield_depth=args.nearfield_depth,
        nearfield_width=args.nearfield_width,
        nearfield_height=args.nearfield_height,
        nearfield_hfov_deg=args.nearfield_hfov_deg,
        nearfield_height_m=args.nearfield_height_m,
    )
    try:
        print("[isaac] reset episode", flush=True)
        obs = server.reset_episode(args.scene_usd, tuple(args.spawn), rgb_device="cpu" if args.save_frame else None)
        print("[isaac] observation captured", flush=True)
        if args.save_frame:
            from PIL import Image

            out = Path(args.save_frame)
            out.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(obs["rgb"]).save(out)
            depth = obs["depth"]
            finite = np.isfinite(depth)
            if np.any(finite):
                depth_clean = np.where(finite, depth, 0.0)
                depth_max = max(float(np.max(depth_clean)), 1e-6)
            else:
                depth_clean = np.zeros_like(depth, dtype=np.float32)
                depth_max = 1.0
            depth_img = np.clip(depth_clean / depth_max * 255.0, 0, 255).astype(np.uint8)
            Image.fromarray(depth_img).save(out.with_name(out.stem + "_depth.png"))
        print({"pose_world": obs["pose_world"], "camera_pose_world": obs["camera_pose_world"]}, flush=True)
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
