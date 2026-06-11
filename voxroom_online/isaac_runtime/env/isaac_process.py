from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from voxroom_online.isaac_runtime.config import str_to_bool
from voxroom_online.isaac_runtime.sensors.camera_geometry import CameraIntrinsics
from voxroom_online.isaac_runtime.sensors.depth_backproject import distance_to_camera_to_image_plane_depth


def yaw_to_quat_wxyz(yaw: float) -> np.ndarray:
    return np.asarray([math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5)], dtype=np.float32)


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
        device = str(camera_annotator_device or "cpu").strip().lower()
        self.camera_annotator_device = device if device in {"cpu", "cuda"} else "cpu"
        self.app = None
        self.world = None
        self.robot = None
        self.controller = None
        self.camera = None
        self.nearfield_camera = None
        self.camera_prim_path = "/World/Kaya/camera_rgbd"
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
        self._logged_cuda_rgb_fallback = False
        self._kaya_physics_disabled_for_kinematic = False
        self.robot_pose_sync_failures = 0
        self.camera_pose_sync_failures = 0
        self.nearfield_camera_pose_sync_failures = 0

    def log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    def start(self) -> None:
        from isaacsim import SimulationApp

        self.log("[isaac] starting SimulationApp")
        self.app = SimulationApp({"headless": self.headless})
        self.log("[isaac] SimulationApp ready")

    def load_scene(self, usd_path: str) -> None:
        if self.app is None:
            self.start()
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import is_stage_loading, open_stage

        self.log("[isaac] opening stage %s" % usd_path)
        open_stage(str(usd_path))
        while is_stage_loading():
            self.app.update()
        self.disable_imported_scene_rigid_bodies()
        self.world = World(stage_units_in_meters=1.0)
        self.log("[isaac] stage loaded")

    def disable_imported_scene_rigid_bodies(self) -> None:
        try:
            import omni.usd
            from pxr import UsdPhysics
        except Exception as exc:
            self.log("[isaac] scene rigid-body cleanup skipped: %s" % exc)
            return
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
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
        self.configure_camera_intrinsics()
        self.bind_viewport_to_robot_camera()
        if self.enable_nearfield_depth:
            self.attach_nearfield_depth_camera(pose_world)

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
        self.log("[isaac] resetting world")
        self.world.reset()
        self.set_pose_world(pose_world)
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
        if self.enable_nearfield_depth and self.nearfield_camera is not None:
            try:
                self.nearfield_camera.add_distance_to_image_plane_to_frame()
            except Exception as exc:
                self.log("[isaac] nearfield camera annotator fallback: %s" % exc)
        self.log("[isaac] rendering warmup frames")
        for _ in range(60):
            self.app.update()
        return self.get_observation(read_rgb=read_rgb, read_depth=self.enable_depth, rgb_device=rgb_device)

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
        previous_frame_token = self._camera_frame_token()
        if self.robot is not None and self.controller is not None:
            self.robot.apply_wheel_actions(self.controller.forward(command=[float(vx), float(vy), float(wz)]))
        for _ in range(int(frames)):
            self.world.step(render=True)
        self._wait_for_fresh_camera_frame(previous_frame_token, read_depth=self.enable_depth if read_depth is None else bool(read_depth))
        return self.get_observation(
            read_rgb=read_rgb,
            read_depth=self.enable_depth if read_depth is None else bool(read_depth),
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
        previous_frame_token = self._camera_frame_token()
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
        self.set_pose_world((x + dx * dt, y + dy * dt, z, yaw), sync_robot=True)
        for _ in range(int(render_updates)):
            self.app.update()
        self._wait_for_fresh_camera_frame(previous_frame_token, read_depth=self.enable_depth if read_depth is None else bool(read_depth))
        return self.get_observation(
            read_rgb=read_rgb,
            read_depth=self.enable_depth if read_depth is None else bool(read_depth),
            rgb_device=rgb_device,
        )

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
        max_updates: int = 10,
        min_fresh_frames: int = 2,
    ) -> None:
        """Avoid pairing a stale RGB-D frame with a newly-updated pose."""
        self.last_camera_frame_sync_updates = 0
        if self.app is None or self.camera is None or previous_token is None:
            return
        fresh_frames = 0
        last_token = previous_token
        for idx in range(int(max_updates) + 1):
            token = self._camera_frame_token()
            if token is not None and token != last_token:
                fresh_frames += 1
                last_token = token
            if fresh_frames >= max(1, int(min_fresh_frames)):
                self.last_camera_frame_token = last_token
                self.last_camera_frame_sync_updates = idx
                return
            self.app.update()
        self.last_camera_frame_token = self._camera_frame_token()
        self.last_camera_frame_sync_updates = int(max_updates)

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

    def get_observation(self, read_rgb: bool = True, read_depth: bool = False, rgb_device: Optional[str] = None) -> dict:
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
        if self.camera is not None:
            if read_depth and self.enable_depth:
                self.last_depth = None
                self.last_depth_source = "none"
                self.last_depth_semantics = "none"
                frame = self.camera.get_current_frame() or {}
            if read_rgb:
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
            if read_depth and self.enable_depth:
                depth_value = frame.get("distance_to_image_plane")
                depth_source = "distance_to_image_plane"
                depth_semantics = "image_plane_z"
                if not isinstance(depth_value, np.ndarray):
                    depth_value = frame.get("distance_to_camera")
                    depth_source = "distance_to_camera"
                    depth_semantics = "euclidean_distance_to_camera"
                if not isinstance(depth_value, np.ndarray):
                    try:
                        depth_value = self.camera.get_depth(device="cpu")
                        depth_source = "camera_get_depth"
                        # Isaac Sim Camera.get_depth() is documented and implemented
                        # as the distance_to_image_plane annotator.
                        depth_semantics = "image_plane_z"
                    except Exception:
                        depth_value = None
                if isinstance(depth_value, np.ndarray):
                    depth_frame = np.asarray(depth_value, dtype=np.float32)
                    if depth_frame.ndim == 2:
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
            if read_depth and self.enable_nearfield_depth and self.nearfield_camera is not None:
                self.last_nearfield_depth = None
                self.last_nearfield_depth_source = "none"
                self.last_nearfield_depth_semantics = "none"
                nearfield_depth_value = None
                nearfield_depth_source = "nearfield_distance_to_image_plane"
                nearfield_depth_semantics = "image_plane_z"
                try:
                    nearfield_frame = self.nearfield_camera.get_current_frame() or {}
                    nearfield_depth_value = nearfield_frame.get("distance_to_image_plane")
                except Exception:
                    nearfield_depth_value = None
                if not isinstance(nearfield_depth_value, np.ndarray):
                    try:
                        nearfield_depth_value = self.nearfield_camera.get_depth(device="cpu")
                        nearfield_depth_source = "nearfield_camera_get_depth"
                        nearfield_depth_semantics = "image_plane_z"
                    except Exception:
                        nearfield_depth_value = None
                if isinstance(nearfield_depth_value, np.ndarray):
                    nearfield_depth_frame = np.asarray(nearfield_depth_value, dtype=np.float32)
                    if nearfield_depth_frame.ndim == 2:
                        nearfield_depth = nearfield_depth_frame
                        self.last_nearfield_depth = nearfield_depth
                        self.last_nearfield_depth_source = nearfield_depth_source
                        self.last_nearfield_depth_semantics = nearfield_depth_semantics
        pose = self.get_pose_world()
        camera_frame_token = self._camera_frame_token()
        if camera_frame_token is not None:
            self.last_camera_frame_token = camera_frame_token
        return {
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
            "sim_time": 0.0,
            "collided": False,
        }

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
