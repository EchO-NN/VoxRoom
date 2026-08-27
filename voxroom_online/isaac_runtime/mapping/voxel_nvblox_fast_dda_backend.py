from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from voxroom_online.isaac_runtime.mapping.voxel_occupancy_grid import (
    VOXEL_FREE,
    VOXEL_OCCUPIED,
    VOXEL_UNKNOWN,
    VoxelIntegrationStats,
)
from voxroom_online.isaac_runtime.sensors.camera_geometry import pose_world_to_matrix


@dataclass
class NvbloxFastDdaRuntime:
    mapper: object
    sensor: object
    grid_uid: int
    device: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    voxel_size_m: float


def occupancy_logodds_to_voxroom_state(
    logodds: np.ndarray,
    free_thr: float,
    occ_thr: float,
    unk_eps: float,
    unobserved_thr: float = -6.0,
) -> np.ndarray:
    values = np.asarray(logodds, dtype=np.float32)
    out = np.full(values.shape, int(VOXEL_UNKNOWN), dtype=np.uint8)
    unknown = (np.abs(values) <= float(unk_eps)) | (values <= float(unobserved_thr))
    free = (values <= float(free_thr)) & ~unknown
    occ = (values >= float(occ_thr)) & ~unknown
    out[free] = int(VOXEL_FREE)
    out[occ] = int(VOXEL_OCCUPIED)
    return out


class VoxelNvbloxFastDdaBackend:
    _runtime: NvbloxFastDdaRuntime | None = None

    @classmethod
    def reset_runtime(cls) -> None:
        cls._runtime = None

    @classmethod
    def integrate_depth_image(
        cls,
        grid,
        *,
        depth: np.ndarray,
        intr,
        camera_pose_world: Sequence[float] | np.ndarray,
        floor_z: float,
    ) -> VoxelIntegrationStats:
        started_at = time.perf_counter()
        stats = VoxelIntegrationStats(integration_backend="nvblox_fast_dda")

        cls._require_runtime_dependencies(grid)
        depth_np = cls._prepare_depth(depth)
        h, w = int(depth_np.shape[0]), int(depth_np.shape[1])
        runtime = cls._get_or_create_runtime(grid, intr=intr, width=w, height=h)

        import torch

        device = str(getattr(grid.config, "nvblox_device", "cuda:0"))
        depth_t = torch.as_tensor(depth_np, dtype=torch.float32, device=device).contiguous()
        t_w_c = cls._voxroom_pose_to_nvblox_optical_pose(camera_pose_world)
        t_w_c_t = torch.as_tensor(t_w_c, dtype=torch.float32, device="cpu").contiguous()

        runtime.mapper.add_depth_frame(depth_t, t_w_c_t, runtime.sensor)

        update = cls._query_frustum_bbox_and_write_state(
            grid,
            runtime=runtime,
            intr=intr,
            camera_pose_world=camera_pose_world,
            floor_z=float(floor_z),
        )

        finite = np.isfinite(depth_np)
        depth_min = float(getattr(grid.config, "depth_min_m", 0.2))
        depth_max = float(getattr(grid.config, "depth_max_m", 3.0))
        hit = finite & (depth_np > depth_min) & (depth_np < depth_max)
        stats.depth_rays_integrated = int(np.count_nonzero(finite))
        stats.depth_hit_rays_integrated = int(np.count_nonzero(hit))
        stats.depth_free_only_rays_integrated = max(0, int(stats.depth_rays_integrated) - int(stats.depth_hit_rays_integrated))
        stats.free_update_count = int(update["free_updates"])
        stats.occupied_update_count = int(update["occupied_updates"])
        stats.refresh_changed_voxels = int(update["changed_voxels"])
        stats.voxel_integrate_dirty_rc_count = int(update["dirty_rc_count"])
        stats.voxel_integrate_changed_flag_count = int(update["changed_voxels"])
        stats.voxel_integrate_total_samples = int(update["query_voxels"])
        stats.nvblox_query_mode = str(getattr(grid.config, "nvblox_query_mode", "frustum_bbox_query_z"))
        stats.nvblox_query_voxels = int(update["query_voxels"])
        stats.nvblox_dirty_rc_count = int(update["dirty_rc_count"])
        stats.nvblox_active_z_bin_count = int(update["query_z_bin_count"])
        stats.nvblox_frustum_bbox_rc = tuple(int(v) for v in update["frustum_bbox_rc"])
        stats.refresh_mode = "nvblox_fast_dda_frustum_bbox"
        stats.integrate_total_ms = float((time.perf_counter() - started_at) * 1000.0)
        return stats

    @staticmethod
    def _require_runtime_dependencies(grid) -> None:
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("nvblox_fast_dda requires torch") from exc

        if not torch.cuda.is_available():
            raise RuntimeError("nvblox_fast_dda requires CUDA")

        try:
            import nvblox_torch  # noqa: F401
            from nvblox_torch.mapper import Mapper, QueryType  # noqa: F401
            from nvblox_torch.mapper_params import MapperParams, ProjectiveIntegratorParams  # noqa: F401
            from nvblox_torch.projective_integrator_types import ProjectiveIntegratorType  # noqa: F401
            from nvblox_torch.sensor import Sensor  # noqa: F401
        except Exception as exc:
            raise RuntimeError("nvblox_fast_dda requires nvblox_torch") from exc

        layer_type = str(getattr(grid.config, "nvblox_projective_layer_type", "occupancy")).strip().lower()
        if layer_type != "occupancy":
            raise RuntimeError("nvblox_fast_dda only supports occupancy projective layer in this implementation")
        query_mode = str(getattr(grid.config, "nvblox_query_mode", "frustum_bbox_query_z")).strip().lower()
        if query_mode != "frustum_bbox_query_z":
            raise RuntimeError("nvblox_fast_dda only supports nvblox_query_mode=frustum_bbox_query_z")

    @classmethod
    def _get_or_create_runtime(cls, grid, *, intr, width: int, height: int) -> NvbloxFastDdaRuntime:
        device = str(getattr(grid.config, "nvblox_device", "cuda:0"))
        voxel_size_m = cls._nvblox_voxel_size_m(grid)
        grid_uid = int(id(grid))
        if cls._runtime is not None:
            same_runtime = (
                cls._runtime.grid_uid == grid_uid
                and cls._runtime.device == device
                and cls._runtime.width == int(width)
                and cls._runtime.height == int(height)
                and np.isclose(cls._runtime.fx, float(intr.fx))
                and np.isclose(cls._runtime.fy, float(intr.fy))
                and np.isclose(cls._runtime.cx, float(intr.cx))
                and np.isclose(cls._runtime.cy, float(intr.cy))
                and np.isclose(cls._runtime.voxel_size_m, float(voxel_size_m))
            )
            if same_runtime:
                return cls._runtime
            cls._runtime = None

        from nvblox_torch.mapper import Mapper
        from nvblox_torch.mapper_params import MapperParams, ProjectiveIntegratorParams
        from nvblox_torch.projective_integrator_types import ProjectiveIntegratorType
        from nvblox_torch.sensor import Sensor

        projective_params = ProjectiveIntegratorParams()
        projective_params.projective_integrator_max_integration_distance_m = float(getattr(grid.config, "depth_max_m", 3.0))

        mapper_params = MapperParams()
        mapper_params.set_projective_integrator_params(projective_params)

        mapper = Mapper(
            voxel_sizes_m=float(voxel_size_m),
            integrator_types=ProjectiveIntegratorType.OCCUPANCY,
            mapper_parameters=mapper_params,
        )
        sensor = Sensor.from_camera(
            fu=float(intr.fx),
            fv=float(intr.fy),
            cu=float(intr.cx),
            cv=float(intr.cy),
            width=int(width),
            height=int(height),
        )

        cls._runtime = NvbloxFastDdaRuntime(
            mapper=mapper,
            sensor=sensor,
            grid_uid=grid_uid,
            device=device,
            width=int(width),
            height=int(height),
            fx=float(intr.fx),
            fy=float(intr.fy),
            cx=float(intr.cx),
            cy=float(intr.cy),
            voxel_size_m=float(voxel_size_m),
        )
        return cls._runtime

    @staticmethod
    def _nvblox_voxel_size_m(grid) -> float:
        xy_resolution_m = float(getattr(grid.map_info, "resolution_m", 0.0))
        z_resolution_m = float(getattr(grid, "z_resolution_m", 0.0))
        if not np.isfinite(xy_resolution_m) or xy_resolution_m <= 0.0:
            raise RuntimeError("nvblox_fast_dda requires a positive VoxRoom map resolution")
        if not np.isfinite(z_resolution_m) or z_resolution_m <= 0.0:
            raise RuntimeError("nvblox_fast_dda requires a positive voxel z_resolution_m")
        if not np.isclose(xy_resolution_m, z_resolution_m, rtol=1.0e-5, atol=1.0e-6):
            raise RuntimeError(
                "nvblox_fast_dda requires cubic voxels aligned to VoxRoom map resolution: "
                f"map resolution={xy_resolution_m:.6f}m, z_resolution={z_resolution_m:.6f}m"
            )
        return xy_resolution_m

    @staticmethod
    def _prepare_depth(depth: np.ndarray) -> np.ndarray:
        arr = np.asarray(depth, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        if arr.ndim != 2:
            raise ValueError(f"depth must be HxW, got {arr.shape}")
        if arr.size == 0:
            raise ValueError("depth image is empty")
        return arr.astype(np.float32, copy=False)

    @staticmethod
    def _voxroom_pose_to_nvblox_optical_pose(
        camera_pose_world: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        t_w_b = pose_world_to_matrix(camera_pose_world)
        t_b_c = np.eye(4, dtype=np.float32)
        t_b_c[:3, :3] = np.array(
            [
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float32,
        )
        return (t_w_b @ t_b_c).astype(np.float32)

    @staticmethod
    def _frustum_bbox_rc(
        grid,
        *,
        intr,
        camera_pose_world: Sequence[float] | np.ndarray,
        floor_z: float,
    ) -> tuple[int, int, int, int]:
        _ = floor_z
        depth_min = max(0.0, float(getattr(grid.config, "depth_min_m", 0.2)))
        depth_max = float(getattr(grid.config, "depth_max_m", 3.0))
        margin = max(0.0, float(getattr(grid.config, "nvblox_frustum_xy_margin_m", 0.25)))
        if not np.isfinite(depth_max) or depth_max <= depth_min:
            raise RuntimeError("nvblox_fast_dda requires a finite positive depth range")

        corners = []
        for z in (depth_min, depth_max):
            for u, v in (
                (0.0, 0.0),
                (float(intr.width - 1), 0.0),
                (0.0, float(intr.height - 1)),
                (float(intr.width - 1), float(intr.height - 1)),
            ):
                x_right = (u - float(intr.cx)) * z / float(intr.fx)
                y_down = (v - float(intr.cy)) * z / float(intr.fy)
                corners.append([z, -x_right, -y_down])

        t_w_b = pose_world_to_matrix(camera_pose_world)
        pts = np.asarray(corners, dtype=np.float32)
        homo = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
        world = (t_w_b @ homo.T).T[:, :3]

        min_x = float(np.min(world[:, 0])) - margin
        max_x = float(np.max(world[:, 0])) + margin
        min_y = float(np.min(world[:, 1])) - margin
        max_y = float(np.max(world[:, 1])) + margin

        mi = grid.map_info
        res = float(mi.resolution_m)
        c0 = max(0, int(np.floor((min_x - float(mi.min_x)) / res)))
        c1 = min(int(mi.width) - 1, int(np.floor((max_x - float(mi.min_x)) / res)))
        r0 = max(0, int(np.floor((float(mi.max_y) - max_y) / res)))
        r1 = min(int(mi.height) - 1, int(np.floor((float(mi.max_y) - min_y) / res)))
        if c1 < c0 or r1 < r0:
            return 0, -1, 0, -1
        return int(r0), int(r1), int(c0), int(c1)

    @classmethod
    def _query_frustum_bbox_and_write_state(
        cls,
        grid,
        *,
        runtime: NvbloxFastDdaRuntime,
        intr,
        camera_pose_world: Sequence[float] | np.ndarray,
        floor_z: float,
    ) -> dict[str, int]:
        import torch
        from nvblox_torch.mapper import QueryType

        r0, r1, c0, c1 = cls._frustum_bbox_rc(
            grid,
            intr=intr,
            camera_pose_world=camera_pose_world,
            floor_z=float(floor_z),
        )
        if r1 < r0 or c1 < c0:
            raise RuntimeError("nvblox_fast_dda frustum bbox is outside VoxRoom map")

        z_idx = cls._strict_query_z_indices(grid)
        if z_idx.size == 0:
            raise RuntimeError("nvblox_fast_dda query z range is empty")

        rows = np.arange(r0, r1 + 1, dtype=np.int32)
        cols = np.arange(c0, c1 + 1, dtype=np.int32)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        rc_flat = np.stack([rr.reshape(-1), cc.reshape(-1)], axis=1).astype(np.int32)

        chunk = max(1, int(getattr(grid.config, "nvblox_query_chunk_voxels", 1048576)))
        free_thr = float(getattr(grid.config, "nvblox_occupancy_free_logodds_threshold", -0.10))
        occ_thr = float(getattr(grid.config, "nvblox_occupancy_occupied_logodds_threshold", 0.10))
        unk_eps = float(getattr(grid.config, "nvblox_occupancy_unknown_abs_logodds_epsilon", 0.10))
        unobserved_thr = float(getattr(grid.config, "nvblox_occupancy_unobserved_logodds_threshold", -6.0))
        device = str(getattr(grid.config, "nvblox_device", "cuda:0"))

        old_state = grid.state.copy()
        free_updates = 0
        occ_updates = 0
        query_voxels = 0
        rr_i = rc_flat[:, 0].astype(np.int64)
        cc_i = rc_flat[:, 1].astype(np.int64)

        for zi in z_idx:
            z_world = float(floor_z) + grid.voxel_center_z(int(zi))
            points = np.empty((int(rc_flat.shape[0]), 3), dtype=np.float32)
            points[:, 0] = float(grid.map_info.min_x) + (rc_flat[:, 1].astype(np.float32) + 0.5) * float(grid.map_info.resolution_m)
            points[:, 1] = float(grid.map_info.max_y) - (rc_flat[:, 0].astype(np.float32) + 0.5) * float(grid.map_info.resolution_m)
            points[:, 2] = float(z_world)

            out_state = np.full((int(points.shape[0]),), int(VOXEL_UNKNOWN), dtype=np.uint8)
            for start in range(0, int(points.shape[0]), chunk):
                end = min(int(points.shape[0]), start + chunk)
                q = torch.as_tensor(points[start:end], dtype=torch.float32, device=device).contiguous()
                result = runtime.mapper.query_layer(QueryType.OCCUPANCY, q, mapper_id=-1)
                if hasattr(result, "detach"):
                    logodds = result.reshape(-1).detach().cpu().numpy().astype(np.float32, copy=False)
                else:
                    logodds = np.asarray(result, dtype=np.float32).reshape(-1)
                out_state[start:end] = occupancy_logodds_to_voxroom_state(
                    logodds,
                    free_thr,
                    occ_thr,
                    unk_eps,
                    unobserved_thr,
                )
                query_voxels += int(end - start)

            grid.state[int(zi), rr_i, cc_i] = out_state
            free_updates += int(np.count_nonzero(out_state == int(VOXEL_FREE)))
            occ_updates += int(np.count_nonzero(out_state == int(VOXEL_OCCUPIED)))

        changed = grid.state != old_state
        changed_voxels = int(np.count_nonzero(changed))
        dirty_rc = np.any(changed, axis=0).reshape(-1).astype(np.uint8)
        grid.last_dirty_rc_flags = dirty_rc
        return {
            "free_updates": int(free_updates),
            "occupied_updates": int(occ_updates),
            "changed_voxels": int(changed_voxels),
            "dirty_rc_count": int(np.count_nonzero(dirty_rc)),
            "query_voxels": int(query_voxels),
            "query_z_bin_count": int(z_idx.size),
            "frustum_bbox_rc": (int(r0), int(r1), int(c0), int(c1)),
        }

    @staticmethod
    def _strict_query_z_indices(grid) -> np.ndarray:
        zmin_cfg = getattr(grid.config, "nvblox_query_z_min_m", None)
        zmax_cfg = getattr(grid.config, "nvblox_query_z_max_m", None)
        zmin = float(getattr(grid.config, "z_min_m", getattr(grid, "z_min_m", -0.10)) if zmin_cfg is None else zmin_cfg)
        zmax = float(getattr(grid.config, "z_max_m", getattr(grid, "z_max_m", 4.0)) if zmax_cfg is None else zmax_cfg)
        if not np.isfinite(zmin) or not np.isfinite(zmax) or zmax < zmin:
            return np.zeros(0, dtype=np.int32)
        z = np.asarray(grid.z_centers_m, dtype=np.float32)
        query = np.nonzero((z >= zmin) & (z <= zmax))[0].astype(np.int32)
        if query.size == 0:
            return query
        margin = max(0.0, float(getattr(grid.config, "nvblox_frustum_z_margin_m", 0.10)))
        nav_min = max(float(getattr(grid.config, "z_min_m", 0.0)), 0.10 - margin)
        nav_max = min(float(getattr(grid.config, "z_max_m", 4.0)), 0.90 + margin)
        required = np.nonzero((z >= nav_min) & (z <= nav_max))[0].astype(np.int32)
        if required.size == 0:
            return query
        return np.unique(np.concatenate([query, required])).astype(np.int32)
