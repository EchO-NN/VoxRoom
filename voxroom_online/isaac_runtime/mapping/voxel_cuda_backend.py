from __future__ import annotations

import time

import numpy as np

from voxroom_online.isaac_runtime.mapping.voxel_occupancy_grid import VoxelIntegrationStats
from voxroom_online.isaac_runtime.mapping.voxel_cpu_fast_backend import _endpoint_range_column_lines, _endpoint_splat_lines, _world_to_voxel_float_array


class VoxelCudaBackend:
    @staticmethod
    def _torch():
        try:
            import torch
        except Exception:
            return None
        return torch

    @classmethod
    def is_available(cls, device: str = "cuda:0") -> bool:
        torch = cls._torch()
        if torch is None:
            return False
        try:
            return bool(torch.cuda.is_available()) and str(device).startswith("cuda")
        except Exception:
            return False

    @classmethod
    def integrate(
        cls,
        grid,
        *,
        camera_origin_world,
        points_world: np.ndarray,
        floor_z: float,
        valid_mask: np.ndarray,
        endpoint_is_hit: np.ndarray | None = None,
    ) -> VoxelIntegrationStats:
        torch = cls._torch()
        if torch is None or not cls.is_available(str(grid.config.cuda_device)):
            raise RuntimeError("cuda_torch voxel backend requested but torch CUDA is unavailable")

        started_at = time.perf_counter()
        device = torch.device(str(grid.config.cuda_device))
        stats = VoxelIntegrationStats(integration_backend="cuda_torch", python_debug_backend_used=False)
        points = np.asarray(points_world, dtype=np.float32)
        valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if endpoint_is_hit is None:
            hit = np.ones(points.shape[0], dtype=bool)
        else:
            hit = np.asarray(endpoint_is_hit, dtype=bool).reshape(-1)
            if hit.size != points.shape[0]:
                raise ValueError("endpoint_is_hit must match points_world length")
        hit = hit[valid]
        points = points[valid]
        finite = np.all(np.isfinite(points[:, :3]), axis=1) if points.size else np.zeros(0, dtype=bool)
        hit = hit[finite]
        points = points[finite]
        if points.size == 0:
            stats.skipped_empty_rays = int(np.count_nonzero(valid))
            stats.integrate_total_ms = float((time.perf_counter() - started_at) * 1000.0)
            return stats

        origin_np = _world_to_voxel_float_array(
            grid,
            np.asarray(camera_origin_world, dtype=np.float32).reshape(1, -1)[:, :3],
            float(floor_z),
        )[0]
        endpoints_np = _world_to_voxel_float_array(grid, points[:, :3], float(floor_z))
        finite_endpoint = np.all(np.isfinite(endpoints_np), axis=1) & np.all(np.isfinite(origin_np))
        hit = hit[finite_endpoint]
        endpoints_np = endpoints_np[finite_endpoint]
        if endpoints_np.size == 0:
            stats.skipped_empty_rays = int(points.shape[0])
            stats.integrate_total_ms = float((time.perf_counter() - started_at) * 1000.0)
            return stats

        chunk_rays = max(1, int(grid.config.cuda_chunk_rays))
        step_voxels = max(0.10, float(grid.config.cuda_ray_step_voxels))
        max_samples = max(1, int(grid.config.cuda_max_samples_per_ray))
        stats.cuda_chunk_rays = int(chunk_rays)
        stats.cuda_max_samples_per_ray = int(max_samples)
        height, width = int(grid.state.shape[1]), int(grid.state.shape[2])
        z_bins = int(grid.state.shape[0])
        total_voxels = int(grid.state.size)
        log_flat = torch.as_tensor(grid.log_odds.reshape(-1).astype(np.int32), dtype=torch.int32, device=device).clone()
        sensor_enabled = bool(getattr(grid.config, "sensor_range_tracking_enabled", True))
        sensor_ray_enabled = sensor_enabled and bool(getattr(grid.config, "sensor_range_mark_ray_samples_enabled", True))
        sensor_endpoint_enabled = sensor_enabled and bool(getattr(grid.config, "sensor_range_mark_endpoint_column_enabled", True))
        sensor_delta = max(0, int(getattr(grid.config, "sensor_range_count_delta", 1)))
        sensor_max = int(np.clip(int(getattr(grid.config, "sensor_range_count_max", 255)), 0, 255))
        sensor_flat = torch.as_tensor(grid.sensor_range_count.reshape(-1).astype(np.int32), dtype=torch.int32, device=device).clone()
        origin = torch.as_tensor(origin_np, dtype=torch.float32, device=device)
        endpoints = torch.as_tensor(endpoints_np, dtype=torch.float32, device=device)
        endpoint_hit = torch.as_tensor(hit.astype(np.bool_, copy=False), dtype=torch.bool, device=device)
        free_delta = int(grid.config.free_logodds_delta)
        occ_delta = int(grid.config.occupied_logodds_delta)
        exclude_n = max(0, int(grid.config.free_excludes_last_n_voxels_before_endpoint))
        free_vote_cap = max(0, int(getattr(grid.config, "free_vote_cap_per_voxel", 1)))
        occ_vote_cap = max(0, int(getattr(grid.config, "occupied_vote_cap_per_voxel", 3)))
        hit_wins = bool(getattr(grid.config, "occupied_wins_over_free_same_voxel", True))
        integrated = 0
        hit_integrated = 0
        free_only_integrated = 0

        for start in range(0, int(endpoints.shape[0]), int(chunk_rays)):
            end = min(int(endpoints.shape[0]), start + int(chunk_rays))
            p1 = endpoints[start:end]
            p1_is_hit = endpoint_hit[start:end]
            sample_started = time.perf_counter()
            delta = p1 - origin[None, :]
            ray_len = torch.max(torch.abs(delta), dim=1).values
            steps = torch.ceil(ray_len / float(step_voxels)).to(torch.int64).clamp_(1, int(max_samples))
            s = torch.arange(int(torch.max(steps).item()) + 1, dtype=torch.float32, device=device)
            valid_step = s[None, :] <= steps[:, None]
            t = s[None, :] / steps[:, None].to(torch.float32)
            coords = origin[None, None, :] + t[:, :, None] * delta[:, None, :]
            vox = torch.floor(coords).to(torch.int64)
            z = vox[:, :, 0]
            r = vox[:, :, 1]
            c = vox[:, :, 2]
            in_bounds = valid_step & (z >= 0) & (z < z_bins) & (r >= 0) & (r < height) & (c >= 0) & (c < width)
            ray_has_sample = torch.any(in_bounds, dim=1)
            integrated += int(torch.count_nonzero(ray_has_sample).item())
            hit_integrated += int(torch.count_nonzero(ray_has_sample & p1_is_hit).item())
            free_only_integrated += int(torch.count_nonzero(ray_has_sample & ~p1_is_hit).item())
            lin = z * int(height * width) + r * int(width) + c
            if sensor_ray_enabled and int(torch.count_nonzero(in_bounds).item()):
                range_unique = torch.unique(lin[in_bounds])
                if int(range_unique.numel()):
                    if sensor_delta > 0 and sensor_max > 0:
                        sensor_flat[range_unique] = torch.clamp(sensor_flat[range_unique] + int(sensor_delta), max=int(sensor_max))
                    stats.sensor_range_update_count += int(range_unique.numel())
            stats.ray_sample_ms += float((time.perf_counter() - sample_started) * 1000.0)

            unique_started = time.perf_counter()
            if bool(grid.config.free_excludes_endpoint):
                # Endpoint is excluded by step index, so runtime does not need
                # expensive per-ray torch.isin over ray_id * total_voxels keys.
                hit_free_step = s[None, :] < torch.clamp(steps - int(exclude_n), min=0)[:, None]
                free_step = torch.where(p1_is_hit[:, None], hit_free_step, valid_step)
                free_mask = in_bounds & free_step
            else:
                free_mask = in_bounds
            free_lin = lin[free_mask]
            endpoint_vox = torch.floor(p1).to(torch.int64)
            endpoint_valid = (
                (endpoint_vox[:, 0] >= 0)
                & (endpoint_vox[:, 0] < z_bins)
                & (endpoint_vox[:, 1] >= 0)
                & (endpoint_vox[:, 1] < height)
                & (endpoint_vox[:, 2] >= 0)
                & (endpoint_vox[:, 2] < width)
            )
            endpoint_valid_hit = endpoint_valid & p1_is_hit
            endpoint_lin = endpoint_vox[:, 0] * int(height * width) + endpoint_vox[:, 1] * int(width) + endpoint_vox[:, 2]
            endpoint_lin_valid = endpoint_lin[endpoint_valid_hit]
            if sensor_endpoint_enabled and int(endpoint_valid_hit.sum().item()):
                endpoint_cpu = endpoint_vox[endpoint_valid_hit].detach().cpu().numpy().astype(np.int32)
                range_np = _endpoint_range_column_lines(grid, endpoint_cpu, height=height, width=width, z_bins=z_bins)
                if range_np.size:
                    range_unique = torch.as_tensor(range_np, dtype=torch.int64, device=device)
                    if sensor_delta > 0 and sensor_max > 0:
                        sensor_flat[range_unique] = torch.clamp(sensor_flat[range_unique] + int(sensor_delta), max=int(sensor_max))
                    stats.sensor_range_update_count += int(range_unique.numel())
            if int(free_lin.numel()):
                free_unique, free_counts = torch.unique(free_lin, return_counts=True)
            else:
                free_unique = torch.empty(0, dtype=torch.int64, device=device)
                free_counts = torch.empty(0, dtype=torch.int64, device=device)

            if bool(grid.config.mark_endpoint_occupied) and int(endpoint_lin_valid.numel()):
                if int(grid.config.endpoint_splat_z_radius_cells) == 0 and int(grid.config.endpoint_splat_xy_radius_cells) == 0:
                    occ_unique, occ_counts = torch.unique(endpoint_lin_valid, return_counts=True)
                else:
                    endpoint_cpu = endpoint_vox[endpoint_valid_hit].detach().cpu().numpy().astype(np.int32)
                    occ_np = _endpoint_splat_lines(grid, endpoint_cpu, height=height, width=width, z_bins=z_bins)
                    if occ_np.size:
                        occ_lin = torch.as_tensor(occ_np, dtype=torch.int64, device=device)
                        occ_unique, occ_counts = torch.unique(occ_lin, return_counts=True)
                    else:
                        occ_unique = torch.empty(0, dtype=torch.int64, device=device)
                        occ_counts = torch.empty(0, dtype=torch.int64, device=device)
            else:
                occ_unique = torch.empty(0, dtype=torch.int64, device=device)
                occ_counts = torch.empty(0, dtype=torch.int64, device=device)
            if hit_wins and int(free_unique.numel()) and int(occ_unique.numel()):
                try:
                    keep = ~torch.isin(free_unique, occ_unique)
                except Exception:
                    keep_np = ~np.isin(
                        free_unique.detach().cpu().numpy(),
                        occ_unique.detach().cpu().numpy(),
                        assume_unique=False,
                    )
                    keep = torch.as_tensor(keep_np, dtype=torch.bool, device=device)
                free_unique = free_unique[keep]
                free_counts = free_counts[keep]
            if free_vote_cap > 0 and int(free_counts.numel()):
                free_counts = torch.clamp(free_counts, max=int(free_vote_cap))
            if occ_vote_cap > 0 and int(occ_counts.numel()):
                occ_counts = torch.clamp(occ_counts, max=int(occ_vote_cap))
            stats.unique_ms += float((time.perf_counter() - unique_started) * 1000.0)

            scatter_started = time.perf_counter()
            if int(free_unique.numel()):
                log_flat.index_add_(0, free_unique, free_counts.to(torch.int32) * int(free_delta))
                stats.free_update_count += int(torch.sum(free_counts).item())
            if int(occ_unique.numel()):
                log_flat.index_add_(0, occ_unique, occ_counts.to(torch.int32) * int(occ_delta))
                stats.occupied_update_count += int(torch.sum(occ_counts).item())
            log_flat.clamp_(int(grid.config.logodds_min), int(grid.config.logodds_max))
            stats.scatter_ms += float((time.perf_counter() - scatter_started) * 1000.0)

        grid.log_odds.reshape(-1)[:] = log_flat.detach().cpu().numpy().astype(np.int16)
        if sensor_enabled:
            grid.sensor_range_count.reshape(-1)[:] = sensor_flat.detach().cpu().numpy().astype(np.uint8)
        stats.depth_rays_integrated = int(integrated)
        stats.depth_hit_rays_integrated = int(hit_integrated)
        stats.depth_free_only_rays_integrated = int(free_only_integrated)
        stats.skipped_empty_rays = max(0, int(endpoints_np.shape[0]) - int(integrated))
        refresh_started = time.perf_counter()
        grid.refresh_state()
        stats.refresh_state_ms = float((time.perf_counter() - refresh_started) * 1000.0)
        stats.integrate_total_ms = float((time.perf_counter() - started_at) * 1000.0)
        return stats
