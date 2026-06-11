from __future__ import annotations

import time

import numpy as np

from voxroom_online.isaac_runtime.mapping.voxel_cpu_autotune import (
    autotune_cache_key,
    load_cached_thread_count,
    parse_thread_candidates,
    store_cached_thread_count,
)
from voxroom_online.isaac_runtime.mapping.voxel_cpu_fast_backend import _endpoint_splat_lines, _world_to_voxel_float_array
from voxroom_online.isaac_runtime.mapping.voxel_occupancy_grid import VOXEL_FREE, VOXEL_OCCUPIED, VOXEL_UNKNOWN, VoxelIntegrationStats


_WARMED_THREAD_COUNTS: set[int] = set()


def numba_available() -> bool:
    try:
        import numba  # noqa: F401
    except Exception:
        return False
    return True


class VoxelCpuNumbaBackend:
    @staticmethod
    def warmup(grid) -> None:
        if not numba_available():
            return
        import numba

        thread_count = _set_numba_threads(grid, numba)
        if int(thread_count) in _WARMED_THREAD_COUNTS:
            return

        height, width = int(grid.state.shape[1]), int(grid.state.shape[2])
        z_bins = int(grid.state.shape[0])
        origin = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        endpoints = np.zeros((max(1, int(thread_count)), 3), dtype=np.float32)
        endpoints[:, 0] = min(float(z_bins - 1), 2.0)
        endpoints[:, 1] = min(float(height - 1), 2.0)
        endpoints[:, 2] = min(float(width - 1), 2.0)
        ray_count = int(endpoints.shape[0])
        endpoint_hit = np.ones(ray_count, dtype=np.uint8)
        free_counts = np.zeros(ray_count, dtype=np.int32)
        sensor_counts = np.zeros(ray_count, dtype=np.int32)
        occ_lin = np.full(ray_count, -1, dtype=np.int32)
        endpoint_vox = np.zeros((ray_count, 3), dtype=np.int32)
        ray_has_sample = np.zeros(ray_count, dtype=np.uint8)
        _count_ray_events_kernel(
            origin,
            endpoints,
            int(z_bins),
            int(height),
            int(width),
            1.0e-6,
            8,
            0,
            True,
            True,
            True,
            endpoint_hit,
            free_counts,
            sensor_counts,
            occ_lin,
            endpoint_vox,
            ray_has_sample,
        )
        free_offsets = _prefix_offsets(free_counts)
        sensor_offsets = _prefix_offsets(sensor_counts)
        free_events = np.empty(int(free_offsets[-1]), dtype=np.int32)
        sensor_events = np.empty(int(sensor_offsets[-1]), dtype=np.int32)
        _write_ray_events_kernel(
            origin,
            endpoints,
            int(z_bins),
            int(height),
            int(width),
            1.0e-6,
            8,
            0,
            True,
            True,
            endpoint_hit,
            free_counts,
            free_offsets,
            sensor_offsets,
            free_events,
            sensor_events,
        )
        log = np.zeros(int(grid.state.size), dtype=np.int16)
        state = np.zeros(int(grid.state.size), dtype=np.uint8)
        sensor = np.zeros(int(grid.state.size), dtype=np.uint8)
        block_size = 256
        free_grouped, free_block_offsets = _bucket_events_by_block(
            free_events,
            total_voxels=int(grid.state.size),
            block_size=block_size,
            thread_count=int(thread_count),
            chunk_multiplier=1,
        )
        occ_events = occ_lin[occ_lin >= 0].astype(np.int32, copy=False)
        occ_grouped, occ_block_offsets = _bucket_events_by_block(
            occ_events,
            total_voxels=int(grid.state.size),
            block_size=block_size,
            thread_count=int(thread_count),
            chunk_multiplier=1,
        )
        _apply_logodds_blocked_kernel(
            log,
            state,
            free_grouped,
            free_block_offsets,
            occ_grouped,
            occ_block_offsets,
            block_size,
            -1,
            3,
            -20,
            20,
            -1,
            1,
            -6,
            1,
            3,
            True,
            np.zeros(0, dtype=np.uint8),
            np.zeros(int(grid.state.shape[1] * grid.state.shape[2]), dtype=np.uint8),
            int(grid.state.shape[1] * grid.state.shape[2]),
            np.zeros(free_block_offsets.size - 1, dtype=np.int64),
            np.zeros(free_block_offsets.size - 1, dtype=np.int64),
            np.zeros(free_block_offsets.size - 1, dtype=np.int64),
            np.zeros(free_block_offsets.size - 1, dtype=np.int64),
            np.zeros(free_block_offsets.size - 1, dtype=np.int64),
        )
        sensor_grouped, sensor_block_offsets = _bucket_events_by_block(
            sensor_events,
            total_voxels=int(grid.state.size),
            block_size=block_size,
            thread_count=int(thread_count),
            chunk_multiplier=1,
        )
        _apply_sensor_blocked_kernel(
            sensor,
            sensor_grouped,
            sensor_block_offsets,
            block_size,
            1,
            255,
            np.zeros(int(grid.state.shape[1] * grid.state.shape[2]), dtype=np.uint8),
            int(grid.state.shape[1] * grid.state.shape[2]),
            np.zeros(sensor_block_offsets.size - 1, dtype=np.int64),
        )
        _WARMED_THREAD_COUNTS.add(int(thread_count))

    @staticmethod
    def integrate(
        grid,
        *,
        camera_origin_world,
        points_world: np.ndarray,
        floor_z: float,
        valid_mask: np.ndarray,
        endpoint_is_hit: np.ndarray | None = None,
    ) -> VoxelIntegrationStats:
        if not numba_available():
            if bool(getattr(grid.config, "cpu_numba_strict_required", True)):
                raise RuntimeError(
                    "integration_backend=cpu_numba requested, but numba is unavailable. "
                    "Do not silently fall back to cpu_vectorized in runtime."
                )
            from voxroom_online.isaac_runtime.mapping.voxel_cpu_fast_backend import VoxelCpuVectorizedBackend

            stats = VoxelCpuVectorizedBackend.integrate(
                grid,
                camera_origin_world=camera_origin_world,
                points_world=points_world,
                floor_z=float(floor_z),
                valid_mask=valid_mask,
                endpoint_is_hit=endpoint_is_hit,
                backend_name="cpu_vectorized",
            )
            stats.voxel_numba_requested_unavailable = True
            return stats
        return _integrate_numba(
            grid,
            camera_origin_world=camera_origin_world,
            points_world=points_world,
            floor_z=float(floor_z),
            valid_mask=valid_mask,
            endpoint_is_hit=endpoint_is_hit,
        )


def _set_numba_threads(grid, numba_module, requested: int | None = None) -> int:
    requested = max(1, int(getattr(grid.config, "cpu_numba_threads", 28) if requested is None else requested))
    try:
        numba_module.set_num_threads(requested)
        return int(numba_module.get_num_threads())
    except Exception:
        return 1


def _prefix_offsets(counts: np.ndarray) -> np.ndarray:
    offsets = np.empty(int(counts.size) + 1, dtype=np.int64)
    offsets[0] = 0
    if counts.size:
        np.cumsum(np.asarray(counts, dtype=np.int64), out=offsets[1:])
    return offsets


def _chunk_offsets(event_count: int, chunk_count: int) -> np.ndarray:
    chunks = max(1, min(int(chunk_count), max(1, int(event_count))))
    return np.linspace(0, int(event_count), chunks + 1, dtype=np.int64)


def _bucket_events_by_block(
    events: np.ndarray,
    *,
    total_voxels: int,
    block_size: int,
    thread_count: int,
    chunk_multiplier: int,
) -> tuple[np.ndarray, np.ndarray]:
    events_i32 = np.asarray(events, dtype=np.int32).reshape(-1)
    num_blocks = int((int(total_voxels) + int(block_size) - 1) // int(block_size))
    block_offsets = np.zeros(num_blocks + 1, dtype=np.int64)
    if events_i32.size == 0 or num_blocks <= 0:
        return np.zeros(0, dtype=np.int32), block_offsets

    chunk_count = max(1, int(thread_count) * max(1, int(chunk_multiplier)))
    offsets = _chunk_offsets(int(events_i32.size), chunk_count)
    local_counts = np.zeros((int(offsets.size) - 1, int(num_blocks)), dtype=np.int32)
    _count_event_blocks_kernel(events_i32, int(total_voxels), int(block_size), offsets, local_counts)

    block_counts = np.sum(local_counts.astype(np.int64), axis=0)
    np.cumsum(block_counts, out=block_offsets[1:])
    grouped = np.empty(int(block_offsets[-1]), dtype=np.int32)
    if grouped.size == 0:
        return grouped, block_offsets

    chunk_block_offsets = np.empty(local_counts.shape, dtype=np.int64)
    running = block_offsets[:-1].copy()
    for chunk_idx in range(local_counts.shape[0]):
        chunk_block_offsets[chunk_idx, :] = running
        running += local_counts[chunk_idx, :].astype(np.int64)
    cursor_offsets = chunk_block_offsets.copy()
    _scatter_events_to_blocks_kernel(
        events_i32,
        int(total_voxels),
        int(block_size),
        offsets,
        cursor_offsets,
        grouped,
    )
    return grouped, block_offsets


def _unique_endpoint_rc_flags(
    endpoint_vox: np.ndarray,
    endpoint_valid: np.ndarray,
    *,
    height: int,
    width: int,
    radius: int,
) -> np.ndarray:
    flags = np.zeros(int(height) * int(width), dtype=np.uint8)
    if endpoint_vox.size == 0 or not np.any(endpoint_valid):
        return flags
    _mark_endpoint_rc_flags_kernel(
        np.asarray(endpoint_vox, dtype=np.int32),
        np.asarray(endpoint_valid, dtype=np.uint8),
        int(height),
        int(width),
        max(0, int(radius)),
        flags,
    )
    return flags


def _integrate_numba(
    grid,
    *,
    camera_origin_world,
    points_world: np.ndarray,
    floor_z: float,
    valid_mask: np.ndarray,
    endpoint_is_hit: np.ndarray | None = None,
    override_thread_count: int | None = None,
    allow_autotune: bool = True,
) -> VoxelIntegrationStats:
    import numba

    started_at = time.perf_counter()
    stats = VoxelIntegrationStats(integration_backend="cpu_numba", python_debug_backend_used=False)

    point_filter_started = time.perf_counter()
    points = np.asarray(points_world, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if points.ndim != 2 or points.shape[1] < 3 or valid.size != points.shape[0]:
        stats.integrate_total_ms = float((time.perf_counter() - started_at) * 1000.0)
        return stats
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
    stats.voxel_integrate_point_filter_ms = float((time.perf_counter() - point_filter_started) * 1000.0)
    if points.size == 0:
        stats.skipped_empty_rays = int(np.count_nonzero(valid))
        stats.integrate_total_ms = float((time.perf_counter() - started_at) * 1000.0)
        return stats

    world_to_voxel_started = time.perf_counter()
    origin = _world_to_voxel_float_array(
        grid,
        np.asarray(camera_origin_world, dtype=np.float32).reshape(1, -1)[:, :3],
        float(floor_z),
    )[0].astype(np.float32)
    endpoints = _world_to_voxel_float_array(grid, points[:, :3], float(floor_z)).astype(np.float32, copy=False)
    finite_endpoint = np.all(np.isfinite(endpoints), axis=1) & np.all(np.isfinite(origin))
    hit = hit[finite_endpoint]
    endpoints = np.asarray(endpoints[finite_endpoint], dtype=np.float32)
    stats.voxel_integrate_world_to_voxel_ms = float((time.perf_counter() - world_to_voxel_started) * 1000.0)
    if endpoints.size == 0:
        stats.skipped_empty_rays = int(points.shape[0])
        stats.integrate_total_ms = float((time.perf_counter() - started_at) * 1000.0)
        return stats

    total_voxels = int(grid.state.size)
    if total_voxels >= np.iinfo(np.int32).max:
        raise RuntimeError("cpu_numba compact event backend requires total_voxels < int32 max")
    height, width = int(grid.state.shape[1]), int(grid.state.shape[2])
    z_bins = int(grid.state.shape[0])
    log_flat = grid.log_odds.reshape(-1)
    state_flat = grid.state.reshape(-1)
    sensor_flat = grid.sensor_range_count.reshape(-1)

    configured_chunk_rays = int(getattr(grid.config, "cpu_numba_chunk_rays", getattr(grid.config, "cuda_chunk_rays", 131072)))
    chunk_rays = int(endpoints.shape[0]) if configured_chunk_rays <= 0 else min(max(1, configured_chunk_rays), int(endpoints.shape[0]))
    ray_traversal_mode = str(getattr(grid.config, "ray_traversal_mode", "exact_dda") or "exact_dda").strip().lower()
    if ray_traversal_mode not in {"exact_dda", "dda", "thin_dda"}:
        raise RuntimeError("cpu_numba only supports exact thin DDA traversal, got %s" % ray_traversal_mode)
    tie_epsilon = max(1.0e-12, float(getattr(grid.config, "ray_traversal_tie_epsilon", 1.0e-6)))
    max_samples = max(1, int(getattr(grid.config, "cpu_numba_max_samples_per_ray", getattr(grid.config, "cuda_max_samples_per_ray", 320))))
    block_size = max(64, int(getattr(grid.config, "cpu_numba_event_block_size", 4096)))
    chunk_multiplier = max(1, int(getattr(grid.config, "cpu_numba_event_chunk_count_multiplier", 4)))
    stats.cuda_chunk_rays = int(chunk_rays)
    stats.cuda_max_samples_per_ray = int(max_samples)

    requested_threads = max(1, int(getattr(grid.config, "cpu_numba_threads", 28) if override_thread_count is None else override_thread_count))
    threads_mode = "manual" if override_thread_count is not None else str(getattr(grid.config, "cpu_numba_threads_mode", "manual") or "manual").strip().lower()
    selected_threads = int(requested_threads)
    stats.voxel_integrate_numba_requested_thread_count = int(getattr(grid.config, "cpu_numba_threads", requested_threads))
    stats.voxel_integrate_numba_threads_mode = str(threads_mode)
    if allow_autotune and threads_mode == "auto":
        autotune_started = time.perf_counter()
        selected_threads = _autotune_thread_count(
            grid,
            numba,
            camera_origin_world=camera_origin_world,
            points_world=points,
            floor_z=float(floor_z),
            endpoint_is_hit=hit,
        )
        stats.voxel_integrate_numba_autotune_ms = float((time.perf_counter() - autotune_started) * 1000.0)
    thread_count = _set_numba_threads(grid, numba, selected_threads)
    stats.voxel_integrate_backend_thread_count = int(thread_count)
    stats.voxel_integrate_backend_effective_thread_count = int(thread_count)
    try:
        stats.voxel_numba_threading_layer = str(numba.threading_layer())
    except Exception:
        stats.voxel_numba_threading_layer = "uninitialized"
    stats.voxel_integrate_numba_threading_layer = str(stats.voxel_numba_threading_layer)
    if int(thread_count) < int(getattr(grid.config, "cpu_numba_fail_if_thread_count_below", 2)):
        raise RuntimeError(
            "integration_backend=cpu_numba requested, but numba thread count is %d below required %d"
            % (int(thread_count), int(getattr(grid.config, "cpu_numba_fail_if_thread_count_below", 2)))
        )

    log_min = int(grid.config.logodds_min)
    log_max = int(grid.config.logodds_max)
    free_delta = int(grid.config.free_logodds_delta)
    occ_delta = int(grid.config.occupied_logodds_delta)
    free_threshold = int(grid.config.free_logodds_threshold)
    occupied_threshold = int(grid.config.occupied_logodds_threshold)
    occupied_to_free_threshold = min(
        int(free_threshold),
        int(getattr(grid.config, "occupied_to_free_logodds_threshold", free_threshold)),
    )
    free_vote_cap = max(0, int(getattr(grid.config, "free_vote_cap_per_voxel", 1)))
    occ_vote_cap = max(0, int(getattr(grid.config, "occupied_vote_cap_per_voxel", 3)))
    hit_wins = bool(getattr(grid.config, "occupied_wins_over_free_same_voxel", True))
    exclude_n = max(0, int(grid.config.free_excludes_last_n_voxels_before_endpoint))
    sensor_enabled = bool(getattr(grid.config, "sensor_range_tracking_enabled", True))
    sensor_ray_enabled = sensor_enabled and bool(getattr(grid.config, "sensor_range_mark_ray_samples_enabled", True))
    sensor_endpoint_enabled = sensor_enabled and bool(getattr(grid.config, "sensor_range_mark_endpoint_column_enabled", True))
    sensor_delta = max(0, int(getattr(grid.config, "sensor_range_count_delta", 1)))
    sensor_max = int(np.clip(int(getattr(grid.config, "sensor_range_count_max", 255)), 0, 255))
    endpoint_splat_enabled = bool(int(grid.config.endpoint_splat_z_radius_cells) != 0 or int(grid.config.endpoint_splat_xy_radius_cells) != 0)

    changed_flags = (
        np.zeros(0, dtype=np.uint8)
        if bool(getattr(grid.config, "cpu_numba_disable_changed_flatnonzero", True))
        else np.zeros(total_voxels, dtype=np.uint8)
    )
    dirty_rc_flags = np.zeros(int(height * width), dtype=np.uint8)
    integrated = 0
    hit_integrated = 0
    free_only_integrated = 0

    for start in range(0, int(endpoints.shape[0]), int(chunk_rays)):
        end = min(int(endpoints.shape[0]), start + int(chunk_rays))
        p1 = np.asarray(endpoints[start:end], dtype=np.float32)
        p1_hit = np.asarray(hit[start:end], dtype=np.uint8)
        ray_count = int(p1.shape[0])

        pass1_started = time.perf_counter()
        free_counts = np.zeros(ray_count, dtype=np.int32)
        sensor_counts = np.zeros(ray_count, dtype=np.int32)
        occ_lin = np.full(ray_count, -1, dtype=np.int32)
        endpoint_vox = np.empty((ray_count, 3), dtype=np.int32)
        ray_has_sample = np.zeros(ray_count, dtype=np.uint8)
        _count_ray_events_kernel(
            origin,
            p1,
            int(z_bins),
            int(height),
            int(width),
            float(tie_epsilon),
            int(max_samples),
            int(exclude_n),
            bool(grid.config.free_excludes_endpoint),
            bool(grid.config.mark_endpoint_occupied),
            bool(sensor_ray_enabled),
            p1_hit,
            free_counts,
            sensor_counts,
            occ_lin,
            endpoint_vox,
            ray_has_sample,
        )
        pass1_ms = float((time.perf_counter() - pass1_started) * 1000.0)
        stats.voxel_integrate_pass1_ms += pass1_ms
        stats.voxel_integrate_count_kernel_ms += pass1_ms
        ray_has_sample_bool = ray_has_sample.astype(bool)
        integrated += int(np.count_nonzero(ray_has_sample_bool))
        hit_integrated += int(np.count_nonzero(ray_has_sample_bool & p1_hit.astype(bool)))
        free_only_integrated += int(np.count_nonzero(ray_has_sample_bool & ~p1_hit.astype(bool)))

        prefix_started = time.perf_counter()
        free_offsets = _prefix_offsets(free_counts)
        sensor_offsets = _prefix_offsets(sensor_counts) if sensor_ray_enabled else np.zeros(ray_count + 1, dtype=np.int64)
        free_events = np.empty(int(free_offsets[-1]), dtype=np.int32)
        sensor_events = np.empty(int(sensor_offsets[-1]), dtype=np.int32) if sensor_ray_enabled else np.zeros(0, dtype=np.int32)
        prefix_ms = float((time.perf_counter() - prefix_started) * 1000.0)
        stats.voxel_integrate_event_prefix_ms += prefix_ms
        stats.voxel_integrate_prefix_ms += prefix_ms

        pass2_started = time.perf_counter()
        _write_ray_events_kernel(
            origin,
            p1,
            int(z_bins),
            int(height),
            int(width),
            float(tie_epsilon),
            int(max_samples),
            int(exclude_n),
            bool(grid.config.free_excludes_endpoint),
            bool(sensor_ray_enabled),
            p1_hit,
            free_counts,
            free_offsets,
            sensor_offsets,
            free_events,
            sensor_events,
        )
        pass2_ms = float((time.perf_counter() - pass2_started) * 1000.0)
        stats.voxel_integrate_pass2_ms += pass2_ms
        stats.voxel_integrate_write_events_ms += pass2_ms
        stats.voxel_integrate_total_samples += int(free_events.size)
        stats.voxel_integrate_total_sensor_events += int(sensor_events.size)
        stats.voxel_integrate_free_event_count += int(free_events.size)
        stats.voxel_integrate_sensor_event_count += int(sensor_events.size)

        endpoint_valid = (
            (endpoint_vox[:, 0] >= 0)
            & (endpoint_vox[:, 0] < z_bins)
            & (endpoint_vox[:, 1] >= 0)
            & (endpoint_vox[:, 1] < height)
            & (endpoint_vox[:, 2] >= 0)
            & (endpoint_vox[:, 2] < width)
        )
        endpoint_valid_hit = endpoint_valid & p1_hit.astype(bool)
        if bool(grid.config.mark_endpoint_occupied):
            if endpoint_splat_enabled:
                occ_events = _endpoint_splat_lines(grid, endpoint_vox[endpoint_valid_hit], height=height, width=width, z_bins=z_bins).astype(np.int32, copy=False)
            else:
                occ_events = occ_lin[occ_lin >= 0].astype(np.int32, copy=False)
        else:
            occ_events = np.zeros(0, dtype=np.int32)
        stats.voxel_integrate_total_occ_events += int(occ_events.size)
        stats.voxel_integrate_occ_event_count += int(occ_events.size)

        bucket_free_started = time.perf_counter()
        free_grouped, free_block_offsets = _bucket_events_by_block(
            free_events,
            total_voxels=total_voxels,
            block_size=block_size,
            thread_count=int(thread_count),
            chunk_multiplier=chunk_multiplier,
        )
        bucket_free_ms = float((time.perf_counter() - bucket_free_started) * 1000.0)
        stats.voxel_integrate_bucket_free_ms += bucket_free_ms
        bucket_occ_started = time.perf_counter()
        occ_grouped, occ_block_offsets = _bucket_events_by_block(
            occ_events,
            total_voxels=total_voxels,
            block_size=block_size,
            thread_count=int(thread_count),
            chunk_multiplier=chunk_multiplier,
        )
        bucket_occ_ms = float((time.perf_counter() - bucket_occ_started) * 1000.0)
        stats.voxel_integrate_bucket_occ_ms += bucket_occ_ms
        stats.voxel_integrate_event_bucket_ms += bucket_free_ms + bucket_occ_ms
        stats.voxel_integrate_num_blocks = max(stats.voxel_integrate_num_blocks, int(free_block_offsets.size) - 1)
        if free_block_offsets.size > 1:
            touched = (free_block_offsets[1:] > free_block_offsets[:-1]) | (occ_block_offsets[1:] > occ_block_offsets[:-1])
            stats.voxel_integrate_touched_block_count += int(np.count_nonzero(touched))

        apply_started = time.perf_counter()
        num_blocks = int(free_block_offsets.size) - 1
        free_update_counts = np.zeros(num_blocks, dtype=np.int64)
        occ_update_counts = np.zeros(num_blocks, dtype=np.int64)
        changed_counts = np.zeros(num_blocks, dtype=np.int64)
        unique_free_counts = np.zeros(num_blocks, dtype=np.int64)
        unique_occ_counts = np.zeros(num_blocks, dtype=np.int64)
        _apply_logodds_blocked_kernel(
            log_flat,
            state_flat,
            free_grouped,
            free_block_offsets,
            occ_grouped,
            occ_block_offsets,
            int(block_size),
            int(free_delta),
            int(occ_delta),
            int(log_min),
            int(log_max),
            int(free_threshold),
            int(occupied_threshold),
            int(occupied_to_free_threshold),
            int(free_vote_cap),
            int(occ_vote_cap),
            bool(hit_wins),
            changed_flags,
            dirty_rc_flags,
            int(height * width),
            free_update_counts,
            occ_update_counts,
            changed_counts,
            unique_free_counts,
            unique_occ_counts,
        )
        stats.free_update_count += int(np.sum(free_update_counts))
        stats.occupied_update_count += int(np.sum(occ_update_counts))
        stats.voxel_integrate_total_unique_free_voxels += int(np.sum(unique_free_counts))
        stats.voxel_integrate_total_unique_occ_voxels += int(np.sum(unique_occ_counts))
        stats.voxel_integrate_changed_flag_count += int(np.sum(changed_counts))
        stats.voxel_integrate_apply_logodds_ms += float((time.perf_counter() - apply_started) * 1000.0)

        if sensor_ray_enabled and sensor_events.size:
            sensor_bucket_started = time.perf_counter()
            sensor_grouped, sensor_block_offsets = _bucket_events_by_block(
                sensor_events,
                total_voxels=total_voxels,
                block_size=block_size,
                thread_count=int(thread_count),
                chunk_multiplier=chunk_multiplier,
            )
            sensor_bucket_ms = float((time.perf_counter() - sensor_bucket_started) * 1000.0)
            stats.voxel_integrate_bucket_sensor_ms += sensor_bucket_ms
            stats.voxel_integrate_event_bucket_ms += sensor_bucket_ms
            sensor_apply_started = time.perf_counter()
            sensor_update_counts = np.zeros(int(sensor_block_offsets.size) - 1, dtype=np.int64)
            _apply_sensor_blocked_kernel(
                sensor_flat,
                sensor_grouped,
                sensor_block_offsets,
                int(block_size),
                int(sensor_delta),
                int(sensor_max),
                dirty_rc_flags,
                int(height * width),
                sensor_update_counts,
            )
            sensor_updates = int(np.sum(sensor_update_counts))
            stats.sensor_range_update_count += sensor_updates
            stats.voxel_integrate_total_unique_sensor_voxels += sensor_updates
            stats.voxel_integrate_apply_sensor_ms += float((time.perf_counter() - sensor_apply_started) * 1000.0)

        endpoint_started = time.perf_counter()
        if sensor_endpoint_enabled and np.any(endpoint_valid_hit):
            rc_flags = _unique_endpoint_rc_flags(
                endpoint_vox,
                endpoint_valid_hit,
                height=height,
                width=width,
                radius=int(getattr(grid.config, "sensor_range_endpoint_column_xy_radius_cells", 0)),
            )
            active_z = (
                np.asarray(grid.active_z_indices(), dtype=np.int32)
                if bool(getattr(grid.config, "sensor_range_mark_active_z_only", True))
                else np.arange(int(z_bins), dtype=np.int32)
            )
            endpoint_updates = np.zeros(1, dtype=np.int64)
            _mark_endpoint_columns_sensor_kernel(
                sensor_flat,
                rc_flags,
                active_z,
                int(height),
                int(width),
                int(sensor_delta),
                int(sensor_max),
                endpoint_updates,
            )
            dirty_rc_flags[:] = np.maximum(dirty_rc_flags, rc_flags)
            stats.sensor_range_update_count += int(endpoint_updates[0])
            stats.voxel_integrate_total_unique_sensor_voxels += int(endpoint_updates[0])
        stats.voxel_integrate_endpoint_column_ms += float((time.perf_counter() - endpoint_started) * 1000.0)

    stats.depth_rays_integrated = int(integrated)
    stats.depth_hit_rays_integrated = int(hit_integrated)
    stats.depth_free_only_rays_integrated = int(free_only_integrated)
    stats.skipped_empty_rays = max(0, int(endpoints.shape[0]) - int(integrated))
    stats.ray_sample_ms = float(stats.voxel_integrate_pass1_ms + stats.voxel_integrate_pass2_ms)
    stats.voxel_integrate_sample_kernel_ms = float(stats.ray_sample_ms)
    stats.unique_ms = float(stats.voxel_integrate_event_bucket_ms)
    stats.scatter_ms = float(stats.voxel_integrate_apply_logodds_ms + stats.voxel_integrate_apply_sensor_ms)

    if changed_flags.size and not bool(getattr(grid.config, "cpu_numba_disable_changed_flatnonzero", True)):
        refresh_started = time.perf_counter()
        changed = np.flatnonzero(changed_flags).astype(np.int64)
        scan_ms = float((time.perf_counter() - refresh_started) * 1000.0)
        stats.voxel_integrate_changed_extract_ms = scan_ms
        stats.voxel_integrate_changed_scan_ms = scan_ms
        stats.refresh_changed_voxels = int(changed.size)
        stats.refresh_mode = "inline_apply_with_changed_scan"
    else:
        stats.voxel_integrate_changed_extract_ms = 0.0
        stats.voxel_integrate_changed_scan_ms = 0.0
        stats.refresh_changed_voxels = int(stats.voxel_integrate_changed_flag_count)
        stats.refresh_mode = "inline_apply"
    stats.refresh_state_ms = 0.0
    stats.voxel_integrate_refresh_state_ms = 0.0
    grid.last_dirty_rc_flags = dirty_rc_flags
    stats.voxel_integrate_dirty_rc_count = int(np.count_nonzero(dirty_rc_flags))
    stats.integrate_total_ms = float((time.perf_counter() - started_at) * 1000.0)
    try:
        stats.voxel_numba_threading_layer = str(numba.threading_layer())
        stats.voxel_integrate_numba_threading_layer = str(stats.voxel_numba_threading_layer)
    except Exception:
        pass
    return stats


def _autotune_thread_count(
    grid,
    numba_module,
    *,
    camera_origin_world,
    points_world: np.ndarray,
    floor_z: float,
    endpoint_is_hit: np.ndarray | None = None,
) -> int:
    requested = max(1, int(getattr(grid.config, "cpu_numba_threads", 28)))
    candidates = parse_thread_candidates(getattr(grid.config, "cpu_numba_autotune_candidates", (2, 4, 8, 14, 28)), requested_max=requested)
    if len(candidates) == 1:
        return int(candidates[0])
    try:
        threading_layer = str(numba_module.threading_layer())
    except Exception:
        threading_layer = "uninitialized"
    key = autotune_cache_key(grid, candidates=candidates, threading_layer=threading_layer)
    cache_path = str(getattr(grid.config, "cpu_numba_autotune_cache_path", "debug/voxel_cpu_numba_autotune.json"))
    cached = load_cached_thread_count(cache_path, key)
    if cached is not None and int(cached) in set(int(v) for v in candidates):
        return int(cached)

    points = np.asarray(points_world, dtype=np.float32)
    hit = np.ones(points.shape[0], dtype=bool) if endpoint_is_hit is None else np.asarray(endpoint_is_hit, dtype=bool).reshape(-1)
    if hit.size != points.shape[0]:
        hit = np.ones(points.shape[0], dtype=bool)
    if points.ndim != 2 or points.shape[0] == 0:
        return min(candidates)
    max_rays = max(1, int(getattr(grid.config, "cpu_numba_autotune_rays", 30000)))
    if points.shape[0] > max_rays:
        sample_idx = np.linspace(0, int(points.shape[0]) - 1, int(max_rays), dtype=np.int64)
        points = np.asarray(points[sample_idx], dtype=np.float32)
        hit = np.asarray(hit[sample_idx], dtype=bool)
    valid = np.ones(int(points.shape[0]), dtype=bool)
    repeat = max(1, int(getattr(grid.config, "cpu_numba_autotune_repeat", 2)))
    results: list[dict[str, object]] = []

    from voxroom_online.isaac_runtime.mapping.voxel_occupancy_grid import VoxelOccupancyGrid3D, VoxelOccupancyGridConfig

    for candidate in candidates:
        times: list[float] = []
        for run_idx in range(repeat + 1):
            cfg = VoxelOccupancyGridConfig.from_mapping(grid.config)
            cfg.cpu_numba_threads_mode = "manual"
            cfg.cpu_numba_threads = int(candidate)
            cfg.cpu_numba_autotune_candidates = (int(candidate),)
            temp = VoxelOccupancyGrid3D.zeros(grid.shape, grid.map_info, cfg)
            temp.active_z_min_m = float(grid.active_z_min_m)
            temp.active_z_max_m = None if grid.active_z_max_m is None else float(grid.active_z_max_m)
            temp.ceiling_height_m = None if grid.ceiling_height_m is None else float(grid.ceiling_height_m)
            stats = _integrate_numba(
                temp,
                camera_origin_world=camera_origin_world,
                points_world=points,
                floor_z=float(floor_z),
                valid_mask=valid,
                endpoint_is_hit=hit,
                override_thread_count=int(candidate),
                allow_autotune=False,
            )
            if run_idx > 0:
                times.append(float(stats.integrate_total_ms))
        median_ms = float(np.median(np.asarray(times, dtype=np.float64))) if times else float("inf")
        results.append({"threads": int(candidate), "median_integrate_ms": median_ms})
    best = min(results, key=lambda item: float(item.get("median_integrate_ms", float("inf"))))
    best_threads = int(best["threads"])
    store_cached_thread_count(cache_path, key, best_threads=best_threads, results=results)
    return int(best_threads)


try:
    import numba as _numba

    @_numba.njit(inline="always", cache=True)
    def _inside_voxel_numba(z: int, r: int, c: int, z_bins: int, height: int, width: int) -> bool:
        return z >= 0 and z < z_bins and r >= 0 and r < height and c >= 0 and c < width

    @_numba.njit(inline="always", cache=True)
    def _linear_index_numba(z: int, r: int, c: int, height: int, width: int) -> int:
        return int(z) * int(height * width) + int(r) * int(width) + int(c)

    @_numba.njit(inline="always", cache=True)
    def _clip_one_axis_numba(p: float, d: float, upper: float, t0: float, t1: float, eps: float):
        if abs(d) < eps:
            if p < 0.0 or p > upper:
                return False, t0, t1
            return True, t0, t1
        ta = (0.0 - p) / d
        tb = (upper - p) / d
        if ta > tb:
            temp = ta
            ta = tb
            tb = temp
        if ta > t0:
            t0 = ta
        if tb < t1:
            t1 = tb
        if t0 > t1:
            return False, t0, t1
        return True, t0, t1

    @_numba.njit(inline="always", cache=True)
    def _clip_segment_to_volume_numba(
        oz: float,
        orow: float,
        oc: float,
        ez: float,
        er: float,
        ec: float,
        z_bins: int,
        height: int,
        width: int,
        eps: float,
    ):
        dz = ez - oz
        dr = er - orow
        dc = ec - oc
        t0 = 0.0
        t1 = 1.0
        ok, t0, t1 = _clip_one_axis_numba(oz, dz, max(0.0, float(z_bins) - eps), t0, t1, eps)
        if not ok:
            return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ok, t0, t1 = _clip_one_axis_numba(orow, dr, max(0.0, float(height) - eps), t0, t1, eps)
        if not ok:
            return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ok, t0, t1 = _clip_one_axis_numba(oc, dc, max(0.0, float(width) - eps), t0, t1, eps)
        if not ok:
            return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        q0z = oz + dz * t0
        q0r = orow + dr * t0
        q0c = oc + dc * t0
        q1z = oz + dz * t1
        q1r = orow + dr * t1
        q1c = oc + dc * t1
        return True, q0z, q0r, q0c, q1z, q1r, q1c

    @_numba.njit(inline="always", cache=True)
    def _init_axis_dda_numba(q0: float, q1: float, eps: float):
        d = q1 - q0
        if abs(d) < eps:
            return 0, np.inf, np.inf
        if d > 0.0:
            boundary = float(np.floor(q0)) + 1.0
            return 1, (boundary - q0) / d, 1.0 / abs(d)
        boundary = float(np.floor(q0))
        return -1, (q0 - boundary) / abs(d), 1.0 / abs(d)

    @_numba.njit(inline="always", cache=True)
    def _clamp_cell_numba(value: int, size: int) -> int:
        if value < 0:
            return 0
        if value >= size:
            return size - 1
        return value

    @_numba.njit(inline="always", cache=True)
    def _dda_count_cells_numba(
        q0z: float,
        q0r: float,
        q0c: float,
        q1z: float,
        q1r: float,
        q1c: float,
        z_bins: int,
        height: int,
        width: int,
        max_samples: int,
        tie_eps: float,
    ) -> int:
        z = _clamp_cell_numba(int(np.floor(q0z)), z_bins)
        r = _clamp_cell_numba(int(np.floor(q0r)), height)
        c = _clamp_cell_numba(int(np.floor(q0c)), width)
        tz = _clamp_cell_numba(int(np.floor(q1z)), z_bins)
        tr = _clamp_cell_numba(int(np.floor(q1r)), height)
        tc = _clamp_cell_numba(int(np.floor(q1c)), width)
        step_z, tmax_z, tdelta_z = _init_axis_dda_numba(q0z, q1z, tie_eps)
        step_r, tmax_r, tdelta_r = _init_axis_dda_numba(q0r, q1r, tie_eps)
        step_c, tmax_c, tdelta_c = _init_axis_dda_numba(q0c, q1c, tie_eps)
        count = 0
        for _step_i in range(max_samples):
            if not _inside_voxel_numba(z, r, c, z_bins, height, width):
                break
            count += 1
            if z == tz and r == tr and c == tc:
                break
            t_next = tmax_z
            if tmax_r < t_next:
                t_next = tmax_r
            if tmax_c < t_next:
                t_next = tmax_c
            if not np.isfinite(t_next):
                break
            if tmax_z <= t_next + tie_eps:
                z += step_z
                tmax_z += tdelta_z
            if tmax_r <= t_next + tie_eps:
                r += step_r
                tmax_r += tdelta_r
            if tmax_c <= t_next + tie_eps:
                c += step_c
                tmax_c += tdelta_c
        return count

    @_numba.njit(parallel=True, nogil=True, cache=True)
    def _count_ray_events_kernel(
        origin: np.ndarray,
        endpoints: np.ndarray,
        z_bins: int,
        height: int,
        width: int,
        tie_epsilon: float,
        max_samples: int,
        exclude_n: int,
        free_excludes_endpoint: bool,
        mark_endpoint_occupied: bool,
        sensor_ray_enabled: bool,
        endpoint_hit: np.ndarray,
        free_counts: np.ndarray,
        sensor_counts: np.ndarray,
        occ_lin: np.ndarray,
        endpoint_vox: np.ndarray,
        ray_has_sample: np.ndarray,
    ) -> None:
        hw = int(height * width)
        for ray_idx in _numba.prange(endpoints.shape[0]):
            ez = int(np.floor(endpoints[ray_idx, 0]))
            er = int(np.floor(endpoints[ray_idx, 1]))
            ec = int(np.floor(endpoints[ray_idx, 2]))
            endpoint_vox[ray_idx, 0] = ez
            endpoint_vox[ray_idx, 1] = er
            endpoint_vox[ray_idx, 2] = ec
            endpoint_valid = _inside_voxel_numba(ez, er, ec, z_bins, height, width)
            endpoint_valid_hit = endpoint_valid and int(endpoint_hit[ray_idx]) != 0
            endpoint_lin = -1
            if endpoint_valid:
                endpoint_lin = _linear_index_numba(ez, er, ec, height, width)
                if endpoint_valid_hit and mark_endpoint_occupied:
                    occ_lin[ray_idx] = endpoint_lin
            ok, q0z, q0r, q0c, q1z, q1r, q1c = _clip_segment_to_volume_numba(
                float(origin[0]),
                float(origin[1]),
                float(origin[2]),
                float(endpoints[ray_idx, 0]),
                float(endpoints[ray_idx, 1]),
                float(endpoints[ray_idx, 2]),
                int(z_bins),
                int(height),
                int(width),
                float(tie_epsilon),
            )
            if not ok:
                free_counts[ray_idx] = 0
                sensor_counts[ray_idx] = 0
                continue
            total_cells = _dda_count_cells_numba(q0z, q0r, q0c, q1z, q1r, q1c, z_bins, height, width, max_samples, tie_epsilon)
            free_count = int(total_cells)
            if free_excludes_endpoint and endpoint_valid_hit:
                free_count = int(total_cells) - 1 - int(exclude_n)
                if free_count < 0:
                    free_count = 0
            free_counts[ray_idx] = free_count
            sensor_counts[ray_idx] = int(total_cells) if sensor_ray_enabled else 0
            if total_cells > 0:
                ray_has_sample[ray_idx] = 1

    @_numba.njit(parallel=True, nogil=True, cache=True)
    def _write_ray_events_kernel(
        origin: np.ndarray,
        endpoints: np.ndarray,
        z_bins: int,
        height: int,
        width: int,
        tie_epsilon: float,
        max_samples: int,
        exclude_n: int,
        free_excludes_endpoint: bool,
        sensor_ray_enabled: bool,
        endpoint_hit: np.ndarray,
        free_counts: np.ndarray,
        free_offsets: np.ndarray,
        sensor_offsets: np.ndarray,
        free_events: np.ndarray,
        sensor_events: np.ndarray,
    ) -> None:
        hw = int(height * width)
        for ray_idx in _numba.prange(endpoints.shape[0]):
            ez = int(np.floor(endpoints[ray_idx, 0]))
            er = int(np.floor(endpoints[ray_idx, 1]))
            ec = int(np.floor(endpoints[ray_idx, 2]))
            endpoint_valid = _inside_voxel_numba(ez, er, ec, z_bins, height, width)
            endpoint_valid_hit = endpoint_valid and int(endpoint_hit[ray_idx]) != 0
            endpoint_lin = -1
            if endpoint_valid:
                endpoint_lin = _linear_index_numba(ez, er, ec, height, width)
            ok, q0z, q0r, q0c, q1z, q1r, q1c = _clip_segment_to_volume_numba(
                float(origin[0]),
                float(origin[1]),
                float(origin[2]),
                float(endpoints[ray_idx, 0]),
                float(endpoints[ray_idx, 1]),
                float(endpoints[ray_idx, 2]),
                int(z_bins),
                int(height),
                int(width),
                float(tie_epsilon),
            )
            if not ok:
                continue
            z = _clamp_cell_numba(int(np.floor(q0z)), z_bins)
            r = _clamp_cell_numba(int(np.floor(q0r)), height)
            c = _clamp_cell_numba(int(np.floor(q0c)), width)
            tz = _clamp_cell_numba(int(np.floor(q1z)), z_bins)
            tr = _clamp_cell_numba(int(np.floor(q1r)), height)
            tc = _clamp_cell_numba(int(np.floor(q1c)), width)
            step_z, tmax_z, tdelta_z = _init_axis_dda_numba(q0z, q1z, tie_epsilon)
            step_r, tmax_r, tdelta_r = _init_axis_dda_numba(q0r, q1r, tie_epsilon)
            step_c, tmax_c, tdelta_c = _init_axis_dda_numba(q0c, q1c, tie_epsilon)
            free_limit = int(free_counts[ray_idx])
            free_pos = int(free_offsets[ray_idx])
            free_end = int(free_offsets[ray_idx + 1])
            sensor_pos = int(sensor_offsets[ray_idx])
            seq_idx = 0
            for _step_i in range(max_samples):
                if not _inside_voxel_numba(z, r, c, z_bins, height, width):
                    break
                lin = int(z) * hw + int(r) * int(width) + int(c)
                if sensor_ray_enabled:
                    sensor_events[sensor_pos] = lin
                    sensor_pos += 1
                if seq_idx < free_limit and free_pos < free_end:
                    if not (free_excludes_endpoint and endpoint_valid_hit and lin == endpoint_lin):
                        free_events[free_pos] = lin
                        free_pos += 1
                seq_idx += 1
                if z == tz and r == tr and c == tc:
                    break
                t_next = tmax_z
                if tmax_r < t_next:
                    t_next = tmax_r
                if tmax_c < t_next:
                    t_next = tmax_c
                if not np.isfinite(t_next):
                    break
                if tmax_z <= t_next + tie_epsilon:
                    z += step_z
                    tmax_z += tdelta_z
                if tmax_r <= t_next + tie_epsilon:
                    r += step_r
                    tmax_r += tdelta_r
                if tmax_c <= t_next + tie_epsilon:
                    c += step_c
                    tmax_c += tdelta_c

    @_numba.njit(parallel=True, nogil=True, cache=True)
    def _count_event_blocks_kernel(
        events: np.ndarray,
        total_voxels: int,
        block_size: int,
        chunk_offsets: np.ndarray,
        local_block_counts: np.ndarray,
    ) -> None:
        for chunk_idx in _numba.prange(chunk_offsets.shape[0] - 1):
            start = int(chunk_offsets[chunk_idx])
            end = int(chunk_offsets[chunk_idx + 1])
            for i in range(start, end):
                value = int(events[i])
                if value >= 0 and value < total_voxels:
                    block = value // block_size
                    local_block_counts[chunk_idx, block] += 1

    @_numba.njit(parallel=True, nogil=True, cache=True)
    def _scatter_events_to_blocks_kernel(
        events: np.ndarray,
        total_voxels: int,
        block_size: int,
        chunk_offsets: np.ndarray,
        cursor_offsets: np.ndarray,
        grouped_events: np.ndarray,
    ) -> None:
        for chunk_idx in _numba.prange(chunk_offsets.shape[0] - 1):
            start = int(chunk_offsets[chunk_idx])
            end = int(chunk_offsets[chunk_idx + 1])
            for i in range(start, end):
                value = int(events[i])
                if value >= 0 and value < total_voxels:
                    block = value // block_size
                    pos = int(cursor_offsets[chunk_idx, block])
                    grouped_events[pos] = value
                    cursor_offsets[chunk_idx, block] = pos + 1

    @_numba.njit(parallel=True, nogil=True, cache=True)
    def _apply_logodds_blocked_kernel(
        log_flat: np.ndarray,
        state_flat: np.ndarray,
        free_grouped: np.ndarray,
        free_block_offsets: np.ndarray,
        occ_grouped: np.ndarray,
        occ_block_offsets: np.ndarray,
        block_size: int,
        free_delta: int,
        occ_delta: int,
        log_min: int,
        log_max: int,
        free_threshold: int,
        occupied_threshold: int,
        occupied_to_free_threshold: int,
        free_vote_cap: int,
        occ_vote_cap: int,
        occupied_wins_over_free_same_voxel: bool,
        changed_flags: np.ndarray,
        dirty_rc_flags: np.ndarray,
        hw: int,
        free_update_counts: np.ndarray,
        occ_update_counts: np.ndarray,
        changed_counts: np.ndarray,
        unique_free_counts: np.ndarray,
        unique_occ_counts: np.ndarray,
    ) -> None:
        num_blocks = free_block_offsets.shape[0] - 1
        total_voxels = log_flat.shape[0]
        for block in _numba.prange(num_blocks):
            block_start = block * block_size
            block_end = block_start + block_size
            if block_end > total_voxels:
                block_end = total_voxels
            block_len = block_end - block_start
            local_free = np.zeros(block_size, dtype=np.int32)
            local_occ = np.zeros(block_size, dtype=np.int32)
            for i in range(int(free_block_offsets[block]), int(free_block_offsets[block + 1])):
                local = int(free_grouped[i]) - block_start
                if local >= 0 and local < block_len:
                    local_free[local] += 1
            for i in range(int(occ_block_offsets[block]), int(occ_block_offsets[block + 1])):
                local = int(occ_grouped[i]) - block_start
                if local >= 0 and local < block_len:
                    local_occ[local] += 1
            free_total = 0
            occ_total = 0
            changed_total = 0
            unique_free_total = 0
            unique_occ_total = 0
            for local in range(block_len):
                raw_f = int(local_free[local])
                raw_o = int(local_occ[local])
                if raw_f == 0 and raw_o == 0:
                    continue
                idx = block_start + local
                value = int(log_flat[idx])
                applied_f = raw_f
                if raw_o > 0 and occupied_wins_over_free_same_voxel:
                    applied_f = 0
                elif free_vote_cap > 0 and applied_f > free_vote_cap:
                    applied_f = free_vote_cap
                applied_o = raw_o
                if occ_vote_cap > 0 and applied_o > occ_vote_cap:
                    applied_o = occ_vote_cap
                if applied_f != 0:
                    value += applied_f * free_delta
                    if value < log_min:
                        value = log_min
                    elif value > log_max:
                        value = log_max
                    free_total += applied_f
                    unique_free_total += 1
                if applied_o != 0:
                    value += applied_o * occ_delta
                    if value < log_min:
                        value = log_min
                    elif value > log_max:
                        value = log_max
                    occ_total += applied_o
                    unique_occ_total += 1
                log_flat[idx] = value
                previous = int(state_flat[idx])
                if value >= occupied_threshold:
                    state_flat[idx] = int(VOXEL_OCCUPIED)
                elif value <= free_threshold:
                    if previous != int(VOXEL_OCCUPIED) or value <= occupied_to_free_threshold:
                        state_flat[idx] = int(VOXEL_FREE)
                    else:
                        state_flat[idx] = previous
                else:
                    state_flat[idx] = previous
                if changed_flags.shape[0] > 0:
                    changed_flags[idx] = 1
                if hw > 0 and dirty_rc_flags.shape[0] == hw:
                    dirty_rc_flags[idx % hw] = 1
                changed_total += 1
            free_update_counts[block] = free_total
            occ_update_counts[block] = occ_total
            changed_counts[block] = changed_total
            unique_free_counts[block] = unique_free_total
            unique_occ_counts[block] = unique_occ_total

    @_numba.njit(parallel=True, nogil=True, cache=True)
    def _apply_sensor_blocked_kernel(
        sensor_flat: np.ndarray,
        sensor_grouped: np.ndarray,
        sensor_block_offsets: np.ndarray,
        block_size: int,
        delta: int,
        max_value: int,
        dirty_rc_flags: np.ndarray,
        hw: int,
        sensor_update_counts: np.ndarray,
    ) -> None:
        num_blocks = sensor_block_offsets.shape[0] - 1
        total_voxels = sensor_flat.shape[0]
        for block in _numba.prange(num_blocks):
            block_start = block * block_size
            block_end = block_start + block_size
            if block_end > total_voxels:
                block_end = total_voxels
            block_len = block_end - block_start
            local_seen = np.zeros(block_size, dtype=np.uint8)
            for i in range(int(sensor_block_offsets[block]), int(sensor_block_offsets[block + 1])):
                local = int(sensor_grouped[i]) - block_start
                if local >= 0 and local < block_len:
                    local_seen[local] = 1
            update_count = 0
            for local in range(block_len):
                if local_seen[local] == 0:
                    continue
                idx = block_start + local
                value = int(sensor_flat[idx]) + delta
                if value > max_value:
                    value = max_value
                sensor_flat[idx] = value
                if hw > 0 and dirty_rc_flags.shape[0] == hw:
                    dirty_rc_flags[idx % hw] = 1
                update_count += 1
            sensor_update_counts[block] = update_count

    @_numba.njit(parallel=True, nogil=True, cache=True)
    def _mark_endpoint_rc_flags_kernel(
        endpoint_vox: np.ndarray,
        endpoint_valid: np.ndarray,
        height: int,
        width: int,
        radius: int,
        rc_flags: np.ndarray,
    ) -> None:
        for i in _numba.prange(endpoint_vox.shape[0]):
            if endpoint_valid[i] == 0:
                continue
            r0 = int(endpoint_vox[i, 1])
            c0 = int(endpoint_vox[i, 2])
            for dr in range(-radius, radius + 1):
                r = r0 + dr
                if r < 0 or r >= height:
                    continue
                for dc in range(-radius, radius + 1):
                    c = c0 + dc
                    if c < 0 or c >= width:
                        continue
                    rc_flags[r * width + c] = 1

    @_numba.njit(parallel=True, nogil=True, cache=True)
    def _mark_endpoint_columns_sensor_kernel(
        sensor_flat: np.ndarray,
        rc_flags: np.ndarray,
        active_z_indices: np.ndarray,
        height: int,
        width: int,
        delta: int,
        max_value: int,
        update_count_out: np.ndarray,
    ) -> None:
        counts = np.zeros(rc_flags.shape[0], dtype=np.int64)
        hw = int(height * width)
        for rc in _numba.prange(rc_flags.shape[0]):
            if rc_flags[rc] == 0:
                continue
            local_count = 0
            for zi in range(active_z_indices.shape[0]):
                z = int(active_z_indices[zi])
                if z < 0:
                    continue
                idx = z * hw + int(rc)
                if idx < 0 or idx >= sensor_flat.shape[0]:
                    continue
                value = int(sensor_flat[idx]) + delta
                if value > max_value:
                    value = max_value
                sensor_flat[idx] = value
                local_count += 1
            counts[rc] = local_count
        total = 0
        for i in range(counts.shape[0]):
            total += int(counts[i])
        update_count_out[0] = total

    @_numba.njit(parallel=True, nogil=True, cache=True)
    def _refresh_state_indices_kernel(
        log_flat: np.ndarray,
        state_flat: np.ndarray,
        changed: np.ndarray,
        free_threshold: int,
        occupied_threshold: int,
    ) -> None:
        for i in _numba.prange(changed.shape[0]):
            idx = int(changed[i])
            if idx < 0 or idx >= state_flat.shape[0]:
                continue
            value = int(log_flat[idx])
            if value >= occupied_threshold:
                state_flat[idx] = int(VOXEL_OCCUPIED)
            elif value <= free_threshold:
                state_flat[idx] = int(VOXEL_FREE)
            else:
                state_flat[idx] = int(VOXEL_UNKNOWN)

except Exception:

    def _missing_numba(*_args, **_kwargs):
        raise RuntimeError("numba is unavailable")

    _count_ray_events_kernel = _missing_numba
    _write_ray_events_kernel = _missing_numba
    _count_event_blocks_kernel = _missing_numba
    _scatter_events_to_blocks_kernel = _missing_numba
    _apply_logodds_blocked_kernel = _missing_numba
    _apply_sensor_blocked_kernel = _missing_numba
    _mark_endpoint_rc_flags_kernel = _missing_numba
    _mark_endpoint_columns_sensor_kernel = _missing_numba
    _refresh_state_indices_kernel = _missing_numba
