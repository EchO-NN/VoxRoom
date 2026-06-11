from __future__ import annotations

import numpy as np


def numba_available() -> bool:
    try:
        import numba  # noqa: F401
    except Exception:
        return False
    return True


def project_navigation_columns(
    state: np.ndarray,
    endpoint_count_xy: np.ndarray | None,
    *,
    occ_z_indices: np.ndarray,
    free_z_indices: np.ndarray,
    endpoint_threshold: int,
    min_free_voxels: int,
    occupied_any_voxel_wins: bool,
    occupied_use_endpoint_hysteresis: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not numba_available():
        raise RuntimeError("numba is unavailable")
    state_arr = np.asarray(state, dtype=np.uint8)
    z_bins, height, width = int(state_arr.shape[0]), int(state_arr.shape[1]), int(state_arr.shape[2])
    endpoint = (
        np.asarray(endpoint_count_xy, dtype=np.uint16).reshape(-1)
        if endpoint_count_xy is not None
        else np.zeros(int(height) * int(width), dtype=np.uint16)
    )
    if endpoint.size != int(height) * int(width):
        raise ValueError("endpoint_count_xy shape does not match voxel grid")
    occupied_from_voxel = np.zeros(endpoint.size, dtype=np.uint8)
    occupied_from_endpoint = np.zeros(endpoint.size, dtype=np.uint8)
    free_raw = np.zeros(endpoint.size, dtype=np.uint8)
    observed_from_voxel = np.zeros(endpoint.size, dtype=np.uint8)
    _project_navigation_columns_kernel(
        state_arr.reshape(-1),
        endpoint,
        int(height),
        int(width),
        int(z_bins),
        np.asarray(occ_z_indices, dtype=np.int32),
        np.asarray(free_z_indices, dtype=np.int32),
        max(1, int(endpoint_threshold)),
        max(1, int(min_free_voxels)),
        bool(occupied_any_voxel_wins),
        bool(occupied_use_endpoint_hysteresis),
        occupied_from_voxel,
        occupied_from_endpoint,
        free_raw,
        observed_from_voxel,
    )
    shape = (int(height), int(width))
    return (
        occupied_from_voxel.reshape(shape).astype(bool),
        occupied_from_endpoint.reshape(shape).astype(bool),
        free_raw.reshape(shape).astype(bool),
        observed_from_voxel.reshape(shape).astype(bool),
    )


def project_navigation_dirty_columns(
    state: np.ndarray,
    endpoint_count_xy: np.ndarray | None,
    dirty_rc_indices: np.ndarray,
    *,
    occ_z_indices: np.ndarray,
    free_z_indices: np.ndarray,
    endpoint_threshold: int,
    min_free_voxels: int,
    occupied_any_voxel_wins: bool,
    occupied_use_endpoint_hysteresis: bool,
    out_occupied_from_voxel_flat: np.ndarray,
    out_occupied_from_endpoint_flat: np.ndarray,
    out_free_raw_flat: np.ndarray,
    out_observed_from_voxel_flat: np.ndarray,
) -> None:
    if not numba_available():
        raise RuntimeError("numba is unavailable")
    state_arr = np.asarray(state, dtype=np.uint8)
    z_bins, height, width = int(state_arr.shape[0]), int(state_arr.shape[1]), int(state_arr.shape[2])
    dirty = np.asarray(dirty_rc_indices, dtype=np.int64).reshape(-1)
    endpoint = (
        np.asarray(endpoint_count_xy, dtype=np.uint16).reshape(-1)
        if endpoint_count_xy is not None
        else np.zeros(int(height) * int(width), dtype=np.uint16)
    )
    hw = int(height) * int(width)
    if endpoint.size != hw:
        raise ValueError("endpoint_count_xy shape does not match voxel grid")
    for out in (
        out_occupied_from_voxel_flat,
        out_occupied_from_endpoint_flat,
        out_free_raw_flat,
        out_observed_from_voxel_flat,
    ):
        if np.asarray(out).reshape(-1).size != hw:
            raise ValueError("dirty projection output shape does not match voxel grid")
    _project_navigation_dirty_columns_kernel(
        state_arr.reshape(-1),
        endpoint,
        dirty,
        int(height),
        int(width),
        int(z_bins),
        np.asarray(occ_z_indices, dtype=np.int32),
        np.asarray(free_z_indices, dtype=np.int32),
        max(1, int(endpoint_threshold)),
        max(1, int(min_free_voxels)),
        bool(occupied_any_voxel_wins),
        bool(occupied_use_endpoint_hysteresis),
        np.asarray(out_occupied_from_voxel_flat).reshape(-1),
        np.asarray(out_occupied_from_endpoint_flat).reshape(-1),
        np.asarray(out_free_raw_flat).reshape(-1),
        np.asarray(out_observed_from_voxel_flat).reshape(-1),
    )


try:
    import numba as _numba

    @_numba.njit(parallel=True, nogil=True, cache=True)
    def _project_navigation_columns_kernel(
        state_flat: np.ndarray,
        endpoint_count_flat: np.ndarray,
        height: int,
        width: int,
        z_bins: int,
        occ_z_indices: np.ndarray,
        free_z_indices: np.ndarray,
        endpoint_threshold: int,
        min_free_voxels: int,
        occupied_any_voxel_wins: bool,
        occupied_use_endpoint_hysteresis: bool,
        occupied_from_voxel: np.ndarray,
        occupied_from_endpoint: np.ndarray,
        free_raw: np.ndarray,
        observed_from_voxel: np.ndarray,
    ) -> None:
        hw = int(height * width)
        for rc in _numba.prange(hw):
            occupied = False
            observed = False
            free_count = 0
            for i in range(occ_z_indices.shape[0]):
                z = int(occ_z_indices[i])
                if z < 0 or z >= z_bins:
                    continue
                value = int(state_flat[z * hw + rc])
                if value != 0:
                    observed = True
                if occupied_any_voxel_wins and value == 2:
                    occupied = True
            for i in range(free_z_indices.shape[0]):
                z = int(free_z_indices[i])
                if z < 0 or z >= z_bins:
                    continue
                value = int(state_flat[z * hw + rc])
                if value != 0:
                    observed = True
                if value == 1:
                    free_count += 1
            occupied_from_voxel[rc] = 1 if occupied else 0
            occupied_from_endpoint[rc] = (
                1 if occupied_use_endpoint_hysteresis and int(endpoint_count_flat[rc]) >= endpoint_threshold else 0
            )
            free_raw[rc] = 1 if free_count >= min_free_voxels else 0
            observed_from_voxel[rc] = 1 if observed else 0

    @_numba.njit(parallel=True, nogil=True, cache=True)
    def _project_navigation_dirty_columns_kernel(
        state_flat: np.ndarray,
        endpoint_count_flat: np.ndarray,
        dirty_rc_indices: np.ndarray,
        height: int,
        width: int,
        z_bins: int,
        occ_z_indices: np.ndarray,
        free_z_indices: np.ndarray,
        endpoint_threshold: int,
        min_free_voxels: int,
        occupied_any_voxel_wins: bool,
        occupied_use_endpoint_hysteresis: bool,
        occupied_from_voxel: np.ndarray,
        occupied_from_endpoint: np.ndarray,
        free_raw: np.ndarray,
        observed_from_voxel: np.ndarray,
    ) -> None:
        hw = int(height * width)
        for k in _numba.prange(dirty_rc_indices.shape[0]):
            rc = int(dirty_rc_indices[k])
            if rc < 0 or rc >= hw:
                continue
            occupied = False
            observed = False
            free_count = 0
            for i in range(occ_z_indices.shape[0]):
                z = int(occ_z_indices[i])
                if z < 0 or z >= z_bins:
                    continue
                value = int(state_flat[z * hw + rc])
                if value != 0:
                    observed = True
                if occupied_any_voxel_wins and value == 2:
                    occupied = True
            for i in range(free_z_indices.shape[0]):
                z = int(free_z_indices[i])
                if z < 0 or z >= z_bins:
                    continue
                value = int(state_flat[z * hw + rc])
                if value != 0:
                    observed = True
                if value == 1:
                    free_count += 1
            occupied_from_voxel[rc] = 1 if occupied else 0
            occupied_from_endpoint[rc] = (
                1 if occupied_use_endpoint_hysteresis and int(endpoint_count_flat[rc]) >= endpoint_threshold else 0
            )
            free_raw[rc] = 1 if free_count >= min_free_voxels else 0
            observed_from_voxel[rc] = 1 if observed else 0

except Exception:

    def _project_navigation_columns_kernel(*_args, **_kwargs):
        raise RuntimeError("numba is unavailable")

    def _project_navigation_dirty_columns_kernel(*_args, **_kwargs):
        raise RuntimeError("numba is unavailable")
