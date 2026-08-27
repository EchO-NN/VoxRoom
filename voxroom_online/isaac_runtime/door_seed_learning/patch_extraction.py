from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from voxroom_online.isaac_runtime.mapping.voxel_occupancy_grid import (
    VOXEL_CONFLICT,
    VOXEL_FREE,
    VOXEL_OCCUPIED,
    VOXEL_UNKNOWN,
)


def sorted_seed_coordinates(seed_mask_or_rc: np.ndarray | Iterable[tuple[int, int]]) -> np.ndarray:
    value = np.asarray(seed_mask_or_rc)
    if value.ndim == 2 and value.dtype == bool:
        return np.argwhere(value).astype(np.int32)
    rc = np.asarray(list(seed_mask_or_rc) if not isinstance(seed_mask_or_rc, np.ndarray) else seed_mask_or_rc, dtype=np.int32)
    if rc.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    if rc.ndim != 2 or rc.shape[1] != 2:
        raise ValueError("seed coordinates must have shape [N,2]")
    order = np.lexsort((rc[:, 1], rc[:, 0]))
    return rc[order].astype(np.int32, copy=False)


def extract_local_voxel_patches(
    voxel_state_zyx: np.ndarray,
    seed_rc: np.ndarray | Iterable[tuple[int, int]],
    *,
    patch_size: int = 9,
    unknown_code: int = int(VOXEL_UNKNOWN),
) -> tuple[np.ndarray, np.ndarray]:
    state = np.asarray(voxel_state_zyx, dtype=np.uint8)
    if state.ndim != 3:
        raise ValueError("voxel_state_zyx must have shape [Z,H,W]")
    size = _validate_odd_size(patch_size)
    rc = sorted_seed_coordinates(seed_rc)
    z_count, height, width = state.shape
    patches = np.full((len(rc), z_count, size, size), int(unknown_code), dtype=np.uint8)
    valid = np.zeros((len(rc), size, size), dtype=bool)
    radius = size // 2
    for index, (row, col) in enumerate(rc):
        src_r0 = max(0, int(row) - radius)
        src_r1 = min(height, int(row) + radius + 1)
        src_c0 = max(0, int(col) - radius)
        src_c1 = min(width, int(col) + radius + 1)
        dst_r0 = src_r0 - (int(row) - radius)
        dst_c0 = src_c0 - (int(col) - radius)
        dst_r1 = dst_r0 + (src_r1 - src_r0)
        dst_c1 = dst_c0 + (src_c1 - src_c0)
        if src_r1 > src_r0 and src_c1 > src_c0:
            patches[index, :, dst_r0:dst_r1, dst_c0:dst_c1] = state[:, src_r0:src_r1, src_c0:src_c1]
            valid[index, dst_r0:dst_r1, dst_c0:dst_c1] = True
    return patches, valid


def extract_seed_voxel_patch_from_snapshot(
    arrays: Mapping[str, np.ndarray],
    *,
    seed_index: int,
    seed_rc: np.ndarray | Iterable[tuple[int, int]],
    patch_size: int,
) -> np.ndarray:
    """Rebuild a seed patch from the full grid when schema v4 provides it."""

    if "voxel_occupancy_state_zyx" in arrays:
        patches, _valid = extract_local_voxel_patches(
            np.asarray(arrays["voxel_occupancy_state_zyx"], dtype=np.uint8),
            seed_rc,
            patch_size=int(patch_size),
        )
        if len(patches) != 1:
            raise ValueError("one seed coordinate is required to rebuild a voxel patch")
        return np.asarray(patches[0], dtype=np.uint8)
    saved = np.asarray(arrays["raw_seed_voxel_state_nzyx"][int(seed_index)], dtype=np.uint8)
    if saved.shape[-2:] != (int(patch_size), int(patch_size)):
        raise ValueError(
            "saved voxel patch does not match requested %dx%d and no full voxel grid is available"
            % (int(patch_size), int(patch_size))
        )
    return saved


def extract_class_patches(
    class_map_xy: np.ndarray,
    seed_rc: np.ndarray | Iterable[tuple[int, int]],
    *,
    patch_size: int = 33,
    unknown_code: int = 0,
) -> np.ndarray:
    class_map = np.asarray(class_map_xy, dtype=np.uint8)
    if class_map.ndim != 2:
        raise ValueError("class_map_xy must have shape [H,W]")
    size = _validate_odd_size(patch_size)
    rc = sorted_seed_coordinates(seed_rc)
    height, width = class_map.shape
    patches = np.full((len(rc), size, size), int(unknown_code), dtype=np.uint8)
    radius = size // 2
    for index, (row, col) in enumerate(rc):
        src_r0 = max(0, int(row) - radius)
        src_r1 = min(height, int(row) + radius + 1)
        src_c0 = max(0, int(col) - radius)
        src_c1 = min(width, int(col) + radius + 1)
        dst_r0 = src_r0 - (int(row) - radius)
        dst_c0 = src_c0 - (int(col) - radius)
        dst_r1 = dst_r0 + (src_r1 - src_r0)
        dst_c1 = dst_c0 + (src_c1 - src_c0)
        if src_r1 > src_r0 and src_c1 > src_c0:
            patches[index, dst_r0:dst_r1, dst_c0:dst_c1] = class_map[src_r0:src_r1, src_c0:src_c1]
    return patches


def voxel_states_to_model_channels(
    voxel_state_nzyx: np.ndarray,
    z_centers_m: np.ndarray,
    *,
    height_scale_m: float = 4.0,
) -> np.ndarray:
    state = np.asarray(voxel_state_nzyx, dtype=np.uint8)
    if state.ndim != 4:
        raise ValueError("voxel_state_nzyx must have shape [N,Z,Y,X]")
    z = np.asarray(z_centers_m, dtype=np.float32)
    if z.shape != (state.shape[1],):
        raise ValueError("z_centers_m length must match voxel Z")
    if float(height_scale_m) <= 0.0:
        raise ValueError("height_scale_m must be positive")
    unknown = (state == int(VOXEL_UNKNOWN)) | (state == int(VOXEL_CONFLICT))
    free = state == int(VOXEL_FREE)
    occupied = state == int(VOXEL_OCCUPIED)
    if np.any((unknown.astype(np.uint8) + free.astype(np.uint8) + occupied.astype(np.uint8)) != 1):
        raise ValueError("voxel state contains unsupported codes")
    height = np.broadcast_to(
        (z / float(height_scale_m))[None, :, None, None],
        state.shape,
    )
    return np.stack((unknown, free, occupied, height), axis=1).astype(np.float32)


def class_patches_to_one_hot(class_patch_nyx: np.ndarray, *, class_count: int = 3) -> np.ndarray:
    classes = np.asarray(class_patch_nyx, dtype=np.uint8)
    if classes.ndim != 3:
        raise ValueError("class_patch_nyx must have shape [N,Y,X]")
    if classes.size and int(np.max(classes)) >= int(class_count):
        raise ValueError("class patch contains an out-of-range class code")
    return np.stack([classes == index for index in range(int(class_count))], axis=1).astype(np.float32)


def apply_synchronized_xy_transform(
    voxel_state_zyx: np.ndarray,
    context_yx: np.ndarray,
    *,
    rotations: int = 0,
    flip_lr: bool = False,
    flip_ud: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    voxel = np.asarray(voxel_state_zyx)
    context = np.asarray(context_yx)
    k = int(rotations) % 4
    if k:
        voxel = np.rot90(voxel, k=k, axes=(-2, -1))
        context = np.rot90(context, k=k, axes=(-2, -1))
    if bool(flip_lr):
        voxel = np.flip(voxel, axis=-1)
        context = np.flip(context, axis=-1)
    if bool(flip_ud):
        voxel = np.flip(voxel, axis=-2)
        context = np.flip(context, axis=-2)
    return np.ascontiguousarray(voxel), np.ascontiguousarray(context)


def observed_ratio(voxel_state_zyx: np.ndarray, context_yx: np.ndarray | None = None) -> float:
    voxel = np.asarray(voxel_state_zyx, dtype=np.uint8)
    observed = (voxel == int(VOXEL_FREE)) | (voxel == int(VOXEL_OCCUPIED))
    observed_count = int(np.count_nonzero(observed))
    total = int(voxel.size)
    if context_yx is not None:
        context = np.asarray(context_yx, dtype=np.uint8)
        observed_count += int(np.count_nonzero(context != 0))
        total += int(context.size)
    return float(observed_count) / float(max(1, total))


def _validate_odd_size(value: int) -> int:
    size = int(value)
    if size <= 0 or size % 2 != 1:
        raise ValueError("patch size must be a positive odd integer")
    return size
