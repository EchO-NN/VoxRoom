from __future__ import annotations

import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np

from voxroom_online.isaac_runtime.door_seed_learning.patch_extraction import (
    apply_synchronized_xy_transform,
    class_patches_to_one_hot,
    extract_class_patches,
    extract_seed_voxel_patch_from_snapshot,
    voxel_states_to_model_channels,
)
from voxroom_online.isaac_runtime.door_seed_learning.schema import (
    CONTEXT_PATCH_SIZE,
    LOCAL_PATCH_SIZE,
    load_snapshot,
)


class DoorSeedDataset:
    def __init__(
        self,
        index_path: str | Path,
        *,
        split: str | Sequence[str] | None = None,
        context_source: str,
        height_scale_m: float = 4.0,
        cache_size: int = 6,
        augment: bool = False,
        rotation_degrees: Sequence[int] | None = None,
        mirror_lr_once: bool = False,
        seed: int = 0,
        local_patch_size: int = LOCAL_PATCH_SIZE,
        context_patch_size: int = CONTEXT_PATCH_SIZE,
    ) -> None:
        self.rows = read_index(index_path)
        allowed_splits = None if split is None else ({str(split)} if isinstance(split, str) else {str(v) for v in split})
        if allowed_splits is not None:
            self.rows = [row for row in self.rows if str(row.get("split")) in allowed_splits]
        self.context_source = str(context_source)
        if self.context_source not in {"vertical", "nav"}:
            raise ValueError("context_source must be vertical or nav")
        mismatched = [row for row in self.rows if str(row.get("context_source")) != self.context_source]
        if mismatched:
            raise ValueError("dataset index context_source mismatch")
        self.height_scale_m = float(height_scale_m)
        self.cache_size = max(1, int(cache_size))
        self.augment = bool(augment)
        self.rotation_degrees = normalize_right_angle_rotations(rotation_degrees)
        self.rotation_count = len(self.rotation_degrees)
        self.mirror_lr_once = bool(mirror_lr_once)
        self.transform_count = self.rotation_count + int(self.mirror_lr_once)
        self.rng = np.random.default_rng(int(seed))
        self.local_patch_size = int(local_patch_size)
        self.context_patch_size = int(context_patch_size)
        self._cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.rows) * self.transform_count

    def __getitem__(self, index: int) -> dict[str, object]:
        import torch

        expanded_index = int(index)
        if expanded_index < 0:
            expanded_index += len(self)
        if expanded_index < 0 or expanded_index >= len(self):
            raise IndexError("door-seed dataset index out of range")
        row_index, transform_slot = divmod(expanded_index, self.transform_count)
        mirrored_lr = transform_slot >= self.rotation_count
        rotation_degrees = (
            0 if mirrored_lr else int(self.rotation_degrees[transform_slot])
        )
        row = self.rows[row_index]
        arrays = self._snapshot(str(row["snapshot_path"]))
        seed_index = int(row["seed_index"])
        rc = np.asarray([[int(row["row"]), int(row["col"])]], dtype=np.int32)
        voxel = extract_seed_voxel_patch_from_snapshot(
            arrays,
            seed_index=seed_index,
            seed_rc=rc,
            patch_size=self.local_patch_size,
        )
        if tuple(voxel.shape[-2:]) != (self.local_patch_size, self.local_patch_size):
            raise ValueError(
                "dataset local voxel patch does not match requested %dx%d input"
                % (self.local_patch_size, self.local_patch_size)
            )
        context_key = "vertical_class_map_xy" if self.context_source == "vertical" else "nav_class_map_xy"
        context = extract_class_patches(
            arrays[context_key], rc, patch_size=self.context_patch_size
        )[0]
        if rotation_degrees:
            voxel, context = apply_synchronized_xy_transform(
                voxel,
                context,
                rotations=rotation_degrees // 90,
            )
        if mirrored_lr:
            voxel, context = apply_synchronized_xy_transform(
                voxel,
                context,
                flip_lr=True,
            )
        if self.augment:
            voxel, context = apply_synchronized_xy_transform(
                voxel,
                context,
                rotations=int(self.rng.integers(0, 4)),
                flip_lr=bool(self.rng.integers(0, 2)),
                flip_ud=bool(self.rng.integers(0, 2)),
            )
        voxel_channels = voxel_states_to_model_channels(
            voxel[None, ...],
            arrays["z_centers_m"],
            height_scale_m=self.height_scale_m,
        )[0]
        context_channels = class_patches_to_one_hot(context[None, ...])[0]
        return {
            "voxel": torch.from_numpy(voxel_channels),
            "context": torch.from_numpy(context_channels),
            "label": torch.tensor([float(row["label"])], dtype=torch.float32),
            "group_id": str(row["group_id"]),
            "scene_id": str(row["scene_id"]),
            "row": int(row["row"]),
            "col": int(row["col"]),
            "rotation_degrees": rotation_degrees,
            "mirrored_lr": mirrored_lr,
        }

    def _snapshot(self, path: str) -> dict[str, np.ndarray]:
        key = str(Path(path).resolve())
        if key in self._cache:
            arrays = self._cache.pop(key)
            self._cache[key] = arrays
            return arrays
        arrays = load_snapshot(key)
        self._cache[key] = arrays
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return arrays


class GroupedCoordinateSampler:
    """Yield one occurrence per seed group and every requested XY transform."""

    def __init__(
        self,
        rows: Sequence[Mapping[str, object]],
        *,
        rotation_count: int = 1,
        seed: int = 0,
        shuffle: bool = True,
    ) -> None:
        self.rows = list(rows)
        groups: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(self.rows):
            groups[str(row["group_id"])].append(index)
        self.groups = list(groups.values())
        self.rotation_count = int(rotation_count)
        if self.rotation_count <= 0:
            raise ValueError("rotation_count must be positive")
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        selected_rows = [int(group[int(rng.integers(0, len(group)))]) for group in self.groups]
        buckets: dict[str, list[int]] = defaultdict(list)
        for row_index in selected_rows:
            buckets[str(self.rows[row_index].get("snapshot_path", ""))].append(row_index)
        snapshot_keys = list(buckets)
        if self.shuffle:
            rng.shuffle(snapshot_keys)
            for rows in buckets.values():
                rng.shuffle(rows)
        selected = []
        for snapshot_key in snapshot_keys:
            for row_index in buckets[snapshot_key]:
                selected.extend(
                    row_index * self.rotation_count + transform_slot
                    for transform_slot in range(self.rotation_count)
                )
        return iter(selected)

    def __len__(self) -> int:
        return len(self.groups) * self.rotation_count


def normalize_right_angle_rotations(degrees: Sequence[int] | None) -> tuple[int, ...]:
    values = (0,) if degrees is None else tuple(int(value) for value in degrees)
    if not values:
        raise ValueError("rotation_degrees must contain at least one angle")
    normalized: list[int] = []
    for value in values:
        if value % 90 != 0:
            raise ValueError("rotation_degrees must be multiples of 90")
        angle = value % 360
        if angle in normalized:
            raise ValueError("rotation_degrees must not contain duplicate orientations")
        normalized.append(angle)
    return tuple(normalized)


def read_index(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("dataset index line %d is not an object" % line_number)
            rows.append(value)
    return rows
