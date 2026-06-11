from __future__ import annotations

from collections import deque
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from scipy import ndimage


FREE_RGB = np.array([145, 220, 230], dtype=np.int16)
WALL_RGB = np.array([230, 40, 45], dtype=np.uint8)
UNKNOWN_RGB = np.array([38, 42, 54], dtype=np.uint8)


def threshold_free_rgb(rgb: np.ndarray, tolerance: int = 60) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.int16)
    return np.abs(arr - FREE_RGB).sum(axis=2) <= int(tolerance)


def full_erosion_marker_roomseg(
    free_mask: np.ndarray,
    *,
    min_child_area: int = 500,
    min_child_ratio: float = 0.02,
    max_iters: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    free = np.asarray(free_mask, dtype=bool)
    conn4 = ndimage.generate_binary_structure(2, 1)
    conn8 = ndimage.generate_binary_structure(2, 2)
    cleaned = ndimage.binary_opening(free, structure=conn4, iterations=1)
    cleaned = ndimage.binary_closing(cleaned, structure=conn8, iterations=1)
    seed_labels = np.zeros(cleaned.shape, dtype=np.int32)
    components, component_count = ndimage.label(cleaned, structure=conn8)
    next_label = 1

    def add_component_seed(component: np.ndarray) -> None:
        nonlocal next_label
        parent_area = int(component.sum())
        current = component.copy()
        last_nonempty = current.copy()
        for _ in range(max(1, int(max_iters))):
            eroded = ndimage.binary_erosion(current, structure=conn4, iterations=1)
            if not eroded.any():
                seed_labels[last_nonempty] = next_label
                next_label += 1
                return
            child_labels, child_count = ndimage.label(eroded, structure=conn8)
            child_ids: list[int] = []
            for child_id in range(1, child_count + 1):
                area = int(np.count_nonzero(child_labels == child_id))
                if area >= int(min_child_area) and area / max(1.0, float(parent_area)) >= float(min_child_ratio):
                    child_ids.append(child_id)
            if len(child_ids) >= 2:
                for child_id in child_ids:
                    add_component_seed(child_labels == child_id)
                return
            last_nonempty = eroded.copy()
            current = eroded
        seed_labels[last_nonempty] = next_label
        next_label += 1

    for component_id in range(1, component_count + 1):
        component = components == component_id
        if int(component.sum()) >= 40:
            add_component_seed(component)

    return grow_seed_labels(seed_labels, cleaned), seed_labels


def grow_seed_labels(seed_labels: np.ndarray, free_mask: np.ndarray) -> np.ndarray:
    labels = np.asarray(seed_labels, dtype=np.int32).copy()
    free = np.asarray(free_mask, dtype=bool)
    queue: deque[tuple[int, int]] = deque(zip(*np.nonzero(labels > 0)))
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while queue:
        row, col = queue.popleft()
        label = int(labels[row, col])
        for dr, dc in neighbors:
            rr = row + dr
            cc = col + dc
            if rr < 0 or rr >= free.shape[0] or cc < 0 or cc >= free.shape[1]:
                continue
            if not free[rr, cc] or labels[rr, cc] != 0:
                continue
            labels[rr, cc] = label
            queue.append((rr, cc))
    return labels


def colorize_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32)
    rgb = np.zeros(labels.shape + (3,), dtype=np.uint8)
    rgb[:] = (18, 20, 26)
    for label in np.unique(labels):
        if int(label) <= 0:
            continue
        digest = hashlib.blake2b(str(int(label)).encode("ascii"), digest_size=3).digest()
        rgb[labels == label] = tuple(int(65 + (byte % 175)) for byte in digest)
    return rgb


def label_boundary(labels: np.ndarray, free_mask: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32)
    free = np.asarray(free_mask, dtype=bool)
    boundary = np.zeros_like(free, dtype=bool)
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        shifted = np.zeros_like(labels)
        src_r = slice(max(0, -dr), labels.shape[0] - max(0, dr))
        src_c = slice(max(0, -dc), labels.shape[1] - max(0, dc))
        dst_r = slice(max(0, dr), labels.shape[0] - max(0, -dr))
        dst_c = slice(max(0, dc), labels.shape[1] - max(0, -dc))
        shifted[dst_r, dst_c] = labels[src_r, src_c]
        boundary |= free & (labels > 0) & (shifted > 0) & (labels != shifted)
    return boundary


def save_roomseg_outputs(image_path: Path, out_dir: Path) -> dict[str, Path | int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    free = threshold_free_rgb(image)
    labels, seeds = full_erosion_marker_roomseg(free)
    color = colorize_labels(labels)
    boundary = label_boundary(labels, free)
    overlay = image.astype(np.float32)
    mask = labels > 0
    overlay[mask] = overlay[mask] * 0.35 + color[mask].astype(np.float32) * 0.65
    overlay[boundary] = WALL_RGB
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    stem = image_path.stem
    paths = {
        "mask": out_dir / f"{stem}.full_erosion_room_mask.png",
        "overlay": out_dir / f"{stem}.full_erosion_overlay.png",
        "seeds": out_dir / f"{stem}.full_erosion_seeds.png",
    }
    Image.fromarray(color, mode="RGB").save(paths["mask"])
    Image.fromarray(overlay, mode="RGB").save(paths["overlay"])
    Image.fromarray((seeds > 0).astype(np.uint8) * 255, mode="L").save(paths["seeds"])
    paths["room_count"] = int(len(np.unique(labels[labels > 0])))
    paths["free_cells"] = int(np.count_nonzero(free))
    return paths


def replay_pngs(paths: Iterable[Path], out_dir: Path) -> list[dict[str, Path | int]]:
    return [save_roomseg_outputs(Path(path), out_dir) for path in paths]
