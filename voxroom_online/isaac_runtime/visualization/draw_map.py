from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


def save_map_png(
    path: str,
    occupancy: np.ndarray,
    navigable: Optional[np.ndarray] = None,
    start: Optional[Tuple[int, int]] = None,
    goals: Optional[Iterable[Tuple[int, int]]] = None,
    path_cells: Optional[Iterable[Tuple[int, int]]] = None,
) -> None:
    try:
        from PIL import Image
    except Exception:
        return
    nav = np.ones_like(occupancy, dtype=bool) if navigable is None else navigable.astype(bool)
    rgb = np.zeros((occupancy.shape[0], occupancy.shape[1], 3), dtype=np.uint8)
    rgb[nav] = (240, 240, 240)
    rgb[~nav] = (170, 170, 170)
    rgb[occupancy.astype(bool)] = (20, 20, 20)
    if goals:
        for r, c in goals:
            if 0 <= r < rgb.shape[0] and 0 <= c < rgb.shape[1]:
                rgb[r, c] = (40, 180, 70)
    if path_cells:
        for r, c in path_cells:
            if 0 <= r < rgb.shape[0] and 0 <= c < rgb.shape[1]:
                rgb[r, c] = (50, 120, 255)
    if start is not None:
        r, c = start
        if 0 <= r < rgb.shape[0] and 0 <= c < rgb.shape[1]:
            rgb[max(0, r - 2) : min(rgb.shape[0], r + 3), max(0, c - 2) : min(rgb.shape[1], c + 3)] = (255, 60, 40)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out)

