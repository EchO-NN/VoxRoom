from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np

from voxroom_online.isaac_runtime.mapping.frontier import FrontierCluster

GridCell = Tuple[int, int]


def save_frontier_debug_snapshot(
    out_dir: str | Path,
    step: int,
    *,
    free: np.ndarray,
    occupied: np.ndarray,
    observed: np.ndarray,
    unknown: np.ndarray,
    unknown_dilated: np.ndarray,
    frontier: np.ndarray,
    traversible: np.ndarray,
    dist_map: np.ndarray,
    agent_grid: GridCell,
    clusters: Iterable[FrontierCluster],
    selected_frontier: Optional[GridCell] = None,
    candidate_centers: Optional[Iterable[GridCell | dict]] = None,
    candidate_target_cells: Optional[Iterable[GridCell]] = None,
    selected_candidate: Optional[GridCell] = None,
    decision_mode: str = "",
) -> tuple[Path, Optional[Path]]:
    """Save frontier extraction layers for one replan."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    clusters_list = list(clusters)
    selected = np.asarray(selected_frontier if selected_frontier is not None else (-1, -1), dtype=np.int32)
    candidate_centers_list = list(candidate_centers or [])
    candidate_center_cells = [_candidate_cell(item) for item in candidate_centers_list]
    candidate_center_status = [_candidate_status(item) for item in candidate_centers_list]
    candidate_targets_list = list(candidate_target_cells or [])
    selected_candidate_arr = np.asarray(selected_candidate if selected_candidate is not None else (-1, -1), dtype=np.int32)
    npz_path = out_path / f"frontier_step_{int(step):06d}.npz"
    np.savez_compressed(
        npz_path,
        free=np.asarray(free).astype(np.uint8),
        occupied=np.asarray(occupied).astype(np.uint8),
        observed=np.asarray(observed).astype(np.uint8),
        unknown=np.asarray(unknown).astype(np.uint8),
        unknown_dilated=np.asarray(unknown_dilated).astype(np.uint8),
        frontier_cells=np.asarray(frontier).astype(np.uint8),
        traversible_for_distance=np.asarray(traversible).astype(np.uint8),
        distance_map=np.asarray(dist_map).astype(np.float32),
        agent_grid=np.asarray(agent_grid, dtype=np.int32),
        cluster_centers=np.asarray([c.center_grid for c in clusters_list], dtype=np.int32),
        cluster_sizes=np.asarray([c.size for c in clusters_list], dtype=np.int32),
        cluster_min_dists=np.asarray([c.min_path_distance for c in clusters_list], dtype=np.float32),
        cluster_mean_dists=np.asarray([c.mean_path_distance for c in clusters_list], dtype=np.float32),
        cluster_center_dists=np.asarray([c.center_path_distance for c in clusters_list], dtype=np.float32),
        selected_frontier=selected,
        candidate_centers=np.asarray(candidate_center_cells, dtype=np.int32),
        candidate_center_status=np.asarray(candidate_center_status),
        candidate_target_cells=np.asarray(candidate_targets_list, dtype=np.int32),
        selected_candidate=selected_candidate_arr,
        decision_mode=np.asarray(str(decision_mode)),
    )
    png_path = _save_frontier_debug_png(
        out_path / f"frontier_step_{int(step):06d}.png",
        free=free,
        occupied=occupied,
        observed=observed,
        unknown=unknown,
        frontier=frontier,
        traversible=traversible,
        agent_grid=agent_grid,
        clusters=clusters_list,
        selected_frontier=selected_frontier,
        candidate_centers=candidate_centers_list,
        candidate_target_cells=candidate_targets_list,
        selected_candidate=selected_candidate,
    )
    return npz_path, png_path


def _save_frontier_debug_png(
    path: Path,
    *,
    free: np.ndarray,
    occupied: np.ndarray,
    observed: np.ndarray,
    unknown: np.ndarray,
    frontier: np.ndarray,
    traversible: np.ndarray,
    agent_grid: GridCell,
    clusters: list[FrontierCluster],
    selected_frontier: Optional[GridCell],
    candidate_centers: list[GridCell | dict],
    candidate_target_cells: list[GridCell],
    selected_candidate: Optional[GridCell],
) -> Optional[Path]:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return None

    free_b = np.asarray(free).astype(bool)
    occ_b = np.asarray(occupied).astype(bool)
    obs_b = np.asarray(observed).astype(bool)
    unknown_b = np.asarray(unknown).astype(bool)
    frontier_b = np.asarray(frontier).astype(bool)
    trav_b = np.asarray(traversible).astype(bool)

    h, w = free_b.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[~obs_b] = (28, 28, 34)
    rgb[unknown_b] = (48, 48, 58)
    rgb[obs_b] = (70, 70, 74)
    rgb[free_b] = (198, 198, 190)
    rgb[trav_b] = (222, 222, 210)
    rgb[occ_b] = (178, 56, 56)
    rgb[frontier_b] = (35, 210, 235)

    scale = max(2, min(6, int(round(900.0 / max(h, w, 1)))))
    resampling = getattr(getattr(Image, "Resampling", Image), "NEAREST")
    image = Image.fromarray(rgb).resize((w * scale, h * scale), resampling)
    draw = ImageDraw.Draw(image)

    def box(cell: GridCell, color: tuple[int, int, int], radius: int = 2) -> None:
        row, col = int(cell[0]), int(cell[1])
        if not (0 <= row < h and 0 <= col < w):
            return
        x = col * scale + scale // 2
        y = row * scale + scale // 2
        r = max(radius, scale)
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=max(1, scale // 2))

    box(agent_grid, (60, 140, 255), radius=3)
    for cluster in clusters:
        box(cluster.center_grid, (255, 210, 50), radius=2)
    if selected_frontier is not None:
        box(selected_frontier, (255, 60, 60), radius=4)
    for cell in candidate_target_cells[:64]:
        box(cell, (220, 70, 255), radius=1)
    for item in candidate_centers[:64]:
        cell = _candidate_cell(item)
        status = _candidate_status(item)
        color = {
            "candidate": (255, 150, 40),
            "selected": (255, 50, 50),
            "accepted": (40, 220, 90),
            "rejected": (135, 135, 135),
        }.get(status, (255, 150, 40))
        box(cell, color, radius=4 if status == "selected" else 3)
    if selected_candidate is not None:
        box(selected_candidate, (255, 30, 30), radius=5)
    image.save(path)
    return path


def _candidate_cell(item) -> GridCell:
    if isinstance(item, dict):
        cell = item.get("center_grid", (-1, -1))
        return (int(cell[0]), int(cell[1]))
    return (int(item[0]), int(item[1]))


def _candidate_status(item) -> str:
    if isinstance(item, dict):
        return str(item.get("status", "candidate"))
    return "candidate"
