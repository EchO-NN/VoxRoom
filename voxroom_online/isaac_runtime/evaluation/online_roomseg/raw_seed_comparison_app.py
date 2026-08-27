from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .hough_raw_seed import (
    rasterize_lines,
    reconstruct_hough_raw_door_seed,
)


VOX_RAW_COLOR = (0, 188, 255)
HOUGH_RAW_COLOR = (255, 58, 58)
UNION_RAW_COLOR = (45, 220, 95)


def run_final_raw_seed_comparison_app(
    *,
    episodes: Sequence[Mapping[str, object]],
    figure_scale: float = 2.4,
    hough_seed_width_cells: int = 3,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    entries = [dict(item) for item in episodes]
    if not entries:
        raise ValueError("final raw seed comparison has no episodes")
    scale = max(1.0, float(figure_scale))
    fig, axes = plt.subplots(1, 3, figsize=(15 * scale, 5 * scale))
    fig.subplots_adjust(left=0.215, right=0.99, top=0.90, bottom=0.08, wspace=0.025)
    fig.patch.set_facecolor("white")
    current_index = 0
    page_size = 24
    page_count = max(1, (len(entries) + page_size - 1) // page_size)
    current_page = 0
    scene_buttons: list[object] = []
    scene_button_indices: list[int | None] = []
    sidebar_axes: set[object] = set()
    refs: list[object] = []
    title = fig.text(0.018, 0.925, "", fontsize=10, fontweight="bold", color="#202124")

    @lru_cache(maxsize=8)
    def load_render_data(index: int):
        episode = entries[int(index)]
        path = Path(str(episode["last_snapshot_path"]))
        with np.load(path, allow_pickle=False) as data:
            raw = np.asarray(data["voxel_door_raw_seed_mask"], dtype=bool).copy()
            vertical, vertical_wall, vertical_unknown = _vertical_free_hough_layers(
                data,
                shape=raw.shape,
            )
        snapshot_paths = _episode_snapshot_paths(episode)
        hough_raw, hough_stats = _accumulate_geometric_raw_history(
            snapshot_paths=snapshot_paths,
            final_shape=raw.shape,
            seed_width_cells=int(hough_seed_width_cells),
        )
        union = raw | hough_raw
        background = _source_background(
            obstacle=vertical_wall,
            unknown=vertical_unknown,
            vertical_free=vertical,
        )
        bounds = _visible_bounds(
            (~vertical_unknown) | vertical_wall | vertical | raw | hough_raw,
            margin=18,
        )
        return background, raw, hough_raw, union, hough_stats, bounds

    def render() -> None:
        episode = entries[current_index]
        fig.suptitle("Loading final raw DoorSeed: %s" % _scene_label(episode), fontsize=13)
        fig.canvas.draw_idle()
        background, vox_raw, hough_raw, union_raw, hough_stats, bounds = load_render_data(current_index)
        masks = (vox_raw, hough_raw, union_raw)
        colors = (VOX_RAW_COLOR, HOUGH_RAW_COLOR, UNION_RAW_COLOR)
        titles = (
            "VoxRoom raw DoorSeed | cells=%d" % int(np.count_nonzero(vox_raw)),
            "TVARS-like raw history | stages=%d points=%d lines=%d cells=%d"
            % (
                int(hough_stats["snapshot_count"]),
                int(hough_stats["point_count"]),
                int(hough_stats["unique_line_count"]),
                int(np.count_nonzero(hough_raw)),
            ),
            "Union raw DoorSeed (Vox OR geometric) | cells=%d overlap=%d"
            % (
                int(np.count_nonzero(union_raw)),
                int(np.count_nonzero(vox_raw & hough_raw)),
            ),
        )
        for ax, mask, color, panel_title in zip(axes, masks, colors, titles):
            ax.clear()
            ax.imshow(_overlay_mask(background, mask, color), origin="upper", interpolation="nearest")
            ax.set_title(panel_title, fontsize=11)
            ax.set_axis_off()
            r0, r1, c0, c1 = bounds
            ax.set_xlim(c0 - 0.5, c1 - 0.5)
            ax.set_ylim(r1 - 0.5, r0 - 0.5)
        fig.suptitle(
            "%s | FINAL cumulative raw comparison on VERTICAL FREE | width=%d cells | history=%d saved stages"
            % (
                _scene_label(episode),
                int(hough_seed_width_cells),
                int(hough_stats["snapshot_count"]),
            ),
            fontsize=13,
        )
        render_sidebar()
        fig.canvas.draw_idle()

    def select_scene(index: int) -> None:
        nonlocal current_index, current_page
        current_index = int(index)
        current_page = current_index // page_size
        render()

    def render_sidebar() -> None:
        nonlocal current_page
        current_page = max(0, min(page_count - 1, int(current_page)))
        title.set_text(
            "FINAL SCENES  %d TOTAL  |  PAGE %d/%d"
            % (len(entries), current_page + 1, page_count)
        )
        start = current_page * page_size
        for slot, button in enumerate(scene_buttons):
            index = start + slot
            if index >= len(entries):
                scene_button_indices[slot] = None
                button.ax.set_visible(False)
                continue
            scene_button_indices[slot] = index
            button.ax.set_visible(True)
            button.label.set_text(_scene_label(entries[index]))
            selected = index == current_index
            color = "#bde0fe" if selected else "#e9ecef"
            hover = "#90caf9" if selected else "#dee2e6"
            button.color = color
            button.hovercolor = hover
            button.ax.set_facecolor(color)
            for spine in button.ax.spines.values():
                spine.set_color("#1261a0" if selected else "#adb5bd")
                spine.set_linewidth(2.6 if selected else 0.8)

    def select_slot(slot: int) -> None:
        index = scene_button_indices[int(slot)]
        if index is not None:
            select_scene(index)

    def change_page(delta: int) -> None:
        nonlocal current_page
        target = max(0, min(page_count - 1, current_page + int(delta)))
        if target != current_page:
            current_page = target
            render_sidebar()
            fig.canvas.draw_idle()

    top, bottom = 0.875, 0.115
    row_height = (top - bottom) / page_size
    for slot in range(page_size):
        y = top - float(slot + 1) * row_height
        axis = fig.add_axes((0.015, y, 0.18, row_height * 0.82))
        button = Button(axis, "")
        button.label.set_fontsize(8)
        button.on_clicked(lambda _event, selected_slot=slot: select_slot(selected_slot))
        scene_buttons.append(button)
        scene_button_indices.append(None)
        sidebar_axes.add(axis)
        refs.append(button)
    prev_axis = fig.add_axes((0.015, 0.065, 0.086, 0.036))
    next_axis = fig.add_axes((0.109, 0.065, 0.086, 0.036))
    prev_button, next_button = Button(prev_axis, "Prev"), Button(next_axis, "Next")
    prev_button.on_clicked(lambda _event: change_page(-1))
    next_button.on_clicked(lambda _event: change_page(1))
    sidebar_axes.update((prev_axis, next_axis))
    refs.extend((prev_button, next_button))

    def on_key(event) -> None:
        key = str(event.key or "").lower()
        if key == "pageup":
            change_page(-1)
        elif key == "pagedown":
            change_page(1)
        elif key in {"right", "down"} and current_index + 1 < len(entries):
            select_scene(current_index + 1)
        elif key in {"left", "up"} and current_index > 0:
            select_scene(current_index - 1)

    def on_scroll(event) -> None:
        if getattr(event, "inaxes", None) in sidebar_axes or (
            getattr(event, "x", None) is not None
            and float(event.x) <= float(fig.bbox.width) * 0.205
        ):
            step = float(getattr(event, "step", 0.0) or 0.0)
            change_page(-1 if step > 0 else 1)

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    render()
    plt.show()
    _ = refs


def _episode_snapshot_paths(episode: Mapping[str, object]) -> tuple[Path, ...]:
    records = list(episode.get("snapshots", []))
    ordered = sorted(
        (
            (int(record.get("step", 0)), Path(str(record["snapshot_path"])))
            for record in records
            if record.get("snapshot_path")
        ),
        key=lambda item: item[0],
    )
    paths: list[Path] = []
    seen: set[Path] = set()
    for _step, path in ordered:
        if path in seen:
            continue
        seen.add(path)
        paths.append(path)
    final_path = Path(str(episode["last_snapshot_path"]))
    if final_path not in seen:
        paths.append(final_path)
    return tuple(paths)


def _accumulate_geometric_raw_history(
    *,
    snapshot_paths: Sequence[Path],
    final_shape: tuple[int, int],
    seed_width_cells: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Accumulate raw TVARS-like proposals up to the saved final state.

    VoxRoom's final raw mask is persistent.  The comparable geometric mask
    therefore has to accumulate proposals from every saved coverage stage,
    rather than reconstructing only the final robot pose.  Every stage is
    recomputed on VoxRoom's Vertical Free / vertical-wall grid.  The old TVARS
    Hough points cannot be reused because those were generated on Nav Free.
    """

    raw_history = np.zeros(final_shape, dtype=bool)
    seen_lines: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    point_count = 0
    snapshot_count = 0
    reconstructed_snapshot_count = 0
    for vox_path in snapshot_paths:
        with np.load(vox_path, allow_pickle=False) as data:
            _vertical_free, vertical_wall, vertical_unknown = (
                _vertical_free_hough_layers(data, shape=final_shape)
            )
            agent_rc_arr = np.asarray(data["agent_rc"], dtype=np.int32).reshape(-1)
            resolution = float(np.asarray(data["map_resolution_m"]).reshape(-1)[0])
            if "demo_pose_world" in data.files:
                yaw_rad = float(np.asarray(data["demo_pose_world"]).reshape(-1)[3])
            elif "base_pose_world_xyzyaw" in data.files:
                yaw_rad = float(
                    np.asarray(data["base_pose_world_xyzyaw"]).reshape(-1)[3]
                )
            else:
                raise KeyError("snapshot missing saved agent yaw: %s" % vox_path)
            reconstructed = reconstruct_hough_raw_door_seed(
                obstacle_mask=vertical_wall,
                unknown_mask=vertical_unknown,
                agent_rc=(int(agent_rc_arr[0]), int(agent_rc_arr[1])),
                yaw_deg=float(np.degrees(yaw_rad)),
                resolution_m=resolution,
                seed_width_cells=int(seed_width_cells),
            )
            points = reconstructed.raw_points_xy
            lines = reconstructed.paired_lines_xyxy
            reconstructed_snapshot_count += 1
        point_count += len(points)
        snapshot_count += 1
        unique_lines: list[tuple[int, int, int, int]] = []
        for x0, y0, x1, y1 in lines:
            identity = tuple(sorted(((int(x0), int(y0)), (int(x1), int(y1)))))
            if identity in seen_lines:
                continue
            seen_lines.add(identity)
            unique_lines.append((int(x0), int(y0), int(x1), int(y1)))
        raw_history |= rasterize_lines(
            unique_lines,
            shape=final_shape,
            width_cells=max(1, int(seed_width_cells)),
        )
    return raw_history, {
        "snapshot_count": snapshot_count,
        "point_count": point_count,
        "unique_line_count": len(seen_lines),
        "reconstructed_snapshot_count": reconstructed_snapshot_count,
    }


def _vertical_free_hough_layers(
    data,
    *,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertical_free = _first_mask(
        data,
        (
            "voxel_vertical_free_xy",
            "height_profile_vertical_free_xy",
            "vertical_free_room_domain",
        ),
        shape,
    )
    if not np.any(vertical_free):
        raise ValueError("snapshot has no non-empty Vertical Free map")
    vertical_wall = _first_mask(
        data,
        (
            "voxel_wall_xy",
            "structural_wall_clean",
            "roomseg_sanitized_wall",
            "initial_roomseg_occupied_after_fusion",
            "initial_roomseg_occupied",
        ),
        shape,
    )
    vertical_wall &= ~vertical_free
    if not np.any(vertical_wall):
        vertical_observed = _first_mask(
            data,
            (
                "voxel_vertical_observed_xy",
                "vertical_observed_map",
                "vertical_observed_0p2_2p0",
            ),
            shape,
        )
        vertical_wall = vertical_observed & ~vertical_free
    if not np.any(vertical_wall):
        raise ValueError("snapshot has no vertical-wall boundary for Hough pairing")
    # The raycaster may traverse only Vertical Free cells.  Everything that is
    # neither Vertical Free nor an explicit vertical wall remains unknown and
    # stops a ray; Nav Free/obstacle is intentionally absent from this domain.
    vertical_unknown = ~(vertical_free | vertical_wall)
    return vertical_free, vertical_wall, vertical_unknown


def _first_mask(data, keys: tuple[str, ...], shape: tuple[int, int]) -> np.ndarray:
    for key in keys:
        if key not in data.files:
            continue
        value = np.asarray(data[key], dtype=bool)
        if value.shape == shape and np.any(value):
            return value.copy()
    return np.zeros(shape, dtype=bool)


def _source_background(
    *,
    obstacle: np.ndarray,
    unknown: np.ndarray,
    vertical_free: np.ndarray,
) -> np.ndarray:
    base = np.zeros(obstacle.shape + (3,), dtype=np.uint8)
    base[:] = (245, 245, 245)
    base[np.asarray(unknown, dtype=bool)] = (225, 225, 225)
    base[np.asarray(vertical_free, dtype=bool)] = (170, 218, 242)
    base[np.asarray(obstacle, dtype=bool)] = (30, 30, 30)
    return base


def _overlay_mask(
    background: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
) -> np.ndarray:
    image = np.asarray(background, dtype=np.uint8).copy()
    image[np.asarray(mask, dtype=bool)] = np.asarray(color, dtype=np.uint8)
    return image


def _visible_bounds(mask: np.ndarray, *, margin: int) -> tuple[int, int, int, int]:
    points = np.argwhere(np.asarray(mask, dtype=bool))
    h, w = mask.shape
    if points.size == 0:
        return 0, h, 0, w
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0) + 1
    return (
        max(0, int(r0) - int(margin)),
        min(h, int(r1) + int(margin)),
        max(0, int(c0) - int(margin)),
        min(w, int(c1) + int(margin)),
    )


def _scene_label(episode: Mapping[str, object]) -> str:
    scene = str(episode.get("scene_id") or episode.get("episode_uid") or "scene")
    parts = {str(part).lower() for part in Path(str(episode.get("coverage_manifest") or "")).parts}
    prefix = "H" if "habitat" in parts else "IA" if "interioragent" in parts else "GR" if "grscene" in parts else "SC"
    return "%s  %s" % (prefix, scene.removeprefix("kujiale_"))
