from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .annotation_schema import LineAnnotation, MergeGroup, ReviewState, approved_review, load_annotation, make_initial_annotation, save_annotation_atomic
from .mask_generation import GtGenerationConfig, generate_gt_from_annotation
from .snapshot_io import SnapshotArrays
from .visualization import colorize_labels


@dataclass(frozen=True)
class AnnotationAppResult:
    action: str
    episode_uid: str | None = None


def run_annotation_app(
    *,
    episode: Mapping[str, object],
    snapshot: SnapshotArrays,
    annotation_path: Path,
    gt_dir: Path,
    line_width_cells: int = 3,
    preclose_radius_cells: int = 1,
    figure_scale: float = 2.8,
    min_room_area_cells: int | None = None,
    source_view: str = "navigation",
    scene_entries: Sequence[Mapping[str, object]] | None = None,
) -> AnnotationAppResult:
    import matplotlib.pyplot as plt

    if Path(annotation_path).exists():
        annotation = load_annotation(Path(annotation_path))
    else:
        annotation = make_initial_annotation(
            episode=episode,
            snapshot_arrays=snapshot,
            line_width_cells=int(line_width_cells),
            preclose_radius_cells=int(preclose_radius_cells),
        )
    if min_room_area_cells is not None:
        annotation = replace(annotation, min_room_area_cells=max(1, int(min_room_area_cells)))
    current_points: list[tuple[int, int]] = []
    merge_points: list[int] = []
    click_mode = "line"
    current_width = int(annotation.line_width_cells_default)
    line_kind = "separator"
    current_source_view = "navigation" if str(source_view) == "navigation" else "segmentation"
    show_prediction = True
    show_obstacle = True
    show_unknown = True
    preview_cache_key = None
    preview_cache = None
    dirty = False
    navigation_result = AnnotationAppResult(action="quit")
    middle_pan: dict[str, object] = {"active": False}
    entries = [dict(item) for item in (scene_entries or ())]
    current_episode_uid = str(episode["episode_uid"])
    entry_status = {
        str(item["episode_uid"]): str(item.get("status", "todo"))
        for item in entries
    }
    sidebar_enabled = len(entries) > 1
    scale = max(0.8, float(figure_scale))
    fig, axes = plt.subplots(1, 3, figsize=(15 * scale, 5 * scale))
    fig.subplots_adjust(
        left=0.215 if sidebar_enabled else 0.02,
        right=0.99,
        top=0.90,
        bottom=0.18,
        wspace=0.02,
    )
    fig.patch.set_facecolor("white")
    _install_help_text(fig)
    button_refs = []
    scene_button_slots: list[object] = []
    scene_button_uids: list[str | None] = []
    scene_sidebar_axes: set[object] = set()
    scene_sidebar_title = None
    sidebar_page_size = 24
    sidebar_page_count = max(
        1,
        (len(entries) + sidebar_page_size - 1) // sidebar_page_size,
    )
    current_entry_index = next(
        (
            index
            for index, item in enumerate(entries)
            if str(item["episode_uid"]) == current_episode_uid
        ),
        0,
    )
    sidebar_page = min(
        sidebar_page_count - 1,
        current_entry_index // sidebar_page_size,
    )

    def set_status(message: str = "") -> None:
        suffix = " | %s" % message if message else ""
        fig.suptitle(
            "VoxRoom roomseg annotation: %s | bg=%s mode=%s width=%d min_area=%d cells%s"
            % (annotation.episode_uid, current_source_view, _mode_text(click_mode, line_kind), int(current_width), int(annotation.min_room_area_cells), suffix)
        )
        fig.canvas.draw_idle()

    set_status("click map once to focus keyboard")

    def compute_preview():
        nonlocal preview_cache_key, preview_cache
        key = (
            tuple(
                (
                    line.id,
                    tuple(line.p0_rc),
                    tuple(line.p1_rc),
                    int(line.width_cells),
                    str(getattr(line, "kind", "separator")),
                )
                for line in annotation.split_lines
            ),
            tuple(tuple(int(v) for v in group.component_ids) for group in annotation.merge_groups),
            int(current_width),
            int(annotation.preclose_radius_cells),
            int(annotation.min_room_area_cells),
        )
        if preview_cache_key == key and preview_cache is not None:
            return preview_cache
        preview_cache = generate_gt_from_annotation(
            eval_domain=snapshot.eval_domain,
            split_lines=annotation.split_lines,
            merge_groups=annotation.merge_groups,
            obstacle_mask=snapshot.obstacle_mask,
            segmentation_domain=snapshot.segmentation_domain,
            config=GtGenerationConfig(
                line_width_cells=current_width,
                preclose_radius_cells=int(annotation.preclose_radius_cells),
                min_room_area_cells=int(annotation.min_room_area_cells),
            ),
        )
        preview_cache_key = key
        return preview_cache

    def render() -> None:
        view_limits = _capture_view_limits(axes)
        for ax in axes:
            ax.clear()
            ax.set_axis_off()
            ax.set_facecolor("white")
        nav = _render_source_background(snapshot=snapshot, navigation_png=annotation.navigation_png, source_view=current_source_view)
        if current_source_view == "segmentation":
            if not show_unknown and snapshot.unknown_mask is not None:
                nav = nav.copy()
                nav[np.asarray(snapshot.unknown_mask, dtype=bool)] = (35, 37, 44)
            if not show_obstacle and snapshot.obstacle_mask is not None:
                nav = nav.copy()
                nav[np.asarray(snapshot.obstacle_mask, dtype=bool)] = (35, 37, 44)
        result = compute_preview()
        axes[0].imshow(nav, origin="upper")
        door_seed_overlay, door_seed_counts = _render_door_seed_overlay(snapshot)
        if door_seed_counts["raw"] > 0:
            from matplotlib.patches import Patch

            axes[0].imshow(
                door_seed_overlay,
                origin="upper",
                interpolation="nearest",
            )
            axes[0].legend(
                handles=[
                    Patch(
                        facecolor="#00e5ff",
                        edgecolor="black",
                        label="DoorSeed kept (%d)" % door_seed_counts["kept"],
                    ),
                    Patch(
                        facecolor="#ff3030",
                        edgecolor="black",
                        label="DoorSeed rejected (%d)" % door_seed_counts["rejected"],
                    ),
                ],
                loc="lower right",
                fontsize=9,
                framealpha=0.92,
            )
        for line in annotation.split_lines:
            color = "#d7191c" if getattr(line, "kind", "separator") == "wall_completion" else "#8b2be2"
            axes[0].plot([line.p0_rc[1], line.p1_rc[1]], [line.p0_rc[0], line.p1_rc[0]], color=color, linewidth=max(2, line.width_cells))
        if len(current_points) == 1:
            axes[0].scatter([current_points[0][1]], [current_points[0][0]], c="yellow", s=20)
        _draw_mode_badge(axes[0], line_kind)
        axes[0].set_title(
            "%s source + manual lines (%s) | DoorSeed raw=%d kept=%d rejected=%d"
            % (
                current_source_view,
                _mode_text(click_mode, line_kind),
                door_seed_counts["raw"],
                door_seed_counts["kept"],
                door_seed_counts["rejected"],
            )
        )
        axes[1].imshow(colorize_labels(result.labels), origin="upper")
        axes[1].set_title("vertical-free split projected to navigation-free; merges=%d" % len(annotation.merge_groups))
        pred_background = _load_or_render_background(annotation.navigation_png, snapshot)
        if show_prediction:
            axes[2].imshow(_render_prediction_on_navigation(pred_background, snapshot.final_room_label_map), origin="upper")
            axes[2].set_title("prediction overlay on navigation map")
        else:
            axes[2].imshow(pred_background, origin="upper")
            axes[2].set_title("navigation map reference")
        _apply_view_limits(axes, view_limits)
        fig.canvas.draw_idle()

    def save(status: str) -> None:
        nonlocal annotation, dirty
        result = compute_preview()
        gt_dir_ep = Path(gt_dir) / annotation.episode_uid
        gt_dir_ep.mkdir(parents=True, exist_ok=True)
        gt_label = gt_dir_ep / "last_step.gt_labels.npy"
        gt_png = gt_dir_ep / "last_step.gt_overlay.png"
        gt_meta = gt_dir_ep / "last_step.gt_metadata.json"
        np.save(gt_label, result.labels.astype(np.int32))
        from .common import write_json_atomic
        from .visualization import save_label_overlay

        save_label_overlay(
            gt_png,
            labels=result.labels,
            domain=snapshot.eval_domain,
            obstacle=snapshot.obstacle_mask,
            unknown=snapshot.unknown_mask,
            split_lines=[line.to_dict() for line in annotation.split_lines],
            title="approved GT" if status == "approved" else "draft GT",
        )
        metadata = {
            **result.metadata,
            "gt_label_npy": str(gt_label),
            "gt_label_png": str(gt_png),
            "gt_metadata_json": str(gt_meta),
            "annotation_path": str(annotation_path),
            "annotation_review_status": str(status),
            "annotation_snapshot_sha256": str(annotation.snapshot_sha256),
            "segmentation_domain_key": str(snapshot.segmentation_domain_key),
            "output_domain_key": str(snapshot.domain_key),
            "navigation_domain_key": str(snapshot.domain_key),
        }
        write_json_atomic(gt_meta, metadata)
        review = approved_review() if status == "approved" else ReviewState(status="draft", notes=annotation.review.notes)
        generated = {
            "gt_label_npy": str(gt_label),
            "gt_label_png": str(gt_png),
            "gt_metadata_json": str(gt_meta),
            "room_count": int(result.metadata["room_count"]),
            "domain_pixels": int(result.metadata["domain_pixels"]),
            "unlabeled_domain_pixels": int(result.metadata["unlabeled_domain_pixels"]),
        }
        annotation = replace(annotation, generated_gt=generated, review=review)
        save_annotation_atomic(annotation, annotation_path)
        dirty = False
        entry_status[current_episode_uid] = str(status)
        render_scene_sidebar()
        set_status("saved %s" % status)

    def undo_line() -> None:
        nonlocal annotation, dirty
        annotation = replace(annotation, split_lines=tuple(annotation.split_lines[:-1]))
        dirty = True
        set_status("undo")
        render()

    def approve_and_close() -> None:
        nonlocal navigation_result
        save("approved")
        navigation_result = AnnotationAppResult(action="next")
        set_status("approved; closing")
        fig.canvas.draw_idle()
        plt.close(fig)

    def reset_view() -> None:
        _reset_zoom(axes, snapshot.shape)
        set_status("reset view")

    def zoom_in() -> None:
        _zoom_axes(axes, factor=0.75)
        set_status("zoom in")

    def zoom_out() -> None:
        _zoom_axes(axes, factor=1.25)
        set_status("zoom out")

    def _set_click_mode(mode: str) -> None:
        nonlocal click_mode, merge_points
        click_mode = mode
        merge_points = []
        set_status("click target")
        render()

    def _set_line_kind(kind: str) -> None:
        nonlocal line_kind, click_mode, merge_points, current_points
        line_kind = kind
        click_mode = "line"
        merge_points = []
        current_points = []
        set_status("%s mode" % _mode_text("line", line_kind))
        render()

    def _clear_merges() -> None:
        nonlocal annotation, merge_points, click_mode, dirty
        annotation = replace(annotation, merge_groups=tuple())
        dirty = True
        merge_points = []
        click_mode = "line"
        set_status("cleared merges")
        render()

    def _toggle_source_view() -> None:
        nonlocal current_source_view
        current_source_view = "segmentation" if current_source_view == "navigation" else "navigation"
        set_status("background %s" % current_source_view)
        render()

    def on_click(event) -> None:
        nonlocal annotation, current_points, merge_points, click_mode, dirty
        _force_keyboard_focus(fig)
        if _mouse_button_number(getattr(event, "button", None)) == 2:
            view_limits = _capture_view_limits(axes)
            if (
                event.inaxes in set(axes)
                and view_limits is not None
                and event.x is not None
                and event.y is not None
            ):
                inverse = event.inaxes.transData.frozen().inverted()
                press_data = inverse.transform((float(event.x), float(event.y)))
                middle_pan.clear()
                middle_pan.update(
                    {
                        "active": True,
                        "inverse": inverse,
                        "press_data": (float(press_data[0]), float(press_data[1])),
                        "view_limits": view_limits,
                    }
                )
            return
        if _mouse_button_number(getattr(event, "button", None)) != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        row = int(round(event.ydata))
        col = int(round(event.xdata))
        if not (0 <= row < snapshot.shape[0] and 0 <= col < snapshot.shape[1]):
            return
        if click_mode == "delete_line":
            idx = _nearest_line_index(annotation.split_lines, (row, col), max_distance=max(4.0, float(current_width) * 2.0))
            if idx is not None:
                lines = list(annotation.split_lines)
                lines.pop(idx)
                annotation = replace(annotation, split_lines=tuple(lines))
                dirty = True
                set_status("deleted line")
            else:
                set_status("no nearby line")
            click_mode = "line"
            render()
            return
        if click_mode == "merge":
            if event.inaxes not in {axes[1], axes[0]}:
                return
            result = compute_preview()
            label = int(result.labels[row, col])
            if label <= 0:
                render()
                return
            merge_points.append(label)
            if len(merge_points) == 2:
                ids = tuple(sorted(set(int(v) for v in merge_points if int(v) > 0)))
                if len(ids) >= 2:
                    group_id = "merge_%04d" % (len(annotation.merge_groups) + 1)
                    group = MergeGroup(id=group_id, component_ids=ids)
                    annotation = replace(annotation, merge_groups=tuple(list(annotation.merge_groups) + [group]))
                    dirty = True
                    set_status("merged %s" % (",".join(str(v) for v in ids)))
                merge_points = []
                click_mode = "line"
            render()
            return
        if event.inaxes != axes[0]:
            return
        current_points.append((row, col))
        if len(current_points) == 2:
            idx = len(annotation.split_lines) + 1
            line = LineAnnotation(
                id="line_%04d" % idx,
                p0_rc=current_points[0],
                p1_rc=current_points[1],
                width_cells=current_width,
                kind=line_kind,
            )
            annotation = replace(annotation, split_lines=tuple(list(annotation.split_lines) + [line]))
            dirty = True
            current_points = []
            set_status("added %s" % _mode_text("line", line_kind))
        render()

    def on_motion(event) -> None:
        if not bool(middle_pan.get("active", False)):
            return
        if event.x is None or event.y is None:
            return
        inverse = middle_pan.get("inverse")
        press_data = middle_pan.get("press_data")
        view_limits = middle_pan.get("view_limits")
        if inverse is None or press_data is None or view_limits is None:
            return
        current_data = inverse.transform((float(event.x), float(event.y)))
        dx = float(press_data[0]) - float(current_data[0])
        dy = float(press_data[1]) - float(current_data[1])
        _pan_axes_from_limits(
            axes,
            view_limits,
            dx=dx,
            dy=dy,
            shape=snapshot.shape,
        )

    def on_release(event) -> None:
        if _mouse_button_number(getattr(event, "button", None)) != 2:
            return
        was_active = bool(middle_pan.get("active", False))
        middle_pan.clear()
        middle_pan["active"] = False
        if was_active:
            set_status("panned")

    def on_key(event) -> None:
        nonlocal annotation, current_points, current_width, click_mode, merge_points, line_kind, show_prediction, show_obstacle, show_unknown, dirty
        _force_keyboard_focus(fig)
        key = str(event.key or "")
        if key == "escape":
            current_points = []
            merge_points = []
            click_mode = "line"
            set_status("cancel")
        elif key in {"u", "ctrl+z", "cmd+z", "control+z"}:
            undo_line()
            return
        elif key == "d":
            click_mode = "delete_line"
            set_status("click line to delete")
        elif key == "[":
            current_width = max(1, current_width - 1)
            set_status("line width %d" % current_width)
        elif key == "]":
            current_width += 1
            set_status("line width %d" % current_width)
        elif key in {"+", "="}:
            zoom_in()
            return
        elif key in {"-", "_"}:
            zoom_out()
            return
        elif key == "0":
            reset_view()
            return
        elif key == "pageup":
            change_sidebar_page(-1)
            return
        elif key == "pagedown":
            change_sidebar_page(1)
            return
        elif key == "r":
            pass
        elif key == "w":
            line_kind = "wall_completion"
            click_mode = "line"
            current_points = []
            set_status("wall completion mode")
        elif key == "v":
            line_kind = "separator"
            click_mode = "line"
            current_points = []
            set_status("split mode")
        elif key == "m":
            click_mode = "merge"
            merge_points = []
            set_status("click two rooms")
        elif key == "M":
            annotation = replace(annotation, merge_groups=tuple())
            dirty = True
            merge_points = []
            click_mode = "line"
            set_status("cleared merges")
        elif key == "b":
            _toggle_source_view()
            return
        elif key == "p":
            show_prediction = not show_prediction
            set_status("prediction %s" % ("on" if show_prediction else "off"))
        elif key == "o":
            show_obstacle = not show_obstacle
            set_status("obstacle %s" % ("on" if show_obstacle else "off"))
        elif key == "k":
            show_unknown = not show_unknown
            set_status("unknown %s" % ("on" if show_unknown else "off"))
        elif key == "s":
            save("draft")
        elif key in {"enter", "return"}:
            approve_and_close()
            return
        elif key == "q":
            plt.close(fig)
            return
        render()

    def render_scene_sidebar() -> None:
        nonlocal sidebar_page
        if scene_sidebar_title is None:
            return
        sidebar_page = max(0, min(sidebar_page_count - 1, int(sidebar_page)))
        approved_count = sum(
            status == "approved" for status in entry_status.values()
        )
        scene_sidebar_title.set_text(
            "SCENES  %d/%d APPROVED  |  PAGE %d/%d"
            % (
                approved_count,
                len(entries),
                sidebar_page + 1,
                sidebar_page_count,
            )
        )
        colors = {
            "approved": ("#b7e4c7", "#95d5b2"),
            "draft": ("#ffe69c", "#ffda6a"),
            "todo": ("#e9ecef", "#dee2e6"),
        }
        prefixes = {"approved": "DONE", "draft": "DRAFT", "todo": "TODO"}
        page_start = sidebar_page * sidebar_page_size
        for slot, button in enumerate(scene_button_slots):
            entry_index = page_start + slot
            if entry_index >= len(entries):
                scene_button_uids[slot] = None
                button.ax.set_visible(False)
                continue
            button.ax.set_visible(True)
            item = entries[entry_index]
            uid = str(item["episode_uid"])
            scene_button_uids[slot] = uid
            status = entry_status.get(uid, "todo")
            if status not in colors:
                status = "draft"
            button.label.set_text(
                "%s  %s" % (prefixes[status], str(item.get("label", uid)))
            )
            color, hover = colors[status]
            button.color = color
            button.hovercolor = hover
            button.ax.set_facecolor(color)
            is_current = uid == current_episode_uid
            for spine in button.ax.spines.values():
                spine.set_color("#1261a0" if is_current else "#adb5bd")
                spine.set_linewidth(2.6 if is_current else 0.8)
        fig.canvas.draw_idle()

    def change_sidebar_page(delta: int) -> None:
        nonlocal sidebar_page
        target = max(
            0,
            min(sidebar_page_count - 1, sidebar_page + int(delta)),
        )
        if target == sidebar_page:
            return
        sidebar_page = target
        render_scene_sidebar()

    def request_scene_slot(slot: int) -> None:
        if not 0 <= int(slot) < len(scene_button_uids):
            return
        selected_uid = scene_button_uids[int(slot)]
        if selected_uid is not None:
            request_scene_switch(selected_uid)

    def on_scroll(event) -> None:
        over_sidebar = (
            getattr(event, "inaxes", None) in scene_sidebar_axes
            or (
                getattr(event, "x", None) is not None
                and float(event.x) <= float(fig.bbox.width) * 0.205
            )
        )
        if sidebar_enabled and over_sidebar:
            step = float(getattr(event, "step", 0.0) or 0.0)
            button = str(getattr(event, "button", "") or "").lower()
            change_sidebar_page(-1 if step > 0.0 or button == "up" else 1)
            return
        _on_scroll_zoom(event, axes, snapshot.shape)

    def request_scene_switch(episode_uid: str) -> None:
        nonlocal navigation_result
        selected_uid = str(episode_uid)
        if selected_uid == current_episode_uid:
            return
        if dirty:
            save("draft")
        navigation_result = AnnotationAppResult(
            action="switch",
            episode_uid=selected_uid,
        )
        plt.close(fig)

    if sidebar_enabled:
        from matplotlib.widgets import Button

        top = 0.875
        bottom = 0.115
        row_height = (top - bottom) / sidebar_page_size
        scene_sidebar_title = fig.text(
            0.018,
            0.925,
            "",
            fontsize=10,
            fontweight="bold",
            color="#202124",
        )
        for slot in range(sidebar_page_size):
            y_pos = top - float(slot + 1) * row_height
            axis = fig.add_axes((0.015, y_pos, 0.18, row_height * 0.82))
            button = Button(axis, "")
            button.label.set_fontsize(8)
            button.on_clicked(
                lambda _event, selected_slot=slot: request_scene_slot(
                    selected_slot
                )
            )
            scene_button_slots.append(button)
            scene_button_uids.append(None)
            scene_sidebar_axes.add(axis)
            button_refs.append(button)
        previous_axis = fig.add_axes((0.015, 0.065, 0.086, 0.036))
        next_axis = fig.add_axes((0.109, 0.065, 0.086, 0.036))
        previous_button = Button(previous_axis, "Prev")
        next_button = Button(next_axis, "Next")
        previous_button.label.set_fontsize(9)
        next_button.label.set_fontsize(9)
        previous_button.on_clicked(lambda _event: change_sidebar_page(-1))
        next_button.on_clicked(lambda _event: change_sidebar_page(1))
        scene_sidebar_axes.update((previous_axis, next_axis))
        button_refs.extend((previous_button, next_button))
        render_scene_sidebar()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    button_refs.extend(
        _install_buttons(
            fig,
            {
                "Undo": undo_line,
                "Del": lambda: _set_click_mode("delete_line"),
                "Wall": lambda: _set_line_kind("wall_completion"),
                "Split": lambda: _set_line_kind("separator"),
                "Merge": lambda: _set_click_mode("merge"),
                "Clear": _clear_merges,
                "Bg": _toggle_source_view,
                "Draft": lambda: save("draft"),
                "Approve": approve_and_close,
                "+": zoom_in,
                "-": zoom_out,
                "Reset": reset_view,
            },
            left=0.215 if sidebar_enabled else 0.02,
            right=0.99,
        )
    )

    def keep_button_refs_alive() -> list:
        return button_refs

    keep_button_refs_alive()
    render()
    _force_keyboard_focus(fig)
    plt.show()
    return navigation_result


def _load_or_render_background(navigation_png: str | None, snapshot: SnapshotArrays) -> np.ndarray:
    # ``*.navigation_room_masks.png`` is a colored prediction artifact, not a
    # clean navigation source.  Never expose it in a GT annotation view.
    _ = navigation_png
    base = np.zeros(snapshot.shape + (3,), dtype=np.uint8)
    base[:] = (20, 22, 28)
    navigation = (
        np.asarray(snapshot.navigation_free_room_domain, dtype=bool)
        if snapshot.navigation_free_room_domain is not None
        else np.asarray(snapshot.eval_domain, dtype=bool)
    )
    base[navigation] = (190, 196, 204)
    if snapshot.unknown_mask is not None:
        base[np.asarray(snapshot.unknown_mask, dtype=bool)] = (60, 60, 66)
    if snapshot.obstacle_mask is not None:
        base[np.asarray(snapshot.obstacle_mask, dtype=bool)] = (0, 0, 0)
    return base


def _infer_navigation_png(snapshot_path: Path) -> Path | None:
    stem = Path(snapshot_path).with_suffix(".navigation_room_masks.png").name
    for path in (Path(snapshot_path), Path(snapshot_path).resolve()):
        parts = path.parts
        if "postprocessed_voronoi_width_jump_no_corner_snap" not in parts:
            continue
        idx = parts.index("postprocessed_voronoi_width_jump_no_corner_snap")
        if idx + 1 >= len(parts):
            continue
        scene = parts[idx + 1]
        root = Path(*parts[:idx])
        candidate = root / "stateless_replay" / scene / "roomseg_snapshots" / stem
        if candidate.exists():
            return candidate
    return None


def _render_source_background(*, snapshot: SnapshotArrays, navigation_png: str | None, source_view: str) -> np.ndarray:
    if str(source_view) == "navigation":
        return _load_or_render_background(navigation_png, snapshot)
    return _render_segmentation_background(snapshot)


def _render_prediction_on_navigation(background: np.ndarray, pred: np.ndarray) -> np.ndarray:
    base = np.asarray(background, dtype=np.float32).copy()
    labels = np.asarray(pred, dtype=np.int32)
    colors = np.asarray(colorize_labels(labels, background=(0, 0, 0)), dtype=np.float32)
    mask = labels > 0
    alpha = 0.58
    base[mask] = (1.0 - alpha) * base[mask] + alpha * colors[mask]
    return np.clip(base, 0, 255).astype(np.uint8)


def _render_segmentation_background(snapshot: SnapshotArrays) -> np.ndarray:
    base = np.zeros(snapshot.shape + (3,), dtype=np.uint8)
    base[:] = (255, 255, 255)
    if snapshot.unknown_mask is not None:
        base[np.asarray(snapshot.unknown_mask, dtype=bool)] = (230, 230, 230)
    if snapshot.obstacle_mask is not None:
        base[np.asarray(snapshot.obstacle_mask, dtype=bool)] = (40, 40, 40)
    base[np.asarray(snapshot.eval_domain, dtype=bool)] = (210, 210, 210)
    base[np.asarray(snapshot.segmentation_domain, dtype=bool)] = (172, 220, 245)
    return base


def _render_door_seed_overlay(
    snapshot: SnapshotArrays,
) -> tuple[np.ndarray, dict[str, int]]:
    raw = snapshot.door_seed_raw_mask
    if raw is None:
        raw_mask = np.zeros(snapshot.shape, dtype=bool)
    else:
        raw_mask = np.asarray(raw, dtype=bool)
    rejected = snapshot.door_seed_reject_mask
    rejected_mask = (
        np.asarray(rejected, dtype=bool) & raw_mask
        if rejected is not None
        else np.zeros(snapshot.shape, dtype=bool)
    )
    kept_mask = raw_mask & ~rejected_mask
    overlay = np.zeros(snapshot.shape + (4,), dtype=np.float32)
    overlay[kept_mask] = (0.0, 0.898, 1.0, 0.88)
    overlay[rejected_mask] = (1.0, 0.188, 0.188, 0.92)
    return overlay, {
        "raw": int(np.count_nonzero(raw_mask)),
        "kept": int(np.count_nonzero(kept_mask)),
        "rejected": int(np.count_nonzero(rejected_mask)),
    }


def _install_help_text(fig) -> None:
    help_text = (
        "Wall/red mode (w or Wall): complete broken wall | Split/purple mode (v or Split): room separator | "
        "Left click x2: add line | Esc: cancel | Ctrl+Z/u: undo | d then click: delete nearest line | "
        "[ / ]: line width | wheel or + / -: zoom | middle-button drag: pan | 0: reset view | "
        "wheel over scene list or PageUp/PageDown: change scene page | "
        "m then click two rooms: merge | M: clear merges | b: switch background | p: prediction overlay | o/k: toggle obstacle/unknown | "
        "s: save draft | Enter: approve GT | q: quit"
    )
    fig.text(
        0.5,
        0.012,
        help_text,
        ha="center",
        va="bottom",
        fontsize=14,
        color="black",
        wrap=True,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "boxstyle": "round,pad=0.35"},
    )


def _install_buttons(
    fig,
    callbacks: Mapping[str, object],
    *,
    left: float = 0.02,
    right: float = 0.98,
) -> list:
    from matplotlib.widgets import Button

    labels = list(callbacks.keys())
    bottom = 0.085
    gap = 0.006
    available = max(0.1, float(right) - float(left))
    width = min(
        0.078,
        (available - gap * max(0, len(labels) - 1)) / max(1, len(labels)),
    )
    refs = []
    for idx, label in enumerate(labels):
        ax = fig.add_axes([left + idx * (width + gap), bottom, width, 0.045])
        ax.set_facecolor("white")
        button = Button(ax, label)
        button.label.set_fontsize(12)
        button.on_clicked(lambda _event, cb=callbacks[label]: cb())
        refs.append(button)
    return refs


def _mode_text(click_mode: str, line_kind: str) -> str:
    if click_mode == "delete_line":
        return "delete"
    if click_mode == "merge":
        return "merge"
    if line_kind == "wall_completion":
        return "wall/red"
    return "split/purple"


def _draw_mode_badge(ax, line_kind: str) -> None:
    if line_kind == "wall_completion":
        label = "DRAWING: WALL COMPLETION / RED"
        color = "#d7191c"
    else:
        label = "DRAWING: SPLIT SEPARATOR / PURPLE"
        color = "#8b2be2"
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
        color="white",
        bbox={"facecolor": color, "edgecolor": "white", "boxstyle": "round,pad=0.35", "alpha": 0.92},
    )


def _force_keyboard_focus(fig) -> None:
    canvas = fig.canvas
    try:
        manager = getattr(canvas, "manager", None)
        window = getattr(manager, "window", None)
        if window is not None:
            if hasattr(window, "raise_"):
                window.raise_()
            if hasattr(window, "activateWindow"):
                window.activateWindow()
    except Exception:
        pass
    try:
        if hasattr(canvas, "setFocusPolicy"):
            canvas.setFocusPolicy(_qt_strong_focus_policy())
        if hasattr(canvas, "setFocus"):
            canvas.setFocus()
        if hasattr(canvas, "SetFocus"):
            canvas.SetFocus()
        if hasattr(canvas, "focus_set"):
            canvas.focus_set()
        if hasattr(canvas, "get_tk_widget"):
            canvas.get_tk_widget().focus_set()
    except Exception:
        pass


def _qt_strong_focus_policy():
    import importlib

    for module_name in ("PySide6.QtCore", "PyQt6.QtCore", "PySide2.QtCore", "PyQt5.QtCore"):
        try:
            qtcore = importlib.import_module(module_name)
        except Exception:
            continue
        qt = getattr(qtcore, "Qt", None)
        if qt is None:
            continue
        focus_policy = getattr(qt, "FocusPolicy", None)
        if focus_policy is not None and hasattr(focus_policy, "StrongFocus"):
            return focus_policy.StrongFocus
        if hasattr(qt, "StrongFocus"):
            return qt.StrongFocus
    return 11


def _nearest_line_index(lines: tuple[LineAnnotation, ...], point_rc: tuple[int, int], *, max_distance: float) -> int | None:
    best_idx: int | None = None
    best_dist = float(max_distance)
    p = np.asarray([float(point_rc[0]), float(point_rc[1])], dtype=float)
    for idx, line in enumerate(lines):
        a = np.asarray([float(line.p0_rc[0]), float(line.p0_rc[1])], dtype=float)
        b = np.asarray([float(line.p1_rc[0]), float(line.p1_rc[1])], dtype=float)
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-9:
            dist = float(np.linalg.norm(p - a))
        else:
            t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
            dist = float(np.linalg.norm(p - (a + t * ab)))
        if dist <= best_dist:
            best_dist = dist
            best_idx = int(idx)
    return best_idx


def _on_scroll_zoom(event, axes, shape: tuple[int, int]) -> None:
    if event.inaxes not in set(axes) or event.xdata is None or event.ydata is None:
        return
    factor = 0.80 if event.button == "up" else 1.25
    _zoom_axes([event.inaxes], factor=factor, center=(float(event.xdata), float(event.ydata)), shape=shape)


def _mouse_button_number(button: object) -> int | None:
    value = getattr(button, "value", button)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _pan_axes_from_limits(
    axes,
    limits: Sequence[tuple[float, float, float, float]],
    *,
    dx: float,
    dy: float,
    shape: tuple[int, int],
) -> None:
    h, w = int(shape[0]), int(shape[1])
    for ax, (x0, x1, y0, y1) in zip(axes, limits):
        ax.set_xlim(
            _translated_clamped_interval(
                x0,
                x1,
                float(dx),
                -0.5,
                float(w) - 0.5,
            )
        )
        ax.set_ylim(
            _translated_clamped_interval(
                y0,
                y1,
                float(dy),
                -0.5,
                float(h) - 0.5,
            )
        )
    if len(axes) > 0:
        axes[0].figure.canvas.draw_idle()


def _translated_clamped_interval(
    a: float,
    b: float,
    delta: float,
    bound_lo: float,
    bound_hi: float,
) -> tuple[float, float]:
    descending = float(a) > float(b)
    lo = min(float(a), float(b)) + float(delta)
    hi = max(float(a), float(b)) + float(delta)
    width = hi - lo
    bound_width = float(bound_hi) - float(bound_lo)
    if width >= bound_width:
        lo, hi = float(bound_lo), float(bound_hi)
    else:
        if lo < float(bound_lo):
            hi += float(bound_lo) - lo
            lo = float(bound_lo)
        if hi > float(bound_hi):
            lo -= hi - float(bound_hi)
            hi = float(bound_hi)
    return (hi, lo) if descending else (lo, hi)


def _capture_view_limits(axes) -> list[tuple[float, float, float, float]] | None:
    if len(axes) == 0 or not getattr(axes[0], "images", None):
        return None
    return [(float(ax.get_xlim()[0]), float(ax.get_xlim()[1]), float(ax.get_ylim()[0]), float(ax.get_ylim()[1])) for ax in axes]


def _apply_view_limits(axes, limits: list[tuple[float, float, float, float]] | None) -> None:
    if limits is None:
        return
    for ax, (x0, x1, y0, y1) in zip(axes, limits):
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)


def _zoom_axes(axes, *, factor: float, center: tuple[float, float] | None = None, shape: tuple[int, int] | None = None) -> None:
    for ax in axes:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        cx = (x0 + x1) * 0.5 if center is None else float(center[0])
        cy = (y0 + y1) * 0.5 if center is None else float(center[1])
        width = abs(x1 - x0) * float(factor)
        height = abs(y1 - y0) * float(factor)
        nx0 = cx - width * 0.5
        nx1 = cx + width * 0.5
        ny0 = cy - height * 0.5
        ny1 = cy + height * 0.5
        if shape is not None:
            h, w = int(shape[0]), int(shape[1])
            nx0, nx1 = _clamp_interval(nx0, nx1, 0.0, max(1.0, float(w - 1)))
            ny0, ny1 = _clamp_interval(ny0, ny1, 0.0, max(1.0, float(h - 1)))
        ax.set_xlim(nx0, nx1)
        ax.set_ylim(ny1, ny0)
    axes[0].figure.canvas.draw_idle()


def _reset_zoom(axes, shape: tuple[int, int]) -> None:
    h, w = int(shape[0]), int(shape[1])
    for ax in axes:
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
    axes[0].figure.canvas.draw_idle()


def _clamp_interval(a: float, b: float, lo: float, hi: float) -> tuple[float, float]:
    width = max(1.0, abs(float(b) - float(a)))
    if width >= hi - lo:
        return lo - 0.5, hi + 0.5
    start = min(float(a), float(b))
    end = max(float(a), float(b))
    if start < lo:
        end += lo - start
        start = lo
    if end > hi:
        start -= end - hi
        end = hi
    return start, end
