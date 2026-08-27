from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from voxroom_online.isaac_runtime.door_seed_learning.annotation_schema import (
    AnnotationReview,
    ManualFinalSeedAnnotation,
    SeedLabel,
    approve_annotation,
    create_initial_annotation,
    load_annotation,
    now_iso,
    save_annotation,
    validate_annotation,
)
from voxroom_online.isaac_runtime.door_seed_learning.schema import load_snapshot, read_json
from voxroom_online.isaac_runtime.mapping.coordinate_transform import MapInfo, grid_to_world_xy


_SEED_POSITIVE_RGBA = np.asarray((21, 148, 71, 255), dtype=np.float32) / 255.0
_SEED_SELECTED_EDGE_RGBA = np.ones(4, dtype=np.float32)
_SEED_SOURCE_RGBA = np.asarray(
    (
        (90, 90, 90, 255),
        (35, 120, 235, 255),
        (145, 70, 210, 255),
        (35, 120, 235, 255),
    ),
    dtype=np.float32,
) / 255.0
_SEED_SOURCE_NAMES = {
    0: "unknown",
    1: "VoxRoom",
    2: "TVARS",
    3: "VoxRoom",
}
_COLUMN_RGBA = np.asarray(
    (
        (188, 188, 188, 255),
        (244, 244, 244, 255),
        (41, 41, 41, 255),
        (143, 90, 168, 255),
    ),
    dtype=np.float32,
) / 255.0


@dataclass(frozen=True)
class AnnotationSceneProgress:
    scene_dir: Path
    scene_id: str
    labeled: int
    total: int
    status: str


def annotation_scene_progress(scene_dir: str | Path) -> AnnotationSceneProgress:
    scene = Path(scene_dir)
    manifest = read_json(scene / "manifest.json")
    finals = [item for item in manifest.get("snapshots", []) if bool(item.get("is_final", False))]
    if len(finals) != 1:
        raise ValueError("collection scene must have exactly one final snapshot")
    total = int(finals[0].get("raw_seed_count", 0))
    annotation_path = scene / "annotations" / "manual_final_seed_labels.json"
    if annotation_path.exists():
        annotation = load_annotation(annotation_path)
        if annotation.scene_uid != str(manifest["scene_uid"]):
            raise ValueError("manual annotation scene_uid mismatch")
        if annotation.final_snapshot_path != str(finals[0]["path"]):
            raise ValueError("manual annotation points to a different final snapshot")
        labeled = len(annotation.labels)
        if labeled > total:
            raise ValueError("manual annotation contains more labels than final raw seeds")
        approved = str(annotation.review.status) == "approved"
    else:
        labeled = 0
        approved = False
    status = "done" if approved and labeled == total else ("partial" if labeled else "todo")
    return AnnotationSceneProgress(
        scene_dir=scene,
        scene_id=str(manifest["scene_id"]),
        labeled=int(labeled),
        total=int(total),
        status=status,
    )


def discover_annotation_scenes(collection_root: str | Path) -> list[AnnotationSceneProgress]:
    root = Path(collection_root).expanduser()
    progress = [annotation_scene_progress(path.parent) for path in root.glob("*/manifest.json")]
    return sorted(progress, key=lambda item: (item.scene_id, item.scene_dir.name))


def _session_progress(session: "AnnotationSession") -> AnnotationSceneProgress:
    labeled, total = session.progress()
    approved = str(session.annotation.review.status) == "approved"
    status = "done" if approved and labeled == total else ("partial" if labeled else "todo")
    return AnnotationSceneProgress(
        scene_dir=session.scene_dir,
        scene_id=str(session.scene_id),
        labeled=int(labeled),
        total=int(total),
        status=status,
    )


def _selected_voxel_center_column(
    voxel_patches: np.ndarray,
    selected_index: int,
) -> np.ndarray:
    patches = np.asarray(voxel_patches)
    if patches.ndim != 4:
        raise ValueError("voxel patches must have shape [N,Z,H,W]")
    height, width = int(patches.shape[2]), int(patches.shape[3])
    if height != width or height <= 0 or height % 2 != 1:
        raise ValueError("voxel patches must have an odd square spatial shape")
    index = int(selected_index)
    if index < 0 or index >= int(patches.shape[0]):
        raise IndexError("selected voxel patch index is outside the batch")
    center = height // 2
    return np.asarray(patches[index, :, center, center])


@dataclass(frozen=True)
class _AnnotationUndoItem:
    row: int
    col: int
    previous_label: SeedLabel | None


@dataclass(frozen=True)
class _AnnotationUndoEntry:
    items: tuple[_AnnotationUndoItem, ...]
    previous_review: AnnotationReview


@dataclass
class AnnotationSession:
    scene_dir: Path
    scene_id: str
    annotation_path: Path
    annotation: ManualFinalSeedAnnotation
    seed_rc: np.ndarray
    seed_source_ids: np.ndarray
    voxel_patches: np.ndarray
    z_centers_m: np.ndarray
    vertical_class_map: np.ndarray
    nav_class_map: np.ndarray
    map_info: MapInfo
    selected_index: int = 0
    background: str = "vertical"
    undo_history: list[_AnnotationUndoEntry] = field(default_factory=list)

    @classmethod
    def open(cls, scene_dir: str | Path) -> "AnnotationSession":
        scene = Path(scene_dir)
        manifest = read_json(scene / "manifest.json")
        finals = [item for item in manifest.get("snapshots", []) if bool(item.get("is_final", False))]
        if len(finals) != 1:
            raise ValueError("collection scene must have exactly one final snapshot")
        snapshot_path = scene / str(finals[0]["path"])
        arrays = load_snapshot(snapshot_path)
        annotation_path = scene / "annotations" / "manual_final_seed_labels.json"
        annotation = load_annotation(annotation_path) if annotation_path.exists() else create_initial_annotation(scene)
        validate_annotation(annotation, scene, snapshot_arrays=arrays)
        map_info = MapInfo.from_dict(dict(manifest["map"]["origin_or_transform"]))
        seed_rc = np.asarray(arrays["raw_seed_rc"], dtype=np.int32)
        source_map = arrays.get("raw_seed_source_id_map_xy")
        if source_map is None:
            default_source_id = (
                1
                if str(dict(manifest.get("collection", {}) or {}).get("raw_seed_source", ""))
                == "voxroom"
                else 0
            )
            seed_source_ids = np.full(len(seed_rc), default_source_id, dtype=np.uint8)
        else:
            source_map = np.asarray(source_map, dtype=np.uint8)
            if source_map.shape != tuple(arrays["raw_seed_mask_xy"].shape):
                raise ValueError("raw seed source ID map shape does not match the raw seed map")
            seed_source_ids = source_map[seed_rc[:, 0], seed_rc[:, 1]].astype(
                np.uint8,
                copy=False,
            )
            if np.any(seed_source_ids > 3):
                raise ValueError("raw seed source ID must be 0, 1, 2, or 3")
        session = cls(
            scene_dir=scene,
            scene_id=str(manifest["scene_id"]),
            annotation_path=annotation_path,
            annotation=annotation,
            seed_rc=seed_rc,
            seed_source_ids=seed_source_ids,
            voxel_patches=np.asarray(arrays["raw_seed_voxel_state_nzyx"], dtype=np.uint8),
            z_centers_m=np.asarray(arrays["z_centers_m"], dtype=np.float32),
            vertical_class_map=np.asarray(arrays["vertical_class_map_xy"], dtype=np.uint8),
            nav_class_map=np.asarray(arrays["nav_class_map_xy"], dtype=np.uint8),
            map_info=map_info,
        )
        session.selected_index = -1
        if not session.select_next_unlabeled(wrap=False) and len(session.seed_rc):
            session.selected_index = 0
        return session

    def prefill_unlabeled_as_reject(self) -> int:
        """Persist label 0 for every raw seed that has no existing manual label."""
        labels = {(item.row, item.col): item for item in self.annotation.labels}
        missing = [
            (int(row), int(col))
            for row, col in self.seed_rc
            if (int(row), int(col)) not in labels
        ]
        if not missing:
            return 0
        updated_at = now_iso()
        for row, col in missing:
            labels[(row, col)] = SeedLabel(
                row=row,
                col=col,
                label=0,
                updated_at=updated_at,
            )
        self.annotation = replace(
            self.annotation,
            labels=tuple(sorted(labels.values(), key=lambda item: (item.row, item.col))),
            review=replace(
                self.annotation.review,
                status="draft",
                approved_by=None,
                approved_at=None,
            ),
        )
        save_annotation(self.annotation, self.annotation_path)
        if len(self.seed_rc) and self.selected_index < 0:
            self.selected_index = 0
        return len(missing)

    def label_selected(self, label: int) -> None:
        if len(self.seed_rc) == 0:
            return
        self.label_indices((self.selected_index,), int(label))

    def label_indices(self, indices: np.ndarray | list[int] | tuple[int, ...], label: int) -> int:
        if int(label) not in {0, 1}:
            raise ValueError("manual seed label must be 0 or 1")
        selected = sorted({int(index) for index in indices if 0 <= int(index) < len(self.seed_rc)})
        if not selected:
            return 0
        self._remember_indices(selected)
        labels = {(item.row, item.col): item for item in self.annotation.labels}
        updated_at = now_iso()
        for index in selected:
            row, col = (int(v) for v in self.seed_rc[index])
            labels[(row, col)] = SeedLabel(row=row, col=col, label=int(label), updated_at=updated_at)
        self.annotation = replace(
            self.annotation,
            labels=tuple(sorted(labels.values(), key=lambda item: (item.row, item.col))),
            review=replace(
                self.annotation.review,
                status="draft",
                approved_by=None,
                approved_at=None,
            ),
        )
        save_annotation(self.annotation, self.annotation_path)
        self.select_next_unlabeled(wrap=True)
        return len(selected)

    def remove_selected_label(self) -> None:
        if len(self.seed_rc) == 0:
            return
        selected = int(self.selected_index)
        row, col = (int(v) for v in self.seed_rc[selected])
        if (row, col) not in self.annotation.label_map():
            return
        self._remember_indices((selected,))
        self.annotation = self.annotation.without_label(row, col)
        save_annotation(self.annotation, self.annotation_path)

    def undo(self) -> bool:
        if not self.undo_history:
            return False
        edit = self.undo_history.pop()
        labels = {(item.row, item.col): item for item in self.annotation.labels}
        for item in edit.items:
            coordinate = (int(item.row), int(item.col))
            if item.previous_label is None:
                labels.pop(coordinate, None)
            else:
                labels[coordinate] = item.previous_label
        self.annotation = replace(
            self.annotation,
            labels=tuple(sorted(labels.values(), key=lambda item: (item.row, item.col))),
            review=edit.previous_review,
        )
        save_annotation(self.annotation, self.annotation_path)
        return True

    def _remember_indices(self, indices: list[int] | tuple[int, ...]) -> None:
        previous_labels = {(item.row, item.col): item for item in self.annotation.labels}
        items = []
        for index in indices:
            row, col = (int(v) for v in self.seed_rc[int(index)])
            items.append(
                _AnnotationUndoItem(
                    row=row,
                    col=col,
                    previous_label=previous_labels.get((row, col)),
                )
            )
        self.undo_history.append(
            _AnnotationUndoEntry(
                items=tuple(items),
                previous_review=self.annotation.review,
            )
        )

    def select_next_unlabeled(self, *, wrap: bool) -> bool:
        if len(self.seed_rc) == 0:
            self.selected_index = 0
            return False
        labels = self.annotation.label_map()
        order = list(range(self.selected_index + 1, len(self.seed_rc)))
        if wrap:
            order += list(range(0, self.selected_index + 1))
        for index in order:
            rc = tuple(int(v) for v in self.seed_rc[index])
            if rc not in labels:
                self.selected_index = int(index)
                return True
        return False

    def move(self, delta: int) -> None:
        if len(self.seed_rc):
            self.selected_index = (int(self.selected_index) + int(delta)) % len(self.seed_rc)

    def move_direction(self, direction: str) -> bool:
        selected = _directional_neighbor_index(self.seed_rc, self.selected_index, direction)
        if selected is None:
            return False
        self.selected_index = int(selected)
        return True

    def approve(self, approved_by: str) -> None:
        self.annotation = approve_annotation(self.annotation, self.scene_dir, approved_by=approved_by)
        save_annotation(self.annotation, self.annotation_path)

    def progress(self) -> tuple[int, int]:
        return len(self.annotation.labels), len(self.seed_rc)


def _seed_style_arrays(
    seed_rc: np.ndarray,
    labels: dict[tuple[int, int], int],
    selected_index: int,
    source_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates = np.asarray(seed_rc, dtype=np.int32)
    count = int(len(coordinates))
    if source_ids is None:
        source_ids = np.zeros(count, dtype=np.uint8)
    ids = np.asarray(source_ids, dtype=np.int64).reshape(-1)
    if len(ids) != count:
        raise ValueError("raw seed source ID count must match the raw seed count")
    facecolors = _seed_source_edgecolors(ids).copy()
    for index, (row, col) in enumerate(coordinates):
        label = labels.get((int(row), int(col)))
        if label == 1:
            facecolors[index] = _SEED_POSITIVE_RGBA
    edgecolors = facecolors.copy()
    sizes = np.full(count, 18.0, dtype=np.float32)
    if count:
        selected = int(selected_index) % count
        edgecolors[selected] = _SEED_SELECTED_EDGE_RGBA
        sizes[selected] = 46.0
    return facecolors, edgecolors, sizes


def _seed_source_edgecolors(source_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(source_ids, dtype=np.int64).reshape(-1)
    if np.any((ids < 0) | (ids >= len(_SEED_SOURCE_RGBA))):
        raise ValueError("raw seed source ID must be 0, 1, 2, or 3")
    return _SEED_SOURCE_RGBA[ids]


def _directional_neighbor_index(
    seed_rc: np.ndarray,
    selected_index: int,
    direction: str,
) -> int | None:
    coordinates = np.asarray(seed_rc, dtype=np.int32)
    count = int(len(coordinates))
    if count <= 1:
        return None
    current = int(selected_index) % count
    delta_row = coordinates[:, 0].astype(np.float32) - float(coordinates[current, 0])
    delta_col = coordinates[:, 1].astype(np.float32) - float(coordinates[current, 1])
    if direction == "left":
        primary = -delta_col
        lateral = np.abs(delta_row)
    elif direction == "right":
        primary = delta_col
        lateral = np.abs(delta_row)
    elif direction == "up":
        primary = -delta_row
        lateral = np.abs(delta_col)
    elif direction == "down":
        primary = delta_row
        lateral = np.abs(delta_col)
    else:
        raise ValueError("direction must be left, right, up, or down")
    candidates = np.flatnonzero(primary > 0.0)
    if not len(candidates):
        return None
    distance = np.hypot(delta_row[candidates], delta_col[candidates])
    score = distance + 2.0 * lateral[candidates]
    order = np.lexsort((candidates, distance, score))
    return int(candidates[int(order[0])])


class _AnnotationFigure:
    def __init__(
        self,
        session: AnnotationSession,
        *,
        annotator: str,
        scene_dirs: Sequence[str | Path] | None = None,
    ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.colors import ListedColormap
        from matplotlib.lines import Line2D
        from matplotlib.patches import Rectangle
        from matplotlib.widgets import Button

        self.session = session
        self.annotator = str(annotator)
        self.requested_scene_dir: Path | None = None
        discovered_dirs = [Path(path) for path in (scene_dirs or ())]
        if session.scene_dir not in discovered_dirs:
            discovered_dirs.append(session.scene_dir)
        self.scene_dirs = tuple(
            sorted(set(discovered_dirs), key=lambda path: annotation_scene_progress(path).scene_id)
        )
        self.scene_progress = {path: annotation_scene_progress(path) for path in self.scene_dirs}
        self.scene_buttons: dict[Path, object] = {}
        self.scene_sidebar_title = None
        self.batch_indices = np.empty(0, dtype=np.int32)
        self.box_mode = False
        self.drag_state: dict[str, object] = {
            "active": False,
            "x": 0.0,
            "y": 0.0,
            "xlim": None,
            "ylim": None,
        }
        self.box_drag_state: dict[str, object] = {
            "active": False,
            "x": 0.0,
            "y": 0.0,
        }

        sidebar_enabled = len(self.scene_dirs) > 1
        self.fig = plt.figure(figsize=(18, 10), facecolor="white")
        manager = getattr(self.fig.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title("VoxRoom DoorSeed Annotation")
        grid = self.fig.add_gridspec(
            1,
            2,
            width_ratios=(4.5, 1.2),
            left=0.215 if sidebar_enabled else 0.035,
            right=0.98,
            top=0.91,
            bottom=0.12,
            wspace=0.08,
        )
        self.map_ax = self.fig.add_subplot(grid[0, 0])
        self.column_ax = self.fig.add_subplot(grid[0, 1])

        map_palette = np.asarray(
            ((238, 238, 238), (255, 255, 255), (45, 45, 45)),
            dtype=np.uint8,
        )
        self.background_arrays = {
            "vertical": map_palette[self.session.vertical_class_map],
            "nav": map_palette[self.session.nav_class_map],
        }
        self.background_image = self.map_ax.imshow(
            self.background_arrays["vertical"],
            origin="upper",
            interpolation="nearest",
        )
        seed_xy = np.column_stack((self.session.seed_rc[:, 1], self.session.seed_rc[:, 0]))
        facecolors, _label_edgecolors, sizes = _seed_style_arrays(
            self.session.seed_rc,
            self.session.annotation.label_map(),
            self.session.selected_index,
            self.session.seed_source_ids,
        )
        self.seed_scatter = self.map_ax.scatter(
            seed_xy[:, 0] if len(seed_xy) else [],
            seed_xy[:, 1] if len(seed_xy) else [],
            s=sizes,
            facecolors=facecolors,
            edgecolors=facecolors,
            linewidths=0.5,
            zorder=3,
        )
        source_legend = [
            Line2D(
                (),
                (),
                marker="o",
                linestyle="none",
                markerfacecolor=_SEED_SOURCE_RGBA[source_id],
                markeredgecolor=_SEED_SOURCE_RGBA[source_id],
                markeredgewidth=0.5,
                markersize=8,
                label=_SEED_SOURCE_NAMES[source_id],
            )
            for source_id in (1, 2)
        ]
        accept_legend = [
            Line2D(
                (),
                (),
                marker="o",
                linestyle="none",
                markerfacecolor=color,
                markeredgecolor=color,
                markersize=7,
                label=label,
            )
            for label, color in (("YES / accept", _SEED_POSITIVE_RGBA),)
        ]
        self.map_ax.legend(
            handles=source_legend + accept_legend,
            title="Fill = source | Green = accept",
            loc="upper right",
            framealpha=0.92,
            fontsize=8,
            title_fontsize=8,
        )
        if len(seed_xy):
            selected_xy = seed_xy[[int(self.session.selected_index)]]
        else:
            selected_xy = np.empty((0, 2), dtype=np.float32)
        self.selected_outer_ring = self.map_ax.scatter(
            selected_xy[:, 0],
            selected_xy[:, 1],
            s=240.0,
            facecolors="none",
            edgecolors="#111111",
            linewidths=5.5,
            zorder=6,
        )
        self.selected_inner_ring = self.map_ax.scatter(
            selected_xy[:, 0],
            selected_xy[:, 1],
            s=240.0,
            facecolors="none",
            edgecolors="#ffd400",
            linewidths=2.5,
            zorder=7,
        )
        self.batch_scatter = self.map_ax.scatter(
            [],
            [],
            s=150.0,
            marker="s",
            facecolors="none",
            edgecolors="#ff2ea6",
            linewidths=2.5,
            zorder=5,
        )
        self.box_rectangle = Rectangle(
            (0.0, 0.0),
            0.0,
            0.0,
            facecolor=(1.0, 0.18, 0.65, 0.16),
            edgecolor="#8a0053",
            linewidth=1.5,
            linestyle="--",
            visible=False,
            zorder=8,
        )
        self.map_ax.add_patch(self.box_rectangle)
        self.map_ax.set_axis_off()

        z_centers = np.asarray(self.session.z_centers_m, dtype=np.float32)
        dz = float(np.median(np.diff(z_centers))) if len(z_centers) > 1 else 0.05
        if len(z_centers):
            z_min = float(z_centers[0] - 0.5 * dz)
            z_max = float(z_centers[-1] + 0.5 * dz)
            initial_column = np.zeros((len(z_centers), 1), dtype=np.uint8)
        else:
            z_min, z_max = 0.0, 1.0
            initial_column = np.zeros((1, 1), dtype=np.uint8)
        self.column_image = self.column_ax.imshow(
            initial_column,
            origin="lower",
            interpolation="nearest",
            aspect="auto",
            extent=(-0.4, 0.4, z_min, z_max),
            cmap=ListedColormap(_COLUMN_RGBA),
            vmin=-0.5,
            vmax=3.5,
        )
        z_edges = np.linspace(z_min, z_max, initial_column.shape[0] + 1, dtype=np.float32)
        self.column_grid = LineCollection(
            [[(-0.4, float(z_value)), (0.4, float(z_value))] for z_value in z_edges],
            colors="#555555",
            linewidths=0.25,
            zorder=2,
        )
        self.column_ax.add_collection(self.column_grid)
        self.column_ax.set_xlim(-0.6, 0.6)
        self.column_ax.set_ylim(z_min, z_max)
        self.column_ax.set_xticks([])
        self.column_ax.set_ylabel("Height above ground (m)")
        self.no_seed_text = self.column_ax.text(
            0.5,
            0.5,
            "No final raw seeds",
            ha="center",
            va="center",
            transform=self.column_ax.transAxes,
            visible=False,
        )
        self.title_text = self.fig.suptitle("", fontsize=14)
        self.status_text = self.fig.text(0.035, 0.052, "", fontsize=9, color="#b42318")

        button_specs = (
            ("Yes [Y]", 0.22, lambda _event: self._label(1)),
            ("No [N]", 0.32, lambda _event: self._label(0)),
            ("Undo [U]", 0.42, lambda _event: self._undo()),
            ("View [V]", 0.52, lambda _event: self._toggle_view()),
            ("Box [B]", 0.62, lambda _event: self._toggle_box_mode()),
            ("Approve", 0.72, lambda _event: self._approve()),
        )
        self.buttons = []
        self.box_button = None
        for label, x_pos, callback in button_specs:
            button_start = 0.31 if sidebar_enabled else 0.22
            axis = self.fig.add_axes((button_start + (x_pos - 0.22), 0.03, 0.085, 0.045))
            button = Button(axis, label)
            button.on_clicked(callback)
            self.buttons.append(button)
            if label == "Box [B]":
                self.box_button = button

        if sidebar_enabled:
            self._build_scene_sidebar(Button)

        self.connections = (
            self.fig.canvas.mpl_connect("key_press_event", self._on_key),
            self.fig.canvas.mpl_connect("button_press_event", self._on_click),
            self.fig.canvas.mpl_connect("button_press_event", self._on_press),
            self.fig.canvas.mpl_connect("button_release_event", self._on_release),
            self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion),
            self.fig.canvas.mpl_connect("scroll_event", self._on_scroll),
        )
        setattr(self.fig, "_voxroom_annotation_view", self)
        self.refresh()

    def _build_scene_sidebar(self, button_class) -> None:
        top = 0.875
        bottom = 0.105
        row_height = min(0.058, (top - bottom) / max(1, len(self.scene_dirs)))
        self.scene_sidebar_title = self.fig.text(
            0.018,
            0.925,
            "",
            fontsize=11,
            fontweight="bold",
            color="#202124",
        )
        for index, scene_dir in enumerate(self.scene_dirs):
            y_pos = top - float(index + 1) * row_height
            axis = self.fig.add_axes((0.015, y_pos, 0.18, row_height * 0.82))
            button = button_class(axis, "")
            button.label.set_fontsize(8)
            button.on_clicked(
                lambda _event, selected_scene=scene_dir: self._request_scene_switch(selected_scene)
            )
            self.scene_buttons[scene_dir] = button
        self._render_scene_sidebar()

    def _render_scene_sidebar(self) -> None:
        if self.scene_sidebar_title is None:
            return
        done_count = sum(item.status == "done" for item in self.scene_progress.values())
        self.scene_sidebar_title.set_text("SCENES  %d/%d DONE" % (done_count, len(self.scene_dirs)))
        colors = {
            "done": ("#d8f3dc", "#b7e4c7"),
            "partial": ("#fff3bf", "#ffe69c"),
            "todo": ("#f1f3f5", "#e9ecef"),
        }
        prefixes = {"done": "DONE", "partial": "PART", "todo": "TODO"}
        for scene_dir, button in self.scene_buttons.items():
            item = self.scene_progress[scene_dir]
            short_name = item.scene_id.removeprefix("kujiale_")
            button.label.set_text(
                "%s  %s    %d/%d" % (prefixes[item.status], short_name, item.labeled, item.total)
            )
            color, hover = colors[item.status]
            button.color = color
            button.hovercolor = hover
            button.ax.set_facecolor(color)
            is_current = scene_dir == self.session.scene_dir
            for spine in button.ax.spines.values():
                spine.set_color("#1261a0" if is_current else "#adb5bd")
                spine.set_linewidth(2.6 if is_current else 0.8)

    def _update_current_scene_progress(self) -> None:
        self.scene_progress[self.session.scene_dir] = _session_progress(self.session)
        self._render_scene_sidebar()

    def _request_scene_switch(self, scene_dir: str | Path) -> None:
        selected = Path(scene_dir)
        if selected == self.session.scene_dir:
            return
        save_annotation(self.session.annotation, self.session.annotation_path)
        self.requested_scene_dir = selected
        import matplotlib.pyplot as plt

        plt.close(self.fig)

    def refresh(self) -> None:
        self.background_image.set_data(self.background_arrays[self.session.background])
        labels = self.session.annotation.label_map()
        facecolors, _label_edgecolors, sizes = _seed_style_arrays(
            self.session.seed_rc,
            labels,
            self.session.selected_index,
            self.session.seed_source_ids,
        )
        self.seed_scatter.set_facecolors(facecolors)
        self.seed_scatter.set_edgecolors(facecolors)
        self.seed_scatter.set_sizes(sizes)

        done, total = self.session.progress()
        has_seed = bool(total)
        if has_seed:
            selected_row, selected_col = (
                int(v) for v in self.session.seed_rc[self.session.selected_index]
            )
            selected_xy = np.asarray(((selected_col, selected_row),), dtype=np.float32)
        else:
            selected_xy = np.empty((0, 2), dtype=np.float32)
        self.selected_outer_ring.set_offsets(selected_xy)
        self.selected_inner_ring.set_offsets(selected_xy)
        self.selected_outer_ring.set_visible(has_seed)
        self.selected_inner_ring.set_visible(has_seed)
        valid_batch = self.batch_indices[
            (self.batch_indices >= 0) & (self.batch_indices < len(self.session.seed_rc))
        ]
        if len(valid_batch):
            batch_rc = self.session.seed_rc[valid_batch]
            batch_xy = np.column_stack((batch_rc[:, 1], batch_rc[:, 0])).astype(np.float32)
        else:
            batch_xy = np.empty((0, 2), dtype=np.float32)
        self.batch_scatter.set_offsets(batch_xy)
        self.batch_scatter.set_visible(bool(len(valid_batch)))
        selected_number = int(self.session.selected_index) + 1 if has_seed else 0
        self.map_ax.set_title(
            "Final raw seeds | selected: %d/%d | batch: %d | mode: %s | background: %s"
            % (
                selected_number,
                total,
                len(valid_batch),
                "box" if self.box_mode else "single",
                self.session.background,
            )
        )
        self.column_image.set_visible(has_seed)
        self.no_seed_text.set_visible(not has_seed)
        if has_seed:
            row, col = (int(v) for v in self.session.seed_rc[self.session.selected_index])
            world_x, world_y = grid_to_world_xy(row, col, self.session.map_info)
            source_id = int(self.session.seed_source_ids[self.session.selected_index])
            source_name = _SEED_SOURCE_NAMES.get(source_id, "unknown")
            state = _selected_voxel_center_column(
                self.session.voxel_patches,
                self.session.selected_index,
            )
            self.column_image.set_data(np.asarray(state, dtype=np.uint8)[:, None])
            counts = {
                name: int(np.count_nonzero(state == code))
                for name, code in (("unknown", 0), ("free", 1), ("occupied", 2), ("conflict", 3))
            }
            current_label = labels.get((row, col))
            self.column_ax.set_title(
                "row=%d col=%d\nworld=(%.2f, %.2f)\nsource=%s\nlabel=%s\nU/F/O/C=%d/%d/%d/%d"
                % (
                    row,
                    col,
                    world_x,
                    world_y,
                    source_name,
                    "unset" if current_label is None else current_label,
                    counts["unknown"],
                    counts["free"],
                    counts["occupied"],
                    counts["conflict"],
                ),
                fontsize=10,
            )
        else:
            self.column_ax.set_title("")
        self.title_text.set_text(
            "VoxRoom DoorSeed annotation | %s | labeled %d/%d | review=%s"
            % (self.session.scene_id, done, total, self.session.annotation.review.status)
        )
        self.fig.canvas.draw_idle()

    def _label(self, label: int) -> None:
        batch_count = int(len(self.batch_indices))
        if batch_count:
            labeled = self.session.label_indices(self.batch_indices, int(label))
            self.batch_indices = np.empty(0, dtype=np.int32)
            self.status_text.set_color("#157347")
            self.status_text.set_text("Batch labeled: %d" % labeled)
        else:
            self.session.label_selected(int(label))
            self.status_text.set_color("#b42318")
            self.status_text.set_text("")
        self._update_current_scene_progress()
        self.refresh()

    def _undo(self) -> None:
        undone = self.session.undo()
        self.batch_indices = np.empty(0, dtype=np.int32)
        if undone:
            self.status_text.set_color("#157347")
            self.status_text.set_text("Last edit undone")
        else:
            self.status_text.set_color("#b42318")
            self.status_text.set_text("")
        self._update_current_scene_progress()
        self.refresh()

    def _toggle_view(self) -> None:
        self.session.background = "nav" if self.session.background == "vertical" else "vertical"
        self.status_text.set_color("#b42318")
        self.status_text.set_text("")
        self.refresh()

    def _toggle_box_mode(self) -> None:
        self.box_mode = not self.box_mode
        self.box_drag_state["active"] = False
        self.box_rectangle.set_visible(False)
        if self.box_button is not None:
            self.box_button.color = "#f5b7d5" if self.box_mode else "0.85"
            self.box_button.ax.set_facecolor(self.box_button.color)
        self.status_text.set_color("#8a0053")
        self.status_text.set_text("Box selection active" if self.box_mode else "")
        self.refresh()

    def _clear_box_selection(self) -> None:
        self.box_mode = False
        self.batch_indices = np.empty(0, dtype=np.int32)
        self.box_drag_state["active"] = False
        self.box_rectangle.set_visible(False)
        if self.box_button is not None:
            self.box_button.color = "0.85"
            self.box_button.ax.set_facecolor(self.box_button.color)
        self.status_text.set_color("#b42318")
        self.status_text.set_text("")
        self.refresh()

    def _approve(self) -> None:
        try:
            self.session.approve(self.annotator)
        except ValueError as exc:
            self.status_text.set_color("#b42318")
            self.status_text.set_text(str(exc))
        else:
            self.status_text.set_color("#157347")
            self.status_text.set_text("Annotation approved")
        self._update_current_scene_progress()
        self.refresh()

    def _save(self) -> None:
        save_annotation(self.session.annotation, self.session.annotation_path)
        self.status_text.set_color("#157347")
        self.status_text.set_text("Annotation saved")
        self._update_current_scene_progress()
        self.refresh()

    def _on_key(self, event) -> None:
        key = str(event.key or "").lower()
        if key == "y":
            self._label(1)
        elif key == "n":
            self._label(0)
        elif key == "s":
            self.session.select_next_unlabeled(wrap=True)
            self.status_text.set_text("")
            self.refresh()
        elif key == "u":
            self._undo()
        elif key == "v":
            self._toggle_view()
        elif key == "b":
            self._toggle_box_mode()
        elif key == "escape":
            self._clear_box_selection()
        elif key == "a":
            self.session.move(-1)
            self.status_text.set_text("")
            self.refresh()
        elif key == "d":
            self.session.move(1)
            self.status_text.set_text("")
            self.refresh()
        elif key in {"left", "right", "up", "down"}:
            self.session.move_direction(key)
            self.status_text.set_text("")
            self.refresh()
        elif key in {"ctrl+s", "cmd+s"}:
            self._save()

    def _on_click(self, event) -> None:
        if (
            event.inaxes is not self.map_ax
            or self.box_mode
            or event.button != 1
            or event.xdata is None
            or event.ydata is None
            or not len(self.session.seed_rc)
        ):
            return
        seed_pixels = self.map_ax.transData.transform(
            np.column_stack((self.session.seed_rc[:, 1], self.session.seed_rc[:, 0]))
        )
        click_pixel = np.asarray([event.x, event.y], dtype=np.float64)
        distances = np.linalg.norm(seed_pixels - click_pixel[None, :], axis=1)
        nearest = int(np.argmin(distances))
        if float(distances[nearest]) <= 14.0:
            self.session.selected_index = nearest
            self.status_text.set_text("")
            self.refresh()

    def _on_scroll(self, event) -> None:
        if event.inaxes is not self.map_ax or event.xdata is None or event.ydata is None:
            return
        scale = 0.78 if event.button == "up" else 1.28
        x0, x1 = self.map_ax.get_xlim()
        y0, y1 = self.map_ax.get_ylim()
        cx, cy = float(event.xdata), float(event.ydata)
        self.map_ax.set_xlim(cx + (x0 - cx) * scale, cx + (x1 - cx) * scale)
        self.map_ax.set_ylim(cy + (y0 - cy) * scale, cy + (y1 - cy) * scale)
        self.fig.canvas.draw_idle()

    def _on_press(self, event) -> None:
        if (
            self.box_mode
            and event.inaxes is self.map_ax
            and event.button == 1
            and event.xdata is not None
            and event.ydata is not None
        ):
            self.box_drag_state.update(
                {
                    "active": True,
                    "x": float(event.xdata),
                    "y": float(event.ydata),
                }
            )
            self.box_rectangle.set_xy((float(event.xdata), float(event.ydata)))
            self.box_rectangle.set_width(0.0)
            self.box_rectangle.set_height(0.0)
            self.box_rectangle.set_visible(True)
            self.fig.canvas.draw_idle()
            return
        if event.inaxes is self.map_ax and event.button in {2, 3}:
            self.drag_state.update(
                {
                    "active": True,
                    "x": float(event.x),
                    "y": float(event.y),
                    "xlim": self.map_ax.get_xlim(),
                    "ylim": self.map_ax.get_ylim(),
                }
            )

    def _on_motion(self, event) -> None:
        if self.box_drag_state["active"]:
            if event.inaxes is not self.map_ax or event.xdata is None or event.ydata is None:
                return
            start_x = float(self.box_drag_state["x"])
            start_y = float(self.box_drag_state["y"])
            current_x = float(event.xdata)
            current_y = float(event.ydata)
            self.box_rectangle.set_xy((min(start_x, current_x), min(start_y, current_y)))
            self.box_rectangle.set_width(abs(current_x - start_x))
            self.box_rectangle.set_height(abs(current_y - start_y))
            self.fig.canvas.draw_idle()
            return
        if not self.drag_state["active"] or event.inaxes is not self.map_ax:
            return
        xlim = self.drag_state["xlim"]
        ylim = self.drag_state["ylim"]
        if xlim is None or ylim is None:
            return
        bbox = self.map_ax.get_window_extent()
        dx = (float(event.x) - float(self.drag_state["x"])) * (xlim[1] - xlim[0]) / max(
            float(bbox.width), 1.0
        )
        dy = (float(event.y) - float(self.drag_state["y"])) * (ylim[1] - ylim[0]) / max(
            float(bbox.height), 1.0
        )
        self.map_ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
        self.map_ax.set_ylim(ylim[0] - dy, ylim[1] - dy)
        self.fig.canvas.draw_idle()

    def _on_release(self, event) -> None:
        if self.box_drag_state["active"]:
            self.box_drag_state["active"] = False
            self.box_rectangle.set_visible(False)
            if event.inaxes is self.map_ax and event.xdata is not None and event.ydata is not None:
                start_x = float(self.box_drag_state["x"])
                start_y = float(self.box_drag_state["y"])
                min_col, max_col = sorted((start_x, float(event.xdata)))
                min_row, max_row = sorted((start_y, float(event.ydata)))
                row = self.session.seed_rc[:, 0]
                col = self.session.seed_rc[:, 1]
                inside = (
                    (row >= min_row)
                    & (row <= max_row)
                    & (col >= min_col)
                    & (col <= max_col)
                )
                self.batch_indices = np.flatnonzero(inside).astype(np.int32)
                if len(self.batch_indices):
                    self.session.selected_index = int(self.batch_indices[0])
                self.status_text.set_color("#8a0053")
                self.status_text.set_text("Batch selected: %d" % len(self.batch_indices))
            self.refresh()
            return
        self.drag_state["active"] = False


def run_annotation_app(
    scene_dir: str | Path | None,
    *,
    annotator: str = "annotator",
    collection_root: str | Path | None = None,
    prefill_unlabeled_reject: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    if scene_dir is None and collection_root is None:
        raise ValueError("scene_dir or collection_root is required")
    root = Path(collection_root).expanduser() if collection_root is not None else Path(scene_dir).parent
    progress = discover_annotation_scenes(root)
    if not progress:
        raise ValueError("no collected annotation scenes found under %s" % root)
    if prefill_unlabeled_reject:
        for item in progress:
            AnnotationSession.open(item.scene_dir).prefill_unlabeled_as_reject()
        progress = discover_annotation_scenes(root)
    current = Path(scene_dir) if scene_dir is not None else next(
        (item.scene_dir for item in progress if item.status != "done"),
        progress[0].scene_dir,
    )
    while current is not None:
        progress = discover_annotation_scenes(root)
        scene_dirs = [item.scene_dir for item in progress]
        session = AnnotationSession.open(current)
        view = _AnnotationFigure(session, annotator=annotator, scene_dirs=scene_dirs)
        plt.show()
        current = view.requested_scene_dir
