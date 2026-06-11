from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from voxroom_online.isaac_runtime.config import repo_root
from voxroom_online.isaac_runtime.dataset.category_normalizer import normalize_category
from voxroom_online.isaac_runtime.graph.decision import NavigationDecision
from voxroom_online.isaac_runtime.mapping.frontier import FrontierCluster
from voxroom_online.isaac_runtime.perception.detection_types import MIN_VALID_DETECTION_CONFIDENCE, Detection2D, detection_confidence_is_valid
from voxroom_online.isaac_runtime.perception.object_memory import ObjectMemory


GridCell = Tuple[int, int]


def _is_green_like(color: Tuple[int, int, int]) -> bool:
    r, g, b = [int(v) for v in color]
    return bool(g >= 150 and g > r + 25 and g >= b)


def _cell_in_crop(cell: GridCell, r0: int, r1: int, c0: int, c1: int) -> bool:
    row, col = int(cell[0]), int(cell[1])
    return bool(r0 <= row < r1 and c0 <= col < c1)


def _top_reason_text(counts: object) -> str:
    if not isinstance(counts, Mapping) or not counts:
        return "none"
    pairs: list[tuple[str, int]] = []
    for key, value in counts.items():
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if str(key) and count > 0:
            pairs.append((str(key), count))
    top = sorted(pairs, key=lambda item: item[1], reverse=True)
    if not top:
        return "none"
    return "%s=%d" % top[0]


class VoxRoomPopupVisualizer:
    def __init__(
        self,
        enabled: bool = True,
        window_name: str = "VoxRoom Isaac Debug",
        save_dir: Optional[str] = None,
        panel_size: Tuple[int, int] = (1440, 900),
        save_every_steps: int = 10,
        ipc_jpeg_quality: int = 75,
        debug_overlay_layers: bool = True,
        save_overlay_layer_metadata: bool = True,
        show_gt_goal_cells: bool = False,
        show_room_proposals: bool = True,
        show_room_masks: bool = True,
        show_room_labels: bool = True,
        show_rose_occupancy_map: bool = True,
        show_frontier_member_cells: bool = True,
        show_object_nodes: bool = True,
        show_candidate_markers: bool = True,
        min_valid_detection_confidence: float = MIN_VALID_DETECTION_CONFIDENCE,
        max_green_like_primitives_before_warning: int = 200,
    ) -> None:
        self.enabled = bool(enabled)
        self.window_name = window_name
        self.save_dir = Path(save_dir) if save_dir else None
        self.panel_size = (int(panel_size[0]), int(panel_size[1]))
        self.save_every_steps = max(1, int(save_every_steps))
        self.ipc_jpeg_quality = max(30, min(95, int(ipc_jpeg_quality)))
        self.debug_overlay_layers = bool(debug_overlay_layers)
        self.save_overlay_layer_metadata = bool(save_overlay_layer_metadata)
        self.show_gt_goal_cells = bool(show_gt_goal_cells)
        self.show_room_proposals = bool(show_room_proposals)
        self.show_room_masks = bool(show_room_masks)
        self.show_room_labels = bool(show_room_labels)
        self.show_rose_occupancy_map = bool(show_rose_occupancy_map)
        self.show_frontier_member_cells = bool(show_frontier_member_cells)
        self.show_object_nodes = bool(show_object_nodes)
        self.show_candidate_markers = bool(show_candidate_markers)
        self.min_valid_detection_confidence = float(min_valid_detection_confidence)
        self.max_green_like_primitives_before_warning = max(0, int(max_green_like_primitives_before_warning))
        self._last_overlay_layers: List[dict] = []
        self._room_masks: List[object] = []
        self._room_semantic_labels: dict[str, object] = {}
        self._room_segmentation_debug: dict = {}
        self._proc: Optional[subprocess.Popen[str]] = None
        self._ipc_dir = Path(tempfile.gettempdir()) / ("voxroom_viz_%d" % os.getpid())
        self._frame_path = self._ipc_dir / "latest.jpg"
        self._font = ImageFont.load_default()
        if self.save_dir:
            try:
                self.save_dir.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                print("[voxroom-viz] disabling frame saves; could not create save dir: %s" % exc, file=sys.stderr, flush=True)
                self.save_dir = None

    def set_room_context(
        self,
        room_masks: Sequence[object],
        room_semantic_labels: Optional[Mapping[str, object]] = None,
        room_segmentation_debug: Optional[Mapping[str, object]] = None,
    ) -> None:
        self._room_masks = list(room_masks or [])
        self._room_semantic_labels = dict(room_semantic_labels or {})
        self._room_segmentation_debug = dict(room_segmentation_debug or {})

    def _try_open_window(self) -> None:
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            print("[voxroom-viz] no DISPLAY/WAYLAND_DISPLAY; popup disabled", file=sys.stderr, flush=True)
            self.enabled = False
            return
        try:
            root = repo_root()
            python_executable = self._default_python_executable()
            cmd = [
                python_executable,
                "-m",
                "voxroom_online.isaac_runtime.visualization.voxroom_popup_worker",
                "--window-name",
                self.window_name,
                "--width",
                str(self.panel_size[0]),
                "--height",
                str(self.panel_size[1]),
            ]
            env = os.environ.copy()
            existing_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = str(root) if not existing_pythonpath else str(root) + os.pathsep + existing_pythonpath
            env["PYTHONUNBUFFERED"] = "1"
            if os.environ.get("ISAAC_BENCH_KEEP_POPUP_LD_LIBRARY_PATH", "").lower() not in {"1", "true", "yes"}:
                env.pop("LD_LIBRARY_PATH", None)
            print("[voxroom-viz] starting popup worker with %s" % python_executable, flush=True)
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(root),
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                bufsize=1,
            )
            self._read_ready(timeout_s=10.0)
        except Exception as exc:
            print("[voxroom-viz] OpenCV popup unavailable: %s" % exc, file=sys.stderr, flush=True)
            self.enabled = False
            self.close()

    @staticmethod
    def _default_python_executable() -> str:
        explicit = os.environ.get("VOXROOM_PYTHON")
        if explicit:
            return explicit
        current = Path(sys.executable)
        if current.exists() and "sgnav-isaac" in str(current):
            return str(current)
        env_root = Path(os.environ.get("VOXROOM_ENV", ""))
        candidate = env_root / "bin" / "python"
        if str(env_root) and candidate.exists():
            return str(candidate)
        return sys.executable

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        if proc.poll() is None:
            try:
                if proc.stdin:
                    proc.stdin.write(json.dumps({"type": "close"}) + "\n")
                    proc.stdin.flush()
            except Exception:
                pass
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
        for stream in (proc.stdin, proc.stdout):
            try:
                if stream:
                    stream.close()
            except Exception:
                pass

    def update(
        self,
        *,
        step: int,
        rgb: np.ndarray,
        detections_2d: Sequence[Detection2D],
        occupancy: np.ndarray,
        navigable: np.ndarray,
        observed: np.ndarray,
        goal_cells: Sequence[GridCell],
        current_grid: GridCell,
        pose: Sequence[float],
        frontiers: Sequence[FrontierCluster],
        nav_decision: Optional[NavigationDecision],
        current_path: Sequence[GridCell],
        full_path: Sequence[GridCell],
        object_memory: ObjectMemory,
        goal_category: str,
        distance_to_goal: float,
        path_length: float,
        scenegraph_backend: str,
        score_debug: Optional[dict] = None,
        failure_reason: Optional[str] = None,
    ) -> np.ndarray:
        panel = self.render(
            step=step,
            rgb=rgb,
            detections_2d=detections_2d,
            occupancy=occupancy,
            navigable=navigable,
            observed=observed,
            goal_cells=goal_cells,
            current_grid=current_grid,
            pose=pose,
            frontiers=frontiers,
            nav_decision=nav_decision,
            current_path=current_path,
            full_path=full_path,
            object_memory=object_memory,
            goal_category=goal_category,
            distance_to_goal=distance_to_goal,
            path_length=path_length,
            scenegraph_backend=scenegraph_backend,
            score_debug=score_debug,
            failure_reason=failure_reason,
        )
        if self.enabled:
            self._send_frame(panel)
        self._save_panel_artifacts(step, panel)
        return panel

    def _save_panel_artifacts(self, step: int, panel: np.ndarray) -> None:
        if not self.save_dir or int(step) % self.save_every_steps != 0:
            return
        try:
            Image.fromarray(panel).save(self.save_dir / ("voxroom_step_%06d.jpg" % int(step)), format="JPEG", quality=85)
            if self.save_overlay_layer_metadata:
                meta_path = self.save_dir / ("voxroom_step_%06d.layers.json" % int(step))
                meta_path.write_text(
                    json.dumps(self.overlay_layer_metadata(step), ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
        except Exception as exc:
            print("[voxroom-viz] frame save failed; popup frame was still sent: %s" % exc, file=sys.stderr, flush=True)

    def overlay_layer_metadata(self, frame_id: int = 0) -> dict:
        layers = [dict(item) for item in self._last_overlay_layers]
        green_like = sum(int(item.get("primitive_count", 0)) for item in layers if bool(item.get("green_like", False)))
        payload = {
            "frame_id": int(frame_id),
            "layers": layers,
            "green_like_primitive_count": int(green_like),
            "max_green_like_primitives_before_warning": int(self.max_green_like_primitives_before_warning),
            "green_like_warning": bool(
                self.max_green_like_primitives_before_warning > 0
                and green_like > self.max_green_like_primitives_before_warning
            ),
        }
        if isinstance(self._room_segmentation_debug, Mapping):
            if self._room_segmentation_debug.get("roomseg_debug_layers"):
                payload["roomseg_debug_layers"] = dict(self._room_segmentation_debug.get("roomseg_debug_layers") or {})
                summary = dict(self._room_segmentation_debug.get("roomseg_debug_summary") or {})
                payload["roomseg_debug_summary"] = {
                    "counts": dict(summary.get("counts") or {}),
                    "likely_cause": summary.get("likely_cause"),
                }
            overlay = self._room_segmentation_debug.get("navigation_free_context_overlay")
            if isinstance(overlay, Mapping):
                payload["context_overlay"] = {
                    "enabled": bool(overlay.get("nav_free_overlay_enabled", False)),
                    "absorbed_cells": int(overlay.get("absorbed_cells", 0) or 0),
                    "remaining_unlabeled_nav_free_cells": int(overlay.get("remaining_unlabeled_nav_free_cells", 0) or 0),
                    "used_for_frontier_room_assignment": True,
                    "used_for_room_nodes": bool(overlay.get("nav_free_overlay_enabled", False)),
                }
        return payload

    def _read_ready(self, timeout_s: float) -> None:
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("popup worker did not start")
        ready, _, _ = select.select([self._proc.stdout], [], [], float(timeout_s))
        if not ready:
            raise RuntimeError("timed out waiting for popup worker")
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("popup worker exited before ready (code=%s)" % self._proc.poll())
        response = json.loads(line)
        if response.get("type") != "ready":
            raise RuntimeError("popup worker protocol error: %s" % response)

    def _send_frame(self, panel: np.ndarray) -> None:
        if self._proc is None:
            self._try_open_window()
            if not self.enabled or self._proc is None:
                return
        proc = self._proc
        if proc is None or proc.stdin is None:
            self.enabled = False
            return
        if proc.poll() is not None:
            print("[voxroom-viz] popup worker exited; continuing without popup", file=sys.stderr, flush=True)
            self.enabled = False
            return
        try:
            self._ipc_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = self._ipc_dir / "latest.tmp.jpg"
            Image.fromarray(panel).save(tmp_path, format="JPEG", quality=self.ipc_jpeg_quality)
            os.replace(tmp_path, self._frame_path)
            proc.stdin.write(json.dumps({"type": "frame_path", "path": str(self._frame_path)}) + "\n")
            proc.stdin.flush()
        except Exception as exc:
            print("[voxroom-viz] popup IPC failed; continuing without popup: %s" % exc, file=sys.stderr, flush=True)
            self.enabled = False
            self.close()

    def render(
        self,
        *,
        step: int,
        rgb: np.ndarray,
        detections_2d: Sequence[Detection2D],
        occupancy: np.ndarray,
        navigable: np.ndarray,
        observed: np.ndarray,
        goal_cells: Sequence[GridCell],
        current_grid: GridCell,
        pose: Sequence[float],
        frontiers: Sequence[FrontierCluster],
        nav_decision: Optional[NavigationDecision],
        current_path: Sequence[GridCell],
        full_path: Sequence[GridCell],
        object_memory: ObjectMemory,
        goal_category: str,
        distance_to_goal: float,
        path_length: float,
        scenegraph_backend: str,
        score_debug: Optional[dict] = None,
        failure_reason: Optional[str] = None,
    ) -> np.ndarray:
        panel_w, panel_h = self.panel_size
        right_w = max(1, int(panel_w * 0.60))
        left_w = panel_w - right_w
        rgb_h = int(panel_h * 0.62)
        text_h = panel_h - rgb_h
        valid_detections_2d = self._valid_detections(detections_2d)

        panel = Image.new("RGB", (panel_w, panel_h), (18, 20, 24))
        rgb_panel = self._render_rgb(
            rgb,
            valid_detections_2d,
            (left_w, rgb_h),
            goal_category=goal_category,
            nav_decision=nav_decision,
        )
        map_panel = self._render_map(
            occupancy=occupancy,
            navigable=navigable,
            observed=observed,
            goal_cells=goal_cells,
            current_grid=current_grid,
            pose=pose,
            frontiers=frontiers,
            nav_decision=nav_decision,
            current_path=current_path,
            full_path=full_path,
            object_memory=object_memory,
            goal_category=goal_category,
            size=(right_w, panel_h),
        )
        text_panel = self._render_text(
            step=step,
            detections_2d=valid_detections_2d,
            frontiers=frontiers,
            nav_decision=nav_decision,
            object_memory=object_memory,
            goal_category=goal_category,
            distance_to_goal=distance_to_goal,
            path_length=path_length,
            scenegraph_backend=scenegraph_backend,
            score_debug=score_debug or {},
            failure_reason=failure_reason,
            size=(left_w, text_h),
        )
        panel.paste(rgb_panel, (0, 0))
        panel.paste(text_panel, (0, rgb_h))
        panel.paste(map_panel, (left_w, 0))
        draw = ImageDraw.Draw(panel)
        draw.line([(left_w, 0), (left_w, panel_h)], fill=(70, 74, 80), width=2)
        draw.line([(0, rgb_h), (left_w, rgb_h)], fill=(70, 74, 80), width=2)
        return np.asarray(panel, dtype=np.uint8)

    def _valid_detections(self, detections: Sequence[Detection2D]) -> List[Detection2D]:
        return [
            det
            for det in detections
            if detection_confidence_is_valid(float(det.confidence), self.min_valid_detection_confidence)
        ]

    def _render_rgb(
        self,
        rgb: np.ndarray,
        detections: Sequence[Detection2D],
        size: Tuple[int, int],
        goal_category: str = "",
        nav_decision: Optional[NavigationDecision] = None,
    ) -> Image.Image:
        width, height = size
        arr = np.asarray(rgb)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim != 3 or arr.shape[2] < 3:
            arr = np.zeros((height, width, 3), dtype=np.uint8)
        src_h, src_w = arr.shape[:2]
        image = Image.new("RGB", (width, height), (12, 14, 18))
        scale = min(width / max(src_w, 1), height / max(src_h, 1))
        render_w = max(1, int(round(src_w * scale)))
        render_h = max(1, int(round(src_h * scale)))
        offset_x = (width - render_w) // 2
        offset_y = (height - render_h) // 2
        rgb_image = Image.fromarray(arr[:, :, :3]).resize((render_w, render_h), Image.BILINEAR)
        image.paste(rgb_image, (offset_x, offset_y))
        draw = ImageDraw.Draw(image)
        sx, sy = render_w / max(src_w, 1), render_h / max(src_h, 1)
        valid_detections = self._valid_detections(detections)
        for det in valid_detections[:30]:
            x1, y1, x2, y2 = det.bbox_xyxy
            box = [
                int(offset_x + x1 * sx),
                int(offset_y + y1 * sy),
                int(offset_x + x2 * sx),
                int(offset_y + y2 * sy),
            ]
            color = self._bbox_color(det, goal_category, nav_decision)
            draw.rectangle(box, outline=color, width=2)
            label = "%s %.2f" % (det.category, float(det.confidence))
            self._label(draw, (box[0], max(0, box[1] - 14)), label, color)
        self._label(draw, (8, 8), "RGB / detector detections: %d  green=normal red=goal" % len(valid_detections), (255, 255, 255))
        return image

    def _render_map(
        self,
        *,
        occupancy: np.ndarray,
        navigable: np.ndarray,
        observed: np.ndarray,
        goal_cells: Sequence[GridCell],
        current_grid: GridCell,
        pose: Sequence[float],
        frontiers: Sequence[FrontierCluster],
        nav_decision: Optional[NavigationDecision],
        current_path: Sequence[GridCell],
        full_path: Sequence[GridCell],
        object_memory: ObjectMemory,
        goal_category: str,
        size: Tuple[int, int],
    ) -> Image.Image:
        width, height = size
        rose_panel_enabled = bool(self.show_rose_occupancy_map)
        rose_h = 0
        divider_h = 0
        map_h_available = int(height)
        if rose_panel_enabled and height >= 180:
            divider_h = 2
            map_h_available = max(1, (int(height) - divider_h) // 2)
            rose_h = max(1, int(height) - divider_h - map_h_available)
        h, w = occupancy.shape
        base = np.zeros((h, w, 3), dtype=np.uint8)
        nav = navigable.astype(bool)
        obs = observed.astype(bool)
        base[nav] = (218, 222, 224)
        base[~nav] = (72, 74, 76)
        base[occupancy.astype(bool)] = (24, 24, 24)
        base[~obs] = (base[~obs].astype(np.float32) * 0.45 + np.array([20, 24, 34], dtype=np.float32)).astype(np.uint8)
        active_room_masks = self._active_room_masks(occupancy.shape)
        proposal_room_masks = self._proposal_room_masks(occupancy.shape)
        proposal_room_cell_count = 0
        if self.show_room_proposals and proposal_room_masks:
            base, proposal_room_cell_count = self._apply_boolean_mask_overlay(
                base,
                proposal_room_masks,
                alpha=0.16,
                boundary_only=True,
            )
        room_mask_cell_count = 0
        room_boundary_cell_count = 0
        if self.show_room_masks and active_room_masks:
            base, room_mask_cell_count, room_boundary_cell_count = self._apply_room_mask_overlay(base, active_room_masks)

        r0, r1, c0, c1 = self._map_crop_bounds(
            occupancy=occupancy,
            navigable=navigable,
            observed=observed,
            current_grid=current_grid,
            frontiers=frontiers,
            nav_decision=nav_decision,
            current_path=current_path,
            full_path=full_path,
            object_memory=object_memory,
            goal_category=goal_category,
        )
        crop = base[r0:r1, c0:c1]
        crop_h, crop_w = crop.shape[:2]
        margin = 12
        scale = min((width - 2 * margin) / max(crop_w, 1), (map_h_available - 2 * margin) / max(crop_h, 1))
        map_w, map_h = max(1, int(crop_w * scale)), max(1, int(crop_h * scale))
        ox, oy = (width - map_w) // 2, (map_h_available - map_h) // 2
        image = Image.new("RGB", (width, map_h_available), (18, 20, 24))
        map_img = Image.fromarray(crop).resize((map_w, map_h), Image.NEAREST)
        image.paste(map_img, (ox, oy))
        draw = ImageDraw.Draw(image)
        layers: List[dict] = []

        def record(name: str, enabled: bool, color: Tuple[int, int, int], count: int, note: str = "", **extra) -> None:
            item = {
                "name": name,
                "enabled": bool(enabled),
                "color": [int(color[0]), int(color[1]), int(color[2])],
                "primitive_count": int(count),
                "count": int(count),
                "green_like": bool(_is_green_like(color)),
                "note": note,
                "description": note,
            }
            item.update(extra)
            layers.append(item)

        def xy(cell: GridCell) -> Tuple[int, int]:
            r, c = int(cell[0]), int(cell[1])
            return int(ox + (c - c0 + 0.5) * scale), int(oy + (r - r0 + 0.5) * scale)

        room_color = (145, 110, 255)
        record("proposal_room_masks", self.show_room_proposals, (150, 150, 160), proposal_room_cell_count, "watershed/proposal basins before doorway-constrained merge")
        record("room_masks", self.show_room_masks, room_color, room_mask_cell_count, "online geometry room mask fill")
        record("final_room_masks", self.show_room_masks, room_color, room_mask_cell_count, "post-merge room masks used by VoxRoom")
        record("room_boundaries", self.show_room_masks, room_color, room_boundary_cell_count, "online geometry room mask boundary")
        room_label_count = self._draw_room_labels(
            draw,
            active_room_masks,
            xy,
            crop_bounds=(r0, r1, c0, c1),
        ) if self.show_room_labels else 0
        record("navigation_map_panel", True, (218, 222, 224), int(np.count_nonzero(obs)), "top-right navigation map panel")
        record("navigation_free", True, (218, 222, 224), int(np.count_nonzero(nav & obs)), "runtime navigation free cells in the top-right map")
        record("navigation_obstacle", True, (24, 24, 24), int(np.count_nonzero(occupancy.astype(bool))), "runtime navigation obstacle cells in the top-right map")
        record("navigation_unknown", True, (72, 74, 76), int(np.count_nonzero(~obs)), "runtime navigation unknown cells in the top-right map")
        record("room_labels", self.show_room_labels, (250, 250, 255), room_label_count, "VLM room category and reliability")

        map_shape = tuple(np.asarray(occupancy).shape[:2])
        clipped_outside_nav = self._room_debug_array("vertical_free_clipped_outside_navigation_map", map_shape, bool)
        free_wall_conflict = self._room_debug_array("free_wall_conflict_map_before_sanitize", map_shape, bool)
        sanitized_free = self._room_debug_array("roomseg_sanitized_free", map_shape, bool)
        sanitized_wall = self._room_debug_array("roomseg_sanitized_wall", map_shape, bool)
        pre_extension_door_detected = self._room_debug_array("pre_extension_door_detected_map", map_shape, bool)
        pre_extension_door_cut = self._room_debug_array("pre_extension_door_cut_mask", map_shape, bool)
        strict_pre_extension_door_cut = self._room_debug_array("strict_pre_extension_door_cut_mask", map_shape, bool)
        partial_door_seed = self._room_debug_array("partial_door_seed_mask", map_shape, bool)
        partial_door_line = self._room_debug_array("partial_door_line_mask", map_shape, bool)
        partial_door_extension_cut = self._room_debug_array("partial_door_extension_cut_mask", map_shape, bool)
        rejected_door_extension = self._room_debug_array("rejected_door_extension_mask", map_shape, bool)
        original_step_boundary = self._room_debug_array("original_step1_step2_virtual_boundary_map", map_shape, bool)
        accepted_closure = self._room_debug_array("accepted_closure_map", map_shape, bool)
        wall_extension_boundary = self._room_debug_array("wall_extension_boundary_mask", map_shape, bool)
        door_completion_boundary = self._room_debug_array("door_completion_boundary_mask", map_shape, bool)
        if not np.any(wall_extension_boundary):
            wall_extension_boundary = original_step_boundary
        if not np.any(door_completion_boundary):
            door_completion_boundary = pre_extension_door_cut
        pre_extension_room_labels = self._room_debug_array("pre_extension_room_label_map", map_shape, np.int32)
        pre_extension_room_boundary = self._room_label_adjacency_boundary(pre_extension_room_labels)
        show_roomseg_diagnostics_on_main = bool(
            self._room_segmentation_debug.get("show_roomseg_diagnostics_on_main", False)
        )
        diagnostic_note = "hidden on main map; use the ROSE roomseg debug panel or snapshot metadata"

        def crop_mask_cells(mask: np.ndarray) -> List[GridCell]:
            return self._mask_cells_in_crop(mask, (r0, r1, c0, c1))

        nav_occ_endpoint = self._room_debug_array("voxel_nav_occupied_from_endpoint_xy", map_shape, bool)
        nav_suppressed_free = self._room_debug_array("voxel_nav_free_suppressed_by_occupied_xy", map_shape, bool)
        current_override = self._room_debug_array("current_pose_navigation_override_mask", map_shape, bool)
        low_clearance = self._room_debug_array("astar_clearance_low_cells", map_shape, bool)
        nav_endpoint_count = self._draw_cells(
            draw,
            crop_mask_cells(nav_occ_endpoint),
            xy,
            (255, 120, 40),
            radius=1,
            max_cells=1200,
        )
        record(
            "voxel_nav_occupied_from_endpoint",
            nav_endpoint_count > 0,
            (255, 120, 40),
            nav_endpoint_count,
            "navigation occupied cells protected by endpoint hysteresis",
            total_cell_count=int(np.count_nonzero(nav_occ_endpoint)),
        )
        nav_suppressed_count = self._draw_cells(
            draw,
            crop_mask_cells(nav_suppressed_free),
            xy,
            (245, 210, 55),
            radius=1,
            max_cells=1200,
        )
        record(
            "voxel_nav_free_suppressed_by_occupied",
            nav_suppressed_count > 0,
            (245, 210, 55),
            nav_suppressed_count,
            "raw free cells suppressed because occupied projection has priority",
            total_cell_count=int(np.count_nonzero(nav_suppressed_free)),
        )
        low_clearance_count = self._draw_cells(
            draw,
            crop_mask_cells(low_clearance),
            xy,
            (180, 90, 230),
            radius=1,
            max_cells=1200,
        )
        record(
            "astar_clearance_low_cells",
            low_clearance_count > 0,
            (180, 90, 230),
            low_clearance_count,
            "A* cells below configured lookahead clearance threshold",
            total_cell_count=int(np.count_nonzero(low_clearance)),
        )
        current_override_count = self._draw_cells(
            draw,
            crop_mask_cells(current_override),
            xy,
            (35, 235, 95),
            radius=2,
            max_cells=500,
        )
        record(
            "current_pose_navigation_override",
            current_override_count > 0,
            (35, 235, 95),
            current_override_count,
            "current robot footprint dynamic occupied override; static walls remain blocked",
            total_cell_count=int(np.count_nonzero(current_override)),
        )

        def record_roomseg_diagnostic_mask(
            name: str,
            mask: np.ndarray,
            color: Tuple[int, int, int],
            radius: int,
            max_cells: int,
            note: str,
            **extra,
        ) -> int:
            cells = crop_mask_cells(mask)
            if show_roomseg_diagnostics_on_main:
                count = self._draw_cells(draw, cells, xy, color, radius=radius, max_cells=max_cells)
                enabled = count > 0
                display_note = note
            else:
                count = 0
                enabled = False
                display_note = "%s; %s" % (diagnostic_note, note)
            record(
                name,
                enabled,
                color,
                count,
                display_note,
                available_cell_count=int(len(cells)),
                total_cell_count=int(np.count_nonzero(mask)),
                **extra,
            )
            return count

        clipped_count = record_roomseg_diagnostic_mask(
            "roomseg_vertical_free_outside_navigation",
            clipped_outside_nav,
            (45, 135, 255),
            2,
            1200,
            "vertical-free cells clipped before VFGC because they were outside navigation free",
        )
        conflict_count = record_roomseg_diagnostic_mask(
            "roomseg_free_wall_conflict",
            free_wall_conflict,
            (255, 45, 45),
            2,
            1200,
            "free/wall overlap removed before VFGC so wall cores cannot be erased by vertical free",
        )
        sanitized_wall_count = record_roomseg_diagnostic_mask(
            "roomseg_sanitized_wall",
            sanitized_wall,
            (10, 10, 10),
            1,
            1600,
            "wall mask after input sanitizer, including terminal ray-wall evidence",
        )
        record(
            "roomseg_sanitized_free",
            bool(np.any(sanitized_free)),
            (166, 170, 174),
            int(np.count_nonzero(sanitized_free)),
            "free mask after input sanitizer; it must be a subset of navigation free when available",
            subset_navigation_ok=bool(self._room_segmentation_debug.get("sanitized_free_subset_navigation_ok", False)),
        )
        pre_door_count = record_roomseg_diagnostic_mask(
            "pre_extension_doors",
            pre_extension_door_detected,
            (0, 210, 255),
            2,
            1200,
            "doors detected before wall-line extension by strict free/occupied/unknown pattern rules",
        )
        pre_door_cut_count = record_roomseg_diagnostic_mask(
            "pre_extension_door_cuts",
            pre_extension_door_cut,
            (255, 190, 40),
            3,
            1200,
            "virtual free-space cuts created by pre-extension door pattern detection",
        )
        wall_extension_cells = crop_mask_cells(wall_extension_boundary | accepted_closure)
        wall_extension_boundary_count = int(len(wall_extension_cells))
        record(
            "wall_extension_boundaries",
            wall_extension_boundary_count > 0,
            (80, 170, 255),
            wall_extension_boundary_count,
            "boundaries produced by the original wall-endpoint/wall-line extension closure path",
        )
        door_completion_cells = crop_mask_cells(door_completion_boundary)
        door_completion_boundary_count = int(len(door_completion_cells))
        record(
            "door_completion_boundaries",
            door_completion_boundary_count > 0,
            (255, 120, 40),
            door_completion_boundary_count,
            "boundaries produced by detected strict/partial door completion, separated from wall-line extension",
        )
        strict_pre_cut_count = record_roomseg_diagnostic_mask(
            "strict_pre_extension_door_cuts",
            strict_pre_extension_door_cut,
            (255, 120, 40),
            2,
            1200,
            "door cuts from the original strict pattern rules",
        )
        partial_seed_count = record_roomseg_diagnostic_mask(
            "partial_door_seed_points",
            partial_door_seed,
            (135, 245, 255),
            2,
            1200,
            "partial door-like occupied seed cells detected before line extension",
        )
        partial_line_count = record_roomseg_diagnostic_mask(
            "accepted_partial_door_extension_lines",
            partial_door_line,
            (60, 250, 180),
            2,
            1600,
            "accepted partial-door lines extended to structural wall anchors",
        )
        partial_cut_count = record_roomseg_diagnostic_mask(
            "partial_door_extension_cuts",
            partial_door_extension_cut,
            (255, 120, 40),
            3,
            1600,
            "accepted partial-door cut cells; these are ORed only into the final boundary",
        )
        rejected_line_count = record_roomseg_diagnostic_mask(
            "rejected_partial_door_extension_lines",
            rejected_door_extension,
            (255, 60, 180),
            2,
            1600,
            "partial-door extension lines rejected by width/unknown/wall/other-door guards",
            reject_reason_counts=dict(self._room_segmentation_debug.get("partial_door_line_reject_reason_counts") or {}),
        )
        degenerate = bool(self._room_segmentation_debug.get("segmentation_degenerate_one_room", False))
        record(
            "segmentation_degenerate_warning",
            degenerate,
            (255, 80, 80),
            1 if degenerate else 0,
            "VFGC produced one large room without any virtual boundary; inspect sanitizer and terminal-wall evidence",
        )
        pre_room_boundary_available_count = int(len(crop_mask_cells(pre_extension_room_boundary)))
        pre_room_boundary_count = record_roomseg_diagnostic_mask(
            "pre_extension_room_boundaries",
            pre_extension_room_boundary,
            (120, 180, 255),
            1,
            2000,
            "boundaries between room labels produced immediately after pre-extension door cuts",
        )
        pre_room_count = int(len([v for v in np.unique(pre_extension_room_labels) if int(v) > 0]))
        record(
            "pre_extension_room_labels",
            show_roomseg_diagnostics_on_main and pre_room_boundary_count > 0,
            (120, 180, 255),
            pre_room_count if show_roomseg_diagnostics_on_main else 0,
            "room labels produced immediately after pre-extension door cuts and before original step1/step2",
            boundary_cell_count=pre_room_boundary_available_count,
            available_room_count=pre_room_count,
        )

        if show_roomseg_diagnostics_on_main:
            wall_line_count, wall_extension_count = self._draw_roomseg_wall_debug_lines(draw, xy, (r0, r1, c0, c1))
        else:
            wall_line_count, wall_extension_count = (0, 0)
        record(
            "roomseg_wall_lines_red",
            show_roomseg_diagnostics_on_main and wall_line_count > 0,
            (255, 35, 35),
            wall_line_count,
            "solid bright-red filtered wall lines used by the current room segmentation pass",
            hidden_on_main=not show_roomseg_diagnostics_on_main,
        )
        record(
            "roomseg_wall_extensions_red_dashed",
            show_roomseg_diagnostics_on_main and wall_extension_count > 0,
            (255, 0, 0),
            wall_extension_count,
            "pure-red dashed attempted wall-line extensions with dark outline; drawn whether or not the final split succeeds",
            hidden_on_main=not show_roomseg_diagnostics_on_main,
        )
        merge_reasons = []
        if show_roomseg_diagnostics_on_main:
            merged_count, doorway_count, merge_reasons = self._draw_room_adjacency_debug_lines(draw, xy, (r0, r1, c0, c1))
        else:
            merged_count, doorway_count = (0, 0)
        record(
            "room_merged_boundaries",
            show_roomseg_diagnostics_on_main and merged_count > 0,
            (150, 150, 155),
            merged_count,
            "dashed proposal boundaries removed by doorway-constrained merge",
            adjacency_merge_reasons=merge_reasons,
            hidden_on_main=not show_roomseg_diagnostics_on_main,
        )
        record(
            "room_doorway_cuts",
            show_roomseg_diagnostics_on_main and doorway_count > 0,
            (255, 170, 40),
            doorway_count,
            "bold verified doorway/gateway cuts preserved as room splits",
            adjacency_merge_reasons=[item for item in merge_reasons if item.get("verified_doorway")],
            hidden_on_main=not show_roomseg_diagnostics_on_main,
        )
        if show_roomseg_diagnostics_on_main:
            corridor_merge_count, small_region_merge_count = self._draw_corridor_merge_debug_lines(draw, xy, (r0, r1, c0, c1))
        else:
            corridor_merge_count, small_region_merge_count = (0, 0)
        record(
            "corridor_merge_edges",
            show_roomseg_diagnostics_on_main and corridor_merge_count > 0,
            (255, 24, 24),
            corridor_merge_count,
            "bright red dashed shared edges that triggered strict corridor/door-neck region merge",
            hidden_on_main=not show_roomseg_diagnostics_on_main,
        )
        record(
            "post_corridor_small_region_merges",
            show_roomseg_diagnostics_on_main and small_region_merge_count > 0,
            (255, 190, 24),
            small_region_merge_count,
            "bright amber dashed circles mark small regions merged after strict corridor merge",
            hidden_on_main=not show_roomseg_diagnostics_on_main,
        )

        goal_color = (30, 220, 80)
        goal_count = self._draw_cells(draw, goal_cells, xy, goal_color, radius=2, max_cells=500) if self.show_gt_goal_cells else 0
        record("gt_goal_cells", self.show_gt_goal_cells, goal_color, goal_count, "disabled by default; oracle GT overlay")
        full_path_color = (80, 130, 255)
        record("full_path", True, full_path_color, self._draw_cells(draw, full_path, xy, full_path_color, radius=1, max_cells=1200))
        current_path_color = (245, 245, 245)
        record("current_path", True, current_path_color, self._draw_cells(draw, current_path, xy, current_path_color, radius=2, max_cells=500))
        frontier_raw_cells: List[GridCell] = []
        for frontier in frontiers[:64]:
            frontier_raw_cells.extend(frontier.members)
        frontier_cell_color = (0, 180, 220)
        frontier_cell_count = (
            self._draw_cells(draw, frontier_raw_cells, xy, frontier_cell_color, radius=1, max_cells=500)
            if self.show_frontier_member_cells
            else 0
        )
        record("frontier_member_cells", self.show_frontier_member_cells, frontier_cell_color, frontier_cell_count)
        frontier_center_count = 0
        for frontier in frontiers[:64]:
            self._triangle(draw, xy(frontier.center_grid), (0, 225, 255), radius=5)
            frontier_center_count += 1
        record("frontier_centers", True, (0, 225, 255), frontier_center_count)

        selected_frontier = None
        if nav_decision and nav_decision.frontier_decision:
            selected_frontier = nav_decision.frontier_decision.selected_frontier
        if selected_frontier is not None:
            self._star(draw, xy(selected_frontier.center_grid), (255, 225, 40), radius=8)
        record("selected_frontier", selected_frontier is not None, (255, 225, 40), 1 if selected_frontier is not None else 0)
        candidate_marker_color = (220, 70, 255)
        candidate_marker_count = 0
        if self.show_candidate_markers and nav_decision and nav_decision.target_cells and nav_decision.mode == "candidate":
            candidate_marker_count = self._draw_crosses(draw, nav_decision.target_cells, xy, candidate_marker_color, radius=5, max_cells=16)
        record("candidate_standoff_markers", self.show_candidate_markers, candidate_marker_color, candidate_marker_count)
        planner_target = None
        if current_path:
            planner_target = current_path[-1]
        elif nav_decision and nav_decision.target_cells:
            planner_target = nav_decision.target_cells[0]
        if planner_target is not None:
            self._star(draw, xy(planner_target), (255, 150, 40), radius=7)
        record("planner_target", planner_target is not None, (255, 150, 40), 1 if planner_target is not None else 0)
        selected_id = self._selected_candidate_id(nav_decision)
        object_node_count = 0
        accepted_candidate_count = 0
        if self.show_object_nodes:
            for node in self._visible_map_nodes(object_memory, goal_category, nav_decision)[:300]:
                radius = 8 if selected_id is not None and int(node.node_id) == selected_id else 5
                color = self._candidate_node_color(node, selected_id, nav_decision)
                if color == (40, 220, 90):
                    accepted_candidate_count += 1
                self._dot(draw, xy(node.center_grid), color, radius=radius)
                object_node_count += 1
        record("object_nodes", self.show_object_nodes, (255, 150, 40), object_node_count)
        record("accepted_candidate", self.show_object_nodes, (40, 220, 90), accepted_candidate_count)

        self._draw_cells(
            draw,
            self._mask_cells_in_crop(wall_extension_boundary | accepted_closure, (r0, r1, c0, c1)),
            xy,
            (80, 170, 255),
            radius=3,
            max_cells=2000,
        )
        self._draw_cells(
            draw,
            self._mask_cells_in_crop(door_completion_boundary, (r0, r1, c0, c1)),
            xy,
            (255, 120, 40),
            radius=4,
            max_cells=2000,
        )

        self._draw_agent(draw, xy(current_grid), float(pose[3]) if len(pose) > 3 else 0.0, scale)
        record("agent", True, (255, 60, 60), 1)
        zoom = max(1.0, min(w / max(crop_w, 1), h / max(crop_h, 1)))
        target_count = len(nav_decision.target_cells) if nav_decision is not None else 0
        if self._should_render_voxel_panel(tuple(np.asarray(occupancy).shape[:2])):
            nav_unknown = int(np.count_nonzero(~obs))
            frontier_count = int(sum(len(getattr(frontier, "cells", [])) for frontier in frontiers))
            backend = str(self._room_segmentation_debug.get("voxel_integration_backend", self._room_segmentation_debug.get("integration_backend", "unknown")))
            integrate_ms = self._room_segmentation_debug.get("voxel_integrate_total_ms")
            threads = int(self._room_segmentation_debug.get("voxel_integrate_backend_effective_thread_count", self._room_segmentation_debug.get("voxel_integrate_backend_thread_count", 0)) or 0)
            requested_threads = int(self._room_segmentation_debug.get("voxel_integrate_numba_requested_thread_count", threads) or threads)
            threads_mode = str(self._room_segmentation_debug.get("voxel_integrate_numba_threads_mode", "manual"))
            pass1_ms = float(self._room_segmentation_debug.get("voxel_integrate_pass1_ms", 0.0) or 0.0)
            pass2_ms = float(self._room_segmentation_debug.get("voxel_integrate_pass2_ms", 0.0) or 0.0)
            bucket_ms = float(self._room_segmentation_debug.get("voxel_integrate_event_bucket_ms", 0.0) or 0.0)
            bucket_free_ms = float(self._room_segmentation_debug.get("voxel_integrate_bucket_free_ms", 0.0) or 0.0)
            bucket_occ_ms = float(self._room_segmentation_debug.get("voxel_integrate_bucket_occ_ms", 0.0) or 0.0)
            bucket_sensor_ms = float(self._room_segmentation_debug.get("voxel_integrate_bucket_sensor_ms", 0.0) or 0.0)
            apply_ms = float(self._room_segmentation_debug.get("voxel_integrate_apply_logodds_ms", 0.0) or 0.0)
            sensor_ms = float(self._room_segmentation_debug.get("voxel_integrate_apply_sensor_ms", 0.0) or 0.0)
            project_ms = float(self._room_segmentation_debug.get("voxel_project_navigation_ms", 0.0) or 0.0)
            project_mode = str(self._room_segmentation_debug.get("voxel_project_navigation_mode", "unknown"))
            roomseg_reason = str(self._room_segmentation_debug.get("roomseg_frontier_update_reason", "unknown"))
            fallback = bool(self._room_segmentation_debug.get("voxel_numba_requested_unavailable", False))
            try:
                integrate_text = "%.1fms" % float(integrate_ms)
            except (TypeError, ValueError):
                integrate_text = "NA"
            clearance_min = self._room_segmentation_debug.get("astar_path_min_clearance_m", self._room_segmentation_debug.get("astar_clearance_min_m"))
            try:
                clearance_text = "%.2f" % float(clearance_min)
            except (TypeError, ValueError):
                clearance_text = "NA"
            title = "nav voxel | free=%d occ=%d unk=%d frontier=%d path_clear=%s backend=%s%s th=%d/%d %s integ=%s proj=%.1f/%s roomseg=%s bucket=%.1f f/o/s=%.1f/%.1f/%.1f zoom %.1fx" % (
                int(np.count_nonzero(nav & obs)),
                int(np.count_nonzero(occupancy.astype(bool))),
                nav_unknown,
                frontier_count,
                clearance_text,
                backend,
                " fallback=true" if fallback else "",
                threads,
                requested_threads,
                threads_mode,
                integrate_text,
                project_ms,
                project_mode,
                roomseg_reason,
                bucket_ms,
                bucket_free_ms,
                bucket_occ_ms,
                bucket_sensor_ms,
                zoom,
            )
        else:
            title = "Map / frontiers / A* / goal candidates  zoom %.1fx target_cells=%d" % (zoom, target_count)
        self._label(draw, (10, 8), title[:112], (255, 255, 255))
        self._legend(draw, (10, max(32, map_h_available - 148)))
        if rose_panel_enabled and rose_h > 0:
            rose_panel, rose_layers = self._render_rose_occupancy_panel(
                occupancy=occupancy,
                navigable=navigable,
                observed=observed,
                size=(width, rose_h),
                crop_bounds=(r0, r1, c0, c1),
            )
            final = Image.new("RGB", (width, height), (18, 20, 24))
            final.paste(image, (0, 0))
            final_draw = ImageDraw.Draw(final)
            final_draw.line([(0, map_h_available), (width, map_h_available)], fill=(70, 74, 80), width=divider_h)
            final.paste(rose_panel, (0, map_h_available + divider_h))
            layers.extend(rose_layers)
            self._last_overlay_layers = layers if self.debug_overlay_layers else []
            return final
        self._last_overlay_layers = layers if self.debug_overlay_layers else []
        return image

    def _render_rose_occupancy_panel(
        self,
        *,
        occupancy: np.ndarray,
        navigable: np.ndarray,
        observed: np.ndarray,
        size: Tuple[int, int],
        crop_bounds: Tuple[int, int, int, int],
    ) -> Tuple[Image.Image, List[dict]]:
        width, height = size
        shape = tuple(np.asarray(occupancy).shape[:2])
        if self._should_render_voxel_panel(shape):
            return self._render_voxel_roomseg_panel(
                occupancy=occupancy,
                navigable=navigable,
                observed=observed,
                size=size,
                crop_bounds=crop_bounds,
            )
        if self._has_height_profile_roomseg_layers(shape):
            return self._render_height_profile_roomseg_panel(
                occupancy=occupancy,
                navigable=navigable,
                observed=observed,
                size=size,
                crop_bounds=crop_bounds,
            )
        occ = np.asarray(occupancy, dtype=bool)
        nav = np.asarray(navigable, dtype=bool)
        obs = np.asarray(observed, dtype=bool)
        roomseg_free = self._room_debug_array("initial_roomseg_free", shape, bool)
        roomseg_occupied = self._room_debug_array("initial_roomseg_occupied", shape, bool)
        pass2_extension_intersection_targets = self._room_debug_array("pass2_extension_intersection_targets", shape, bool)
        pass2_line_extension_completion = self._room_debug_array("pass2_line_extension_completion", shape, bool)
        wall_target_after_line_extension = self._room_debug_array("wall_target_after_line_extension", shape, bool)
        completed_wall_after_line_extension = self._room_debug_array("completed_wall_after_line_extension", shape, bool)
        clipped_outside_nav = self._room_debug_array("vertical_free_clipped_outside_navigation_map", shape, bool)
        free_wall_conflict = self._room_debug_array("free_wall_conflict_map_before_sanitize", shape, bool)
        sanitized_free = self._room_debug_array("roomseg_sanitized_free", shape, bool)
        sanitized_wall = self._room_debug_array("roomseg_sanitized_wall", shape, bool)
        terminal_wall_roomseg = self._room_debug_array("terminal_wall_roomseg_mask", shape, bool)
        pre_extension_door_detected = self._room_debug_array("pre_extension_door_detected_map", shape, bool)
        pre_extension_door_cut = self._room_debug_array("pre_extension_door_cut_mask", shape, bool)
        strict_pre_extension_door_cut = self._room_debug_array("strict_pre_extension_door_cut_mask", shape, bool)
        partial_door_seed = self._room_debug_array("partial_door_seed_mask", shape, bool)
        partial_door_line = self._room_debug_array("partial_door_line_mask", shape, bool)
        partial_door_extension_cut = self._room_debug_array("partial_door_extension_cut_mask", shape, bool)
        rejected_door_extension = self._room_debug_array("rejected_door_extension_mask", shape, bool)
        original_step_boundary = self._room_debug_array("original_step1_step2_virtual_boundary_map", shape, bool)
        wall_extension_boundary = self._room_debug_array("wall_extension_boundary_mask", shape, bool)
        door_completion_boundary = self._room_debug_array("door_completion_boundary_mask", shape, bool)
        if not np.any(wall_extension_boundary):
            wall_extension_boundary = original_step_boundary
        if not np.any(door_completion_boundary):
            door_completion_boundary = pre_extension_door_cut
        pre_extension_room_labels = self._room_debug_array("pre_extension_room_label_map", shape, np.int32)
        pre_extension_room_boundary = self._room_label_adjacency_boundary(pre_extension_room_labels)
        initial_unknown_after_fusion = self._room_debug_array("initial_roomseg_unknown_after_fusion", shape, bool)
        vertical_free_room_domain = self._room_debug_array("vertical_free_room_domain", shape, bool)
        vertical_occupied_0p2_2p0 = self._room_debug_array("vertical_occupied_0p2_2p0", shape, bool)
        vertical_observed = self._room_debug_array("vertical_observed_map", shape, bool)
        vertical_observed_0p2_2p0 = self._room_debug_array("vertical_observed_0p2_2p0", shape, bool)
        vertical_unknown_before_overlay = self._room_debug_array("vertical_unknown_before_overlay", shape, bool)
        roomseg_ray_covered_count = self._room_debug_array("roomseg_ray_covered_count", shape, np.uint16)
        roomseg_terminal_wall_count = self._room_debug_array("roomseg_terminal_wall_count", shape, np.uint16)
        roomseg_terminal_wall_splat = self._room_debug_array("roomseg_terminal_wall_splat", shape, bool)
        ray_valid_wall = self._room_debug_array("ray_valid_wall_inference", shape, bool)
        unknown_removed_by_ray_wall = self._room_debug_array("unknown_removed_by_ray_wall", shape, bool)
        nav_raw_obstacle = self._room_debug_array("nav_raw_obstacle", shape, bool)
        static_structural = self._room_debug_array("roomseg_static_structural_occupied", shape, bool)
        nav_obstacle_overlay_accepted = self._room_debug_array("nav_obstacle_overlay_accepted", shape, bool)
        walls_rescued_from_unknown = self._room_debug_array("walls_rescued_from_unknown", shape, bool)
        vertical_free_over_nav_obstacle = self._room_debug_array("vertical_free_over_nav_obstacle", shape, bool)
        repaired_free = self._room_debug_array("repaired_roomseg_free", shape, bool)
        repaired_occupied = self._room_debug_array("repaired_roomseg_occupied", shape, bool)
        boundary_map = self._room_debug_array("boundary_map", shape, bool)
        virtual_boundary = self._room_debug_array("virtual_boundary_map", shape, bool)
        accepted_closure = self._room_debug_array("accepted_closure_map", shape, bool)
        vertical_or_free = self._room_debug_array("vertical_or_free_map", shape, bool)
        vertical_carved = self._room_debug_array("vertical_carved_map", shape, bool)
        structural = self._room_debug_array("structural_wall_mask", shape, bool)
        clean_structure = self._room_debug_array("clean_structure_map", shape, bool)
        rejected_structure = self._room_debug_array("structural_component_rejected_mask", shape, bool)
        interior_clutter = self._room_debug_array("interior_clutter_suppression_mask", shape, bool)
        furniture_suppressed = self._room_debug_array("furniture_suppression_mask", shape, bool)
        suppressed_clutter = rejected_structure | interior_clutter | furniture_suppressed
        wall_conf = self._room_debug_array("wall_confidence_map", shape, np.float32)
        context_labels = self._room_debug_array("context_room_label_map", shape, np.int32)
        final_labels = self._room_debug_array("final_room_label_map", shape, np.int32)
        context_absorbed = (context_labels > 0) & (final_labels <= 0)
        threshold = float(self._room_segmentation_debug.get("wall_confidence_threshold", 0.55) or 0.55)
        wall_conf_hot = wall_conf >= threshold if wall_conf.shape == shape else np.zeros(shape, dtype=bool)
        debug_only = bool(self._room_segmentation_debug.get("roomseg_debug_only", False))
        vertical_debug_free = (
            vertical_free_room_domain
            if np.any(vertical_free_room_domain)
            else (repaired_free if np.any(repaired_free) else roomseg_free)
        )

        has_rose_input = bool(
            np.any(vertical_debug_free)
            or np.any(roomseg_free)
            or np.any(roomseg_occupied)
            or np.any(pass2_extension_intersection_targets)
            or np.any(pass2_line_extension_completion)
            or np.any(wall_target_after_line_extension)
            or np.any(completed_wall_after_line_extension)
            or np.any(clipped_outside_nav)
            or np.any(free_wall_conflict)
            or np.any(sanitized_free)
            or np.any(sanitized_wall)
            or np.any(pre_extension_door_detected)
            or np.any(pre_extension_door_cut)
            or np.any(wall_extension_boundary)
            or np.any(door_completion_boundary)
            or np.any(partial_door_seed)
            or np.any(partial_door_line)
            or np.any(partial_door_extension_cut)
            or np.any(rejected_door_extension)
            or np.any(pre_extension_room_labels > 0)
            or np.any(vertical_observed)
            or np.any(vertical_observed_0p2_2p0)
            or np.any(ray_valid_wall)
            or np.any(roomseg_terminal_wall_splat)
            or np.any(nav_raw_obstacle)
            or np.any(static_structural)
            or np.any(nav_obstacle_overlay_accepted)
            or np.any(vertical_or_free)
            or np.any(vertical_carved)
            or np.any(structural)
            or np.any(clean_structure)
            or np.any(suppressed_clutter)
            or np.any(wall_conf > 0.0)
        )
        rose_occupied = (
            repaired_occupied
            if debug_only and np.any(repaired_occupied)
            else (
                completed_wall_after_line_extension
                if np.any(completed_wall_after_line_extension)
                else (
                    wall_target_after_line_extension
                    if np.any(wall_target_after_line_extension)
                    else (roomseg_occupied if np.any(roomseg_occupied) else (structural if np.any(structural) else (clean_structure | wall_conf_hot)))
                )
            )
        )
        vertical_free_overridden_occupied = occ & vertical_or_free & ~roomseg_occupied
        canvas = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        if debug_only:
            canvas[:, :] = (12, 14, 18)
            canvas[obs] = (28, 31, 36)
            canvas[initial_unknown_after_fusion] = (8, 10, 14)
            canvas[vertical_debug_free] = (166, 170, 174)
            canvas[rose_occupied | boundary_map] = (0, 0, 0)
            label_palette = [
                (125, 104, 235),
                (100, 185, 245),
                (95, 210, 155),
                (235, 185, 80),
                (235, 120, 145),
                (180, 140, 240),
            ]
            for label in np.unique(final_labels):
                if int(label) <= 0:
                    continue
                mask = final_labels == int(label)
                color = np.asarray(label_palette[(int(label) - 1) % len(label_palette)], dtype=np.float32)
                base = canvas[mask].astype(np.float32)
                canvas[mask] = np.clip(base * 0.45 + color * 0.55, 0, 255).astype(np.uint8)
            canvas[virtual_boundary] = (245, 245, 245)
            canvas[accepted_closure | original_step_boundary | wall_extension_boundary] = (80, 170, 255)
            canvas[door_completion_boundary] = (255, 120, 40)
            canvas[ray_valid_wall] = (255, 80, 40)
            canvas[roomseg_terminal_wall_splat] = (255, 135, 25)
            canvas[terminal_wall_roomseg] = (255, 150, 35)
            canvas[clipped_outside_nav] = (45, 135, 255)
            canvas[free_wall_conflict] = (255, 45, 45)
            canvas[pass2_extension_intersection_targets] = (255, 240, 0)
            canvas[pass2_line_extension_completion] = (255, 0, 0)
            canvas[pre_extension_room_boundary] = (120, 180, 255)
            canvas[pre_extension_door_detected] = (0, 210, 255)
            canvas[strict_pre_extension_door_cut] = (255, 120, 40)
            canvas[partial_door_seed] = (135, 245, 255)
            canvas[partial_door_line] = (60, 250, 180)
            canvas[partial_door_extension_cut] = (255, 120, 40)
            canvas[rejected_door_extension] = (255, 60, 180)
            canvas[pre_extension_door_cut & ~door_completion_boundary] = (255, 120, 40)
            canvas[nav_obstacle_overlay_accepted] = (230, 40, 230)
            canvas[walls_rescued_from_unknown] = (255, 35, 35)
            canvas[vertical_free_over_nav_obstacle] = (45, 135, 255)
        else:
            canvas[:, :] = (36, 40, 48)
            canvas[nav] = (86, 92, 96)
            canvas[obs & nav] = (118, 126, 130)
            canvas[obs & ~nav] = (54, 56, 60)
            if not has_rose_input:
                canvas[occ] = (8, 8, 8)
            canvas[roomseg_free] = (150, 156, 160)
            canvas[suppressed_clutter] = (58, 82, 132)
            canvas[vertical_or_free] = (116, 96, 74)
            canvas[vertical_carved] = (126, 104, 75)
            canvas[vertical_free_overridden_occupied] = (225, 132, 45)
            canvas[ray_valid_wall] = (255, 80, 40)
            canvas[roomseg_terminal_wall_splat] = (255, 135, 25)
            canvas[terminal_wall_roomseg] = (255, 150, 35)
            canvas[clipped_outside_nav] = (45, 135, 255)
            canvas[free_wall_conflict] = (255, 45, 45)
            canvas[pass2_extension_intersection_targets] = (255, 240, 0)
            canvas[pass2_line_extension_completion] = (255, 0, 0)
            canvas[pre_extension_room_boundary] = (120, 180, 255)
            canvas[pre_extension_door_detected] = (0, 210, 255)
            canvas[virtual_boundary] = (245, 245, 245)
            canvas[accepted_closure | original_step_boundary | wall_extension_boundary] = (80, 170, 255)
            canvas[door_completion_boundary] = (255, 120, 40)
            canvas[strict_pre_extension_door_cut] = (255, 120, 40)
            canvas[partial_door_seed] = (135, 245, 255)
            canvas[partial_door_line] = (60, 250, 180)
            canvas[partial_door_extension_cut] = (255, 120, 40)
            canvas[rejected_door_extension] = (255, 60, 180)
            canvas[pre_extension_door_cut & ~door_completion_boundary] = (255, 120, 40)
            canvas[unknown_removed_by_ray_wall] = (255, 35, 35)
            canvas[wall_conf_hot] = (255, 105, 75)
            canvas[rose_occupied] = (0, 0, 0)
            canvas[clean_structure] = (255, 190, 70)
            canvas[context_absorbed] = (120, 210, 255)
            canvas[nav_obstacle_overlay_accepted] = (230, 40, 230)
            canvas[walls_rescued_from_unknown] = (255, 35, 35)
            canvas[vertical_free_over_nav_obstacle] = (45, 135, 255)

        r0, r1, c0, c1 = crop_bounds
        crop = canvas[r0:r1, c0:c1]
        crop_h, crop_w = crop.shape[:2]
        label_h = 36
        margin = 8
        available_h = max(1, int(height) - label_h - margin)
        scale = min((width - 2 * margin) / max(crop_w, 1), available_h / max(crop_h, 1))
        map_w, map_h = max(1, int(crop_w * scale)), max(1, int(crop_h * scale))
        ox = (width - map_w) // 2
        oy = label_h + max(0, (available_h - map_h) // 2)
        image = Image.new("RGB", (width, height), (15, 17, 21))
        image.paste(Image.fromarray(crop).resize((map_w, map_h), Image.NEAREST), (ox, oy))
        draw = ImageDraw.Draw(image)

        def xy(cell: GridCell) -> Tuple[int, int]:
            r, c = int(cell[0]), int(cell[1])
            return int(ox + (c - c0 + 0.5) * scale), int(oy + (r - r0 + 0.5) * scale)

        window_gap_count = self._draw_rose_gap_markers(
            draw,
            list(self._room_segmentation_debug.get("repaired_window_gaps") or []),
            xy,
            crop_bounds,
            (255, 80, 130),
        )
        doorway_gap_count = self._draw_rose_gap_markers(
            draw,
            list(self._room_segmentation_debug.get("verified_doorway_gaps") or []),
            xy,
            crop_bounds,
            (80, 255, 130),
        )
        wall_line_count, wall_extension_count = self._draw_roomseg_wall_debug_lines(draw, xy, crop_bounds)
        title = "vertical-free roomseg debug" if debug_only else "ROSE roomseg input after vertical-free operation"
        if has_rose_input:
            title += " | occupied=%d vfree=%d ray_wall=%d room_pixels=%d closures=%d win=%d door=%d" % (
                int(np.count_nonzero(rose_occupied)),
                int(np.count_nonzero(vertical_debug_free)),
                int(np.count_nonzero(ray_valid_wall)),
                int(np.count_nonzero(final_labels > 0)),
                int(np.count_nonzero(accepted_closure | virtual_boundary)),
                int(window_gap_count),
                int(doorway_gap_count),
            )
        else:
            title += " | waiting for ROSE debug; showing current map underlay"
        self._label(draw, (8, 5), title[:120], (255, 255, 255))
        self._rose_legend(draw, (8, max(label_h + 4, height - 50)))

        layers = [
            self._overlay_record(
                "rose_occupancy_map",
                True,
                (0, 0, 0),
                int(np.count_nonzero(rose_occupied)),
                "room segmentation occupancy after vertical-free operation, shown below the runtime occupancy map",
                has_rose_input=has_rose_input,
                current_occupancy_underlay_cells=int(np.count_nonzero(occ)),
                current_navigable_underlay_cells=int(np.count_nonzero(nav)),
                roomseg_free_cells=int(np.count_nonzero(roomseg_free)),
                vertical_free_overridden_occupied_cells=int(np.count_nonzero(vertical_free_overridden_occupied)),
            ),
            self._overlay_record(
                "roomseg_ray_valid_wall_inference",
                True,
                (255, 80, 40),
                int(np.count_nonzero(ray_valid_wall)),
                "roomseg occupied cells inferred only from vertical occupied evidence or valid depth-ray terminal wall evidence",
                terminal_wall_cells=int(np.count_nonzero(roomseg_terminal_wall_count)),
                terminal_wall_splat_cells=int(np.count_nonzero(roomseg_terminal_wall_splat)),
                ray_covered_cells=int(np.count_nonzero(roomseg_ray_covered_count)),
                unknown_removed_by_ray_wall_cells=int(np.count_nonzero(unknown_removed_by_ray_wall)),
            ),
            self._overlay_record(
                "roomseg_vertical_free_outside_navigation",
                bool(np.any(clipped_outside_nav)),
                (45, 135, 255),
                int(np.count_nonzero(clipped_outside_nav)),
                "vertical-free cells clipped before VFGC because they were outside navigation free",
            ),
            self._overlay_record(
                "roomseg_free_wall_conflict",
                bool(np.any(free_wall_conflict)),
                (255, 45, 45),
                int(np.count_nonzero(free_wall_conflict)),
                "free/wall overlap removed before VFGC so wall cores cannot be erased by vertical free",
            ),
            self._overlay_record(
                "roomseg_sanitized_free",
                bool(np.any(sanitized_free)),
                (166, 170, 174),
                int(np.count_nonzero(sanitized_free)),
                "free mask after input sanitizer",
                subset_navigation_ok=bool(self._room_segmentation_debug.get("sanitized_free_subset_navigation_ok", False)),
            ),
            self._overlay_record(
                "roomseg_sanitized_wall",
                bool(np.any(sanitized_wall)),
                (10, 10, 10),
                int(np.count_nonzero(sanitized_wall)),
                "wall mask after input sanitizer, including terminal ray-wall evidence",
                terminal_wall_roomseg_cells=int(np.count_nonzero(terminal_wall_roomseg)),
            ),
            self._overlay_record(
                "roomseg_nav_obstacle_overlay_accepted",
                True,
                (230, 40, 230),
                int(np.count_nonzero(nav_obstacle_overlay_accepted)),
                "debug-only audit; strict ray-valid roomseg must keep this at zero",
                nav_raw_obstacle_cells=int(np.count_nonzero(nav_raw_obstacle)),
                roomseg_static_structural_occupied_cells=int(np.count_nonzero(static_structural)),
            ),
            self._overlay_record(
                "roomseg_walls_rescued_from_unknown",
                True,
                (255, 35, 35),
                int(np.count_nonzero(walls_rescued_from_unknown)),
                "wall cells rescued from roomseg unknown before the structural map is consumed",
            ),
            self._overlay_record(
                "roomseg_vertical_free_over_nav_obstacle",
                True,
                (45, 135, 255),
                int(np.count_nonzero(vertical_free_over_nav_obstacle)),
                "vertical-free cells that override raw/static navigation obstacle evidence",
            ),
            self._overlay_record(
                "vertical_free_roomseg_input",
                True,
                (150, 156, 160),
                int(np.count_nonzero(vertical_debug_free)),
                "vertical-free room segmentation input used by the current latest room segmenter",
                vertical_observed_cells=int(np.count_nonzero(vertical_observed)),
                vertical_observed_0p2_2p0_cells=int(np.count_nonzero(vertical_observed_0p2_2p0)),
                vertical_occupied_0p2_2p0_cells=int(np.count_nonzero(vertical_occupied_0p2_2p0)),
                vertical_unknown_before_overlay_cells=int(np.count_nonzero(vertical_unknown_before_overlay)),
                initial_roomseg_unknown_after_fusion_cells=int(np.count_nonzero(initial_unknown_after_fusion)),
            ),
            self._overlay_record(
                "vertical_free_room_labels",
                bool(np.any(final_labels > 0)),
                (125, 104, 235),
                int(np.count_nonzero(final_labels > 0)),
                "final room labels overlaid on the vertical-free roomseg input",
            ),
            self._overlay_record(
                "vertical_free_gap_closure_boundaries",
                bool(np.any(virtual_boundary)),
                (245, 245, 245),
                int(np.count_nonzero(virtual_boundary)),
                "final virtual boundaries after combining wall extension and detected door completion",
                original_step1_step2_virtual_boundary_cells=int(np.count_nonzero(original_step_boundary)),
            ),
            self._overlay_record(
                "wall_extension_boundaries",
                bool(np.any(wall_extension_boundary | accepted_closure)),
                (80, 170, 255),
                int(np.count_nonzero(wall_extension_boundary | accepted_closure)),
                "boundaries produced by the original wall-endpoint/wall-line extension closure path",
            ),
            self._overlay_record(
                "door_completion_boundaries",
                bool(np.any(door_completion_boundary)),
                (255, 120, 40),
                int(np.count_nonzero(door_completion_boundary)),
                "boundaries produced by detected strict/partial door completion, separated from wall-line extension",
                strict_door_completion_cells=int(np.count_nonzero(strict_pre_extension_door_cut)),
                partial_door_completion_cells=int(np.count_nonzero(partial_door_extension_cut)),
            ),
            self._overlay_record(
                "strict_pre_extension_door_cuts",
                bool(np.any(strict_pre_extension_door_cut)),
                (255, 120, 40),
                int(np.count_nonzero(strict_pre_extension_door_cut)),
                "door cuts from the original strict pre-extension pattern rules",
            ),
            self._overlay_record(
                "partial_door_seed_points",
                bool(np.any(partial_door_seed)),
                (135, 245, 255),
                int(np.count_nonzero(partial_door_seed)),
                "partial door-like occupied seed cells detected before line extension",
            ),
            self._overlay_record(
                "accepted_partial_door_extension_lines",
                bool(np.any(partial_door_line)),
                (60, 250, 180),
                int(np.count_nonzero(partial_door_line)),
                "accepted partial-door lines extended to structural wall anchors",
            ),
            self._overlay_record(
                "partial_door_extension_cuts",
                bool(np.any(partial_door_extension_cut)),
                (255, 120, 40),
                int(np.count_nonzero(partial_door_extension_cut)),
                "accepted partial-door cut cells; these are ORed only into the final boundary",
            ),
            self._overlay_record(
                "rejected_partial_door_extension_lines",
                bool(np.any(rejected_door_extension)),
                (255, 60, 180),
                int(np.count_nonzero(rejected_door_extension)),
                "partial-door extension lines rejected by width/unknown/wall/other-door guards",
                reject_reason_counts=dict(self._room_segmentation_debug.get("partial_door_line_reject_reason_counts") or {}),
            ),
            self._overlay_record(
                "segmentation_degenerate_warning",
                bool(self._room_segmentation_debug.get("segmentation_degenerate_one_room", False)),
                (255, 80, 80),
                1 if bool(self._room_segmentation_debug.get("segmentation_degenerate_one_room", False)) else 0,
                "VFGC produced one large room without any virtual boundary; inspect sanitizer and terminal-wall evidence",
            ),
            self._overlay_record(
                "pass2_extension_intersection_targets",
                bool(np.any(pass2_extension_intersection_targets)),
                (255, 240, 0),
                int(np.count_nonzero(pass2_extension_intersection_targets)),
                "virtual neck targets generated from crossing pass2 line-extension probes",
            ),
            self._overlay_record(
                "pass2_line_extension_completion",
                bool(np.any(pass2_line_extension_completion)),
                (255, 0, 0),
                int(np.count_nonzero(pass2_line_extension_completion)),
                "topology-accepted pass2 line extensions merged into the completed roomseg wall target",
                completed_wall_after_line_extension_cells=int(np.count_nonzero(completed_wall_after_line_extension)),
            ),
            self._overlay_record(
                "roomseg_wall_lines_red",
                wall_line_count > 0,
                (255, 35, 35),
                int(wall_line_count),
                "solid bright-red filtered wall lines used by the current room segmentation pass",
            ),
            self._overlay_record(
                "roomseg_wall_extensions_red_dashed",
                wall_extension_count > 0,
                (255, 0, 0),
                int(wall_extension_count),
                "pure-red dashed attempted wall-line extensions with dark outline; drawn whether or not the final split succeeds",
            ),
            self._overlay_record(
                "rose_vertical_free_overrides",
                True,
                (225, 132, 45),
                int(np.count_nonzero(vertical_free_overridden_occupied)),
                "runtime occupied cells changed to free in the ROSE-only room segmentation input",
            ),
            self._overlay_record(
                "rose_vertical_carved_map",
                True,
                (126, 104, 75),
                int(np.count_nonzero(vertical_carved)),
                "furniture-suppressed vertical carved map passed into ROSE preprocessing",
            ),
            self._overlay_record(
                "rose_wall_confidence_map",
                True,
                (255, 105, 75),
                int(np.count_nonzero(wall_conf_hot)),
                "wall-confidence cells above room segmentation threshold",
                threshold=threshold,
            ),
            self._overlay_record(
                "rose_structural_rejected_clutter",
                True,
                (58, 82, 132),
                int(np.count_nonzero(suppressed_clutter)),
                "occupied clutter/furniture components rejected before ROSE structural occupancy",
            ),
            self._overlay_record(
                "rose_repaired_window_gaps",
                True,
                (255, 80, 130),
                int(window_gap_count),
                "window/non-traversable gaps closed as walls for room segmentation",
            ),
            self._overlay_record(
                "rose_verified_doorway_gaps",
                True,
                (80, 255, 130),
                int(doorway_gap_count),
                "floor-traversable doorway gaps recorded as portals",
            ),
            self._overlay_record(
                "roomseg_context_absorbed_nav_free",
                bool(np.any(context_absorbed)),
                (120, 210, 255),
                int(np.count_nonzero(context_absorbed)),
                "navigation-free cells absorbed only into room context overlay",
            ),
        ]
        return image, layers

    def _has_height_profile_roomseg_layers(self, shape: Tuple[int, int]) -> bool:
        keys = (
            "height_profile_vertical_free_xy",
            "height_profile_wall_xy",
            "height_profile_unknown_xy",
            "height_profile_final_room_label_map",
            "height_profile_door_cut_mask",
            "height_profile_step2_extension_separator_map",
        )
        for key in keys:
            raw = self._room_segmentation_debug.get(key)
            if raw is None:
                continue
            try:
                arr = np.asarray(raw)
            except Exception:
                continue
            if arr.shape == tuple(shape) and np.any(arr):
                return True
        return False

    def _is_voxel_roomseg_debug(self, shape: Tuple[int, int]) -> bool:
        return self._should_render_voxel_panel(shape)

    def _should_render_voxel_panel(self, shape: Tuple[int, int]) -> bool:
        backend_text = " ".join(
            str(self._room_segmentation_debug.get(key, ""))
            for key in (
                "backend",
                "actual_backend",
                "source_backend",
                "roomseg_backend",
                "algorithm",
                "source",
                "context_source",
                "room_map_mode",
                "frontier_source",
                "source_grid",
            )
        ).lower()
        if "voxel_occupancy_door_wall_v29" in backend_text or "voxel_occupancy_door_wall_v9" in backend_text or "voxel_vertical_free" in backend_text:
            return True
        if "voxel" in backend_text:
            return True
        for key in (
            "voxel_vertical_free_xy",
            "voxel_wall_xy",
            "voxel_unknown_xy",
            "voxel_final_room_label_map",
            "voxel_door_cut_mask",
            "voxel_step2_extension_separator_map",
        ):
            raw = self._room_segmentation_debug.get(key)
            if raw is None:
                continue
            try:
                arr = np.asarray(raw)
            except Exception:
                continue
            if arr.shape == tuple(shape):
                return True
        return False

    def _render_voxel_roomseg_panel(
        self,
        *,
        occupancy: np.ndarray,
        navigable: np.ndarray,
        observed: np.ndarray,
        size: Tuple[int, int],
        crop_bounds: Tuple[int, int, int, int],
    ) -> Tuple[Image.Image, List[dict]]:
        _ = navigable, observed
        width, height = size
        shape = tuple(np.asarray(occupancy).shape[:2])
        required = (
            "voxel_vertical_free_xy",
            "voxel_wall_xy",
            "voxel_unknown_xy",
            "voxel_final_room_label_map",
            "voxel_door_cut_mask",
            "voxel_step2_extension_separator_map",
            "voxel_display_wall_xy",
        )
        missing = []
        for key in required:
            raw = self._room_segmentation_debug.get(key)
            try:
                ok = raw is not None and np.asarray(raw).shape == tuple(shape)
            except Exception:
                ok = False
            if not ok:
                missing.append(key)
        vertical_free = self._room_debug_array("voxel_vertical_free_xy", shape, bool)
        wall = self._room_debug_array("voxel_wall_xy", shape, bool)
        raw_wall = self._room_debug_array("voxel_raw_occupied_wall_support_xy", shape, bool)
        strict_raw_wall = self._room_debug_array("voxel_strict_raw_wall_xy", shape, bool)
        wall_line_support = self._room_debug_array("voxel_wall_line_support_xy", shape, bool)
        wall_suppressed_by_free = self._room_debug_array("voxel_wall_suppressed_by_free_xy", shape, bool)
        ratio_wall_debug = self._room_debug_array("voxel_wall_ratio_raw_xy", shape, bool)
        if not np.any(ratio_wall_debug):
            ratio_wall_debug = self._room_debug_array("voxel_ratio_wall_debug_xy", shape, bool)
        nonstructural_occupied = self._room_debug_array("voxel_nonstructural_occupied_xy", shape, bool)
        projected_wall = self._room_debug_array("voxel_projected_structural_wall_map", shape, bool)
        if not np.any(projected_wall):
            projected_wall = self._room_debug_array("voxel_projected_wall_map", shape, bool)
        if not np.any(projected_wall):
            projected_wall = self._room_debug_array("voxel_wall_projected_xy", shape, bool)
        anchor_projected_wall = self._room_debug_array("voxel_anchor_projected_wall_map", shape, bool)
        rejected_wall_support = self._room_debug_array("voxel_wall_projection_rejected_support_map", shape, bool)
        unknown_dominant = self._room_debug_array("voxel_unknown_dominant_xy", shape, bool)
        unknown_rejected_wall = self._room_debug_array("voxel_wall_support_rejected_unknown_xy", shape, bool)
        unknown_gated_wall = self._room_debug_array("voxel_wall_support_unknown_gated_xy", shape, bool)
        unknown = self._room_debug_array("voxel_unknown_xy", shape, bool)
        outside_boundary = self._room_debug_array("voxel_outside_xy", shape, bool)
        outside_candidate = self._room_debug_array("voxel_outside_candidate_xy", shape, bool)
        outside_candidate_base = self._room_debug_array("voxel_outside_candidate_base_xy", shape, bool)
        outside_disconnected_candidate = self._room_debug_array("voxel_outside_disconnected_candidate_xy", shape, bool)
        outside_pre_traversible = self._room_debug_array("voxel_outside_pre_traversible_xy", shape, bool)
        outside_robot_component = self._room_debug_array("voxel_outside_robot_component_xy", shape, bool)
        outside_connected_rejected = self._room_debug_array("voxel_outside_connected_to_robot_rejected_xy", shape, bool)
        outside_floor_seen = self._room_debug_array("voxel_outside_floor_frustum_seen_xy", shape, bool)
        outside_holdout = self._room_debug_array("voxel_outside_holdout_floor_unseen_xy", shape, bool)
        conflict = self._room_debug_array("voxel_free_wall_conflict_xy", shape, bool)
        line_wall = self._room_debug_array("voxel_filtered_wall_line_mask", shape, bool)
        real_wall = self._room_debug_array("voxel_real_wall_barrier_map", shape, bool)
        door_seed = self._room_debug_array("voxel_door_seed_mask", shape, bool)
        raw_door_seed = self._room_debug_array("voxel_raw_door_seed_mask", shape, bool)
        if np.any(raw_door_seed):
            door_seed = raw_door_seed
        extensible_door_seed = self._room_debug_array("voxel_extensible_door_seed_group_mask", shape, bool)
        nonextensible_door_seed = self._room_debug_array("voxel_nonextensible_door_seed_mask", shape, bool)
        step2_block = self._room_debug_array("voxel_step2_block_mask", shape, bool)
        door_primitive = self._room_debug_array("voxel_door_seed_line_primitive_mask", shape, bool)
        door_extensible_primitive = self._room_debug_array("voxel_door_extensible_primitive_mask", shape, bool)
        door_rejected_primitive = self._room_debug_array("voxel_door_rejected_primitive_mask", shape, bool)
        door_attempt = self._room_debug_array("voxel_door_extension_attempt_all_mask", shape, bool)
        if not np.any(door_attempt):
            door_attempt = self._room_debug_array("voxel_door_trial_candidate_lines_map", shape, bool)
        door_attempt_rejected = self._room_debug_array("voxel_door_extension_attempt_rejected_mask", shape, bool)
        if not np.any(door_attempt_rejected):
            door_attempt_rejected = self._room_debug_array("voxel_door_trial_rejected_lines_map", shape, bool)
        accepted_door_visual = self._room_debug_array("voxel_accepted_door_centerline_mask", shape, bool)
        door_visual_all = self._room_debug_array("voxel_door_provisional_accepted_visual_mask", shape, bool)
        if not np.any(door_visual_all):
            door_visual_all = self._room_debug_array("voxel_door_current_centerline_visual_mask", shape, bool)
        if not np.any(door_visual_all):
            door_visual_all = self._room_debug_array("voxel_door_centerline_visual_mask", shape, bool)
        if not np.any(door_visual_all):
            door_visual_all = accepted_door_visual.copy()
        door_visual = accepted_door_visual.copy()
        door_visual_only = self._room_debug_array("voxel_door_visual_only_mask", shape, bool)
        if not np.any(door_visual_only):
            door_visual_only = door_visual_all & ~door_visual & ~self._room_debug_array("voxel_door_cut_mask", shape, bool)
        door_geometry_warning = self._room_debug_array("voxel_door_geometry_warning_cut_mask", shape, bool)
        door_geometry_only = self._room_debug_array("voxel_door_geometry_only_mask", shape, bool)
        door_attachment_only = self._room_debug_array("voxel_door_attachment_only_mask", shape, bool)
        door_not_closed = self._room_debug_array("voxel_door_cut_not_closed_to_wall_mask", shape, bool)
        door_topology_effective = self._room_debug_array("voxel_door_partition_effective_verified_mask", shape, bool)
        if not np.any(door_topology_effective):
            door_topology_effective = self._room_debug_array("voxel_door_topology_effective_cut_mask", shape, bool)
        if not np.any(door_topology_effective):
            door_topology_effective = self._room_debug_array("voxel_door_final_cut_mask", shape, bool)
        door_partition_candidate = self._room_debug_array("voxel_door_partition_cut_candidate_mask", shape, bool)
        door_partition_rejected = self._room_debug_array("voxel_door_partition_cut_rejected_mask", shape, bool)
        door_topology_warning = self._room_debug_array("voxel_door_topology_warning_cut_mask", shape, bool)
        stable_door_cut = self._room_debug_array("voxel_stable_door_cut_mask", shape, bool)
        stable_door_visual = self._room_debug_array("voxel_stable_door_visual_mask", shape, bool)
        door_cut = self._room_debug_array("voxel_door_cut_mask", shape, bool)
        if not np.any(door_cut):
            door_cut = self._room_debug_array("voxel_door_partition_cut_accepted_mask", shape, bool)
        door_visual_debug = door_visual_all & ~door_topology_effective & ~stable_door_cut
        door_trial = self._room_debug_array("voxel_door_trial_candidate_lines_map", shape, bool)
        door_trial_rejected = self._room_debug_array("voxel_door_trial_rejected_lines_map", shape, bool)
        door_selected = self._room_debug_array("voxel_door_selected_candidate_lines_map", shape, bool)
        step1 = self._room_debug_array("voxel_step1_wall_gap_fill_map", shape, bool)
        step1_completed = self._room_debug_array("voxel_step1_completed_wall_map", shape, bool)
        if not np.any(step1_completed):
            step1_completed = self._room_debug_array("voxel_wall_after_step1_map", shape, bool)
        step2 = self._room_debug_array("voxel_step2_extension_separator_map", shape, bool)
        stable_step2 = self._room_debug_array("voxel_stable_step2_separator_mask", shape, bool)
        projected_step2_source = self._room_debug_array("voxel_step2_projected_source_line_map", shape, bool)
        step2_hits = self._room_debug_array("voxel_step2_extension_hits_all_map", shape, bool)
        if not np.any(step2_hits):
            step2_hits = self._room_debug_array("voxel_step2_extension_candidate_map", shape, bool)
        step2_pre_topology = self._room_debug_array("voxel_step2_extension_hits_pre_topology_map", shape, bool)
        step2_candidate = self._room_debug_array("voxel_step2_separator_candidates_pre_topology_map", shape, bool)
        step2_partition_candidate = self._room_debug_array("voxel_step2_partition_cut_candidate_map", shape, bool)
        if not np.any(step2_partition_candidate):
            step2_partition_candidate = step2_candidate
        step2_partition_accepted = self._room_debug_array("voxel_step2_partition_cut_accepted_map", shape, bool)
        if np.any(step2_partition_accepted):
            step2 = step2_partition_accepted
        step2_topology_rejected = self._room_debug_array("voxel_step2_topology_rejected_separator_map", shape, bool)
        rejected = self._room_debug_array("voxel_rejected_door_centerline_mask", shape, bool) | self._room_debug_array("voxel_step2_rejected_extension_map", shape, bool)
        labels = self._room_debug_array("voxel_final_room_label_map", shape, np.int32)
        frontier = self._room_debug_array("frontier_map", shape, bool)
        if not np.any(frontier):
            frontier = self._room_debug_array("voxel_frontier", shape, bool)
        if not np.any(frontier):
            frontier = self._room_debug_array("voxel_frontier_mask", shape, bool)
        show_diag = bool(self._room_segmentation_debug.get("voxel_show_wall_diagnostics", False))
        clean_display_wall = self._room_debug_array("voxel_display_wall_xy", shape, bool)

        canvas = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        canvas[:, :] = (30, 32, 36)
        canvas[unknown] = (26, 28, 34)
        canvas[vertical_free] = (170, 220, 245)
        palette = [
            (120, 150, 230),
            (90, 190, 160),
            (232, 174, 82),
            (218, 128, 155),
            (166, 134, 226),
            (96, 178, 222),
        ]
        for label in np.unique(labels):
            if int(label) <= 0:
                continue
            mask = labels == int(label)
            color = np.asarray(palette[(int(label) - 1) % len(palette)], dtype=np.float32)
            base = canvas[mask].astype(np.float32)
            canvas[mask] = np.clip(base * 0.72 + color * 0.28, 0, 255).astype(np.uint8)
        if show_diag:
            canvas[outside_pre_traversible & ~outside_boundary] = (122, 150, 122)
            canvas[outside_candidate_base & ~outside_boundary] = (245, 216, 118)
            canvas[outside_robot_component & ~outside_boundary] = (20, 190, 120)
            canvas[outside_connected_rejected & ~outside_boundary] = (120, 80, 225)
            canvas[outside_disconnected_candidate & ~outside_boundary] = (255, 210, 55)
            canvas[outside_floor_seen & ~outside_boundary] = (60, 210, 220)
            canvas[outside_holdout & ~outside_boundary] = (110, 98, 135)
            canvas[outside_candidate & ~outside_boundary] = (245, 216, 118)
        canvas[outside_boundary] = (180, 142, 28)
        wall_visual = clean_display_wall
        canvas[wall_visual] = (255, 30, 30)
        if show_diag:
            canvas[ratio_wall_debug & ~wall_visual] = (112, 64, 180)
            canvas[wall_suppressed_by_free] = (72, 150, 92)
            canvas[conflict] = (255, 126, 45)
            canvas[raw_wall & ~wall_visual] = (92, 78, 64)
            canvas[wall_line_support & ~wall_visual] = (255, 170, 64)
            canvas[nonstructural_occupied & ~wall_visual] = (120, 98, 72)
            canvas[rejected_wall_support] = (255, 142, 45)
            canvas[unknown_rejected_wall] = (255, 118, 36)
            canvas[unknown_dominant & ~unknown_rejected_wall] = (74, 70, 92)
            canvas[unknown_gated_wall & ~wall_visual] = (150, 46, 54)
            canvas[anchor_projected_wall & ~projected_wall] = (255, 156, 42)
            canvas[projected_step2_source & ~wall_visual] = (170, 170, 190)
            canvas[step2_hits & ~step2] = (160, 80, 255)
            canvas[step2_pre_topology & ~step2] = (140, 72, 210)
            canvas[step2_partition_candidate & ~step2] = (116, 76, 140)
            canvas[step2_topology_rejected & ~step2] = (120, 80, 160)
            canvas[step2_block] = (78, 210, 118)
            canvas[door_attempt & ~door_visual & ~door_cut] = (0, 120, 40)
            canvas[door_trial & ~door_selected & ~door_visual & ~door_cut] = (0, 120, 40)
            canvas[door_attempt_rejected & ~door_visual & ~door_cut] = (120, 120, 120)
            canvas[door_trial_rejected & ~door_visual & ~door_cut] = (120, 120, 120)
            canvas[door_partition_candidate & ~door_cut] = (170, 220, 80)
            canvas[door_partition_rejected & ~door_cut] = (255, 100, 50)
            canvas[door_topology_warning & ~door_cut] = (255, 205, 70)
            canvas[door_visual_debug & ~door_cut] = (78, 105, 84)
            canvas[door_visual_only & ~door_cut] = (125, 125, 125)
            canvas[door_geometry_warning & ~door_cut] = (255, 168, 55)
            canvas[door_geometry_only & ~door_cut] = (255, 168, 55)
            canvas[door_attachment_only & ~door_cut] = (255, 112, 55)
            canvas[door_not_closed & ~door_cut] = (210, 95, 70)
            canvas[door_rejected_primitive & ~door_cut] = (190, 140, 80)
            canvas[rejected] = (130, 112, 118)
        canvas[step1] = (245, 215, 55)
        canvas[step2] = (220, 60, 255)
        canvas[stable_step2 & ~step2] = (175, 45, 210)
        if show_diag:
            canvas[door_visual_debug & ~door_topology_effective] = (0, 135, 60)
            canvas[stable_door_visual & ~door_topology_effective] = (115, 205, 175)
        canvas[door_topology_effective] = (90, 255, 70)
        canvas[door_cut] = (90, 255, 70)
        canvas[stable_door_cut] = (255, 95, 190)
        canvas[door_primitive & ~door_extensible_primitive & ~door_cut] = (35, 115, 255)
        canvas[door_extensible_primitive & ~door_cut] = (0, 205, 255)
        canvas[extensible_door_seed] = (50, 125, 255)
        canvas[door_seed] = (0, 80, 255)
        canvas[frontier] = (0, 235, 255)

        r0, r1, c0, c1 = crop_bounds
        crop = canvas[r0:r1, c0:c1]
        crop_h, crop_w = crop.shape[:2]
        label_h = 36
        margin = 8
        available_h = max(1, int(height) - label_h - margin)
        scale = min((width - 2 * margin) / max(crop_w, 1), available_h / max(crop_h, 1))
        map_w, map_h = max(1, int(crop_w * scale)), max(1, int(crop_h * scale))
        ox = (width - map_w) // 2
        oy = label_h + max(0, (available_h - map_h) // 2)
        image = Image.new("RGB", (width, height), (15, 17, 21))
        image.paste(Image.fromarray(crop).resize((map_w, map_h), Image.NEAREST), (ox, oy))
        draw = ImageDraw.Draw(image)
        ceiling = self._room_segmentation_debug.get("voxel_ceiling_height_m")
        active_z = self._room_segmentation_debug.get("voxel_active_z_max_m")
        backend = str(self._room_segmentation_debug.get("voxel_integration_backend", self._room_segmentation_debug.get("integration_backend", "unknown")))
        integrate_ms = self._room_segmentation_debug.get("voxel_integrate_total_ms")

        def fmt_m(value: object) -> str:
            try:
                if value is None:
                    return "NA"
                return "%.2f" % float(value)
            except (TypeError, ValueError):
                return "NA"

        if missing:
            title_a = "VOXEL DEBUG MISSING | missing %s" % missing[0]
            title_b = "missing_keys=%d" % int(len(missing))
        else:
            raw_wall_count = int(np.count_nonzero(raw_wall))
            wall_line_support_count = int(np.count_nonzero(wall_line_support))
            projected_wall_count = int(np.count_nonzero(projected_wall))
            rejected_wall_support_count = int(np.count_nonzero(rejected_wall_support))
            display_wall_count = int(np.count_nonzero(wall_visual))
            door_seed_count = int(np.count_nonzero(door_seed))
            outside_count = int(np.count_nonzero(outside_boundary))
            door_green_count = int(np.count_nonzero(door_visual))
            door_extensible_seed_count = int(np.count_nonzero(extensible_door_seed))
            door_primitive_count = int(self._room_segmentation_debug.get("voxel_door_line_primitive_count", np.count_nonzero(door_primitive)) or 0)
            door_extensible_primitive_count = int(self._room_segmentation_debug.get("voxel_door_extensible_primitive_count", np.count_nonzero(door_extensible_primitive)) or 0)
            door_visual_count = int(np.count_nonzero(door_visual_all))
            door_visual_only_count = int(np.count_nonzero(door_visual_only))
            door_topology_count = int(np.count_nonzero(door_topology_effective))
            door_cut_candidate_count = int(np.count_nonzero(door_partition_candidate))
            door_cut_count = int(np.count_nonzero(door_cut))
            door_cluster_count = int(self._room_segmentation_debug.get("voxel_door_seed_cluster_count", 0) or 0)
            door_group_count = int(self._room_segmentation_debug.get("voxel_door_seed_group_count", door_cluster_count) or 0)
            door_trial_count = int(self._room_segmentation_debug.get("voxel_door_trial_candidate_count", np.count_nonzero(door_attempt)) or 0)
            step2_source_count = int(self._room_segmentation_debug.get("voxel_step2_source_line_count", 0) or 0)
            projected_step2_source_count = int(self._room_segmentation_debug.get("voxel_step2_projected_source_line_count", np.count_nonzero(projected_step2_source)) or 0)
            step2_hit_count = int(np.count_nonzero(step2_hits))
            step2_candidate_count = int(self._room_segmentation_debug.get("voxel_step2_candidate_count", np.count_nonzero(step2_candidate)) or 0)
            step2_accepted_count = int(self._room_segmentation_debug.get("voxel_step2_accepted_count", 0) or 0)
            try:
                integrate_text = "%.1fms" % float(integrate_ms)
            except (TypeError, ValueError):
                integrate_text = "NA"
            stable_count = int(np.count_nonzero(stable_door_cut))
            stable_step2_count = int(np.count_nonzero(stable_step2))
            warning_count = int(np.count_nonzero(door_topology_warning))
            update_reason = str(self._room_segmentation_debug.get("roomseg_frontier_update_reason", "NA"))
            title_a = "voxel v32 | raw_seed=%d seed_cc=%d seed_clusters=%d line_prim=%d ext_prim=%d trials=%d topo_cut=%d stable_door=%d step2_acc=%d" % (
                door_seed_count,
                int(self._room_segmentation_debug.get("voxel_door_seed_component_count", 0) or 0),
                door_cluster_count,
                door_primitive_count,
                door_extensible_primitive_count,
                door_trial_count,
                door_topology_count,
                stable_count,
                step2_accepted_count,
            )
            raw_seed_not_blocking = int(bool(self._room_segmentation_debug.get("voxel_step2_block_topology_effective_door_only", False)))
            seed_not_in_free = int(bool(self._room_segmentation_debug.get("voxel_seed_not_added_to_partition_free", False)))
            title_b = "wall=%d proj=%d red=%d outside=%d step2_src=%d hit=%d door_final_cut=%d stable_s2=%d raw_seed_not_blocking_step2=%d seed_not_in_partition_free=%d" % (
                wall_line_support_count,
                projected_wall_count,
                display_wall_count,
                outside_count,
                step2_source_count,
                step2_hit_count,
                door_cut_count,
                stable_step2_count,
                raw_seed_not_blocking,
                seed_not_in_free,
            )
            door_reject_counts = self._room_segmentation_debug.get("voxel_door_reject_reason_counts", {})
            door_partition_reject_counts = self._room_segmentation_debug.get("voxel_door_partition_reject_reason_counts", {})
            door_topology_reject_counts = self._room_segmentation_debug.get("voxel_door_topology_reject_reason_counts", {})
            step2_reject_counts = self._room_segmentation_debug.get("voxel_step2_topology_reject_reason_counts", self._room_segmentation_debug.get("voxel_step2_reject_reason_counts", {}))
            door_counts = door_topology_reject_counts if isinstance(door_topology_reject_counts, dict) and door_topology_reject_counts else door_partition_reject_counts
            if not isinstance(door_counts, dict) or not door_counts:
                door_counts = door_reject_counts
            door_reject_text = _top_reason_text(door_counts)
            step2_reject_text = _top_reason_text(step2_reject_counts)
            if show_diag:
                title_b = "%s | visual=%d visual_only=%d cut=%d rooms=%d seed_rej=%s partition_rej=%s step2_rej=%s" % (
                    title_b,
                    door_visual_count,
                    door_visual_only_count,
                    door_cut_count,
                    int(len([v for v in np.unique(labels) if int(v) > 0])),
                    _top_reason_text(self._room_segmentation_debug.get("voxel_door_seed_reject_reason_counts", {})),
                    door_reject_text,
                    step2_reject_text,
                )
        title_color = (255, 70, 70) if missing or backend == "python_debug" else (255, 255, 255)
        self._label(draw, (8, 5), title_a[:96], title_color)
        self._label(draw, (8, 18), title_b[:96], title_color if backend == "python_debug" else (255, 255, 255))
        legend_y = oy + map_h + 4
        if legend_y + 50 <= height:
            self._voxel_roomseg_legend(draw, (8, legend_y))
        rooms = int(len([v for v in np.unique(labels) if int(v) > 0]))
        strict_wall_count = int(np.count_nonzero(strict_raw_wall))
        projected_wall_count = int(np.count_nonzero(projected_wall))
        step1_completed_count = int(np.count_nonzero(step1_completed))
        line_wall_count = int(np.count_nonzero(line_wall & ~projected_wall & ~strict_raw_wall))
        layers = [
            self._overlay_record(
                "voxel_vertical_panel",
                True,
                (170, 220, 245),
                int(np.count_nonzero(vertical_free)),
                "bottom-right voxel vertical-free map panel",
                vertical_free_cells=int(np.count_nonzero(vertical_free)),
                strict_wall_cells=strict_wall_count,
                raw_wall_cells=int(np.count_nonzero(raw_wall)),
                wall_line_support_cells=int(np.count_nonzero(wall_line_support)),
                projected_wall_cells=projected_wall_count,
                display_wall_cells=int(np.count_nonzero(wall_visual)),
                anchor_projected_wall_cells=int(np.count_nonzero(anchor_projected_wall)),
                step1_completed_wall_cells=step1_completed_count,
                wall_suppressed_by_free_cells=int(np.count_nonzero(wall_suppressed_by_free)),
                ratio_wall_debug_cells=int(np.count_nonzero(ratio_wall_debug)),
                nonstructural_occupied_cells=int(np.count_nonzero(nonstructural_occupied)),
                rejected_wall_support_cells=int(np.count_nonzero(rejected_wall_support)),
                unknown_dominant_cells=int(np.count_nonzero(unknown_dominant)),
                unknown_rejected_wall_cells=int(np.count_nonzero(unknown_rejected_wall)),
                unknown_gated_wall_cells=int(np.count_nonzero(unknown_gated_wall)),
                unknown_cells=int(np.count_nonzero(unknown)),
                outside_boundary_cells=int(np.count_nonzero(outside_boundary)),
                outside_candidate_cells=int(np.count_nonzero(outside_candidate)),
                outside_candidate_base_cells=int(np.count_nonzero(outside_candidate_base)),
                outside_disconnected_candidate_cells=int(np.count_nonzero(outside_disconnected_candidate)),
                outside_pre_traversible_cells=int(np.count_nonzero(outside_pre_traversible)),
                outside_robot_component_cells=int(np.count_nonzero(outside_robot_component)),
                outside_connected_to_robot_rejected_cells=int(np.count_nonzero(outside_connected_rejected)),
                outside_floor_frustum_seen_cells=int(np.count_nonzero(outside_floor_seen)),
                outside_holdout_floor_unseen_cells=int(np.count_nonzero(outside_holdout)),
                conflict_cells=int(np.count_nonzero(conflict)),
                door_seed_cells=int(np.count_nonzero(door_seed)),
                extensible_door_seed_cells=int(np.count_nonzero(extensible_door_seed)),
                nonextensible_door_seed_cells=int(np.count_nonzero(nonextensible_door_seed)),
                door_seed_line_primitive_cells=int(np.count_nonzero(door_primitive)),
                door_extensible_primitive_cells=int(np.count_nonzero(door_extensible_primitive)),
                door_rejected_primitive_cells=int(np.count_nonzero(door_rejected_primitive)),
                step2_block_cells=int(np.count_nonzero(step2_block)),
                door_attempt_cells=int(np.count_nonzero(door_attempt)),
                door_attempt_rejected_cells=int(np.count_nonzero(door_attempt_rejected)),
                door_trial_cells=int(np.count_nonzero(door_trial)),
                door_selected_cells=int(np.count_nonzero(door_selected)),
                door_visual_cells=int(np.count_nonzero(door_visual)),
                door_provisional_visual_cells=int(np.count_nonzero(door_visual_all)),
                door_visual_only_cells=int(np.count_nonzero(door_visual_only)),
                door_partition_candidate_cells=int(np.count_nonzero(door_partition_candidate)),
                door_partition_rejected_cells=int(np.count_nonzero(door_partition_rejected)),
                door_topology_warning_cells=int(np.count_nonzero(door_topology_warning)),
                door_cut_cells=int(np.count_nonzero(door_cut)),
                stable_door_cut_cells=int(np.count_nonzero(stable_door_cut)),
                stable_door_visual_cells=int(np.count_nonzero(stable_door_visual)),
                stable_step2_separator_cells=int(np.count_nonzero(stable_step2)),
                projected_step2_source_cells=int(np.count_nonzero(projected_step2_source)),
                step2_hit_cells=int(np.count_nonzero(step2_hits)),
                step2_pre_topology_cells=int(np.count_nonzero(step2_pre_topology)),
                step2_candidate_cells=int(np.count_nonzero(step2_candidate)),
                step2_topology_rejected_cells=int(np.count_nonzero(step2_topology_rejected)),
                step1_gap_fill_cells=int(np.count_nonzero(step1)),
                step2_separator_cells=int(np.count_nonzero(step2)),
                frontier_cells=int(np.count_nonzero(frontier)),
                diagnostics_visible=show_diag,
                missing_keys=list(missing),
                ceiling_height_estimate_m=None if ceiling is None else fmt_m(ceiling),
                voxel_active_z_max_m=None if active_z is None else fmt_m(active_z),
                voxel_integration_backend=backend,
                voxel_integrate_total_ms=integrate_ms,
                roomseg_frontier_update_reason=self._room_segmentation_debug.get("roomseg_frontier_update_reason"),
                room_count=rooms,
            ),
            self._overlay_record("vertical_free", True, (170, 220, 245), int(np.count_nonzero(vertical_free)), "voxel vertical free cells used by roomseg"),
            self._overlay_record("voxel_vertical_free", True, (170, 220, 245), int(np.count_nonzero(vertical_free)), "voxel vertical free cells used by roomseg"),
            self._overlay_record("voxel_display_wall", True, (255, 30, 30), int(np.count_nonzero(wall_visual)), "clean v19 display wall from voxel_display_wall_xy only"),
            self._overlay_record("voxel_raw_occupied_wall_support", bool(show_diag and np.any(raw_wall)), (92, 78, 64), int(np.count_nonzero(raw_wall)), "diagnostic raw occupied-any support, not default red wall"),
            self._overlay_record("voxel_wall_line_support", bool(show_diag and np.any(wall_line_support)), (255, 170, 64), int(np.count_nonzero(wall_line_support)), "diagnostic structural wall-line support candidate before projection validation"),
            self._overlay_record("voxel_wall_raw", bool(show_diag and np.any(ratio_wall_debug)), (112, 64, 180), int(np.count_nonzero(ratio_wall_debug)), "diagnostic ratio wall candidates before priority filtering"),
            self._overlay_record("voxel_strict_raw_wall", bool(np.any(strict_raw_wall)), (255, 30, 30), strict_wall_count, "clean structural voxel wall after free and unknown priority"),
            self._overlay_record("voxel_wall_suppressed_by_free", bool(show_diag and np.any(wall_suppressed_by_free)), (72, 150, 92), int(np.count_nonzero(wall_suppressed_by_free)), "diagnostic occupied support suppressed by free voxel threshold"),
            self._overlay_record("voxel_ratio_wall_debug", bool(show_diag and np.any(ratio_wall_debug)), (112, 64, 180), int(np.count_nonzero(ratio_wall_debug)), "diagnostic ratio wall candidates before priority filtering"),
            self._overlay_record("voxel_nonstructural_occupied", bool(show_diag and np.any(nonstructural_occupied)), (120, 98, 72), int(np.count_nonzero(nonstructural_occupied)), "diagnostic occupied cells that are not structural wall"),
            self._overlay_record("voxel_unknown_dominant", bool(show_diag and np.any(unknown_dominant)), (74, 70, 92), int(np.count_nonzero(unknown_dominant)), "unknown-heavy columns gated out of structural wall, projection, Step2, and door anchors"),
            self._overlay_record("voxel_wall_support_unknown_gated", bool(show_diag and np.any(unknown_gated_wall)), (150, 46, 54), int(np.count_nonzero(unknown_gated_wall)), "occupied-any wall support that passed unknown gating"),
            self._overlay_record("voxel_wall_support_rejected_unknown", bool(show_diag and np.any(unknown_rejected_wall)), (255, 118, 36), int(np.count_nonzero(unknown_rejected_wall)), "occupied-any wall support rejected because the column is unknown-dominant"),
            self._overlay_record("voxel_wall_projected", bool(np.any(projected_wall)), (255, 30, 30), projected_wall_count, "projected structural wall line used by room boundaries and anchors"),
            self._overlay_record("voxel_anchor_projected_wall", bool(show_diag and np.any(anchor_projected_wall)), (255, 156, 42), int(np.count_nonzero(anchor_projected_wall)), "diagnostic relaxed short projected wall anchors for door and step2"),
            self._overlay_record("voxel_wall_projection_rejected_support", bool(show_diag and np.any(rejected_wall_support)), (255, 142, 45), int(np.count_nonzero(rejected_wall_support)), "diagnostic raw wall support rejected by wall projection"),
            self._overlay_record("voxel_step1_completed_wall", bool(np.any(step1_completed)), (226, 192, 46), step1_completed_count, "wall map after step1 gap completion"),
            self._overlay_record("voxel_wall_red", True, (255, 30, 30), int(np.count_nonzero(wall_visual)), "default clean red wall display"),
            self._overlay_record("voxel_wall", True, (255, 30, 30), int(np.count_nonzero(wall)), "final voxel wall layer"),
            self._overlay_record("voxel_line_supported_wall", bool(np.any(line_wall)), (148, 20, 24), line_wall_count, "line-supported wall derived from projected wall evidence"),
            self._overlay_record("voxel_conflict", bool(show_diag and np.any(conflict)), (255, 126, 45), int(np.count_nonzero(conflict)), "diagnostic xy cells with simultaneous vertical-free and strict-wall evidence"),
            self._overlay_record("voxel_unknown", True, (26, 28, 34), int(np.count_nonzero(unknown)), "voxel unknown cells preserved as unknown"),
            self._overlay_record("voxel_outside_boundary", bool(np.any(outside_boundary)), (180, 142, 28), int(np.count_nonzero(outside_boundary)), "sensor-aware outside boundary: not free, not occupied, not frontier domain"),
            self._overlay_record("voxel_outside_candidate", bool(show_diag and np.any(outside_candidate)), (245, 216, 118), int(np.count_nonzero(outside_candidate)), "diagnostic outside candidate from navigation free plus full-height no occupied plus floor frustum"),
            self._overlay_record("voxel_outside_candidate_base", bool(show_diag and np.any(outside_candidate_base)), (245, 216, 118), int(np.count_nonzero(outside_candidate_base)), "diagnostic outside candidate before robot-component connectivity gating"),
            self._overlay_record("voxel_outside_disconnected_candidate", bool(show_diag and np.any(outside_disconnected_candidate)), (255, 210, 55), int(np.count_nonzero(outside_disconnected_candidate)), "diagnostic outside candidate after rejecting the robot reachable component"),
            self._overlay_record("voxel_outside_robot_component", bool(show_diag and np.any(outside_robot_component)), (20, 190, 120), int(np.count_nonzero(outside_robot_component)), "pre-outside A* traversible component reachable from the robot"),
            self._overlay_record("voxel_outside_connected_rejected", bool(show_diag and np.any(outside_connected_rejected)), (120, 80, 225), int(np.count_nonzero(outside_connected_rejected)), "candidate outside cells rejected because they are connected to the robot"),
            self._overlay_record("voxel_outside_pre_traversible", bool(show_diag and np.any(outside_pre_traversible)), (122, 150, 122), int(np.count_nonzero(outside_pre_traversible)), "pre-outside traversible mask used for robot component flood fill"),
            self._overlay_record("voxel_outside_floor_frustum_seen", bool(show_diag and np.any(outside_floor_seen)), (60, 210, 220), int(np.count_nonzero(outside_floor_seen)), "diagnostic XY cells whose floor height entered the projective camera frustum"),
            self._overlay_record("voxel_outside_holdout_floor_unseen", bool(show_diag and np.any(outside_holdout)), (110, 98, 135), int(np.count_nonzero(outside_holdout)), "diagnostic candidate-like cells held out because floor frustum was not seen"),
            self._overlay_record("voxel_door_seed", bool(np.any(door_seed)), (0, 80, 255), int(np.count_nonzero(door_seed)), "door seed cells from per-xy voxel z-pattern detection"),
            self._overlay_record("voxel_door_seed_line_primitive", bool(np.any(door_primitive)), (0, 205, 255), int(np.count_nonzero(door_primitive)), "v32 line primitives extracted from raw door seed before extension"),
            self._overlay_record("voxel_door_extensible_primitive", bool(np.any(door_extensible_primitive)), (0, 205, 255), int(np.count_nonzero(door_extensible_primitive)), "v32 accepted seed line primitives eligible for door extension"),
            self._overlay_record("voxel_door_rejected_primitive", bool(show_diag and np.any(door_rejected_primitive)), (190, 140, 80), int(np.count_nonzero(door_rejected_primitive)), "diagnostic v32 seed line primitives rejected before extension"),
            self._overlay_record("voxel_extensible_door_seed", bool(show_diag and np.any(extensible_door_seed)), (50, 125, 255), int(np.count_nonzero(extensible_door_seed)), "diagnostic v26 seed groups that passed accepted-extension gating"),
            self._overlay_record("voxel_nonextensible_door_seed", bool(show_diag and np.any(nonextensible_door_seed)), (0, 55, 150), int(np.count_nonzero(nonextensible_door_seed)), "diagnostic raw door seed kept as evidence only and not used to block Step2"),
            self._overlay_record("voxel_step2_block_mask", bool(show_diag and np.any(step2_block)), (78, 210, 118), int(np.count_nonzero(step2_block)), "accepted or stable doors that may block Step2; raw seed is excluded"),
            self._overlay_record("voxel_door_extension_attempt", bool(show_diag and np.any(door_attempt)), (0, 120, 40), int(np.count_nonzero(door_attempt)), "diagnostic all door extension attempt paths"),
            self._overlay_record("voxel_door_trial_candidates", bool(show_diag and np.any(door_trial)), (54, 96, 74), int(np.count_nonzero(door_trial)), "diagnostic all door orientation trial lines"),
            self._overlay_record("voxel_door_selected_candidates", bool(show_diag and np.any(door_selected)), (0, 180, 70), int(np.count_nonzero(door_selected)), "diagnostic selected door candidate line per seed cluster"),
            self._overlay_record("voxel_door_trial_rejected", bool(show_diag and np.any(door_attempt_rejected | door_trial_rejected)), (120, 120, 120), int(np.count_nonzero(door_attempt_rejected | door_trial_rejected)), "diagnostic rejected door orientation trial lines"),
            self._overlay_record("voxel_door_centerline", bool(np.any(door_visual)), (0, 255, 70), int(np.count_nonzero(door_visual)), "topology accepted door partition centerline"),
            self._overlay_record("voxel_door_visual_only", bool(show_diag and np.any(door_visual_only)), (120, 220, 140), int(np.count_nonzero(door_visual_only)), "diagnostic door visual line rejected by partition topology"),
            self._overlay_record("voxel_door_partition_cut_candidate", bool(show_diag and np.any(door_partition_candidate)), (170, 220, 80), int(np.count_nonzero(door_partition_candidate)), "diagnostic door partition cut candidates"),
            self._overlay_record("voxel_door_partition_cut_rejected", bool(show_diag and np.any(door_partition_rejected)), (255, 100, 50), int(np.count_nonzero(door_partition_rejected)), "diagnostic door partition cut rejected by topology"),
            self._overlay_record("voxel_door_topology_warning_cut", bool(show_diag and np.any(door_topology_warning)), (255, 205, 70), int(np.count_nonzero(door_topology_warning)), "door partition cut accepted by geometry while topology no-gain is recorded as warning"),
            self._overlay_record("voxel_stable_door_cut", bool(np.any(stable_door_cut)), (55, 235, 155), int(np.count_nonzero(stable_door_cut)), "stable door memory cut ORed into final partition"),
            self._overlay_record("voxel_stable_door_visual", bool(show_diag and np.any(stable_door_visual)), (55, 210, 155), int(np.count_nonzero(stable_door_visual)), "stable door memory visual centerline"),
            self._overlay_record("voxel_door_cut", bool(np.any(door_cut)), (80, 255, 80), int(np.count_nonzero(door_cut)), "final door partition cut including geometry-first and stable memory"),
            self._overlay_record("voxel_step1_gap_fill", bool(np.any(step1)), (245, 215, 55), int(np.count_nonzero(step1)), "real-wall short gap fill before virtual separators"),
            self._overlay_record("voxel_step2_extension", bool(np.any(step2)), (220, 60, 255), int(np.count_nonzero(step2)), "accepted Step2 wall-line extension separators"),
            self._overlay_record("voxel_stable_step2_separator", bool(np.any(stable_step2)), (175, 45, 210), int(np.count_nonzero(stable_step2)), "stable Step2 corridor separators kept by v26 memory"),
            self._overlay_record("voxel_step2_projected_source_line", bool(show_diag and np.any(projected_step2_source)), (170, 170, 190), int(np.count_nonzero(projected_step2_source)), "diagnostic projected wall line objects included in Step2 source pool"),
            self._overlay_record("voxel_step2_source_line", bool(show_diag and "voxel_step2_source_line_count" in self._room_segmentation_debug), (160, 160, 170), int(self._room_segmentation_debug.get("voxel_step2_source_line_count", 0) or 0), "diagnostic Step2 source line pool after filtered+relaxed dedup"),
            self._overlay_record("voxel_step2_extension_hits", bool(show_diag and np.any(step2_hits)), (160, 80, 255), int(np.count_nonzero(step2_hits)), "diagnostic all Step2 extension hit traces"),
            self._overlay_record("voxel_step2_partition_cut_candidate", bool(show_diag and np.any(step2_partition_candidate)), (116, 76, 140), int(np.count_nonzero(step2_partition_candidate)), "diagnostic Step2 partition cut candidates"),
            self._overlay_record("voxel_step2_topology_rejected", bool(show_diag and np.any(step2_topology_rejected)), (120, 80, 160), int(np.count_nonzero(step2_topology_rejected)), "diagnostic Step2 topology rejected separators"),
            self._overlay_record("voxel_rejected_candidates", bool(show_diag and np.any(rejected)), (130, 112, 118), int(np.count_nonzero(rejected)), "diagnostic rejected door centerlines and Step2 extension candidates"),
            self._overlay_record("voxel_frontier", bool(np.any(frontier)), (0, 235, 255), int(np.count_nonzero(frontier)), "frontier cells generated from voxel vertical free source"),
            self._overlay_record("voxel_final_room_labels", bool(np.any(labels > 0)), (120, 150, 230), int(np.count_nonzero(labels > 0)), "final 4-connectivity voxel room labels"),
        ]
        return image, layers

    def _render_height_profile_roomseg_panel(
        self,
        *,
        occupancy: np.ndarray,
        navigable: np.ndarray,
        observed: np.ndarray,
        size: Tuple[int, int],
        crop_bounds: Tuple[int, int, int, int],
    ) -> Tuple[Image.Image, List[dict]]:
        width, height = size
        shape = tuple(np.asarray(occupancy).shape[:2])
        vertical_free = self._room_debug_array("height_profile_vertical_free_xy", shape, bool)
        wall = self._room_debug_array("height_profile_wall_xy", shape, bool)
        unknown = self._room_debug_array("height_profile_unknown_xy", shape, bool)
        conflict = self._room_debug_array("height_profile_free_wall_conflict_xy", shape, bool)
        line_wall = self._room_debug_array("height_profile_filtered_wall_line_mask", shape, bool)
        door_seed = self._room_debug_array("height_profile_door_seed_mask", shape, bool)
        door_cut = self._room_debug_array("height_profile_accepted_door_centerline_mask", shape, bool)
        if not np.any(door_cut):
            door_cut = self._room_debug_array("height_profile_door_cut_mask", shape, bool)
        step1 = self._room_debug_array("height_profile_step1_wall_gap_fill_map", shape, bool)
        step2 = self._room_debug_array("height_profile_step2_extension_separator_map", shape, bool)
        rejected = self._room_debug_array("height_profile_rejected_door_centerline_mask", shape, bool) | self._room_debug_array("height_profile_step2_line_extensions_rejected", shape, bool)
        labels = self._room_debug_array("height_profile_final_room_label_map", shape, np.int32)
        frontier = self._room_debug_array("frontier_map", shape, bool)
        if not np.any(frontier):
            frontier = self._room_debug_array("height_profile_frontier", shape, bool)
        if not np.any(frontier):
            frontier = self._room_debug_array("height_profile_frontier_mask", shape, bool)

        canvas = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        canvas[:, :] = (30, 32, 36)
        canvas[unknown] = (26, 28, 34)
        canvas[vertical_free] = (180, 216, 238)
        palette = [
            (120, 150, 230),
            (90, 190, 160),
            (232, 174, 82),
            (218, 128, 155),
            (166, 134, 226),
            (96, 178, 222),
        ]
        for label in np.unique(labels):
            if int(label) <= 0:
                continue
            mask = labels == int(label)
            color = np.asarray(palette[(int(label) - 1) % len(palette)], dtype=np.float32)
            base = canvas[mask].astype(np.float32)
            canvas[mask] = np.clip(base * 0.72 + color * 0.28, 0, 255).astype(np.uint8)
        canvas[conflict] = (255, 126, 45)
        canvas[wall] = (238, 42, 42)
        canvas[line_wall & ~wall] = (148, 20, 24)
        canvas[step1] = (245, 215, 55)
        canvas[rejected] = (130, 112, 118)
        canvas[step2] = (40, 175, 255)
        canvas[door_seed] = (150, 245, 255)
        canvas[door_cut] = (255, 95, 220)
        canvas[frontier] = (0, 235, 255)

        r0, r1, c0, c1 = crop_bounds
        crop = canvas[r0:r1, c0:c1]
        crop_h, crop_w = crop.shape[:2]
        label_h = 22
        margin = 8
        available_h = max(1, int(height) - label_h - margin)
        scale = min((width - 2 * margin) / max(crop_w, 1), available_h / max(crop_h, 1))
        map_w, map_h = max(1, int(crop_w * scale)), max(1, int(crop_h * scale))
        ox = (width - map_w) // 2
        oy = label_h + max(0, (available_h - map_h) // 2)
        image = Image.new("RGB", (width, height), (15, 17, 21))
        image.paste(Image.fromarray(crop).resize((map_w, map_h), Image.NEAREST), (ox, oy))
        draw = ImageDraw.Draw(image)
        ceiling = self._room_segmentation_debug.get("ceiling_height_estimate_m")
        active_z = self._room_segmentation_debug.get("height_profile_active_z_max_m")

        def fmt_m(value: object) -> str:
            try:
                if value is None:
                    return "NA"
                return "%.2f" % float(value)
            except (TypeError, ValueError):
                return "NA"

        title_a = "height-profile vertical free | free=%d wall=%d unknown=%d" % (
            int(np.count_nonzero(vertical_free)),
            int(np.count_nonzero(wall)),
            int(np.count_nonzero(unknown)),
        )
        title_b = "conflict=%d door=%d step2=%d ceiling=%s active_z=%s" % (
            int(np.count_nonzero(conflict)),
            int(np.count_nonzero(door_cut)),
            int(np.count_nonzero(step2)),
            fmt_m(ceiling),
            fmt_m(active_z),
        )
        self._label(draw, (8, 5), title_a[:96], (255, 255, 255))
        self._label(draw, (8, 18), title_b[:96], (255, 255, 255))
        self._height_profile_legend(draw, (8, max(label_h + 4, height - 58)))
        rooms = int(len([v for v in np.unique(labels) if int(v) > 0]))
        strict_wall_count = int(np.count_nonzero(wall))
        line_wall_count = int(np.count_nonzero(line_wall & ~wall))
        layers = [
            self._overlay_record(
                "height_profile_vertical_panel",
                True,
                (180, 216, 238),
                int(np.count_nonzero(vertical_free)),
                "bottom-right height-profile vertical-free map panel",
                vertical_free_cells=int(np.count_nonzero(vertical_free)),
                strict_wall_cells=strict_wall_count,
                unknown_cells=int(np.count_nonzero(unknown)),
                conflict_cells=int(np.count_nonzero(conflict)),
                door_seed_cells=int(np.count_nonzero(door_seed)),
                door_cut_cells=int(np.count_nonzero(door_cut)),
                step1_gap_fill_cells=int(np.count_nonzero(step1)),
                step2_separator_cells=int(np.count_nonzero(step2)),
                frontier_cells=int(np.count_nonzero(frontier)),
                ceiling_height_estimate_m=None if ceiling is None else fmt_m(ceiling),
                height_profile_active_z_max_m=None if active_z is None else fmt_m(active_z),
                room_count=rooms,
            ),
            self._overlay_record("vertical_free", True, (180, 216, 238), int(np.count_nonzero(vertical_free)), "height-profile vertical free cells used by roomseg"),
            self._overlay_record("height_profile_vertical_free", True, (180, 216, 238), int(np.count_nonzero(vertical_free)), "height-profile vertical free cells used by roomseg"),
            self._overlay_record("height_profile_wall_red", True, (238, 42, 42), strict_wall_count, "strict known wall from 95 percent of active height bins"),
            self._overlay_record("height_profile_wall", True, (238, 42, 42), strict_wall_count, "strict known wall from 95 percent of active height bins"),
            self._overlay_record("height_profile_line_supported_wall", bool(np.any(line_wall)), (148, 20, 24), line_wall_count, "line-supported wall derived only from strict wall evidence"),
            self._overlay_record("height_profile_conflict", bool(np.any(conflict)), (255, 126, 45), int(np.count_nonzero(conflict)), "xy cells with simultaneous vertical-free and strict-wall evidence"),
            self._overlay_record("height_profile_unknown", True, (26, 28, 34), int(np.count_nonzero(unknown)), "height-profile unknown cells preserved as unknown"),
            self._overlay_record("height_profile_door_seed", bool(np.any(door_seed)), (150, 245, 255), int(np.count_nonzero(door_seed)), "door seed cells from per-xy z-pattern detection"),
            self._overlay_record("height_profile_door_centerline", bool(np.any(door_cut)), (255, 95, 220), int(np.count_nonzero(door_cut)), "door separators detected from vertical z-pattern and centerline projection"),
            self._overlay_record("height_profile_step1_gap_fill", bool(np.any(step1)), (245, 215, 55), int(np.count_nonzero(step1)), "real-wall short gap fill before virtual separators"),
            self._overlay_record("height_profile_step2_extension", bool(np.any(step2)), (40, 175, 255), int(np.count_nonzero(step2)), "Step2 wall-line extension separators"),
            self._overlay_record("height_profile_rejected_candidates", bool(np.any(rejected)), (130, 112, 118), int(np.count_nonzero(rejected)), "rejected door centerlines and Step2 extension candidates"),
            self._overlay_record("height_profile_frontier", bool(np.any(frontier)), (0, 235, 255), int(np.count_nonzero(frontier)), "frontier cells generated from vertical free source"),
            self._overlay_record("height_profile_final_room_labels", bool(np.any(labels > 0)), (120, 150, 230), int(np.count_nonzero(labels > 0)), "final 4-connectivity height-profile room labels"),
        ]
        return image, layers

    def _room_debug_array(self, key: str, shape: Tuple[int, int], dtype) -> np.ndarray:
        raw = self._room_segmentation_debug.get(key)
        if raw is None:
            return np.zeros(shape, dtype=dtype)
        try:
            arr = np.asarray(raw, dtype=dtype)
        except Exception:
            return np.zeros(shape, dtype=dtype)
        if arr.shape != tuple(shape):
            return np.zeros(shape, dtype=dtype)
        return arr

    def _overlay_record(
        self,
        name: str,
        enabled: bool,
        color: Tuple[int, int, int],
        count: int,
        note: str = "",
        **extra,
    ) -> dict:
        item = {
            "name": name,
            "enabled": bool(enabled),
            "color": [int(color[0]), int(color[1]), int(color[2])],
            "primitive_count": int(count),
            "count": int(count),
            "green_like": bool(_is_green_like(color)),
            "note": note,
            "description": note,
        }
        item.update(extra)
        return item

    def _draw_rose_gap_markers(
        self,
        draw: ImageDraw.ImageDraw,
        gaps: Sequence[object],
        xy_func,
        crop_bounds: Tuple[int, int, int, int],
        color: Tuple[int, int, int],
    ) -> int:
        r0, r1, c0, c1 = crop_bounds
        count = 0
        for gap in gaps[:128]:
            if not isinstance(gap, Mapping):
                continue
            axis = str(gap.get("axis", "vertical"))
            index = int(gap.get("index", 0) or 0)
            start = int(gap.get("start", 0) or 0)
            end = int(gap.get("end", start) or start)
            p0 = (start, index) if axis == "vertical" else (index, start)
            p1 = (end, index) if axis == "vertical" else (index, end)
            if not (_cell_in_crop(p0, r0, r1, c0, c1) or _cell_in_crop(p1, r0, r1, c0, c1)):
                continue
            draw.line([xy_func(p0), xy_func(p1)], fill=color, width=3)
            count += 1
        return count

    def _rose_legend(self, draw: ImageDraw.ImageDraw, xy: Tuple[int, int]) -> None:
        x, y = xy
        items = [
            ((0, 0, 0), "ROSE roomseg occupied"),
            ((150, 156, 160), "ROSE roomseg free"),
            ((225, 132, 45), "occupied -> free"),
            ((230, 40, 230), "nav obstacle overlay"),
            ((255, 35, 35), "rescued wall"),
            ((45, 135, 255), "vfree wins warning"),
            ((58, 82, 132), "rejected clutter"),
            ((126, 104, 75), "vertical-carved"),
            ((255, 190, 70), "ROSE line"),
            ((255, 35, 35), "wall line / dashed extension"),
            ((255, 80, 130), "closed window"),
            ((80, 255, 130), "doorway portal"),
        ]
        for color, label in items:
            self._dot(draw, (x + 5, y + 7), color, radius=4)
            draw.text((x + 14, y), label, fill=(230, 232, 235), font=self._font)
            y += 10

    def _height_profile_legend(self, draw: ImageDraw.ImageDraw, xy: Tuple[int, int]) -> None:
        x, y = xy
        items = [
            ((180, 216, 238), "vertical free"),
            ((255, 126, 45), "free/wall conflict"),
            ((238, 42, 42), "strict wall"),
            ((148, 20, 24), "line wall"),
            ((245, 215, 55), "step1 gap"),
            ((40, 175, 255), "step2 extension"),
            ((150, 245, 255), "door seed"),
            ((255, 95, 220), "door centerline"),
            ((130, 112, 118), "rejected"),
            ((0, 235, 255), "frontier"),
        ]
        for color, label in items:
            self._dot(draw, (x + 5, y + 7), color, radius=4)
            draw.text((x + 14, y), label, fill=(230, 232, 235), font=self._font)
            y += 10

    def _voxel_roomseg_legend(self, draw: ImageDraw.ImageDraw, xy: Tuple[int, int]) -> None:
        x, y = xy
        items = [
            ((170, 220, 245), "vertical free"),
            ((255, 30, 30), "known wall"),
            ((0, 80, 255), "door seed"),
            ((0, 255, 70), "door extension"),
            ((80, 255, 80), "door accepted cut"),
            ((245, 215, 55), "step1 gap"),
            ((220, 60, 255), "step2 accepted"),
            ((0, 235, 255), "frontier"),
        ]
        for color, label in items:
            self._dot(draw, (x + 5, y + 7), color, radius=4)
            draw.text((x + 14, y), label, fill=(230, 232, 235), font=self._font)
            y += 10

    def _map_crop_bounds(
        self,
        *,
        occupancy: np.ndarray,
        navigable: np.ndarray,
        observed: np.ndarray,
        current_grid: GridCell,
        frontiers: Sequence[FrontierCluster],
        nav_decision: Optional[NavigationDecision],
        current_path: Sequence[GridCell],
        full_path: Sequence[GridCell],
        object_memory: ObjectMemory,
        goal_category: str,
    ) -> Tuple[int, int, int, int]:
        h, w = occupancy.shape
        rows: List[int] = []
        cols: List[int] = []

        active = observed.astype(bool) | occupancy.astype(bool) | navigable.astype(bool)
        rr, cc = np.nonzero(active)
        if rr.size:
            rows.extend(int(v) for v in rr)
            cols.extend(int(v) for v in cc)

        def add_cell(cell: GridCell) -> None:
            r, c = int(cell[0]), int(cell[1])
            if 0 <= r < h and 0 <= c < w:
                rows.append(r)
                cols.append(c)

        add_cell(current_grid)
        for cell in current_path:
            add_cell(cell)
        for cell in full_path:
            add_cell(cell)
        for frontier in frontiers[:128]:
            add_cell(frontier.center_grid)
        if nav_decision is not None:
            target_cells = nav_decision.target_cells
            if nav_decision.mode != "candidate":
                target_cells = target_cells[:1]
            for cell in target_cells[:32]:
                add_cell(cell)
            if nav_decision.frontier_decision is not None and nav_decision.frontier_decision.selected_frontier is not None:
                add_cell(nav_decision.frontier_decision.selected_frontier.center_grid)
            if nav_decision.selected_candidate is not None:
                add_cell(nav_decision.selected_candidate.center_grid)
        for node in self._visible_map_nodes(object_memory, goal_category, nav_decision)[:500]:
            add_cell(node.center_grid)
        for room in self._active_room_masks((h, w))[:64]:
            mask = np.asarray(getattr(room, "mask", None), dtype=bool)
            rr, cc = np.nonzero(mask)
            if rr.size:
                rows.extend(int(v) for v in rr[:: max(1, rr.size // 128)])
                cols.extend(int(v) for v in cc[:: max(1, cc.size // 128)])

        if not rows or not cols:
            return 0, h, 0, w

        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        padding = max(24, int(round(min(h, w) * 0.04)))
        min_r = max(0, min_r - padding)
        max_r = min(h - 1, max_r + padding)
        min_c = max(0, min_c - padding)
        max_c = min(w - 1, max_c + padding)

        crop_h = max_r - min_r + 1
        crop_w = max_c - min_c + 1
        min_crop = min(max(h, w), 160)
        if crop_h < min_crop:
            extra = min_crop - crop_h
            min_r = max(0, min_r - extra // 2)
            max_r = min(h - 1, max_r + extra - extra // 2)
        if crop_w < min_crop:
            extra = min_crop - crop_w
            min_c = max(0, min_c - extra // 2)
            max_c = min(w - 1, max_c + extra - extra // 2)

        return min_r, max_r + 1, min_c, max_c + 1

    def _render_text(
        self,
        *,
        step: int,
        detections_2d: Sequence[Detection2D],
        frontiers: Sequence[FrontierCluster],
        nav_decision: Optional[NavigationDecision],
        object_memory: ObjectMemory,
        goal_category: str,
        distance_to_goal: float,
        path_length: float,
        scenegraph_backend: str,
        score_debug: dict,
        failure_reason: Optional[str],
        size: Tuple[int, int],
    ) -> Image.Image:
        width, height = size
        image = Image.new("RGB", (width, height), (24, 26, 30))
        draw = ImageDraw.Draw(image)
        goal_norm = normalize_category(goal_category)
        goal_nodes = [
            node
            for node in object_memory.nodes
            if normalize_category(node.category) == goal_norm
            or (goal_norm and (goal_norm in normalize_category(node.category) or normalize_category(node.category) in goal_norm))
        ]
        mode = nav_decision.mode if nav_decision else "init"
        reason = nav_decision.reason if nav_decision else ""
        lines = [
            "VoxRoom decision panel",
            "step=%d goal=%s mode=%s reason=%s" % (int(step), goal_category, mode, reason),
            "dtg=%.2fm path=%.2fm objects=%d goal_candidates=%d detections=%d frontiers=%d backend=%s"
            % (
                float(distance_to_goal),
                float(path_length),
                len(object_memory.nodes),
                len(goal_nodes),
                len(detections_2d),
                len(frontiers),
                scenegraph_backend,
            ),
        ]
        if failure_reason:
            lines.append("failure=%s" % failure_reason)
        if nav_decision and nav_decision.selected_candidate is not None:
            node = nav_decision.selected_candidate
            lines.append(
                "candidate id=%d cat=%s conf=%.2f hits=%d grid=%s"
                % (node.node_id, node.category, node.confidence, node.observed_count, tuple(node.center_grid))
            )
        if nav_decision:
            meta = nav_decision.metadata or {}
            lines.append(
                "selected_frontier=%s selected_candidate=%s target_cells=%d"
                % (
                    nav_decision.frontier_decision.selected_index
                    if nav_decision.frontier_decision is not None
                    else None,
                    self._selected_candidate_id(nav_decision),
                    len(nav_decision.target_cells),
                )
            )
            if meta:
                commit = meta.get("frontier_commitment", {}) if isinstance(meta, dict) else {}
                if isinstance(commit, dict) and commit:
                    lines.append(
                        "frontier_commit id=%s age=%s dist=%s reason=%s"
                        % (
                            self._short(commit.get("active_frontier_id", "n/a")),
                            self._short(commit.get("active_frontier_age", "n/a")),
                            self._short(commit.get("active_frontier_distance_m", "n/a")),
                            self._short(commit.get("frontier_commitment_reason", "n/a")),
                        )
                    )
                lines.append(
                    "candidate credibility=%s track_obs=%s rep_steps=%s accepted=%s rejected=%s"
                    % (
                        self._short(meta.get("candidate_credibility", "n/a")),
                        self._short(meta.get("candidate_track_observation_count", "n/a")),
                        self._short(meta.get("candidate_reperception_steps", "n/a")),
                        self._short(meta.get("candidate_accepted", "n/a")),
                        self._short(meta.get("candidate_rejected", "n/a")),
                    )
                )
        if nav_decision and nav_decision.frontier_decision is not None:
            fd = nav_decision.frontier_decision
            lines.append("frontier selected=%s" % (fd.selected_index,))
            top = sorted(
                enumerate(fd.total_scores),
                key=lambda item: item[1],
                reverse=True,
            )[:3]
            for idx, total in top:
                sg = fd.scenegraph_scores[idx] if idx < len(fd.scenegraph_scores) else 0.0
                dist = fd.distance_scores[idx] if idx < len(fd.distance_scores) else 0.0
                lines.append("  #%d sg=%.3f dist=%.3f total=%.3f" % (idx, sg, dist, total))
        if score_debug:
            compact = ", ".join("%s=%s" % (k, self._short(v)) for k, v in list(score_debug.items())[:5])
            lines.append("score_debug: %s" % compact)
        if detections_2d:
            det_line = ", ".join("%s %.2f" % (d.category, d.confidence) for d in detections_2d[:5])
            lines.append("detections: %s" % det_line)

        y = 8
        for i, line in enumerate(lines[:11]):
            fill = (255, 255, 255) if i == 0 else (215, 220, 225)
            draw.text((10, y), line[:110], fill=fill, font=self._font)
            y += 18
        return image

    def _active_room_masks(self, shape: Tuple[int, int]) -> List[object]:
        out = []
        for room in self._room_masks:
            if bool(getattr(room, "stale", False)):
                continue
            mask = getattr(room, "mask", None)
            if mask is None:
                continue
            arr = np.asarray(mask, dtype=bool)
            if arr.shape != tuple(shape) or not np.any(arr):
                continue
            out.append(room)
        return out

    def _proposal_room_masks(self, shape: Tuple[int, int]) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for item in list(self._room_segmentation_debug.get("proposal_room_masks") or []):
            if not isinstance(item, Mapping):
                continue
            raw_mask = item.get("mask")
            if raw_mask is None:
                continue
            arr = np.asarray(raw_mask, dtype=bool)
            if arr.shape != tuple(shape) or not np.any(arr):
                continue
            out.append(arr)
        return out

    def _apply_boolean_mask_overlay(
        self,
        base: np.ndarray,
        masks: Sequence[np.ndarray],
        alpha: float,
        boundary_only: bool = False,
    ) -> Tuple[np.ndarray, int]:
        out = np.asarray(base, dtype=np.uint8).copy()
        total_cells = 0
        for idx, mask in enumerate(masks[:96]):
            arr = np.asarray(mask, dtype=bool)
            if arr.shape != out.shape[:2] or not np.any(arr):
                continue
            draw_mask = self._mask_boundary(arr) if boundary_only else arr
            total_cells += int(np.count_nonzero(draw_mask))
            color = np.asarray(self._room_color(idx), dtype=np.float32)
            blended = out[draw_mask].astype(np.float32) * (1.0 - float(alpha)) + color[None, :] * float(alpha)
            out[draw_mask] = np.clip(blended, 0, 255).astype(np.uint8)
        return out, total_cells

    def _apply_room_mask_overlay(self, base: np.ndarray, room_masks: Sequence[object]) -> Tuple[np.ndarray, int, int]:
        out = np.asarray(base, dtype=np.uint8).copy()
        owner = np.zeros(out.shape[:2], dtype=np.int32)
        total_mask_cells = 0
        total_boundary_cells = 0
        for idx, room in enumerate(room_masks[:64]):
            mask = np.asarray(getattr(room, "mask", None), dtype=bool)
            if mask.shape != out.shape[:2] or not np.any(mask):
                continue
            owner[(owner <= 0) & mask] = int(idx) + 1
            color = np.asarray(self._room_color(idx), dtype=np.float32)
            total_mask_cells += int(np.count_nonzero(mask))
            blended = out[mask].astype(np.float32) * 0.62 + color[None, :] * 0.38
            out[mask] = np.clip(blended, 0, 255).astype(np.uint8)
            boundary = self._mask_boundary(mask)
            total_boundary_cells += int(np.count_nonzero(boundary))
            out[boundary] = np.asarray(np.clip(color * 1.08, 0, 255), dtype=np.uint8)
        adjacency_boundary = self._room_label_adjacency_boundary(owner)
        if np.any(adjacency_boundary):
            total_boundary_cells += int(np.count_nonzero(adjacency_boundary))
            out[adjacency_boundary] = np.asarray((245, 250, 255), dtype=np.uint8)
        return out, total_mask_cells, total_boundary_cells

    @staticmethod
    def _mask_boundary(mask: np.ndarray) -> np.ndarray:
        arr = np.asarray(mask, dtype=bool)
        padded = np.pad(arr, 1, mode="constant", constant_values=False)
        neighbors = (
            padded[1:-1, :-2]
            & padded[1:-1, 2:]
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
        )
        return arr & ~neighbors

    @staticmethod
    def _room_label_adjacency_boundary(owner: np.ndarray) -> np.ndarray:
        labels = np.asarray(owner, dtype=np.int32)
        if labels.size == 0:
            return np.zeros_like(labels, dtype=bool)
        pos = labels > 0
        boundary = np.zeros_like(pos, dtype=bool)
        boundary[:, 1:] |= pos[:, 1:] & pos[:, :-1] & (labels[:, 1:] != labels[:, :-1])
        boundary[:, :-1] |= pos[:, :-1] & pos[:, 1:] & (labels[:, :-1] != labels[:, 1:])
        boundary[1:, :] |= pos[1:, :] & pos[:-1, :] & (labels[1:, :] != labels[:-1, :])
        boundary[:-1, :] |= pos[:-1, :] & pos[1:, :] & (labels[:-1, :] != labels[1:, :])
        return boundary

    def _draw_room_labels(
        self,
        draw: ImageDraw.ImageDraw,
        room_masks: Sequence[object],
        xy_func,
        crop_bounds: Tuple[int, int, int, int],
    ) -> int:
        r0, r1, c0, c1 = crop_bounds
        count = 0
        for idx, room in enumerate(room_masks[:64]):
            center = self._room_center_cell(room)
            if center is None:
                continue
            r, c = center
            if not (r0 <= r < r1 and c0 <= c < c1):
                continue
            label = self._room_label_text(room)
            if not label:
                continue
            x, y = xy_func(center)
            color = self._room_color(idx)
            self._label(draw, (x + 5, y - 9), label[:36], color)
            self._dot(draw, (x, y), color, radius=4)
            count += 1
        return count

    def _draw_roomseg_wall_debug_lines(self, draw: ImageDraw.ImageDraw, xy_func, crop_bounds: Tuple[int, int, int, int]) -> Tuple[int, int]:
        debug = self._room_segmentation_debug if isinstance(self._room_segmentation_debug, Mapping) else {}
        wall_color = (255, 35, 35)
        extension_color = (255, 0, 0)
        extension_outline_color = (8, 8, 8)
        wall_count = 0
        extension_count = 0

        filtered_report = debug.get("filtered_wall_lines_report") or {}
        if isinstance(filtered_report, Mapping):
            raw_lines = filtered_report.get("filtered_wall_lines") or []
        else:
            raw_lines = []
        line_items: List[Mapping[str, object]] = []
        for item in list(raw_lines)[:512]:
            if not isinstance(item, Mapping):
                continue
            line_items.append(item)
            p0 = self._debug_rc_point(item.get("p0_rc"))
            p1 = self._debug_rc_point(item.get("p1_rc"))
            if p0 is None or p1 is None or not self._line_overlaps_crop(p0, p1, crop_bounds):
                continue
            draw.line([xy_func(p0), xy_func(p1)], fill=wall_color, width=3)
            wall_count += 1

        visible_extension_starts: set[GridCell] = set()
        extension_report = debug.get("line_extension_report") or {}
        if isinstance(extension_report, Mapping):
            pass_reports = [extension_report.get("pass1"), extension_report.get("pass2")]
        else:
            pass_reports = []
        for report in pass_reports:
            if not isinstance(report, Mapping):
                continue
            for item in list(report.get("extensions") or [])[:1024]:
                if not isinstance(item, Mapping):
                    continue
                p0 = self._debug_rc_point(item.get("p_start_rc"))
                p1 = self._debug_rc_point(item.get("p_hit_rc"))
                if p0 is None or p1 is None or not self._line_overlaps_crop(p0, p1, crop_bounds):
                    continue
                if float(np.hypot(float(p1[0] - p0[0]), float(p1[1] - p0[1]))) >= 3.0:
                    visible_extension_starts.add((int(p0[0]), int(p0[1])))
                p0_xy = xy_func(p0)
                p1_xy = xy_func(p1)
                self._draw_dashed_line(draw, p0_xy, p1_xy, extension_outline_color, width=7, dash_px=10, gap_px=5)
                self._draw_dashed_line(draw, p0_xy, p1_xy, extension_color, width=4, dash_px=10, gap_px=5)
                for x, y in (p0_xy, p1_xy):
                    draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=extension_color, outline=extension_outline_color)
                extension_count += 1
        resolution_m = max(1e-6, float(debug.get("resolution_m", 0.05) or 0.05))
        probe_cells = max(6, int(round(1.80 / resolution_m)))
        for item in line_items[:512]:
            for endpoint in ("p0", "p1"):
                probe = self._line_endpoint_probe(item, endpoint, probe_cells)
                if probe is None:
                    continue
                p0, p1 = probe
                if (int(p0[0]), int(p0[1])) in visible_extension_starts:
                    continue
                if not self._line_overlaps_crop(p0, p1, crop_bounds):
                    continue
                p0_xy = xy_func(p0)
                p1_xy = xy_func(p1)
                self._draw_dashed_line(draw, p0_xy, p1_xy, extension_outline_color, width=5, dash_px=7, gap_px=6)
                self._draw_dashed_line(draw, p0_xy, p1_xy, extension_color, width=3, dash_px=7, gap_px=6)
                x, y = p0_xy
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=extension_color, outline=extension_outline_color)
                extension_count += 1
        return wall_count, extension_count

    def _draw_room_adjacency_debug_lines(self, draw: ImageDraw.ImageDraw, xy_func, crop_bounds: Tuple[int, int, int, int]) -> Tuple[int, int, List[dict]]:
        r0, r1, c0, c1 = crop_bounds
        merged_count = 0
        doorway_count = 0
        reasons: List[dict] = []
        for item in list(self._room_segmentation_debug.get("adjacency_evidence") or []):
            if not isinstance(item, Mapping):
                continue
            cells = []
            for raw_cell in list(item.get("boundary_cells_sample") or []):
                try:
                    row, col = int(raw_cell[0]), int(raw_cell[1])
                except Exception:
                    continue
                if r0 <= row < r1 and c0 <= col < c1:
                    cells.append((row, col))
            if not cells:
                continue
            verified = bool(item.get("verified_doorway", False))
            reason = str(item.get("merge_reason", ""))
            reasons.append(
                {
                    "room_a_label": item.get("room_a_label"),
                    "room_b_label": item.get("room_b_label"),
                    "verified_doorway": verified,
                    "merge_reason": reason,
                }
            )
            if verified:
                for cell in cells:
                    self._dot(draw, xy_func(cell), (255, 170, 40), radius=3)
                    doorway_count += 1
            else:
                for idx, cell in enumerate(cells):
                    if idx % 2 == 0:
                        self._dot(draw, xy_func(cell), (150, 150, 155), radius=2)
                        merged_count += 1
        return merged_count, doorway_count, reasons

    def _draw_corridor_merge_debug_lines(self, draw: ImageDraw.ImageDraw, xy_func, crop_bounds: Tuple[int, int, int, int]) -> Tuple[int, int]:
        debug = self._room_segmentation_debug if isinstance(self._room_segmentation_debug, Mapping) else {}
        report = debug.get("corridor_merge_report") or debug.get("corridor_merge") or {}
        if not isinstance(report, Mapping):
            return 0, 0
        merge_count = 0
        small_region_count = 0
        corridor_color = (255, 24, 24)
        small_region_color = (255, 190, 24)
        for item in list(report.get("merge_events") or []):
            if not isinstance(item, Mapping):
                continue
            p0 = self._debug_rc_point(item.get("shared_edge_total_p0_rc")) or self._debug_rc_point(item.get("shared_edge_p0_rc"))
            p1 = self._debug_rc_point(item.get("shared_edge_total_p1_rc")) or self._debug_rc_point(item.get("shared_edge_p1_rc"))
            if p0 is None or p1 is None:
                continue
            if not self._line_overlaps_crop(p0, p1, crop_bounds):
                continue
            self._draw_dashed_line(draw, xy_func(p0), xy_func(p1), corridor_color, width=4, dash_px=10, gap_px=6)
            midpoint = ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2)
            if _cell_in_crop(midpoint, *crop_bounds):
                mx, my = xy_func(midpoint)
                self._label(draw, (mx + 5, my - 11), "CORRIDOR", corridor_color)
            merge_count += 1
        for item in list(report.get("sliver_merge_events") or []):
            if not isinstance(item, Mapping):
                continue
            if str(item.get("reason", "")) != "merge_post_corridor_small_region_to_larger_neighbor":
                continue
            center = self._debug_rc_point(item.get("source_centroid_rc"))
            if center is None or not _cell_in_crop(center, *crop_bounds):
                continue
            cx, cy = xy_func(center)
            self._draw_dashed_circle(draw, (cx, cy), radius=11, color=small_region_color, width=3)
            neighbors = item.get("neighbors") if isinstance(item.get("neighbors"), list) else []
            label = "SMALL n=%d" % len(neighbors)
            if item.get("target") is not None:
                label += " ->%s" % str(item.get("target"))
            self._label(draw, (cx + 6, cy + 5), label, small_region_color)
            small_region_count += 1
        return merge_count, small_region_count

    @staticmethod
    def _debug_rc_point(raw: object) -> Optional[GridCell]:
        try:
            row, col = raw[:2]  # type: ignore[index]
            return int(round(float(row))), int(round(float(col)))
        except Exception:
            return None

    @classmethod
    def _line_endpoint_probe(cls, item: Mapping[str, object], endpoint: str, probe_cells: int) -> Optional[Tuple[GridCell, GridCell]]:
        p0 = cls._debug_rc_point(item.get("p0_rc"))
        p1 = cls._debug_rc_point(item.get("p1_rc"))
        if p0 is None or p1 is None:
            return None
        if str(endpoint) == "p0":
            start = p0
            delta = (int(p0[0]) - int(p1[0]), int(p0[1]) - int(p1[1]))
        else:
            start = p1
            delta = (int(p1[0]) - int(p0[0]), int(p1[1]) - int(p0[1]))
        if abs(int(delta[0])) >= abs(int(delta[1])):
            direction = (1 if int(delta[0]) >= 0 else -1, 0)
        else:
            direction = (0, 1 if int(delta[1]) >= 0 else -1)
        end = (int(start[0]) + int(direction[0]) * int(probe_cells), int(start[1]) + int(direction[1]) * int(probe_cells))
        return (int(start[0]), int(start[1])), end

    @staticmethod
    def _line_overlaps_crop(p0: GridCell, p1: GridCell, crop_bounds: Tuple[int, int, int, int]) -> bool:
        r0, r1, c0, c1 = crop_bounds
        min_r = min(int(p0[0]), int(p1[0]))
        max_r = max(int(p0[0]), int(p1[0]))
        min_c = min(int(p0[1]), int(p1[1]))
        max_c = max(int(p0[1]), int(p1[1]))
        return bool(max_r >= r0 and min_r < r1 and max_c >= c0 and min_c < c1)

    @staticmethod
    def _draw_dashed_line(
        draw: ImageDraw.ImageDraw,
        p0_xy: Tuple[int, int],
        p1_xy: Tuple[int, int],
        color: Tuple[int, int, int],
        *,
        width: int = 3,
        dash_px: int = 8,
        gap_px: int = 5,
    ) -> None:
        x0, y0 = float(p0_xy[0]), float(p0_xy[1])
        x1, y1 = float(p1_xy[0]), float(p1_xy[1])
        length = float(np.hypot(x1 - x0, y1 - y0))
        if length <= 1e-6:
            draw.point((int(round(x0)), int(round(y0))), fill=color)
            return
        ux = (x1 - x0) / length
        uy = (y1 - y0) / length
        pos = 0.0
        dash = max(1.0, float(dash_px))
        gap = max(0.0, float(gap_px))
        while pos < length:
            end = min(length, pos + dash)
            draw.line(
                [
                    (int(round(x0 + ux * pos)), int(round(y0 + uy * pos))),
                    (int(round(x0 + ux * end)), int(round(y0 + uy * end))),
                ],
                fill=color,
                width=max(1, int(width)),
            )
            pos += dash + gap

    @classmethod
    def _draw_dashed_circle(
        cls,
        draw: ImageDraw.ImageDraw,
        center_xy: Tuple[int, int],
        *,
        radius: int,
        color: Tuple[int, int, int],
        width: int = 2,
    ) -> None:
        cx, cy = float(center_xy[0]), float(center_xy[1])
        points: list[Tuple[int, int]] = []
        for idx in range(25):
            theta = (2.0 * np.pi * float(idx)) / 24.0
            points.append((int(round(cx + float(radius) * np.cos(theta))), int(round(cy + float(radius) * np.sin(theta)))))
        for idx in range(0, 24, 2):
            cls._draw_dashed_line(draw, points[idx], points[idx + 1], color, width=width, dash_px=4, gap_px=3)

    def _room_center_cell(self, room: object) -> Optional[GridCell]:
        metadata = getattr(room, "metadata", {}) or {}
        if isinstance(metadata, Mapping) and metadata.get("centroid_grid") is not None:
            try:
                row, col = metadata["centroid_grid"][:2]
                return int(round(float(row))), int(round(float(col)))
            except Exception:
                pass
        mask = getattr(room, "mask", None)
        if mask is None:
            return None
        rr, cc = np.nonzero(np.asarray(mask, dtype=bool))
        if rr.size == 0:
            return None
        return int(round(float(np.mean(rr)))), int(round(float(np.mean(cc))))

    def _room_label_text(self, room: object) -> str:
        room_id = str(getattr(room, "room_id", "room"))
        label = self._room_semantic_labels.get(room_id)
        if label is None:
            return room_id
        if isinstance(label, Mapping):
            category = str(label.get("category", "unknown"))
            reliability = label.get("label_reliability", label.get("confidence"))
        else:
            category = str(getattr(label, "category", "unknown"))
            reliability = getattr(label, "label_reliability", getattr(label, "confidence", None))
        if reliability is None:
            return "%s | %s" % (room_id, category)
        try:
            return "%s | %s | reliability=%.2f" % (room_id, category, float(reliability))
        except Exception:
            return "%s | %s" % (room_id, category)

    @staticmethod
    def _room_color(idx: int) -> Tuple[int, int, int]:
        palette = [
            (145, 110, 255),
            (255, 120, 120),
            (80, 190, 255),
            (255, 190, 80),
            (180, 130, 255),
            (90, 210, 170),
            (255, 145, 210),
            (210, 210, 90),
        ]
        return palette[int(idx) % len(palette)]

    def _draw_agent(self, draw: ImageDraw.ImageDraw, center: Tuple[int, int], yaw: float, scale: float) -> None:
        x, y = center
        radius = max(5, int(3 * scale))
        self._dot(draw, center, (255, 60, 60), radius=radius)
        end = (int(x + np.cos(yaw) * radius * 2.2), int(y - np.sin(yaw) * radius * 2.2))
        draw.line([center, end], fill=(255, 255, 255), width=2)

    def _draw_cells(
        self,
        draw: ImageDraw.ImageDraw,
        cells: Iterable[GridCell],
        xy_func,
        color: Tuple[int, int, int],
        radius: int,
        max_cells: int,
    ) -> int:
        cells_list = list(cells)
        if not cells_list:
            return 0
        stride = max(1, len(cells_list) // max(1, int(max_cells)))
        drawn = 0
        for cell in cells_list[::stride][:max_cells]:
            self._dot(draw, xy_func(cell), color, radius=radius)
            drawn += 1
        return drawn

    @staticmethod
    def _mask_cells_in_crop(mask: np.ndarray, crop_bounds: Tuple[int, int, int, int]) -> List[GridCell]:
        r0, r1, c0, c1 = crop_bounds
        arr = np.asarray(mask, dtype=bool)
        if arr.ndim != 2 or r1 <= r0 or c1 <= c0:
            return []
        rr, cc = np.nonzero(arr[int(r0):int(r1), int(c0):int(c1)])
        return [(int(r) + int(r0), int(c) + int(c0)) for r, c in zip(rr, cc)]

    def _dot(self, draw: ImageDraw.ImageDraw, xy: Tuple[int, int], color: Tuple[int, int, int], radius: int = 3) -> None:
        x, y = int(xy[0]), int(xy[1])
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)

    def _triangle(self, draw: ImageDraw.ImageDraw, xy: Tuple[int, int], color: Tuple[int, int, int], radius: int = 4) -> None:
        x, y = int(xy[0]), int(xy[1])
        pts = [(x, y - radius), (x - radius, y + radius), (x + radius, y + radius)]
        draw.polygon(pts, fill=color)

    def _cross(self, draw: ImageDraw.ImageDraw, xy: Tuple[int, int], color: Tuple[int, int, int], radius: int = 4) -> None:
        x, y = int(xy[0]), int(xy[1])
        draw.line([(x - radius, y - radius), (x + radius, y + radius)], fill=color, width=2)
        draw.line([(x - radius, y + radius), (x + radius, y - radius)], fill=color, width=2)

    def _star(self, draw: ImageDraw.ImageDraw, xy: Tuple[int, int], color: Tuple[int, int, int], radius: int = 6) -> None:
        x, y = int(xy[0]), int(xy[1])
        draw.line([(x - radius, y), (x + radius, y)], fill=color, width=2)
        draw.line([(x, y - radius), (x, y + radius)], fill=color, width=2)
        draw.line([(x - radius, y - radius), (x + radius, y + radius)], fill=color, width=1)
        draw.line([(x - radius, y + radius), (x + radius, y - radius)], fill=color, width=1)

    def _draw_crosses(
        self,
        draw: ImageDraw.ImageDraw,
        cells: Iterable[GridCell],
        xy_func,
        color: Tuple[int, int, int],
        radius: int,
        max_cells: int,
    ) -> int:
        cells_list = list(cells)
        if not cells_list:
            return 0
        drawn = 0
        for cell in cells_list[: max(1, int(max_cells))]:
            self._cross(draw, xy_func(cell), color, radius=radius)
            drawn += 1
        return drawn

    def _label(self, draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, color: Tuple[int, int, int]) -> None:
        x, y = int(xy[0]), int(xy[1])
        bbox = draw.textbbox((x, y), text, font=self._font)
        draw.rectangle([bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1], fill=(10, 12, 16))
        draw.text((x, y), text, fill=color, font=self._font)

    def _legend(self, draw: ImageDraw.ImageDraw, xy: Tuple[int, int]) -> None:
        x, y = xy
        items = [
            ((255, 60, 60), "agent"),
            ((145, 110, 255), "online room"),
            ((80, 170, 255), "wall extension"),
            ((255, 120, 40), "door completion"),
            ((245, 245, 245), "A*"),
            ((0, 225, 255), "frontier center"),
            ((255, 225, 40), "chosen frontier"),
            ((255, 150, 40), "goal cand"),
            ((255, 50, 50), "selected cand"),
            ((40, 220, 90), "accepted goal"),
            ((220, 70, 255), "standoff"),
            ((255, 150, 40), "planner target"),
        ]
        for color, label in items:
            self._dot(draw, (x + 6, y + 8), color, radius=5)
            draw.text((x + 16, y), label, fill=(230, 232, 235), font=self._font)
            y += 12

    @classmethod
    def _bbox_color(
        cls,
        det: Detection2D,
        goal_category: str,
        nav_decision: Optional[NavigationDecision],
    ) -> Tuple[int, int, int]:
        if cls._category_matches_goal(det.category, goal_category):
            return (255, 60, 60)
        candidate = nav_decision.selected_candidate if nav_decision and nav_decision.selected_candidate is not None else None
        if candidate is not None and cls._category_matches_goal(det.category, candidate.category):
            return (255, 60, 60)
        return (80, 230, 120)

    @classmethod
    def _visible_map_nodes(
        cls,
        object_memory: ObjectMemory,
        goal_category: str,
        nav_decision: Optional[NavigationDecision],
    ) -> List[object]:
        selected_id = cls._selected_candidate_id(nav_decision)
        visible: List[object] = []
        seen_ids = set()
        for node in object_memory.nodes:
            node_id = int(node.node_id)
            if node_id == selected_id or cls._category_matches_goal(node.category, goal_category):
                visible.append(node)
                seen_ids.add(node_id)
        candidate = nav_decision.selected_candidate if nav_decision and nav_decision.selected_candidate is not None else None
        if candidate is not None and int(candidate.node_id) not in seen_ids:
            visible.append(candidate)
        return visible

    @staticmethod
    def _selected_candidate_id(nav_decision: Optional[NavigationDecision]) -> Optional[int]:
        if nav_decision is None or nav_decision.selected_candidate is None:
            return None
        return int(nav_decision.selected_candidate.node_id)

    @classmethod
    def _candidate_node_color(
        cls,
        node,
        selected_id: Optional[int],
        nav_decision: Optional[NavigationDecision],
    ) -> Tuple[int, int, int]:
        if selected_id is None or int(node.node_id) != int(selected_id):
            return (255, 150, 40)
        meta = dict(getattr(nav_decision, "metadata", {}) or {}) if nav_decision is not None else {}
        if bool(meta.get("candidate_rejected", False)):
            return (135, 135, 135)
        if bool(meta.get("candidate_accepted", False)) or (nav_decision is not None and nav_decision.mode == "stop"):
            return (40, 220, 90)
        return (255, 50, 50)

    @staticmethod
    def _category_matches_goal(category: str, goal_category: str) -> bool:
        cat = normalize_category(category)
        goal = normalize_category(goal_category)
        return bool(goal and cat and (cat == goal or goal in cat or cat in goal))

    @staticmethod
    def _short(value) -> str:
        if isinstance(value, float):
            return "%.3f" % value
        return str(value)[:32]
