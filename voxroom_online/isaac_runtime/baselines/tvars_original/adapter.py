from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from ..data_contract import MapInfo, resolve_map_info
from ..mask_io import build_metric_domain_from_source, relabel_consecutive, save_baseline_snapshot_npz
from ..offline.fallback_utils import wavefront_fill_unlabeled
from ..topology_active.detector import (
    CameraIntrinsics,
    OriginalDetrDoorDetector,
    camera_intrinsics_from_mapping,
    project_door_bbox_to_grid_with_status,
)


BASELINE_NAME = "tvars_original_isaac"


@dataclass
class _OriginalModules:
    repo_dir: Path
    convert_2_laser: Any
    DoorDetection: Any
    FrontierDetection: Any
    TopomapConstruction: Any
    map_size_cells: int
    resolution_m: float


class TVARSOriginalIsaacBaseline:
    """Run TVARS room segmentation modules on Isaac observations.

    Habitat is intentionally not launched here. Isaac provides RGB-D, robot pose,
    and online maps; the door/frontier/topomap update calls come from the
    original Active_room_segmentation source checkout.
    """

    baseline_name = BASELINE_NAME

    def __init__(
        self,
        *,
        output_dir: Path,
        repo_dir: Path | str | None = None,
        detector: OriginalDetrDoorDetector | None = None,
        save_stream: bool = False,
        save_every_snapshot: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.repo_dir = _resolve_repo_dir(repo_dir)
        self.detector = detector or OriginalDetrDoorDetector(repo_dir=self.repo_dir)
        self.save_stream = bool(save_stream)
        self.save_every_snapshot = bool(save_every_snapshot)
        self._modules: _OriginalModules | None = None
        self._door_detect: Any | None = None
        self._frontier_detector: Any | None = None
        self._topomap: Any | None = None
        self._detected_door_list: list[dict[str, Any]] = []
        self._raw_detect_list: list[list[int]] = []
        self.latest_label_map: np.ndarray | None = None
        self.latest_debug_arrays: dict[str, np.ndarray] = {}
        self.latest_metadata: dict[str, Any] = {}
        self._stream_manifest_path = self.output_dir.parent.parent / "tvars_original_stream" / "stream_manifest.jsonl"
        self._last_stream_step: int | None = None

    def on_episode_start(self, episode_metadata: Mapping[str, Any]) -> None:
        if not self.save_stream:
            return
        self._stream_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._stream_manifest_path.exists():
            self._stream_manifest_path.write_text("", encoding="utf-8")
        meta_path = self._stream_manifest_path.parent / "episode_metadata.json"
        meta_path.write_text(json.dumps(_json_ready(dict(episode_metadata)), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def update(
        self,
        *,
        step: int,
        obs: Mapping[str, Any] | None,
        sgnav_obs: Mapping[str, Any] | None = None,
        map_state: Mapping[str, Any],
        mapper: Any | None = None,
        room_segmenter: Any | None = None,
        frontier_map: np.ndarray | None = None,
        selected_frontier_center_rc: tuple[int, int] | None = None,
        camera_intrinsics: Any | None = None,
    ) -> None:
        _ = sgnav_obs, room_segmenter, frontier_map, selected_frontier_center_rc
        arrays = _arrays_from_map_state(map_state)
        map_info = resolve_map_info(map_state=map_state, mapper=mapper, snapshot_arrays=arrays)
        shape = tuple(int(v) for v in arrays["occupancy_map"].shape)
        modules = self._ensure_original_modules(shape=shape, resolution_m=float(map_info.resolution_m))
        agent_rc = tuple(int(v) for v in np.asarray(map_state["current_grid"]).reshape(-1)[:2])

        if self.save_stream:
            self._save_stream_frame(step=int(step), arrays=arrays, obs=obs or {}, map_info=map_info)

        metadata: dict[str, Any] = {
            "method": BASELINE_NAME,
            "runner_type": "original_tvars_modules_isaac_adapter",
            "original_repo": "FreeformRobotics/Active_room_segmentation",
            "original_repo_path": str(self.repo_dir),
            "original_repo_commit": _git_head(self.repo_dir),
            "environment_adapter": "isaac_no_habitat",
            "habitat_runtime_launched": False,
            "uses_rgb": True,
            "uses_depth": True,
            "uses_occupancy": True,
            "uses_oracle_semantics": False,
            "policy_control": "never",
            "baseline_policy_control": "never",
            "snapshot_step": int(step),
            "map_info": map_info.to_metadata(),
            "tvars_original_map_size_cells": int(modules.map_size_cells),
            "tvars_original_resolution_m": float(modules.resolution_m),
        }

        obs_map = np.asarray(arrays["obstacle_mask"], dtype=np.uint8)
        exp_map = (~np.asarray(arrays["unknown_mask"], dtype=bool)).astype(np.uint8)
        free_mask = np.asarray(arrays["observed_free_mask"], dtype=bool)
        h, w = obs_map.shape
        agent_r = int(np.clip(agent_rc[0], 0, h - 1))
        agent_c = int(np.clip(agent_rc[1], 0, w - 1))
        yaw_deg = _yaw_deg_from_map_state(map_state)
        bot_pose_m = np.asarray([agent_r * modules.resolution_m, agent_c * modules.resolution_m, yaw_deg], dtype=np.float32)
        lmb = np.asarray([0, h, 0, w], dtype=np.int32)

        with _suppress_original_stdout():
            hough_door_list, laser_list = modules.convert_2_laser(obs_map.copy(), exp_map.copy(), bot_pose_m.copy())
        hough_door_list = _clip_xy_points(hough_door_list, shape=shape, margin=22)
        vision_candidates, vision_map, vision_meta = self._project_visual_door_candidates(
            obs=obs or {},
            camera_intrinsics=camera_intrinsics,
            map_info=map_info,
            shape=shape,
        )
        filtered_hough = _filter_hough_with_vision(hough_door_list, vision_map=vision_map, radius_cells=8)
        vision_fallback_points: list[list[int]] = []
        loose_hough_fallback_points: list[list[int]] = []
        used_vision_loose_hough_fallback = False
        used_vision_direct_fallback = False
        if not filtered_hough and vision_candidates:
            loose_hough_fallback_points = _filter_hough_with_vision(
                hough_door_list,
                vision_map=vision_map,
                radius_cells=20,
            )
            if loose_hough_fallback_points:
                filtered_hough = list(loose_hough_fallback_points)
                used_vision_loose_hough_fallback = True
        if not filtered_hough and vision_candidates:
            vision_fallback_points = _vision_candidates_to_xy_points(vision_candidates, shape=shape, margin=22)
            filtered_hough = list(vision_fallback_points)
            used_vision_direct_fallback = bool(filtered_hough)

        if self._door_detect is None:
            self._door_detect = modules.DoorDetection([agent_c, min(agent_c + 1, w - 1)], [agent_r, agent_r])
        if self._frontier_detector is None:
            self._frontier_detector = modules.FrontierDetection(int(h))
        if self._topomap is None:
            self._topomap = modules.TopomapConstruction()

        door_list: list[dict[str, Any]] = []
        raw_list: list[list[int]] = []
        room_exp_list: list[list[int]] = []
        door_grid: list[list[int]] = []
        close_door_list: list[dict[str, Any]] = []
        door_filter_debug_list: list[dict[str, Any]] = []
        checked_doors: list[dict[str, Any]] = []
        door_remove_list: list[dict[str, Any]] = []
        return_flag = False
        update_error: str | None = None
        try:
            with _suppress_original_stdout():
                door_list, raw_list = self._door_detect.door_filter(
                    obs_map.copy(),
                    obs_map.copy(),
                    exp_map.copy(),
                    [agent_c, agent_r],
                    [],
                    self._detected_door_list,
                    use_12point=False,
                    external_door_point=filtered_hough,
                )
                close_door_list = list(door_list)
                door_filter_debug_list = [dict(door) for door in door_list]
                self._detected_door_list.extend(door_list)
                self._raw_detect_list.extend(raw_list)
                _, _, show_map, room_exp_list, door_grid = self._frontier_detector.frontier_detection(
                    np.asarray([agent_r, agent_c], dtype=np.int32),
                    np.asarray([agent_r, agent_c], dtype=np.int32),
                    obs_map.copy(),
                    exp_map.copy(),
                    lmb.copy(),
                    self._detected_door_list,
                    laser_list,
                )
                self._detected_door_list = self._topomap.same_node_check(room_exp_list, self._detected_door_list)
                checked_doors, door_remove_list = self._topomap.check_topomap(
                    close_door_list,
                    self._detected_door_list,
                    room_exp_list,
                    [agent_c, agent_r],
                    obs_map.copy(),
                    exp_map.copy(),
                    lmb.copy(),
                    door_grid,
                    0,
                    "isaac",
                    laser_list,
                )
                _, return_flag = self._topomap.add_room(
                    checked_doors,
                    [agent_c, agent_r],
                    obs_map.copy(),
                    exp_map.copy(),
                    lmb.copy(),
                )
            for door in list(door_remove_list):
                if door in self._detected_door_list:
                    self._detected_door_list.remove(door)
        except Exception as exc:  # Keep the Isaac run alive; metadata records the failure.
            update_error = f"{type(exc).__name__}: {exc}"
            show_map = np.zeros(shape, dtype=np.float32)

        original_topomap_label_map = self._label_map_from_original_topomap(
            shape=shape,
            domain=free_mask,
            fallback_room_exp=room_exp_list,
        )
        frontier_room_exp_map = _rc_points_to_map(room_exp_list, shape=shape)
        frontier_door_grid_map = _rc_points_to_map(door_grid, shape=shape)
        door_filter_line_map = _doors_to_line_map(door_filter_debug_list, shape=shape)
        extended_door_filter_line_map = _extend_separator_line_components(
            door_filter_line_map | frontier_door_grid_map,
            seed_domain=frontier_room_exp_map,
            max_extend_cells=120,
            allowed_gap_cells=2,
        )
        isaac_adapter_separator_map = door_filter_line_map | frontier_door_grid_map | extended_door_filter_line_map
        label_map = original_topomap_label_map
        isaac_adapter_label_count = int(np.max(label_map)) if label_map.size else 0
        isaac_adapter_used = False
        if int(np.max(original_topomap_label_map)) <= 1 and bool(np.any(isaac_adapter_separator_map)):
            adapted_label_map = self._label_map_from_isaac_door_separators(
                domain=free_mask,
                cut_seed_domain=frontier_room_exp_map,
                separator_map=isaac_adapter_separator_map,
            )
            adapted_label_count = int(np.max(adapted_label_map)) if adapted_label_map.size else 0
            if adapted_label_count > int(np.max(original_topomap_label_map)):
                label_map = adapted_label_map
                isaac_adapter_label_count = adapted_label_count
                isaac_adapter_used = True
        self.latest_label_map = label_map
        self.latest_debug_arrays = {
            "tvars_original_hough_door_seed_map": _xy_points_to_map(hough_door_list, shape=shape),
            "tvars_original_filtered_hough_door_seed_map": _xy_points_to_map(filtered_hough, shape=shape),
            "tvars_original_vision_door_seed_map": vision_map.astype(bool),
            "tvars_original_door_filter_line_map": door_filter_line_map,
            "tvars_original_extended_door_filter_line_map": extended_door_filter_line_map,
            "tvars_original_door_line_map": _doors_to_line_map(self._detected_door_list, shape=shape),
            "tvars_original_frontier_room_exp_map": frontier_room_exp_map,
            "tvars_original_frontier_door_grid_map": frontier_door_grid_map,
            "tvars_original_isaac_adapter_separator_map": isaac_adapter_separator_map,
            "tvars_original_frontier_show_map": np.asarray(show_map, dtype=np.float32),
            "tvars_original_pre_isaac_adapter_topomap_label_map": original_topomap_label_map.astype(np.int32),
            "tvars_original_topomap_label_map": label_map.astype(np.int32),
        }
        metadata.update(
            {
                "detector_name": str(getattr(self.detector, "name", "unknown")),
                "detector_available": bool(getattr(self.detector, "available", False)),
                "detector_adapter_verified": bool(getattr(self.detector, "detector_adapter_verified", False)),
                "checkpoint_path": getattr(self.detector, "checkpoint_path", None),
                "checkpoint_sha256": getattr(self.detector, "checkpoint_sha256", None),
                "vision_num_detections": int(vision_meta.get("num_detections", 0)),
                "vision_num_candidates": int(len(vision_candidates)),
                "vision_projection_attempted": bool(vision_meta.get("projection_attempted", False)),
                "vision_projection_missing_inputs": list(vision_meta.get("missing_inputs", [])),
                "vision_projection_status_counts": dict(vision_meta.get("projection_status_counts", {})),
                "vision_rgb_square_crop_applied": bool(vision_meta.get("rgb_square_crop_applied", False)),
                "vision_rgb_square_crop_xyxy": vision_meta.get("rgb_square_crop_xyxy"),
                "vision_rgb_original_shape_hw": vision_meta.get("rgb_original_shape_hw"),
                "vision_rgb_detector_shape_hw": vision_meta.get("rgb_detector_shape_hw"),
                "vision_depth_square_crop_applied": bool(vision_meta.get("depth_square_crop_applied", False)),
                "hough_num_candidates": int(len(hough_door_list)),
                "hough_num_rgb_filtered_candidates": int(len(filtered_hough)),
                "hough_rgb_filter_used_vision_loose_hough_fallback": bool(used_vision_loose_hough_fallback),
                "vision_loose_hough_fallback_radius_cells": 20,
                "vision_loose_hough_fallback_num_points": int(len(loose_hough_fallback_points)),
                "hough_rgb_filter_used_vision_direct_fallback": bool(used_vision_direct_fallback),
                "vision_direct_fallback_num_points": int(len(vision_fallback_points)),
                "door_filter_num_raw": int(len(raw_list)),
                "door_filter_num_accepted_this_step": int(len(door_list)),
                "topomap_num_checked_doors": int(len(checked_doors)),
                "topomap_num_removed_doors": int(len(door_remove_list)),
                "door_memory_num_detected": int(len(self._detected_door_list)),
                "topomap_num_vertices": int(_topomap_vertex_count(self._topomap)),
                "original_topomap_label_count": int(np.max(original_topomap_label_map))
                if original_topomap_label_map.size
                else 0,
                "isaac_adapter_used_door_filter_separator": bool(isaac_adapter_used),
                "isaac_adapter_frontier_room_exp_cells": int(np.count_nonzero(frontier_room_exp_map)),
                "isaac_adapter_extended_separator_line_cells": int(np.count_nonzero(extended_door_filter_line_map)),
                "isaac_adapter_separator_line_cells": int(np.count_nonzero(isaac_adapter_separator_map)),
                "isaac_adapter_separator_label_count": int(isaac_adapter_label_count),
                "return_flag": bool(return_flag),
                "update_error": update_error,
                "main_experiment_allowed": bool(
                    getattr(self.detector, "available", False)
                    and getattr(self.detector, "detector_adapter_verified", False)
                    and update_error is None
                ),
                "approximation_note": (
                    "Original TVARS door/frontier/topomap modules are used, but Habitat is not launched; Isaac supplies RGB-D and maps."
                ),
            }
        )
        self.latest_metadata = metadata

    def save_snapshot_like_voxroom(
        self,
        *,
        source_snapshot_npz: Path | Mapping[str, Any],
        step: int,
        source_summary_json: Path | None = None,
    ) -> Path:
        source_snapshot_npz = _snapshot_npz_path(source_snapshot_npz)
        with np.load(source_snapshot_npz, allow_pickle=False) as data:
            occupancy = np.asarray(data["occupancy_map"])
        label_map = self.latest_label_map
        if label_map is None:
            label_map = np.zeros(occupancy.shape, dtype=np.int32)
        output_npz = self.output_dir / "roomseg_snapshots" / source_snapshot_npz.name
        metadata = {
            **dict(self.latest_metadata),
            "source_snapshot": str(source_snapshot_npz),
            "source_summary_json": None if source_summary_json is None else str(source_summary_json),
            "snapshot_step": int(step),
            "save_every_snapshot": bool(self.save_every_snapshot),
        }
        save_baseline_snapshot_npz(
            source_npz_path=source_snapshot_npz,
            output_npz_path=output_npz,
            baseline_label_map=label_map,
            baseline_name=BASELINE_NAME,
            metadata=metadata,
            debug_arrays=self.latest_debug_arrays,
        )
        _write_summary_json(
            output_npz.with_suffix(".summary.json"),
            output_npz=output_npz,
            source_npz=source_snapshot_npz,
            metadata=metadata,
            label_map=label_map,
        )
        return output_npz

    def on_episode_end(self) -> None:
        return None

    def _ensure_original_modules(self, *, shape: tuple[int, int], resolution_m: float) -> _OriginalModules:
        if self._modules is not None:
            return self._modules
        if shape[0] != shape[1]:
            raise ValueError("TVARS original map expects square maps, got %s" % (shape,))
        if abs(float(resolution_m) - 0.05) > 1e-4:
            raise ValueError("TVARS original hard-codes 0.05 m cells, got %.6f" % float(resolution_m))
        self._modules = _load_original_modules(
            repo_dir=self.repo_dir,
            map_size_cells=int(shape[0]),
            resolution_m=float(resolution_m),
        )
        return self._modules

    def _project_visual_door_candidates(
        self,
        *,
        obs: Mapping[str, Any],
        camera_intrinsics: Any | None,
        map_info: MapInfo,
        shape: tuple[int, int],
    ) -> tuple[list[Any], np.ndarray, dict[str, Any]]:
        vision_map = np.zeros(shape, dtype=bool)
        meta = {
            "num_detections": 0,
            "projection_attempted": False,
            "missing_inputs": [],
            "projection_status_counts": {},
        }
        if not bool(getattr(self.detector, "available", False)):
            meta["missing_inputs"].append("detector_unavailable")
            return [], vision_map, meta
        if not (obs.get("has_rgb") and "rgb" in obs):
            meta["missing_inputs"].append("rgb")
            return [], vision_map, meta
        depth = obs.get("depth")
        camera_pose_world = obs.get("camera_pose_world")
        if depth is None:
            meta["missing_inputs"].append("depth")
        if camera_intrinsics is None:
            meta["missing_inputs"].append("camera_intrinsics")
        if camera_pose_world is None:
            meta["missing_inputs"].append("camera_pose_world")
        rgb = _safe_numpy(obs.get("rgb"))
        if rgb is None:
            meta["missing_inputs"].append("rgb_numpy")
        if meta["missing_inputs"]:
            return [], vision_map, meta
        rgb_for_detector, depth_for_projection, intr_for_projection, crop_meta = _square_crop_camera_frame(
            np.asarray(rgb),
            np.asarray(depth, dtype=np.float32),
            camera_intrinsics,
        )
        meta.update(crop_meta)
        detections = self.detector.detect(np.asarray(rgb_for_detector))
        meta["num_detections"] = int(len(detections))
        meta["projection_attempted"] = True
        candidates = []
        for detection in detections:
            candidate, status = project_door_bbox_to_grid_with_status(
                detection,
                depth=np.asarray(depth_for_projection, dtype=np.float32),
                camera_intrinsics=intr_for_projection,
                camera_pose_world=np.asarray(camera_pose_world, dtype=np.float32),
                map_info=map_info,
            )
            status_counts = meta.setdefault("projection_status_counts", {})
            status_counts[str(status)] = int(status_counts.get(str(status), 0)) + 1
            if candidate is None:
                continue
            r, c = int(candidate.rc[0]), int(candidate.rc[1])
            if 0 <= r < shape[0] and 0 <= c < shape[1]:
                vision_map[r, c] = True
                candidates.append(candidate)
        if bool(vision_map.any()):
            vision_map = cv2.dilate(vision_map.astype(np.uint8), np.ones((17, 17), np.uint8), iterations=1).astype(bool)
        return candidates, vision_map, meta

    def _label_map_from_original_topomap(
        self,
        *,
        shape: tuple[int, int],
        domain: np.ndarray,
        fallback_room_exp: list[list[int]],
    ) -> np.ndarray:
        labels = np.zeros(shape, dtype=np.int32)
        graph = getattr(self._topomap, "g", None)
        if graph is not None:
            try:
                for vertex_idx in range(int(graph.vcount())):
                    room_exp = graph.vs[vertex_idx]["room_exp"]
                    for rc in room_exp or []:
                        r, c = int(rc[0]), int(rc[1])
                        if 0 <= r < shape[0] and 0 <= c < shape[1]:
                            labels[r, c] = int(vertex_idx) + 1
            except Exception:
                labels.fill(0)
        if not bool(np.any(labels > 0)) and fallback_room_exp:
            for rc in fallback_room_exp:
                r, c = int(rc[0]), int(rc[1])
                if 0 <= r < shape[0] and 0 <= c < shape[1]:
                    labels[r, c] = 1
        labels[~np.asarray(domain, dtype=bool)] = 0
        labels = wavefront_fill_unlabeled(labels, domain=np.asarray(domain, dtype=bool))
        return relabel_consecutive(labels)

    def _label_map_from_isaac_door_separators(
        self,
        *,
        domain: np.ndarray,
        cut_seed_domain: np.ndarray,
        separator_map: np.ndarray,
    ) -> np.ndarray:
        domain_mask = np.asarray(domain, dtype=bool)
        seed_mask = np.asarray(cut_seed_domain, dtype=bool)
        if not bool(np.any(seed_mask)):
            seed_mask = domain_mask
        seed_mask &= domain_mask
        separator_mask = np.asarray(separator_map, dtype=bool) & seed_mask
        cut_domain = seed_mask & ~separator_mask
        if not bool(np.any(cut_domain)):
            return np.zeros(domain_mask.shape, dtype=np.int32)
        _, labels = cv2.connectedComponents(cut_domain.astype(np.uint8), connectivity=8)
        labels = np.asarray(labels, dtype=np.int32)
        labels[~cut_domain] = 0
        labels = wavefront_fill_unlabeled(labels, domain=domain_mask)
        labels[~domain_mask] = 0
        return relabel_consecutive(labels)

    def _save_stream_frame(
        self,
        *,
        step: int,
        arrays: Mapping[str, np.ndarray],
        obs: Mapping[str, Any],
        map_info: MapInfo,
    ) -> None:
        if self._last_stream_step == int(step):
            return
        self._last_stream_step = int(step)
        self._stream_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        frames_dir = self._stream_manifest_path.parent / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        frame_npz = frames_dir / ("frame_%06d.npz" % int(step))
        frame_arrays: dict[str, Any] = {
            "occupancy_map": np.asarray(arrays["occupancy_map"]),
            "observed_free_mask": np.asarray(arrays["observed_free_mask"]),
            "obstacle_mask": np.asarray(arrays["obstacle_mask"]),
            "unknown_mask": np.asarray(arrays["unknown_mask"]),
            "agent_rc": np.asarray(arrays["agent_rc"], dtype=np.int32),
            "map_resolution_m": np.asarray(float(map_info.resolution_m), dtype=np.float32),
            "control_step": np.asarray(int(step), dtype=np.int64),
        }
        if obs.get("has_rgb") and "rgb" in obs:
            rgb = _safe_numpy(obs["rgb"])
            if rgb is not None:
                frame_arrays["rgb"] = np.asarray(rgb)
        if obs.get("has_depth") and "depth" in obs:
            frame_arrays["depth"] = np.asarray(obs["depth"], dtype=np.float32)
        if "camera_pose_world" in obs:
            frame_arrays["camera_pose_world"] = np.asarray(obs["camera_pose_world"], dtype=np.float32)
        np.savez_compressed(frame_npz, **frame_arrays)
        with self._stream_manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": int(step), "frame_npz": str(frame_npz)}, sort_keys=True) + "\n")


def _load_original_modules(*, repo_dir: Path, map_size_cells: int, resolution_m: float) -> _OriginalModules:
    repo_dir = Path(repo_dir).resolve()
    if not repo_dir.exists():
        raise FileNotFoundError(f"Active_room_segmentation repo not found: {repo_dir}")
    map_resolution_cm = int(round(float(resolution_m) * 100.0))
    map_size_cm = int(map_size_cells * map_resolution_cm)
    argv = [
        "tvars_original_isaac",
        "--no_cuda",
        "--visualize",
        "0",
        "--map_size_cm",
        str(map_size_cm),
        "--map_resolution",
        str(map_resolution_cm),
    ]
    old_argv = list(sys.argv)
    old_path = list(sys.path)
    try:
        sys.argv = argv
        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))
        _patch_matplotlib_for_original()
        door_module = importlib.import_module("door_detection")
        frontier_module = importlib.import_module("frontier_detection")
        topo_module = importlib.import_module("topomap_construction")
        hough_module = _load_hough_module(repo_dir)
        _patch_original_plotting(topo_module)
        return _OriginalModules(
            repo_dir=repo_dir,
            convert_2_laser=hough_module.convert_2_laser,
            DoorDetection=door_module.Door_detection,
            FrontierDetection=frontier_module.Frontier_detection,
            TopomapConstruction=topo_module.Topomap_construction,
            map_size_cells=int(map_size_cells),
            resolution_m=float(resolution_m),
        )
    finally:
        sys.argv = old_argv
        sys.path[:] = old_path


def _load_hough_module(repo_dir: Path) -> Any:
    path = repo_dir / "env" / "habitat" / "hough_door_detection.py"
    spec = importlib.util.spec_from_file_location("_tvars_original_hough_door_detection", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patch_matplotlib_for_original() -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt

        plt.ion = lambda *args, **kwargs: None
        plt.ioff = lambda *args, **kwargs: None
        plt.show = lambda *args, **kwargs: None
        plt.pause = lambda *args, **kwargs: None
    except Exception:
        return


def _patch_original_plotting(topo_module: Any) -> None:
    try:
        topo_module.plt.ion = lambda *args, **kwargs: None
        topo_module.plt.ioff = lambda *args, **kwargs: None
        topo_module.plt.show = lambda *args, **kwargs: None
        topo_module.plt.pause = lambda *args, **kwargs: None
    except Exception:
        pass
    try:
        topo_module.ig.plot = lambda *args, **kwargs: None
    except Exception:
        pass


def _resolve_repo_dir(repo_dir: Path | str | None) -> Path:
    if repo_dir is not None:
        return Path(repo_dir).expanduser().resolve()
    raw = os.environ.get("ACTIVE_ROOM_SEG_ROOT")
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.cwd() / "external_baselines" / "Active_room_segmentation").resolve()


def _arrays_from_map_state(map_state: Mapping[str, Any]) -> dict[str, np.ndarray]:
    occupancy = np.asarray(map_state["occupancy"], dtype=bool)
    free = np.asarray(map_state["free"], dtype=bool)
    observed = np.asarray(map_state["observed"], dtype=bool)
    return {
        "occupancy_map": occupancy,
        "observed_free_mask": free,
        "obstacle_mask": occupancy,
        "unknown_mask": ~observed,
        "navigation_free_room_domain": free,
        "agent_rc": np.asarray(map_state["current_grid"], dtype=np.int32),
    }


def _yaw_deg_from_map_state(map_state: Mapping[str, Any]) -> float:
    pose = map_state.get("pose")
    if pose is None:
        return 0.0
    arr = np.asarray(pose, dtype=np.float32).reshape(-1)
    if arr.size >= 3:
        return float(np.degrees(float(arr[2])))
    return 0.0


def _clip_xy_points(points: Any, *, shape: tuple[int, int], margin: int = 0) -> list[list[int]]:
    out: list[list[int]] = []
    h, w = int(shape[0]), int(shape[1])
    for point in list(points or []):
        if len(point) < 2:
            continue
        x, y = int(point[0]), int(point[1])
        if margin <= x < w - margin and margin <= y < h - margin:
            out.append([x, y])
    return out


def _filter_hough_with_vision(points: list[list[int]], *, vision_map: np.ndarray, radius_cells: int) -> list[list[int]]:
    if not bool(np.any(vision_map)):
        return []
    h, w = vision_map.shape
    radius = int(radius_cells)
    out: list[list[int]] = []
    for x, y in points:
        y0, y1 = max(0, int(y) - radius), min(h, int(y) + radius + 1)
        x0, x1 = max(0, int(x) - radius), min(w, int(x) + radius + 1)
        if bool(np.any(vision_map[y0:y1, x0:x1])):
            out.append([int(x), int(y)])
    return out


def _vision_candidates_to_xy_points(candidates: list[Any], *, shape: tuple[int, int], margin: int = 0) -> list[list[int]]:
    out: list[list[int]] = []
    h, w = int(shape[0]), int(shape[1])
    for candidate in candidates:
        try:
            r, c = candidate.rc
        except Exception:
            continue
        x, y = int(c), int(r)
        if margin <= x < w - margin and margin <= y < h - margin:
            out.append([x, y])
    return out


def _xy_points_to_map(points: list[list[int]], *, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for x, y in points:
        if 0 <= int(y) < shape[0] and 0 <= int(x) < shape[1]:
            out[int(y), int(x)] = True
    return out


def _rc_points_to_map(points: list[list[int]], *, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for point in points or []:
        if len(point) < 2:
            continue
        r, c = int(point[0]), int(point[1])
        if 0 <= r < shape[0] and 0 <= c < shape[1]:
            out[r, c] = True
    return out


def _doors_to_line_map(doors: list[dict[str, Any]], *, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=np.uint8)
    for door in doors:
        try:
            start = tuple(int(v) for v in door["start"])
            end = tuple(int(v) for v in door["end"])
        except Exception:
            continue
        cv2.line(out, start, end, 1, thickness=3)
    return out.astype(bool)


def _extend_separator_line_components(
    line_map: np.ndarray,
    *,
    seed_domain: np.ndarray,
    max_extend_cells: int,
    allowed_gap_cells: int,
) -> np.ndarray:
    line = np.asarray(line_map, dtype=bool)
    seed = np.asarray(seed_domain, dtype=bool)
    out = np.zeros(line.shape, dtype=np.uint8)
    if not bool(np.any(line)) or not bool(np.any(seed)):
        return out.astype(bool)
    num_labels, comp_labels = cv2.connectedComponents(line.astype(np.uint8), connectivity=8)
    for label_id in range(1, int(num_labels)):
        points = np.argwhere(comp_labels == label_id)
        if points.shape[0] < 2:
            continue
        center = points.astype(np.float64).mean(axis=0)
        centered = points.astype(np.float64) - center
        cov = centered.T @ centered / max(int(points.shape[0]), 1)
        try:
            values, vectors = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            continue
        direction = vectors[:, int(np.argmax(values))]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            continue
        direction = direction / norm
        projections = centered @ direction
        p0 = center + direction * float(np.min(projections))
        p1 = center + direction * float(np.max(projections))
        p0 = _extend_separator_endpoint(
            p0,
            -direction,
            seed=seed,
            max_extend_cells=int(max_extend_cells),
            allowed_gap_cells=int(allowed_gap_cells),
        )
        p1 = _extend_separator_endpoint(
            p1,
            direction,
            seed=seed,
            max_extend_cells=int(max_extend_cells),
            allowed_gap_cells=int(allowed_gap_cells),
        )
        r0, c0 = int(round(p0[0])), int(round(p0[1]))
        r1, c1 = int(round(p1[0])), int(round(p1[1]))
        if not (0 <= r0 < line.shape[0] and 0 <= c0 < line.shape[1]):
            continue
        if not (0 <= r1 < line.shape[0] and 0 <= c1 < line.shape[1]):
            continue
        cv2.line(out, (c0, r0), (c1, r1), 1, thickness=3)
    return out.astype(bool) & seed


def _extend_separator_endpoint(
    start_rc: np.ndarray,
    direction_rc: np.ndarray,
    *,
    seed: np.ndarray,
    max_extend_cells: int,
    allowed_gap_cells: int,
) -> np.ndarray:
    h, w = seed.shape
    last = np.asarray(start_rc, dtype=np.float64).copy()
    consecutive_gap = 0
    for step in range(1, int(max_extend_cells) + 1):
        current = np.asarray(start_rc, dtype=np.float64) + np.asarray(direction_rc, dtype=np.float64) * float(step)
        r, c = int(round(current[0])), int(round(current[1]))
        if not (0 <= r < h and 0 <= c < w):
            break
        if bool(seed[r, c]):
            last = current.copy()
            consecutive_gap = 0
        else:
            consecutive_gap += 1
            if consecutive_gap > int(allowed_gap_cells):
                break
    return last


def _square_crop_camera_frame(
    rgb: np.ndarray,
    depth: np.ndarray,
    camera_intrinsics: Any,
) -> tuple[np.ndarray, np.ndarray, Any, dict[str, Any]]:
    rgb_arr = np.asarray(rgb)
    if rgb_arr.ndim < 2:
        return rgb_arr, depth, camera_intrinsics, {"rgb_square_crop_applied": False}
    h, w = int(rgb_arr.shape[0]), int(rgb_arr.shape[1])
    size = int(min(h, w))
    if size <= 0 or h == w:
        return rgb_arr, depth, camera_intrinsics, {"rgb_square_crop_applied": False}
    y0 = int((h - size) // 2)
    x0 = int((w - size) // 2)
    rgb_crop = rgb_arr[y0 : y0 + size, x0 : x0 + size, ...]

    depth_arr = np.asarray(depth)
    depth_crop = depth_arr
    depth_crop_applied = False
    if depth_arr.ndim >= 2 and int(depth_arr.shape[0]) == h and int(depth_arr.shape[1]) == w:
        depth_crop = depth_arr[y0 : y0 + size, x0 : x0 + size, ...]
        depth_crop_applied = True

    intr = _camera_intrinsics_duck(camera_intrinsics)
    intr_crop: Any = camera_intrinsics
    if intr is not None:
        intr_crop = CameraIntrinsics(
            fx=float(intr.fx),
            fy=float(intr.fy),
            cx=float(intr.cx) - float(x0),
            cy=float(intr.cy) - float(y0),
            width=size,
            height=size,
        )
    return (
        rgb_crop,
        depth_crop,
        intr_crop,
        {
            "rgb_square_crop_applied": True,
            "rgb_square_crop_xyxy": [int(x0), int(y0), int(x0 + size), int(y0 + size)],
            "rgb_original_shape_hw": [int(h), int(w)],
            "rgb_detector_shape_hw": [int(size), int(size)],
            "depth_square_crop_applied": bool(depth_crop_applied),
        },
    )


def _camera_intrinsics_duck(value: Any) -> CameraIntrinsics | None:
    if value is not None and any(hasattr(value, name) for name in ("fx", "f_x")):
        try:
            fx = getattr(value, "fx") if hasattr(value, "fx") else getattr(value, "f_x")
            fy = getattr(value, "fy") if hasattr(value, "fy") else getattr(value, "f_y")
            cx = getattr(value, "cx") if hasattr(value, "cx") else getattr(value, "c_x")
            cy = getattr(value, "cy") if hasattr(value, "cy") else getattr(value, "c_y")
            return CameraIntrinsics(
                fx=float(fx),
                fy=float(fy),
                cx=float(cx),
                cy=float(cy),
                width=None if getattr(value, "width", None) is None else int(getattr(value, "width")),
                height=None if getattr(value, "height", None) is None else int(getattr(value, "height")),
            )
        except Exception:
            pass
    return camera_intrinsics_from_mapping(value)


def _safe_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        return np.asarray(value)
    except Exception:
        return None


def _topomap_vertex_count(topo: Any) -> int:
    try:
        return int(topo.g.vcount())
    except Exception:
        return 0


@contextlib.contextmanager
def _suppress_original_stdout():
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        yield


def _snapshot_npz_path(value: Path | Mapping[str, Any]) -> Path:
    if isinstance(value, Mapping):
        paths = value.get("paths")
        if isinstance(paths, Mapping) and paths.get("npz") is not None:
            return Path(str(paths["npz"]))
        if value.get("npz") is not None:
            return Path(str(value["npz"]))
    return Path(value)


def _write_summary_json(
    path: Path,
    *,
    output_npz: Path,
    source_npz: Path,
    metadata: Mapping[str, Any],
    label_map: np.ndarray,
) -> None:
    labels = np.asarray(label_map, dtype=np.int32)
    payload = {
        "step": int(metadata.get("snapshot_step", metadata.get("step", -1))),
        "method": BASELINE_NAME,
        "output_npz": str(output_npz),
        "source_npz": str(source_npz),
        "metadata": dict(metadata),
        "shape": [int(v) for v in labels.shape],
        "counts": {
            "final_labeled": int(np.count_nonzero(labels > 0)),
            "positive_labels": int(len([v for v in np.unique(labels) if int(v) > 0])),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_head(path: Path) -> str | None:
    try:
        import subprocess

        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value
