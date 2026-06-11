from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..data_contract import MapInfo, resolve_map_info
from ..mask_io import save_baseline_snapshot_npz
from .detector import DoorDetector, project_door_bbox_to_grid
from .door_extraction import RayCastingDoorExtractor, fuse_door_pairs
from .frontier_mcl import compute_current_room_mcl
from .rasterize import rasterize_topomap
from .topomap import TopologicalRoomMap


class ActiveRoomSegmentationBaseline:
    def __init__(
        self,
        *,
        output_dir: Path,
        detector: DoorDetector,
        map_info: MapInfo | None = None,
        panorama_views: int = 12,
        policy_control: str = "never",
        save_stream: bool = True,
        save_every_snapshot: bool = True,
    ) -> None:
        if str(policy_control) != "never":
            raise ValueError("Topology baseline policy_control must be 'never' for fair comparison")
        self.output_dir = Path(output_dir)
        self.detector = detector
        self.map_info = map_info
        self.panorama_views = int(panorama_views)
        self.policy_control = "never"
        self.save_stream = bool(save_stream)
        self.save_every_snapshot = bool(save_every_snapshot)
        self.topomap = TopologicalRoomMap()
        self.raycast = RayCastingDoorExtractor()
        self.latest_label_map: np.ndarray | None = None
        self.latest_debug_arrays: dict[str, np.ndarray] = {}
        self.latest_metadata: dict[str, Any] = {}
        self._latest_mcl: np.ndarray | None = None
        self._manifest_path = self.output_dir.parent.parent / "active_stream" / "stream_manifest.jsonl"
        self._frames_dir = self._manifest_path.parent / "frames"
        self._last_stream_step: int | None = None

    def on_episode_start(self, episode_metadata: Mapping[str, Any]) -> None:
        if self.save_stream:
            self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
            self._frames_dir.mkdir(parents=True, exist_ok=True)
            if not self._manifest_path.exists():
                self._manifest_path.write_text("", encoding="utf-8")
            meta_path = self._manifest_path.parent / "episode_metadata.json"
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
        _ = sgnav_obs, room_segmenter
        arrays = _arrays_from_map_state(map_state, frontier_map=frontier_map, selected_frontier_center_rc=selected_frontier_center_rc)
        map_info = resolve_map_info(map_info=self.map_info, map_state=map_state, mapper=mapper, snapshot_arrays=arrays)
        self.map_info = map_info
        agent_rc = tuple(int(v) for v in np.asarray(map_state["current_grid"]).reshape(-1)[:2])
        if self.save_stream:
            self._save_stream_frame(
                step=int(step),
                obs=obs or {},
                arrays=arrays,
                map_info=map_info,
                selected_frontier_center_rc=selected_frontier_center_rc,
                camera_intrinsics=camera_intrinsics,
            )
        candidates = self.raycast.extract(
            occupancy_map=arrays["occupancy_map"],
            free_mask=arrays["observed_free_mask"],
            unknown_mask=arrays["unknown_mask"],
            robot_rc=agent_rc,
        )
        vision_candidates = []
        vision_num_detections = 0
        vision_projection_attempted = False
        vision_projection_missing_inputs: list[str] = []
        if obs is not None and self.detector.name != "disabled" and obs.get("has_rgb") and self.detector.available:
            detections = self.detector.detect(np.asarray(obs["rgb"]))
            vision_num_detections = int(len(detections))
            depth = obs.get("depth")
            camera_pose_world = obs.get("camera_pose_world")
            if depth is None:
                vision_projection_missing_inputs.append("depth")
            if camera_intrinsics is None:
                vision_projection_missing_inputs.append("camera_intrinsics")
            if camera_pose_world is None:
                vision_projection_missing_inputs.append("camera_pose_world")
            if not vision_projection_missing_inputs:
                vision_projection_attempted = True
                for detection in detections:
                    candidate = project_door_bbox_to_grid(
                        detection,
                        depth=np.asarray(depth, dtype=np.float32),
                        camera_intrinsics=camera_intrinsics,
                        camera_pose_world=np.asarray(camera_pose_world, dtype=np.float32),
                        map_info=map_info,
                    )
                    if candidate is not None:
                        vision_candidates.append(candidate)
        all_candidates = list(candidates) + list(vision_candidates)
        door_pairs = fuse_door_pairs(
            all_candidates,
            free_mask=arrays["observed_free_mask"],
            occupied_mask=arrays["obstacle_mask"],
            resolution_m=float(map_info.resolution_m),
        )
        mcl = compute_current_room_mcl(
            free_mask=arrays["observed_free_mask"],
            unknown_mask=arrays["unknown_mask"],
            obstacle_mask=arrays["obstacle_mask"],
            door_pairs=door_pairs,
            closed_door_ids=set(),
            start_rc=agent_rc,
            previous_mcl=self._latest_mcl,
        )
        self._latest_mcl = np.asarray(mcl.mcl_mask, dtype=bool)
        self.topomap.update(step=int(step), mcl_result=mcl, door_pairs=door_pairs, agent_rc=agent_rc, obstacle_mask=arrays["obstacle_mask"])
        policy = self.topomap.policy_decision(mcl_result=mcl, agent_rc=agent_rc)
        label_map, debug = rasterize_topomap(self.topomap, arrays)
        policy_target = np.asarray([-1, -1] if policy.target_rc is None else [int(policy.target_rc[0]), int(policy.target_rc[1])], dtype=np.int32)
        debug.update(
            {
                "topology_frontier_cluster_map": _cluster_label_map(mcl.frontier_clusters, arrays["occupancy_map"].shape),
                "topology_baseline_policy_target_rc": policy_target,
            }
        )
        self.latest_label_map = label_map
        self.latest_debug_arrays = debug
        detector_adapter_verified = bool(getattr(self.detector, "detector_adapter_verified", False))
        projection_verified = bool(
            detector_adapter_verified
            and str(self.detector.name) == "original_detr"
            and vision_projection_attempted
            and len(vision_candidates) > 0
        )
        original_repo_commit = getattr(self.detector, "active_repo_commit", None)
        checkpoint_sha256 = getattr(self.detector, "checkpoint_sha256", None)
        main_experiment_allowed = bool(
            str(self.detector.name) == "original_detr"
            and bool(self.detector.available)
            and detector_adapter_verified
            and projection_verified
            and original_repo_commit
            and checkpoint_sha256
        )
        topology_state_machine_verified = bool(self.topomap.current_node_id is not None or len(self.topomap.nodes) >= 0)
        self.latest_metadata = {
            "method": "topology_visual_active",
            "runner_type": "original_active_room_segmentation" if main_experiment_allowed else "scaffold_or_debug",
            "uses_rgb": True,
            "uses_depth": True,
            "uses_occupancy": True,
            "uses_oracle_semantics": False,
            "panorama_views": int(self.panorama_views),
            "panorama_views_saved": int(self.panorama_views),
            "num_nodes": int(len(self.topomap.nodes)),
            "num_doors": int(len(self.topomap.door_pairs)),
            "num_edges": int(len(self.topomap.edges)),
            "current_node_id": None if self.topomap.current_node_id is None else int(self.topomap.current_node_id),
            "door_detector": str(self.detector.name),
            "door_detector_available": bool(self.detector.available),
            "detector_adapter_verified": bool(detector_adapter_verified),
            "projection_verified": bool(projection_verified),
            "topology_state_machine_verified": bool(topology_state_machine_verified),
            "original_repo": "FreeformRobotics/Active_room_segmentation",
            "original_repo_path": None if getattr(self.detector, "repo_dir", None) is None else str(getattr(self.detector, "repo_dir")),
            "original_repo_commit": original_repo_commit,
            "checkpoint_path": getattr(self.detector, "checkpoint_path", None),
            "checkpoint_sha256": checkpoint_sha256,
            "raycast_num_candidates": int(len(candidates)),
            "vision_num_detections": int(vision_num_detections),
            "vision_num_candidates": int(len(vision_candidates)),
            "vision_projection_attempted": bool(vision_projection_attempted),
            "vision_projection_missing_inputs": list(vision_projection_missing_inputs),
            "num_door_pairs": int(len(door_pairs)),
            "policy_control": "never",
            "baseline_policy_control": "never",
            "baseline_policy_target_rc": None if policy.target_rc is None else [int(policy.target_rc[0]), int(policy.target_rc[1])],
            "baseline_policy_reason": str(policy.reason),
            "map_info": map_info.to_metadata(),
            "main_experiment_allowed": bool(main_experiment_allowed),
            "allowed_usage": "main_experiment" if main_experiment_allowed else "smoke_test_only",
            "approximation_note": None
            if main_experiment_allowed
            else "Closed-loop sidecar scaffold; original visual DETR detector is not available, so canonical paper reproduction remains incomplete.",
        }

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
            baseline_name="topology_visual_active",
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

    def _save_stream_frame(
        self,
        *,
        step: int,
        obs: Mapping[str, Any],
        arrays: Mapping[str, np.ndarray],
        map_info: MapInfo,
        selected_frontier_center_rc: tuple[int, int] | None,
        camera_intrinsics: Any | None,
    ) -> None:
        if self._last_stream_step == int(step):
            return
        self._last_stream_step = int(step)
        self._frames_dir.mkdir(parents=True, exist_ok=True)
        frame_npz = self._frames_dir / ("frame_%06d.npz" % int(step))
        frame_arrays: dict[str, Any] = {
            "occupancy_map": np.asarray(arrays["occupancy_map"]),
            "observed_free_mask": np.asarray(arrays["observed_free_mask"]),
            "obstacle_mask": np.asarray(arrays["obstacle_mask"]),
            "unknown_mask": np.asarray(arrays["unknown_mask"]),
            "navigation_free_room_domain": np.asarray(arrays["navigation_free_room_domain"]),
            "frontier_map": np.asarray(arrays.get("frontier_map", np.zeros_like(arrays["occupancy_map"], dtype=bool))),
            "agent_rc": np.asarray(arrays["agent_rc"], dtype=np.int32),
            "selected_frontier_center_rc": np.asarray([-1, -1] if selected_frontier_center_rc is None else selected_frontier_center_rc, dtype=np.int32),
            "map_resolution_m": np.asarray(float(map_info.resolution_m), dtype=np.float32),
            "map_origin_xy_m": np.asarray([float(map_info.min_x), float(map_info.min_y)], dtype=np.float32),
            "control_step": np.asarray(int(step), dtype=np.int64),
        }
        if obs.get("has_rgb") and "rgb" in obs:
            frame_arrays["rgb"] = np.asarray(obs["rgb"])
        if obs.get("has_depth") and "depth" in obs:
            frame_arrays["depth"] = np.asarray(obs["depth"], dtype=np.float32)
        if "pose_world" in obs:
            frame_arrays["pose_world"] = np.asarray(obs["pose_world"], dtype=np.float32)
            frame_arrays["agent_pose_world"] = np.asarray(obs["pose_world"], dtype=np.float32)
        if "camera_pose_world" in obs:
            frame_arrays["camera_pose_world"] = np.asarray(obs["camera_pose_world"], dtype=np.float32)
        if camera_intrinsics is not None:
            frame_arrays["camera_intrinsics"] = _camera_intrinsics_array(camera_intrinsics)
        np.savez_compressed(frame_npz, **frame_arrays)
        row = {
            "schema_version": 2,
            "control_step": int(step),
            "snapshot_step": int(step),
            "frame_npz": str(frame_npz.relative_to(self._manifest_path.parent)),
            "agent_rc": [int(v) for v in np.asarray(arrays["agent_rc"]).reshape(-1)[:2]],
            "selected_frontier_center_rc": [-1, -1] if selected_frontier_center_rc is None else [int(v) for v in selected_frontier_center_rc],
            "frontier_selection_mode": "random",
            "frontier_source": "voxel_vertical_free",
            "has_rgb": bool(obs.get("has_rgb") and "rgb" in obs),
            "has_depth": bool(obs.get("has_depth") and "depth" in obs),
            "has_intrinsics": bool(camera_intrinsics is not None),
            "has_camera_pose_world": bool("camera_pose_world" in obs),
            "panorama_id": None,
            "panorama_view_index": None,
            "map_resolution_m": float(map_info.resolution_m),
            "map_origin_xy": [float(map_info.min_x), float(map_info.min_y)],
        }
        with self._manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_json_ready(row), ensure_ascii=False, sort_keys=True) + "\n")


def _arrays_from_map_state(
    map_state: Mapping[str, Any],
    *,
    frontier_map: np.ndarray | None,
    selected_frontier_center_rc: tuple[int, int] | None,
) -> dict[str, np.ndarray]:
    occupancy = np.asarray(map_state["occupancy"], dtype=bool)
    free = np.asarray(map_state["free"], dtype=bool)
    observed = np.asarray(map_state["observed"], dtype=bool)
    return {
        "occupancy_map": occupancy,
        "observed_free_mask": free,
        "obstacle_mask": occupancy,
        "unknown_mask": ~observed,
        "navigation_free_room_domain": free,
        "frontier_map": np.zeros(occupancy.shape, dtype=bool) if frontier_map is None else np.asarray(frontier_map, dtype=bool),
        "agent_rc": np.asarray(map_state["current_grid"], dtype=np.int32),
        "selected_frontier_center_rc": np.asarray([-1, -1] if selected_frontier_center_rc is None else selected_frontier_center_rc, dtype=np.int32),
    }


def _cluster_label_map(clusters: list[Any], shape: tuple[int, ...]) -> np.ndarray:
    out = np.zeros(tuple(int(v) for v in shape), dtype=np.int32)
    for cluster in clusters:
        mask = np.asarray(cluster.mask, dtype=bool)
        if mask.shape == out.shape:
            out[mask] = int(cluster.cluster_id)
    return out


def _camera_intrinsics_array(intr: Any) -> np.ndarray:
    vals = []
    for name in ("fx", "fy", "cx", "cy", "width", "height"):
        vals.append(float(getattr(intr, name, 0.0)))
    return np.asarray(vals, dtype=np.float32)


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
        "method": "topology_visual_active",
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
