#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from voxroom_online.isaac_runtime.config import get_nested, load_config
from voxroom_online.isaac_runtime.door_seed_learning.config import (
    DoorSeedLearningConfig,
)
from voxroom_online.isaac_runtime.door_seed_learning.hybrid_raw_seed import (
    VoxroomTvarsRawSeedAccumulator,
)
from voxroom_online.isaac_runtime.door_seed_learning.inference import (
    DoorSeedFilterOutput,
)
from voxroom_online.isaac_runtime.mapping.coordinate_transform import MapInfo
from voxroom_online.isaac_runtime.mapping.voxel_door_detector import (
    rebuild_voxel_door_seed_result,
)
from voxroom_online.isaac_runtime.mapping.voxel_occupancy_door_wall_roomseg import (
    VoxelOccupancyDoorWallRoomSegmenter,
)
from voxroom_online.isaac_runtime.scripts.replay_voxel_roomseg_snapshots import (
    _voxel_grid_from_snapshot,
)


VARIANTS = {
    "no_tvars_raw_seed",
    "no_voxel_raw_seed",
    "no_neural_filter",
    "saved_full_model_control",
    "nav_no_clearance_full",
    "vertical_2d_only",
    "vertical_3d_only",
    "vertical_full_retrain",
    "vertical_full_retrain_context_feature_shuffle",
    "all_vertical_free_cells",
}


class SavedUnionAccumulator:
    def __init__(self, config: DoorSeedLearningConfig) -> None:
        self.base = VoxroomTvarsRawSeedAccumulator(config)
        self.saved_mask: np.ndarray | None = None

    def combine(self, stage, **kwargs):
        combined = self.base.combine(stage, **kwargs)
        if self.saved_mask is None:
            raise RuntimeError("saved union raw-seed mask was not set")
        mask = np.asarray(self.saved_mask, dtype=bool)
        if mask.shape != tuple(combined.map_shape):
            raise ValueError("saved union raw-seed mask shape mismatch")
        return replace(
            combined,
            raw_seed_mask_xy=mask.copy(),
            debug={
                **dict(combined.debug),
                "ablation_candidate_mask_source": "saved_exact_voxel_door_raw_seed_mask",
                "ablation_candidate_cells": int(np.count_nonzero(mask)),
            },
        )


class SavedKeepFilter:
    def __init__(self) -> None:
        self.keep_mask: np.ndarray | None = None
        self.probability_xy: np.ndarray | None = None

    def filter_stage(self, *, stage, voxel_grid, seed_connectivity: int):
        del voxel_grid
        raw = np.asarray(stage.raw_seed_mask_xy, dtype=bool)
        if self.keep_mask is None:
            raise RuntimeError("saved keep mask was not set")
        keep = np.asarray(self.keep_mask, dtype=bool) & raw
        probability = (
            np.asarray(self.probability_xy, dtype=np.float32).copy()
            if self.probability_xy is not None
            else np.full(raw.shape, np.nan, dtype=np.float32)
        )
        result = rebuild_voxel_door_seed_result(
            stage.raw_seed_result,
            keep,
            seed_connectivity=int(seed_connectivity),
            model_probability_xy=probability,
            eligible_raw_seed_mask=raw,
        )
        result.debug.update(
            {
                "voxel_door_seed_model_fallback": False,
                "voxel_door_seed_model_source": "saved_exact_keep_mask",
                "voxel_door_seed_model_keep_mask": keep.copy(),
                "voxel_door_seed_model_reject_mask": raw & ~keep,
            }
        )
        return DoorSeedFilterOutput(
            seed_result=result,
            probability_xy=probability,
            keep_mask_xy=keep,
            reject_mask_xy=raw & ~keep,
            fallback_reason=None,
            latency_ms=0.0,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _approved(annotation_dir: Path, episode_uid: str) -> bool:
    path = annotation_dir / episode_uid / "last_step.annotation.json"
    if not path.is_file():
        return False
    raw = json.loads(path.read_text(encoding="utf-8"))
    review = raw.get("review", {})
    return isinstance(review, Mapping) and str(review.get("status")) == "approved"


def _scalar(arrays: Mapping[str, np.ndarray], key: str, default: float) -> float:
    if key not in arrays:
        return float(default)
    value = np.asarray(arrays[key]).reshape(-1)
    return float(value[0]) if value.size else float(default)


def _map_info(arrays: Mapping[str, np.ndarray], shape: tuple[int, int]) -> MapInfo:
    resolution = _scalar(arrays, "map_resolution_m", 0.05)
    min_x = _scalar(arrays, "map_min_x_m", 0.0)
    min_y = _scalar(arrays, "map_min_y_m", 0.0)
    return MapInfo(
        resolution_m=resolution,
        min_x=min_x,
        max_x=_scalar(arrays, "map_max_x_m", min_x + shape[1] * resolution),
        min_y=min_y,
        max_y=_scalar(arrays, "map_max_y_m", min_y + shape[0] * resolution),
        width=int(shape[1]),
        height=int(shape[0]),
    )


def _learning_mapping(args: argparse.Namespace) -> dict[str, Any]:
    exact_saved_union = args.variant in {
        "no_neural_filter",
        "saved_full_model_control",
        "nav_no_clearance_full",
        "vertical_2d_only",
        "vertical_3d_only",
        "vertical_full_retrain",
        "vertical_full_retrain_context_feature_shuffle",
    }
    rules_only = args.variant in {"no_neural_filter", "saved_full_model_control"}
    raw_source = {
        "no_tvars_raw_seed": "voxroom",
        "no_voxel_raw_seed": "tvars_vertical",
        "all_vertical_free_cells": "vertical_free_all",
    }.get(args.variant, "voxroom_tvars_vertical_union")
    context_source = "nav" if args.variant == "nav_no_clearance_full" else "vertical"
    mapping: dict[str, Any] = {
        "mode": "rules_only" if rules_only else "inference",
        "raw_seed_source": raw_source,
        "collection_every_steps": 5,
        "save_full_voxel_milestones": True,
        "persistent_final_raw_seed_union": True,
        "tvars_seed_width_cells": 3,
        "local_voxel_patch_size": 19,
        "context_patch_size": 41,
        "context_source": context_source,
        "checkpoint_path": str(Path(args.checkpoint).expanduser().resolve()) if args.checkpoint else "",
        "device": str(args.device),
        "inference_batch_size": int(args.inference_batch_size),
        "keep_threshold": 0.5,
        "fallback_to_rule_seed_on_error": False,
        "keep_uninformative_seed": False,
        "height_scale_m": 4.0,
        "ablation_allow_checkpoint_mismatch": True,
    }
    if mapping["mode"] == "inference" and not mapping["checkpoint_path"]:
        raise ValueError("variant %s requires --checkpoint" % args.variant)
    mapping["ablation_exact_saved_union"] = exact_saved_union
    return mapping


def _segmenter(
    *,
    roomseg_mapping: Mapping[str, Any],
    learning_mapping: Mapping[str, Any],
    map_info: MapInfo,
    variant: str,
) -> tuple[VoxelOccupancyDoorWallRoomSegmenter, SavedUnionAccumulator | None, SavedKeepFilter | None]:
    config = dict(roomseg_mapping)
    config["door_seed_learning"] = {
        key: value
        for key, value in learning_mapping.items()
        if key != "ablation_exact_saved_union"
    }
    config["min_room_area_m2"] = 0.5
    segmenter = VoxelOccupancyDoorWallRoomSegmenter(config=config, map_info=map_info)
    saved_union = None
    saved_keep = None
    if bool(learning_mapping["ablation_exact_saved_union"]):
        saved_union = SavedUnionAccumulator(segmenter.config.door_seed_learning)
        segmenter.door_seed_raw_seed_accumulator = saved_union
    elif variant == "no_tvars_raw_seed":
        # Preserve the source-specific history at the same sparse checkpoint cadence.
        segmenter.door_seed_raw_seed_accumulator = VoxroomTvarsRawSeedAccumulator(
            segmenter.config.door_seed_learning
        )
    if variant == "saved_full_model_control":
        saved_keep = SavedKeepFilter()
        segmenter.door_seed_inference_engine = saved_keep
    return segmenter, saved_union, saved_keep


def _install_context_feature_channel_shuffle(
    segmenter: VoxelOccupancyDoorWallRoomSegmenter,
    permutation: np.ndarray,
) -> None:
    """Shuffle the pooled 2D-context channels before the fusion classifier."""

    import torch

    engine = segmenter.door_seed_inference_engine
    model = getattr(engine, "model", None)
    if model is None:
        raise RuntimeError("context-feature shuffle requires a loaded inference model")
    config = model.model_config
    if not bool(config.use_voxel_branch) or not bool(config.use_context_branch):
        raise ValueError("context-feature shuffle requires both model branches")
    order = np.asarray(permutation, dtype=np.int64)
    if order.shape != (224,) or not np.array_equal(np.sort(order), np.arange(224)):
        raise ValueError("context-feature shuffle must be a permutation of 224 channels")
    order_cpu = torch.from_numpy(order.copy())

    def shuffle_context(_module, inputs):
        if len(inputs) != 1:
            raise RuntimeError("fusion classifier expected one input tensor")
        fused = inputs[0]
        if fused.ndim != 2 or int(fused.shape[1]) != 448:
            raise RuntimeError("full-model fusion feature must have shape [B,448]")
        order_device = order_cpu.to(device=fused.device)
        return (
            torch.cat(
                (fused[:, :224], fused[:, 224:].index_select(1, order_device)),
                dim=1,
            ),
        )

    model.classifier.register_forward_pre_hook(shuffle_context)


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]).copy() for key in data.files}


def _write_prediction(
    path: Path,
    *,
    arrays: Mapping[str, np.ndarray],
    labels: np.ndarray,
    result,
    variant: str,
    input_path: Path,
    candidate_policy: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    debug = dict(result.debug)
    raw_mask = np.asarray(
        debug.get("voxel_door_raw_seed_mask", np.zeros(labels.shape, dtype=bool)),
        dtype=bool,
    )
    keep_mask = np.asarray(
        debug.get("voxel_door_seed_model_keep_mask", debug.get("voxel_door_seed_mask", raw_mask)),
        dtype=bool,
    )
    payload: dict[str, np.ndarray] = {
        "final_room_label_map": np.asarray(labels, dtype=np.int32),
        "voxel_final_room_label_map": np.asarray(labels, dtype=np.int32),
        "voxel_final_separator_map": np.asarray(result.separator_map, dtype=bool),
        "voxel_door_raw_seed_mask": raw_mask,
        "voxel_door_seed_model_keep_mask": keep_mask,
        "ablation_variant": np.asarray(str(variant)),
        "ablation_candidate_policy": np.asarray(str(candidate_policy)),
        "ablation_input_snapshot": np.asarray(str(input_path)),
    }
    for key in (
        "roomseg_eval_explored_reference_mask",
        "roomseg_eval_reference_explorable_mask",
        "roomseg_eval_coverage_ratio",
        "roomseg_eval_event_id",
        "roomseg_eval_event_kind",
        "roomseg_eval_threshold",
        "map_resolution_m",
    ):
        if key in arrays:
            payload[key] = np.asarray(arrays[key])
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay one VoxRoom DoorSeed ablation on an approved snapshot index.")
    parser.add_argument("--index", required=True)
    parser.add_argument("--annotation-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--config", default="configs/voxroom_online.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--inference-batch-size", type=int, default=256)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--max-snapshots-per-episode", type=int, default=0)
    parser.add_argument("--feature-shuffle-seed", type=int, default=0)
    args = parser.parse_args()

    index_path = Path(args.index).expanduser().resolve()
    annotation_dir = Path(args.annotation_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    episodes = [
        dict(episode)
        for episode in index.get("episodes", [])
        if _approved(annotation_dir, str(episode["episode_uid"]))
    ]
    if int(args.max_episodes) > 0:
        episodes = episodes[: int(args.max_episodes)]
    if int(args.max_snapshots_per_episode) > 0:
        for episode in episodes:
            episode["snapshots"] = list(episode.get("snapshots", []))[
                : int(args.max_snapshots_per_episode)
            ]
    if not episodes:
        raise ValueError("no approved episodes selected")
    config_path = Path(args.config).expanduser().resolve()
    root_config = load_config(config_path)
    roomseg_mapping = dict(get_nested(root_config, "mapping.room_segmentation", {}) or {})
    voxel_grid_mapping = dict(roomseg_mapping.get("voxel_grid", {}) or {})
    learning_mapping = _learning_mapping(args)
    context_feature_permutation = None
    context_feature_shuffle = None
    if args.variant == "vertical_full_retrain_context_feature_shuffle":
        context_feature_permutation = np.random.default_rng(
            int(args.feature_shuffle_seed)
        ).permutation(224).astype(np.int64)
        context_feature_shuffle = {
            "enabled": True,
            "mode": "fixed_channel_permutation_after_global_pool_before_classifier",
            "seed": int(args.feature_shuffle_seed),
            "feature_offset": 224,
            "feature_count": 224,
            "permutation_sha256": hashlib.sha256(
                context_feature_permutation.tobytes()
            ).hexdigest(),
            "permutation": context_feature_permutation.tolist(),
        }
    manifest: dict[str, Any] = {
        "schema_version": "voxroom_door_seed_ablation_replay_v1",
        "variant": str(args.variant),
        "candidate_policy": (
            "saved_exact_union"
            if bool(learning_mapping["ablation_exact_saved_union"])
            else "vertical_free_all"
            if args.variant == "all_vertical_free_cells"
            else "source_specific_history_at_saved_checkpoint_cadence"
        ),
        "source_history_limitation": (
            "evaluation snapshots omit the every-decision source-id histories; no_tvars/no_voxel are reconstructed at saved checkpoint cadence"
            if args.variant in {"no_tvars_raw_seed", "no_voxel_raw_seed"}
            else None
        ),
        "index": str(index_path),
        "index_sha256": _sha256(index_path),
        "annotation_dir": str(annotation_dir),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()) if args.checkpoint else None,
        "checkpoint_sha256": _sha256(Path(args.checkpoint).expanduser().resolve()) if args.checkpoint else None,
        "learning_config": learning_mapping,
        "context_feature_shuffle": context_feature_shuffle,
        "episode_count": len(episodes),
        "snapshot_count": sum(len(episode.get("snapshots", [])) for episode in episodes),
        "processed": 0,
        "started_at_unix": time.time(),
        "episodes": [],
    }
    candidate_policy = str(manifest["candidate_policy"])
    for episode_number, episode in enumerate(episodes, start=1):
        uid = str(episode["episode_uid"])
        snapshots = list(episode.get("snapshots", []))
        if not snapshots:
            continue
        first_arrays = _load_arrays(Path(str(snapshots[0]["snapshot_path"])))
        shape = tuple(np.asarray(first_arrays["final_room_label_map"]).shape)
        map_info = _map_info(first_arrays, (int(shape[0]), int(shape[1])))
        segmenter, saved_union, saved_keep = _segmenter(
            roomseg_mapping=roomseg_mapping,
            learning_mapping=learning_mapping,
            map_info=map_info,
            variant=str(args.variant),
        )
        if context_feature_permutation is not None:
            _install_context_feature_channel_shuffle(
                segmenter,
                context_feature_permutation,
            )
        episode_record = {
            "episode_uid": uid,
            "scene_id": str(episode.get("scene_id")),
            "snapshots": [],
        }
        for snapshot_number, snapshot in enumerate(snapshots, start=1):
            started = time.perf_counter()
            input_path = Path(str(snapshot["snapshot_path"])).expanduser().resolve()
            arrays = first_arrays if snapshot_number == 1 else _load_arrays(input_path)
            current_shape = tuple(np.asarray(arrays["final_room_label_map"]).shape)
            if current_shape != shape:
                raise ValueError("map shape changed within episode: %s" % uid)
            if saved_union is not None:
                saved_union.saved_mask = np.asarray(arrays["voxel_door_raw_seed_mask"], dtype=bool)
            if saved_keep is not None:
                saved_keep.keep_mask = np.asarray(arrays["voxel_door_seed_model_keep_mask"], dtype=bool)
                saved_keep.probability_xy = np.asarray(
                    arrays.get(
                        "voxel_door_seed_model_probability_xy",
                        np.full(shape, np.nan, dtype=np.float32),
                    ),
                    dtype=np.float32,
                )
            voxel_grid = _voxel_grid_from_snapshot(
                arrays,
                map_info,
                voxel_grid_mapping,
                recompute_state_from_logodds=False,
            )
            pose = np.asarray(
                arrays.get("demo_pose_world", arrays.get("demo_camera_pose_world", np.zeros(4))),
                dtype=np.float64,
            ).reshape(-1)
            yaw_deg = float(np.degrees(pose[3])) if pose.size >= 4 else 0.0
            agent = np.asarray(arrays.get("agent_rc", np.zeros(2)), dtype=np.int32).reshape(-1)
            segmenter.update(
                occupancy_map=np.asarray(arrays["occupancy_map"], dtype=bool),
                observed_free_mask=np.asarray(arrays["observed_free_mask"], dtype=bool),
                obstacle_mask=np.asarray(arrays["obstacle_mask"], dtype=bool),
                unknown_mask=np.asarray(arrays["unknown_mask"], dtype=bool),
                voxel_grid=voxel_grid,
                step=int(snapshot["step"]),
                navigation_free_mask=np.asarray(arrays["voxel_nav_free_xy"], dtype=bool),
                navigation_obstacle_mask=np.asarray(arrays["voxel_nav_occupied_xy"], dtype=bool),
                door_seed_no_clearance_free_mask=np.asarray(arrays["voxel_nav_free_xy"], dtype=bool),
                agent_rc=(int(agent[0]), int(agent[1])),
                agent_yaw_deg=yaw_deg,
            )
            result = segmenter.last_result
            if result is None:
                raise RuntimeError("segmenter returned no result")
            event_id = str(snapshot["coverage_event_id"])
            prediction_path = out_dir / "predictions" / uid / (event_id + ".npz")
            _write_prediction(
                prediction_path,
                arrays=arrays,
                labels=np.asarray(result.room_label_map, dtype=np.int32),
                result=result,
                variant=str(args.variant),
                input_path=input_path,
                candidate_policy=candidate_policy,
            )
            record = {
                **dict(snapshot),
                "source_snapshot_path": str(input_path),
                "snapshot_path": str(prediction_path),
                "runtime_seconds": time.perf_counter() - started,
                "raw_seed_cells": int(
                    np.count_nonzero(
                        result.debug.get("voxel_door_raw_seed_mask", False)
                    )
                ),
                "kept_seed_cells": int(
                    np.count_nonzero(
                        result.debug.get("voxel_door_seed_model_keep_mask", result.debug.get("voxel_door_seed_mask", False))
                    )
                ),
            }
            episode_record["snapshots"].append(record)
            manifest["processed"] += 1
            print(
                "[ablation-replay] variant=%s episode=%d/%d snapshot=%d/%d uid=%s event=%s rooms=%d seconds=%.3f"
                % (
                    args.variant,
                    episode_number,
                    len(episodes),
                    snapshot_number,
                    len(snapshots),
                    uid,
                    event_id,
                    int(np.max(result.room_label_map, initial=0)),
                    float(record["runtime_seconds"]),
                ),
                flush=True,
            )
        manifest["episodes"].append(episode_record)
        first_arrays = {}
        temp_manifest = out_dir / "prediction_index.json.tmp"
        temp_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temp_manifest, out_dir / "prediction_index.json")
    manifest["finished_at_unix"] = time.time()
    manifest["elapsed_seconds"] = manifest["finished_at_unix"] - manifest["started_at_unix"]
    manifest["complete"] = manifest["processed"] == manifest["snapshot_count"]
    (out_dir / "prediction_index.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: manifest[key] for key in ("variant", "episode_count", "snapshot_count", "processed", "complete", "elapsed_seconds")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
