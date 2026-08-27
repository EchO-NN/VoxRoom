from __future__ import annotations

import inspect
import time
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import numpy as np

from voxroom_online.isaac_runtime.door_seed_learning.config import DoorSeedLearningConfig
from voxroom_online.isaac_runtime.door_seed_learning.model import (
    MODEL_ARCHITECTURE_VERSION,
    PREPROCESSOR_VERSION,
    DoorSeedModelConfig,
    build_door_seed_model,
)
from voxroom_online.isaac_runtime.door_seed_learning.patch_extraction import (
    class_patches_to_one_hot,
    extract_class_patches,
    extract_local_voxel_patches,
    observed_ratio,
    sorted_seed_coordinates,
    voxel_states_to_model_channels,
)
from voxroom_online.isaac_runtime.door_seed_learning.schema import (
    DATASET_SCHEMA_VERSION,
    scalar_value,
    sha256_bytes,
    source_tree_hash,
)
from voxroom_online.isaac_runtime.door_seed_learning.stage_extractor import DoorSeedStageResult
from voxroom_online.isaac_runtime.mapping.voxel_door_detector import VoxelDoorSeedResult, rebuild_voxel_door_seed_result
from voxroom_online.isaac_runtime.mapping.voxel_occupancy_grid import VoxelOccupancyGrid3D


@dataclass
class DoorSeedFilterOutput:
    seed_result: VoxelDoorSeedResult
    probability_xy: np.ndarray
    keep_mask_xy: np.ndarray
    reject_mask_xy: np.ndarray
    fallback_reason: str | None
    latency_ms: float


class DoorSeedInferenceEngine:
    def __init__(self, config: DoorSeedLearningConfig | Mapping[str, object]):
        self.config = config if isinstance(config, DoorSeedLearningConfig) else DoorSeedLearningConfig.from_mapping(config)
        if self.config.mode != "inference":
            raise ValueError("DoorSeedInferenceEngine requires inference mode")
        self.model = None
        self.checkpoint: dict[str, object] | None = None
        self.device = None
        self.load_error: str | None = None
        self.fallback_count = 0
        self.inference_count = 0
        try:
            self._load()
        except Exception as exc:
            self.load_error = "%s: %s" % (type(exc).__name__, exc)
            if not bool(self.config.fallback_to_rule_seed_on_error):
                raise
            warnings.warn("door seed learned filter unavailable; rules-only fallback: %s" % self.load_error, RuntimeWarning)

    def filter_stage(
        self,
        *,
        stage: DoorSeedStageResult,
        voxel_grid: VoxelOccupancyGrid3D,
        seed_connectivity: int,
    ) -> DoorSeedFilterOutput:
        started_at = time.perf_counter()
        raw_result = stage.raw_seed_result
        raw_mask = np.asarray(stage.raw_seed_mask_xy, dtype=bool)
        probability_xy = np.full(raw_mask.shape, np.nan, dtype=np.float32)
        if self.load_error is not None:
            return self._fallback(raw_result, raw_mask, probability_xy, self.load_error, started_at, seed_connectivity)
        try:
            rc = sorted_seed_coordinates(stage.raw_seed_mask_xy)
            if len(rc) == 0:
                result = rebuild_voxel_door_seed_result(
                    raw_result,
                    raw_mask,
                    seed_connectivity=int(seed_connectivity),
                    eligible_raw_seed_mask=raw_mask,
                )
                result = replace(result, debug={
                    **dict(result.debug),
                    "voxel_door_raw_seed_mask": np.asarray(stage.raw_seed_mask_xy, dtype=bool).copy(),
                    "voxel_door_seed_model_probability_xy": probability_xy,
                    "voxel_door_seed_model_keep_mask": np.asarray(stage.raw_seed_mask_xy, dtype=bool).copy(),
                    "voxel_door_seed_model_reject_mask": np.zeros(raw_mask.shape, dtype=bool),
                    "voxel_door_seed_model_fallback": False,
                    "voxel_door_seed_model_latency_ms": 0.0,
                })
                return DoorSeedFilterOutput(result, probability_xy, np.asarray(stage.raw_seed_mask_xy, dtype=bool), np.zeros(raw_mask.shape, dtype=bool), None, 0.0)
            voxel_patches, _valid = extract_local_voxel_patches(
                voxel_grid.state,
                rc,
                patch_size=int(self.config.local_voxel_patch_size),
            )
            context_map = stage.vertical_class_map_xy if self.config.context_source == "vertical" else stage.nav_class_map_xy
            self._validate_runtime_metadata(
                z_centers_m=np.asarray(voxel_grid.z_centers_m, dtype=np.float32),
                z_min_m=float(voxel_grid.z_min_m),
                z_resolution_m=float(voxel_grid.z_resolution_m),
                xy_resolution_m=float(voxel_grid.map_info.resolution_m),
                raw_seed_config_hash=str(stage.raw_seed_config_hash),
                input_semantics_hash=str(stage.input_semantics_hash),
            )
            probabilities = self.predict_arrays(
                voxel_state_nzyx=voxel_patches,
                z_centers_m=np.asarray(voxel_grid.z_centers_m, dtype=np.float32),
                class_map_xy=np.asarray(context_map, dtype=np.uint8),
                seed_rc=rc,
            )
            threshold = self.keep_threshold
            keep = probabilities >= threshold
            if bool(self.config.keep_uninformative_seed):
                context_patches = extract_class_patches(context_map, rc, patch_size=int(self.config.context_patch_size))
                uninformative = np.asarray(
                    [
                        observed_ratio(voxel_patches[index], context_patches[index])
                        < float(self.config.uninformative_observed_ratio_min)
                        for index in range(len(rc))
                    ],
                    dtype=bool,
                )
                keep |= uninformative
            filtered = raw_mask.copy()
            for index, (row, col) in enumerate(rc):
                filtered[int(row), int(col)] = bool(keep[index])
                probability_xy[int(row), int(col)] = float(probabilities[index])
            result = rebuild_voxel_door_seed_result(
                raw_result,
                filtered,
                seed_connectivity=int(seed_connectivity),
                model_probability_xy=probability_xy,
                eligible_raw_seed_mask=raw_mask,
            )
            latency_ms = float((time.perf_counter() - started_at) * 1000.0)
            result.debug.update(
                {
                    "voxel_door_seed_model_fallback": False,
                    "voxel_door_seed_model_fallback_reason": None,
                    "voxel_door_seed_model_keep_threshold": float(threshold),
                    "voxel_door_seed_model_latency_ms": latency_ms,
                    "voxel_door_seed_model_context_source": str(self.config.context_source),
                    "voxel_door_seed_model_checkpoint": str(self.config.checkpoint_path),
                    "voxel_door_seed_model_inference_count": int(self.inference_count + 1),
                    "voxel_door_seed_model_fallback_count": int(self.fallback_count),
                }
            )
            self.inference_count += 1
            return DoorSeedFilterOutput(
                seed_result=result,
                probability_xy=probability_xy,
                keep_mask_xy=filtered,
                reject_mask_xy=raw_mask & ~filtered,
                fallback_reason=None,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            if not bool(self.config.fallback_to_rule_seed_on_error):
                raise
            reason = "%s: %s" % (type(exc).__name__, exc)
            try:
                import torch

                if "out of memory" in reason.lower() and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            warnings.warn("door seed learned filter failed; rules-only fallback: %s" % reason, RuntimeWarning)
            return self._fallback(raw_result, raw_mask, probability_xy, reason, started_at, seed_connectivity)

    def predict_arrays(
        self,
        *,
        voxel_state_nzyx: np.ndarray,
        z_centers_m: np.ndarray,
        class_map_xy: np.ndarray,
        seed_rc: np.ndarray,
    ) -> np.ndarray:
        import torch

        if self.model is None or self.device is None:
            raise RuntimeError("door seed model is not loaded")
        voxel = np.asarray(voxel_state_nzyx, dtype=np.uint8)
        rc = sorted_seed_coordinates(seed_rc)
        if voxel.shape[0] != len(rc):
            raise ValueError("voxel patch count does not match raw seed coordinates")
        context = extract_class_patches(class_map_xy, rc, patch_size=int(self.config.context_patch_size))
        output = np.empty((len(rc),), dtype=np.float32)
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(rc), int(self.config.inference_batch_size)):
                end = min(len(rc), start + int(self.config.inference_batch_size))
                voxel_channels = voxel_states_to_model_channels(
                    voxel[start:end],
                    z_centers_m,
                    height_scale_m=float(self.config.height_scale_m),
                )
                context_channels = class_patches_to_one_hot(context[start:end])
                logits = self.model(
                    torch.from_numpy(voxel_channels).to(self.device, non_blocking=True),
                    torch.from_numpy(context_channels).to(self.device, non_blocking=True),
                )
                if logits.shape != (end - start, 1):
                    raise RuntimeError("door seed model must output [B,1]")
                if not torch.all(torch.isfinite(logits)):
                    raise RuntimeError("door seed model produced NaN or Inf logits")
                output[start:end] = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        if not np.all(np.isfinite(output)):
            raise RuntimeError("door seed model produced non-finite probabilities")
        return output

    @property
    def keep_threshold(self) -> float:
        if self.config.keep_threshold is not None:
            return float(self.config.keep_threshold)
        if self.checkpoint is None or "recommended_keep_threshold" not in self.checkpoint:
            raise ValueError("checkpoint has no recommended_keep_threshold")
        threshold = float(self.checkpoint["recommended_keep_threshold"])
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("checkpoint recommended_keep_threshold is outside [0,1]")
        return threshold

    def _load(self) -> None:
        import torch

        checkpoint_path = Path(self.config.checkpoint_path).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError("door seed checkpoint not found: %s" % checkpoint_path)
        load_kwargs = {"map_location": "cpu"}
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_kwargs["weights_only"] = True
        checkpoint = torch.load(checkpoint_path, **load_kwargs)
        if not isinstance(checkpoint, dict):
            raise ValueError("door seed checkpoint must be a mapping")
        required = {
            "model_state_dict",
            "model_config",
            "model_architecture_version",
            "preprocessor_version",
            "context_source",
            "local_patch_size",
            "context_patch_size",
            "z_count",
            "z_min_m",
            "z_resolution_m",
            "xy_resolution_m",
            "z_centers_sha256",
            "raw_seed_config_hash",
            "input_semantics_hash",
            "height_scale_m",
            "voxel_state_mapping",
            "recommended_keep_threshold",
            "dataset_schema_version",
            "source_code_hash",
            "training_scene_ids",
            "validation_scene_ids",
            "achieved_recall",
            "negative_rejection_rate",
        }
        missing = sorted(required - set(checkpoint))
        if missing:
            raise ValueError("door seed checkpoint missing metadata: %s" % ", ".join(missing))
        if checkpoint["model_architecture_version"] != MODEL_ARCHITECTURE_VERSION:
            raise ValueError("door seed checkpoint model architecture mismatch")
        if checkpoint["preprocessor_version"] != PREPROCESSOR_VERSION:
            raise ValueError("door seed checkpoint preprocessor mismatch")
        if checkpoint["dataset_schema_version"] != DATASET_SCHEMA_VERSION:
            raise ValueError("door seed checkpoint dataset schema mismatch")
        if str(checkpoint["context_source"]) != str(self.config.context_source):
            raise ValueError("door seed checkpoint context_source mismatch")
        if int(checkpoint["local_patch_size"]) != int(self.config.local_voxel_patch_size):
            raise ValueError("door seed checkpoint local patch size mismatch")
        if int(checkpoint["context_patch_size"]) != int(self.config.context_patch_size):
            raise ValueError("door seed checkpoint context patch size mismatch")
        if abs(float(checkpoint["height_scale_m"]) - float(self.config.height_scale_m)) > 1e-6:
            raise ValueError("door seed checkpoint height_scale_m mismatch")
        if checkpoint["voxel_state_mapping"] != {"unknown": [0, 3], "free": [1], "occupied": [2]}:
            raise ValueError("door seed checkpoint voxel state mapping mismatch")
        model_cfg = DoorSeedModelConfig.from_mapping(checkpoint["model_config"])
        if int(model_cfg.z_count) != int(checkpoint["z_count"]):
            raise ValueError("door seed checkpoint model z_count mismatch")
        training_scenes = _validated_scene_ids(checkpoint["training_scene_ids"], "training_scene_ids")
        validation_scenes = _validated_scene_ids(checkpoint["validation_scene_ids"], "validation_scene_ids")
        overlap = training_scenes & validation_scenes
        if overlap:
            raise ValueError("door seed checkpoint train/validation scene overlap: %s" % sorted(overlap))
        for key in ("achieved_recall", "negative_rejection_rate"):
            value = float(checkpoint[key])
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("door seed checkpoint %s must be in [0,1]" % key)
        source_hash = str(checkpoint["source_code_hash"])
        current_source_hash = source_tree_hash(Path(__file__).resolve().parent)
        if not source_hash or source_hash != current_source_hash:
            raise ValueError("door seed checkpoint source_code_hash mismatch")
        model = build_door_seed_model(model_cfg)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        device = torch.device(str(self.config.device))
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA inference requested but torch.cuda.is_available() is false")
        model.to(device)
        model.eval()
        self.model = model
        self.checkpoint = checkpoint
        self.device = device
        _ = self.keep_threshold

    def _validate_runtime_metadata(
        self,
        *,
        z_centers_m: np.ndarray,
        z_min_m: float,
        z_resolution_m: float,
        xy_resolution_m: float,
        raw_seed_config_hash: str,
        input_semantics_hash: str,
    ) -> None:
        if self.checkpoint is None:
            raise RuntimeError("door seed checkpoint is not loaded")
        checkpoint = self.checkpoint
        z = np.asarray(z_centers_m, dtype=np.float32)
        checks = (
            (len(z) == int(checkpoint["z_count"]), "z_count"),
            (abs(float(z_min_m) - float(checkpoint["z_min_m"])) <= 1e-6, "z_min_m"),
            (abs(float(z_resolution_m) - float(checkpoint["z_resolution_m"])) <= 1e-6, "z_resolution_m"),
            (abs(float(xy_resolution_m) - float(checkpoint["xy_resolution_m"])) <= 1e-6, "xy_resolution_m"),
            (sha256_bytes(np.ascontiguousarray(z).tobytes()) == str(checkpoint["z_centers_sha256"]), "z_centers_sha256"),
            (str(raw_seed_config_hash) == str(checkpoint["raw_seed_config_hash"]), "raw_seed_config_hash"),
            (str(input_semantics_hash) == str(checkpoint["input_semantics_hash"]), "input_semantics_hash"),
        )
        failed = [name for passed, name in checks if not passed]
        if failed:
            raise ValueError("door seed checkpoint runtime metadata mismatch: %s" % ", ".join(failed))

    def _fallback(
        self,
        raw_result: VoxelDoorSeedResult,
        eligible_raw_mask: np.ndarray,
        probability_xy: np.ndarray,
        reason: str,
        started_at: float,
        seed_connectivity: int,
    ) -> DoorSeedFilterOutput:
        self.fallback_count += 1
        latency_ms = float((time.perf_counter() - started_at) * 1000.0)
        raw_mask = np.asarray(eligible_raw_mask, dtype=bool)
        restricted = rebuild_voxel_door_seed_result(
            raw_result,
            raw_mask,
            seed_connectivity=int(seed_connectivity),
            eligible_raw_seed_mask=raw_mask,
        )
        debug = {
            **dict(restricted.debug),
            "voxel_door_raw_seed_mask": raw_mask.copy(),
            "voxel_door_seed_mask": raw_mask.copy(),
            "voxel_door_seed_model_probability_xy": probability_xy,
            "voxel_door_seed_model_keep_mask": raw_mask.copy(),
            "voxel_door_seed_model_reject_mask": np.zeros(raw_mask.shape, dtype=bool),
            "voxel_door_seed_model_fallback": True,
            "voxel_door_seed_model_fallback_reason": str(reason),
            "voxel_door_seed_model_latency_ms": latency_ms,
            "voxel_door_seed_model_fallback_count": int(self.fallback_count),
            "voxel_door_seed_model_inference_count": int(self.inference_count),
        }
        result = replace(restricted, debug=debug)
        return DoorSeedFilterOutput(result, probability_xy, raw_mask.copy(), np.zeros(raw_mask.shape, dtype=bool), str(reason), latency_ms)


def _validated_scene_ids(value: object, name: str) -> set[str]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("door seed checkpoint %s must be a non-empty list" % name)
    scenes = [str(item).strip() for item in value]
    if any(not item for item in scenes) or len(set(scenes)) != len(scenes):
        raise ValueError("door seed checkpoint %s contains empty or duplicate scene ids" % name)
    return set(scenes)
