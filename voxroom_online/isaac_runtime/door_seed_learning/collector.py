from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image

from voxroom_online.isaac_runtime.door_seed_learning.config import DoorSeedLearningConfig
from voxroom_online.isaac_runtime.door_seed_learning.patch_extraction import extract_local_voxel_patches, sorted_seed_coordinates
from voxroom_online.isaac_runtime.door_seed_learning.schema import (
    COLLECTION_SCHEMA_VERSION,
    COORDINATE_CONVENTION,
    SNAPSHOT_REQUIRED_ARRAYS,
    SNAPSHOT_RAW_SEED_SOURCE_ARRAYS,
    hash_arrays,
    load_snapshot,
    map_info_payload,
    read_json,
    sha256_file,
    validate_snapshot_arrays,
    write_json_atomic,
)
from voxroom_online.isaac_runtime.door_seed_learning.stage_extractor import DoorSeedStageResult
from voxroom_online.isaac_runtime.mapping.voxel_occupancy_grid import VoxelOccupancyGrid3D


class DoorSeedCollector:
    def __init__(
        self,
        *,
        config: DoorSeedLearningConfig | Mapping[str, object],
        scene_id: str,
        episode_id: str,
        run_seed: int,
        map_info: object,
        voxel_grid: VoxelOccupancyGrid3D,
        source_code_hash: str,
    ) -> None:
        self.config = config if isinstance(config, DoorSeedLearningConfig) else DoorSeedLearningConfig.from_mapping(config)
        if self.config.mode != "collect":
            raise ValueError("DoorSeedCollector requires collect mode")
        self.scene_id = str(scene_id)
        self.episode_id = str(episode_id)
        self.run_seed = int(run_seed)
        self.scene_uid = make_scene_uid(self.scene_id, self.episode_id, self.run_seed)
        self.scene_dir = Path(self.config.collection_root).expanduser() / self.scene_uid
        self.snapshot_dir = self.scene_dir / "snapshots"
        self.annotation_dir = self.scene_dir / "annotations"
        self.preview_dir = self.scene_dir / "previews"
        for directory in (self.snapshot_dir, self.annotation_dir, self.preview_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.scene_dir / "manifest.json"
        self.map_payload = map_info_payload(map_info, tuple(voxel_grid.shape))
        self._cleanup_temporary_files()
        if self.manifest_path.exists():
            self.manifest = read_json(self.manifest_path)
            self._validate_existing_manifest(voxel_grid)
        else:
            self.manifest = self._initial_manifest(voxel_grid, source_code_hash)
            write_json_atomic(self.manifest_path, self.manifest)
        self.next_decision_id = 1 + max(
            [-1] + [int(item["decision_id"]) for item in self.manifest.get("snapshots", [])]
        )

    def collect(
        self,
        *,
        stage: DoorSeedStageResult,
        voxel_grid: VoxelOccupancyGrid3D,
        step: int,
        reason: str,
        is_final: bool = False,
        full_voxel_arrays: Mapping[str, np.ndarray] | None = None,
        coverage_metadata: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        self._validate_stage(stage, voxel_grid)
        arrays = self._snapshot_arrays(
            stage,
            voxel_grid,
            step=int(step),
            reason=str(reason),
            is_final=bool(is_final),
            full_voxel_arrays=full_voxel_arrays,
            coverage_metadata=coverage_metadata,
        )
        content_hash = snapshot_content_hash(arrays)
        existing = self._find_existing_decision(step=int(step), reason=str(reason))
        if existing is not None:
            if str(existing.get("content_sha256", "")) != content_hash:
                raise RuntimeError("existing door seed decision has different content")
            if not is_final or bool(existing.get("is_final", False)):
                return dict(existing)
        return self._append_snapshot_arrays(
            arrays,
            step=int(step),
            reason=str(reason),
            is_final=bool(is_final),
            termination_reason=None,
        )

    def finalize(
        self,
        *,
        stage: DoorSeedStageResult,
        voxel_grid: VoxelOccupancyGrid3D,
        step: int,
        termination_reason: str,
        full_voxel_arrays: Mapping[str, np.ndarray] | None = None,
        coverage_metadata: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        self._validate_stage(stage, voxel_grid)
        arrays = self._snapshot_arrays(
            stage,
            voxel_grid,
            step=int(step),
            reason="episode_final",
            is_final=True,
            full_voxel_arrays=full_voxel_arrays,
            coverage_metadata=coverage_metadata,
        )
        existing = self._find_existing_decision(step=int(step), reason="episode_final")
        if existing is not None:
            if str(existing.get("content_sha256", "")) != snapshot_content_hash(arrays):
                raise RuntimeError("existing final door seed decision has different content")
            return dict(existing)
        return self._append_snapshot_arrays(
            arrays,
            step=int(step),
            reason="episode_final",
            is_final=True,
            termination_reason=str(termination_reason),
        )

    def finalize_last_snapshot(self, *, termination_reason: str) -> dict[str, object]:
        """Durably mark the latest committed decision as final after interruption."""

        finals = [item for item in self.manifest.get("snapshots", []) if bool(item.get("is_final", False))]
        if finals:
            return dict(finals[0])
        snapshots = list(self.manifest.get("snapshots", []))
        if not snapshots:
            raise RuntimeError("cannot finalize an interrupted collection before its first snapshot")
        source = snapshots[-1]
        source_path = self.scene_dir / str(source["path"])
        if sha256_file(source_path) != str(source["sha256"]):
            raise RuntimeError("cannot finalize interrupted collection from a corrupt snapshot")
        arrays = load_snapshot(source_path)
        arrays["reason"] = np.asarray("episode_final")
        arrays["is_final"] = np.asarray(True)
        return self._append_snapshot_arrays(
            arrays,
            step=int(source["step"]),
            reason="episode_final",
            is_final=True,
            termination_reason=str(termination_reason),
            original_reason=str(source.get("reason", "")),
        )

    def _initial_manifest(self, voxel_grid: VoxelOccupancyGrid3D, source_code_hash: str) -> dict[str, object]:
        return {
            "schema_version": COLLECTION_SCHEMA_VERSION,
            "scene_uid": self.scene_uid,
            "scene_id": self.scene_id,
            "episode_id": self.episode_id,
            "run_seed": self.run_seed,
            "coordinate_convention": COORDINATE_CONVENTION,
            "map": {
                "shape": [int(voxel_grid.shape[0]), int(voxel_grid.shape[1])],
                "resolution_m": float(voxel_grid.map_info.resolution_m),
                "origin_or_transform": self.map_payload,
                "map_info_hash": "",
            },
            "voxel": {
                "z_min_m": float(voxel_grid.z_min_m),
                "z_max_m": float(voxel_grid.z_max_m),
                "z_resolution_m": float(voxel_grid.z_resolution_m),
                "z_count": int(voxel_grid.z_bin_count),
                "state_codes": {"unknown": 0, "free": 1, "occupied": 2, "conflict": 3},
            },
            "collection": {
                "local_voxel_patch_size": int(self.config.local_voxel_patch_size),
                "context_patch_size": int(self.config.context_patch_size),
                "raw_seed_config_hash": "",
                "input_semantics_hash": "",
                "git_commit_or_source_hash": str(source_code_hash),
                "raw_seed_source": str(self.config.raw_seed_source),
                "collection_every_steps": int(self.config.collection_every_steps),
                "tvars_seed_width_cells": int(self.config.tvars_seed_width_cells),
                "persistent_final_raw_seed_union": bool(
                    self.config.persistent_final_raw_seed_union
                ),
                "save_full_voxel_milestones": bool(
                    self.config.save_full_voxel_milestones
                ),
                "full_voxel_coverage_milestones": list(
                    self.config.parsed_full_voxel_coverage_milestones()
                ),
            },
            "snapshots": [],
            "final_snapshot_decision_id": None,
            "termination_reason": None,
            "created_at_unix": time.time(),
        }

    def _validate_existing_manifest(self, voxel_grid: VoxelOccupancyGrid3D) -> None:
        manifest = self.manifest
        if manifest.get("schema_version") != COLLECTION_SCHEMA_VERSION:
            raise ValueError("unsupported door seed collection manifest schema")
        if manifest.get("scene_uid") != self.scene_uid:
            raise ValueError("door seed collection scene_uid mismatch")
        shape = tuple(int(v) for v in manifest.get("map", {}).get("shape", []))
        if shape != tuple(voxel_grid.shape):
            raise ValueError("door seed collection map shape changed")
        voxel = dict(manifest.get("voxel", {}) or {})
        expected = (float(voxel_grid.z_min_m), float(voxel_grid.z_max_m), float(voxel_grid.z_resolution_m), int(voxel_grid.z_bin_count))
        actual = (
            float(voxel.get("z_min_m")),
            float(voxel.get("z_max_m")),
            float(voxel.get("z_resolution_m")),
            int(voxel.get("z_count")),
        )
        if actual != expected:
            raise ValueError("door seed collection voxel geometry changed")
        final_records = [item for item in manifest.get("snapshots", []) if bool(item.get("is_final", False))]
        if len(final_records) > 1:
            raise ValueError("door seed collection manifest has multiple final snapshots")

    def _validate_stage(self, stage: DoorSeedStageResult, voxel_grid: VoxelOccupancyGrid3D) -> None:
        if tuple(stage.map_shape) != tuple(voxel_grid.shape):
            raise ValueError("door seed stage shape changed during collection")
        map_section = self.manifest["map"]
        expected_map_hash = str(map_section.get("map_info_hash", ""))
        if expected_map_hash and expected_map_hash != str(stage.map_info_hash):
            raise RuntimeError("map_info_hash changed within one door seed collection scene")
        if not expected_map_hash:
            map_section["map_info_hash"] = str(stage.map_info_hash)
        collection = self.manifest["collection"]
        expected_seed_hash = str(collection.get("raw_seed_config_hash", ""))
        if expected_seed_hash and expected_seed_hash != str(stage.raw_seed_config_hash):
            raise RuntimeError("raw_seed_config_hash changed within one door seed collection scene")
        if not expected_seed_hash:
            collection["raw_seed_config_hash"] = str(stage.raw_seed_config_hash)
        expected_semantics_hash = str(collection.get("input_semantics_hash", ""))
        if expected_semantics_hash and expected_semantics_hash != str(stage.input_semantics_hash):
            raise RuntimeError("input_semantics_hash changed within one door seed collection scene")
        if not expected_semantics_hash:
            collection["input_semantics_hash"] = str(stage.input_semantics_hash)

    def _snapshot_arrays(
        self,
        stage: DoorSeedStageResult,
        voxel_grid: VoxelOccupancyGrid3D,
        *,
        step: int,
        reason: str,
        is_final: bool,
        full_voxel_arrays: Mapping[str, np.ndarray] | None = None,
        coverage_metadata: Mapping[str, object] | None = None,
    ) -> dict[str, np.ndarray]:
        rc = sorted_seed_coordinates(stage.raw_seed_mask_xy)
        patches, valid = extract_local_voxel_patches(
            voxel_grid.state,
            rc,
            patch_size=int(self.config.local_voxel_patch_size),
        )
        arrays = {
            "vertical_class_map_xy": np.asarray(stage.vertical_class_map_xy, dtype=np.uint8),
            "nav_class_map_xy": np.asarray(stage.nav_class_map_xy, dtype=np.uint8),
            "raw_seed_mask_xy": np.asarray(stage.raw_seed_mask_xy, dtype=bool),
            "raw_seed_rc": rc.astype(np.int32),
            "raw_seed_voxel_state_nzyx": patches.astype(np.uint8),
            "raw_seed_patch_valid_nyx": valid.astype(bool),
            "z_centers_m": np.asarray(voxel_grid.z_centers_m, dtype=np.float32),
            "schema_version": np.asarray(COLLECTION_SCHEMA_VERSION),
            "step": np.asarray(int(step), dtype=np.int64),
            "reason": np.asarray(str(reason)),
            "is_final": np.asarray(bool(is_final)),
            "scene_uid": np.asarray(self.scene_uid),
            "height": np.asarray(int(stage.map_shape[0]), dtype=np.int32),
            "width": np.asarray(int(stage.map_shape[1]), dtype=np.int32),
            "resolution_m": np.asarray(float(voxel_grid.map_info.resolution_m), dtype=np.float32),
            "z_min_m": np.asarray(float(voxel_grid.z_min_m), dtype=np.float32),
            "z_max_m": np.asarray(float(voxel_grid.z_max_m), dtype=np.float32),
            "z_resolution_m": np.asarray(float(voxel_grid.z_resolution_m), dtype=np.float32),
            "local_patch_size": np.asarray(int(self.config.local_voxel_patch_size), dtype=np.int32),
            "coordinate_convention": np.asarray(COORDINATE_CONVENTION),
            "map_info_hash": np.asarray(str(stage.map_info_hash)),
            "raw_seed_config_hash": np.asarray(str(stage.raw_seed_config_hash)),
            "input_semantics_hash": np.asarray(str(stage.input_semantics_hash)),
            "full_voxel_snapshot": np.asarray(full_voxel_arrays is not None),
        }
        if full_voxel_arrays is not None:
            arrays.update(
                {str(key): np.asarray(value).copy() for key, value in full_voxel_arrays.items()}
            )
        if coverage_metadata is not None:
            coverage = dict(coverage_metadata)
            arrays.update(
                {
                    "coverage_event_id": np.asarray(str(coverage["event_id"])),
                    "coverage_threshold": np.asarray(
                        np.nan if coverage.get("threshold") is None else float(coverage["threshold"]),
                        dtype=np.float64,
                    ),
                    "coverage_ratio": np.asarray(float(coverage["coverage_ratio"]), dtype=np.float64),
                    "coverage_explored_cells": np.asarray(int(coverage["explored_cells"]), dtype=np.int64),
                    "coverage_total_explorable_cells": np.asarray(
                        int(coverage["total_explorable_cells"]), dtype=np.int64
                    ),
                    "coverage_reference_explorable_mask_xy": np.asarray(
                        coverage["reference_explorable_mask_xy"], dtype=bool
                    ).copy(),
                    "coverage_explored_reference_mask_xy": np.asarray(
                        coverage["explored_reference_mask_xy"], dtype=bool
                    ).copy(),
                }
            )
        voxroom_current = _stage_mask_or_fallback(
            stage.voxroom_raw_seed_mask_xy,
            stage.raw_seed_mask_xy,
            stage.map_shape,
        )
        tvars_current = _stage_mask_or_fallback(
            stage.tvars_vertical_raw_seed_mask_xy,
            np.zeros(stage.map_shape, dtype=bool),
            stage.map_shape,
        )
        voxroom_history = _stage_mask_or_fallback(
            stage.voxroom_raw_seed_history_mask_xy,
            voxroom_current,
            stage.map_shape,
        )
        tvars_history = _stage_mask_or_fallback(
            stage.tvars_vertical_raw_seed_history_mask_xy,
            tvars_current,
            stage.map_shape,
        )
        source_id = np.zeros(stage.map_shape, dtype=np.uint8)
        source_id[voxroom_history] = 1
        source_id[tvars_history] = 2
        source_id[voxroom_history & tvars_history] = 3
        arrays.update(
            {
                "voxroom_raw_seed_mask_xy": voxroom_current,
                "tvars_vertical_raw_seed_mask_xy": tvars_current,
                "voxroom_raw_seed_history_mask_xy": voxroom_history,
                "tvars_vertical_raw_seed_history_mask_xy": tvars_history,
                "raw_seed_source_id_map_xy": source_id,
            }
        )
        validate_snapshot_arrays(arrays)
        return arrays

    def _find_existing_decision(self, *, step: int, reason: str) -> dict[str, object] | None:
        for item in self.manifest.get("snapshots", []):
            if int(item.get("step", -1)) == int(step) and str(item.get("reason", "")) == str(reason):
                return item
        return None

    def _append_snapshot_arrays(
        self,
        arrays: Mapping[str, np.ndarray],
        *,
        step: int,
        reason: str,
        is_final: bool,
        termination_reason: str | None,
        original_reason: str | None = None,
    ) -> dict[str, object]:
        if is_final and any(bool(item.get("is_final", False)) for item in self.manifest.get("snapshots", [])):
            raise RuntimeError("door seed collection already has a final snapshot")
        payload = {key: np.asarray(value) for key, value in arrays.items()}
        payload["step"] = np.asarray(int(step), dtype=np.int64)
        payload["reason"] = np.asarray(str(reason))
        payload["is_final"] = np.asarray(bool(is_final))
        content_hash = snapshot_content_hash(payload)
        decision_id = int(self.next_decision_id)
        filename = "decision_%06d_step_%06d.npz" % (decision_id, int(step))
        relative_path = str(Path("snapshots") / filename)
        path = self.scene_dir / relative_path
        payload["decision_id"] = np.asarray(decision_id, dtype=np.int64)
        validate_snapshot_arrays(payload)
        _write_npz_atomic(path, payload)
        record = {
            "decision_id": decision_id,
            "step": int(step),
            "reason": str(reason),
            "path": relative_path,
            "sha256": sha256_file(path),
            "content_sha256": content_hash,
            "raw_seed_count": int(len(payload["raw_seed_rc"])),
            "voxroom_raw_seed_count": int(
                np.count_nonzero(payload.get("voxroom_raw_seed_mask_xy", False))
            ),
            "tvars_vertical_raw_seed_count": int(
                np.count_nonzero(payload.get("tvars_vertical_raw_seed_mask_xy", False))
            ),
            "voxroom_raw_seed_history_count": int(
                np.count_nonzero(payload.get("voxroom_raw_seed_history_mask_xy", False))
            ),
            "tvars_vertical_raw_seed_history_count": int(
                np.count_nonzero(payload.get("tvars_vertical_raw_seed_history_mask_xy", False))
            ),
            "is_final": bool(is_final),
            "full_voxel_snapshot": bool(
                np.asarray(payload.get("full_voxel_snapshot", False)).item()
            ),
            "created_at_unix": time.time(),
        }
        if "coverage_event_id" in payload:
            record.update(
                {
                    "coverage_event_id": str(np.asarray(payload["coverage_event_id"]).item()),
                    "coverage_threshold": (
                        None
                        if not np.isfinite(float(np.asarray(payload["coverage_threshold"]).item()))
                        else float(np.asarray(payload["coverage_threshold"]).item())
                    ),
                    "coverage_ratio": float(np.asarray(payload["coverage_ratio"]).item()),
                    "coverage_explored_cells": int(
                        np.asarray(payload["coverage_explored_cells"]).item()
                    ),
                    "coverage_total_explorable_cells": int(
                        np.asarray(payload["coverage_total_explorable_cells"]).item()
                    ),
                }
            )
        if original_reason and original_reason != str(reason):
            record["original_reason"] = str(original_reason)
        next_manifest = dict(self.manifest)
        next_manifest["snapshots"] = [dict(item) for item in self.manifest.get("snapshots", [])] + [record]
        if is_final:
            next_manifest["final_snapshot_decision_id"] = decision_id
        if termination_reason is not None:
            next_manifest["termination_reason"] = str(termination_reason)
        write_json_atomic(self.manifest_path, next_manifest)
        self.manifest = next_manifest
        self.next_decision_id = decision_id + 1
        if is_final:
            self._save_final_previews(payload)
        return dict(record)

    def _cleanup_temporary_files(self) -> None:
        for path in self.scene_dir.glob("**/*.tmp"):
            path.unlink(missing_ok=True)
        for path in self.scene_dir.glob("**/*.tmp.npz"):
            path.unlink(missing_ok=True)

    def _save_final_previews(self, arrays: Mapping[str, np.ndarray]) -> None:
        seed = np.asarray(arrays["raw_seed_mask_xy"], dtype=bool)
        for key, filename, colors in (
            ("vertical_class_map_xy", "final_vertical.png", ((235, 235, 235), (245, 245, 245), (30, 30, 30))),
            ("nav_class_map_xy", "final_nav.png", ((235, 235, 235), (248, 248, 248), (45, 45, 45))),
        ):
            classes = np.asarray(arrays[key], dtype=np.uint8)
            rgb = np.zeros(classes.shape + (3,), dtype=np.uint8)
            for class_id, color in enumerate(colors):
                rgb[classes == class_id] = color
            rgb[seed] = (30, 110, 230)
            Image.fromarray(rgb, mode="RGB").save(self.preview_dir / filename)

        if "raw_seed_source_id_map_xy" in arrays:
            source_id = np.asarray(arrays["raw_seed_source_id_map_xy"], dtype=np.uint8)
            source_rgb = np.full(source_id.shape + (3,), 240, dtype=np.uint8)
            source_rgb[source_id == 1] = (230, 55, 55)
            source_rgb[source_id == 2] = (35, 120, 235)
            source_rgb[source_id == 3] = (185, 45, 210)
            Image.fromarray(source_rgb, mode="RGB").save(
                self.preview_dir / "final_raw_seed_sources.png"
            )


def snapshot_content_hash(arrays: Mapping[str, np.ndarray]) -> str:
    keys = list(SNAPSHOT_REQUIRED_ARRAYS)
    keys.extend(key for key in SNAPSHOT_RAW_SEED_SOURCE_ARRAYS if key in arrays)
    return hash_arrays(arrays, keys=keys)


def _stage_mask_or_fallback(
    value: np.ndarray | None,
    fallback: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    result = np.asarray(fallback if value is None else value, dtype=bool)
    if result.shape != tuple(shape):
        raise ValueError("raw seed source mask does not match stage map")
    return result.copy()


def make_scene_uid(scene_id: str, episode_id: str, run_seed: int) -> str:
    raw = "%s__episode_%s__seed_%d" % (str(scene_id), str(episode_id), int(run_seed))
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_") or "scene"
    suffix = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    return "%s__%s" % (readable, suffix)


def _write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.stem + ".tmp.npz")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **{key: np.asarray(value) for key, value in arrays.items()})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, target)
