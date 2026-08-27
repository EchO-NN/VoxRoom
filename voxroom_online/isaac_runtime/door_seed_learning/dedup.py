from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from voxroom_online.isaac_runtime.door_seed_learning.config import DoorSeedLearningConfig
from voxroom_online.isaac_runtime.door_seed_learning.label_resolver import load_resolved_labels
from voxroom_online.isaac_runtime.door_seed_learning.patch_extraction import (
    extract_class_patches,
    extract_seed_voxel_patch_from_snapshot,
    observed_ratio,
)
from voxroom_online.isaac_runtime.door_seed_learning.schema import (
    CONTEXT_PATCH_SIZE,
    DATASET_SCHEMA_VERSION,
    LOCAL_PATCH_SIZE,
    hash_arrays,
    load_snapshot,
    read_json,
    sha256_file,
    write_json_atomic,
)


_TRANSIENT_VOXEL_KEY = "_dedup_voxel_state_nzyx"
_TRANSIENT_CONTEXT_KEY = "_dedup_context_class_yx"


def build_dataset_indexes(
    collection_root: str | Path,
    output_dir: str | Path,
    *,
    config: DoorSeedLearningConfig | Mapping[str, object],
    split_assignments: Mapping[str, str] | None = None,
) -> dict[str, object]:
    cfg = config if isinstance(config, DoorSeedLearningConfig) else DoorSeedLearningConfig.from_mapping(config)
    root = Path(collection_root).expanduser()
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    scene_dirs = sorted(path.parent for path in root.glob("*/manifest.json"))
    if not scene_dirs:
        raise ValueError("no door seed collection scenes found under %s" % root)
    manifests = {scene: read_json(scene / "manifest.json") for scene in scene_dirs}
    physical_scene_ids = sorted({str(manifest["scene_id"]) for manifest in manifests.values()})
    splits = dict(split_assignments or deterministic_scene_split(physical_scene_ids))
    unknown_split_scenes = set(physical_scene_ids) - set(splits)
    if unknown_split_scenes:
        raise ValueError("split assignments missing scene ids: %s" % sorted(unknown_split_scenes))
    if any(value not in {"train", "val", "test"} for value in splits.values()):
        raise ValueError("split assignments must use train, val, or test")

    report: dict[str, object] = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "collection_root": str(root),
        "output_dir": str(out),
        "config": cfg.to_dict(),
        "scene_splits": {split: sorted(scene for scene, value in splits.items() if value == split) for split in ("train", "val", "test")},
        "indexes": {},
    }
    for context_source in ("vertical", "nav"):
        raw_entries: list[dict[str, object]] = []
        for scene in scene_dirs:
            manifest = manifests[scene]
            scene_entries = build_scene_occurrence_entries(scene, context_source=context_source)
            split = splits[str(manifest["scene_id"])]
            for entry in scene_entries:
                entry["split"] = split
            raw_entries.extend(scene_entries)
        retained, dedup_report = deduplicate_entries(raw_entries, cfg)
        index_path = out / ("dataset_%s.jsonl" % context_source)
        _write_jsonl_atomic(index_path, retained)
        report["indexes"][context_source] = {
            "path": str(index_path),
            "raw_count": len(raw_entries),
            "retained_count": len(retained),
            "dedup": dedup_report,
            "split_counts": dict(Counter(str(item["split"]) for item in retained)),
            "positive_count": int(sum(int(item["label"]) == 1 for item in retained)),
            "negative_count": int(sum(int(item["label"]) == 0 for item in retained)),
        }
    write_json_atomic(out / "dataset_build_report.json", report)
    return report


def build_scene_occurrence_entries(scene_dir: str | Path, *, context_source: str) -> list[dict[str, object]]:
    scene = Path(scene_dir)
    manifest = read_json(scene / "manifest.json")
    resolved_path = scene / "annotations" / "resolved_seed_labels.json"
    resolved = load_resolved_labels(resolved_path)
    if str(resolved["scene_uid"]) != str(manifest["scene_uid"]):
        raise ValueError("resolved label scene_uid mismatch")
    if str(resolved.get("manifest_sha256", "")) != sha256_file(scene / "manifest.json"):
        raise ValueError("resolved labels are stale relative to the collection manifest")
    annotation_path = scene / "annotations" / "manual_final_seed_labels.json"
    if str(resolved.get("manual_annotation_sha256", "")) != sha256_file(annotation_path):
        raise ValueError("resolved labels are stale relative to the manual annotation")
    if str(resolved.get("map_info_hash", "")) != str(manifest["map"]["map_info_hash"]):
        raise ValueError("resolved label map_info_hash mismatch")
    if int(dict(resolved.get("counts", {}) or {}).get("unresolved", 0)) != 0:
        raise ValueError("unresolved labels may not enter a training dataset")
    context_key = "vertical_class_map_xy" if context_source == "vertical" else "nav_class_map_xy"
    cache: dict[int, dict[str, np.ndarray]] = {}
    records = {int(item["decision_id"]): item for item in manifest.get("snapshots", [])}
    entries: list[dict[str, object]] = []
    for occurrence in resolved.get("occurrences", []):
        if occurrence.get("label") is None:
            continue
        decision_id = int(occurrence["decision_id"])
        if decision_id not in cache:
            cache[decision_id] = load_snapshot(scene / str(records[decision_id]["path"]))
        arrays = cache[decision_id]
        seed_index = int(occurrence["seed_index"])
        row, col = int(occurrence["row"]), int(occurrence["col"])
        rc = np.asarray([[row, col]], dtype=np.int32)
        voxel = extract_seed_voxel_patch_from_snapshot(
            arrays,
            seed_index=seed_index,
            seed_rc=rc,
            patch_size=LOCAL_PATCH_SIZE,
        )
        context = extract_class_patches(
            arrays[context_key], rc, patch_size=CONTEXT_PATCH_SIZE
        )[0]
        z = np.asarray(arrays["z_centers_m"], dtype=np.float32)
        signature = hash_arrays({"context": context, "voxel": voxel, "z_centers_m": z})
        entries.append(
            {
                "schema_version": DATASET_SCHEMA_VERSION,
                "scene_uid": str(manifest["scene_uid"]),
                "scene_id": str(manifest["scene_id"]),
                "episode_id": str(manifest["episode_id"]),
                "decision_id": decision_id,
                "step": int(occurrence["step"]),
                "row": row,
                "col": col,
                "seed_index": seed_index,
                "snapshot_path": str((scene / str(records[decision_id]["path"])).resolve()),
                "label": int(occurrence["label"]),
                "label_source": str(occurrence["label_source"]),
                "context_source": context_source,
                "exact_input_sha256": signature,
                "group_id": "%s:%d:%d" % (manifest["scene_uid"], row, col),
                "observed_ratio": observed_ratio(voxel, context),
                _TRANSIENT_VOXEL_KEY: np.ascontiguousarray(voxel).copy(),
                _TRANSIENT_CONTEXT_KEY: np.ascontiguousarray(context).copy(),
            }
        )
    return entries


def deduplicate_entries(
    entries: Iterable[Mapping[str, object]],
    config: DoorSeedLearningConfig | Mapping[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    cfg = config if isinstance(config, DoorSeedLearningConfig) else DoorSeedLearningConfig.from_mapping(config)
    groups: dict[tuple[str, int, int, int], list[dict[str, object]]] = defaultdict(list)
    for raw in entries:
        item = dict(raw)
        key = (str(item["scene_uid"]), int(item["row"]), int(item["col"]), int(item["label"]))
        groups[key].append(item)
    retained: list[dict[str, object]] = []
    exact_removed = 0
    near_removed = 0
    capped_removed = 0
    group_sizes: list[int] = []
    for group in groups.values():
        ordered = sorted(group, key=lambda item: (int(item["step"]), int(item["decision_id"])))
        exact_unique: list[dict[str, object]] = []
        seen: dict[str, dict[str, object]] = {}
        for item in ordered:
            signature = str(item["exact_input_sha256"])
            if bool(cfg.dedup_exact) and signature in seen:
                seen[signature]["duplicate_count"] = int(seen[signature].get("duplicate_count", 0)) + 1
                seen[signature]["last_duplicate_step"] = int(item["step"])
                exact_removed += 1
                continue
            item["duplicate_count"] = 0
            seen[signature] = item
            exact_unique.append(item)
        near_retained = _near_dedup(exact_unique, cfg)
        near_removed += len(exact_unique) - len(near_retained)
        capped = _cap_group(near_retained, int(cfg.max_samples_per_scene_column))
        capped_removed += len(near_retained) - len(capped)
        group_sizes.append(len(capped))
        retained.extend(capped)
    retained.sort(key=lambda item: (str(item["scene_uid"]), int(item["step"]), int(item["row"]), int(item["col"])))
    for item in retained:
        item.pop(_TRANSIENT_VOXEL_KEY, None)
        item.pop(_TRANSIENT_CONTEXT_KEY, None)
    return retained, {
        "input_count": int(sum(len(group) for group in groups.values())),
        "retained_count": len(retained),
        "exact_removed": exact_removed,
        "near_removed": near_removed,
        "cap_removed": capped_removed,
        "group_count": len(groups),
        "max_retained_per_group": max(group_sizes, default=0),
        "mean_retained_per_group": float(np.mean(group_sizes)) if group_sizes else 0.0,
    }


def input_change_metrics(previous: Mapping[str, object], current: Mapping[str, object]) -> tuple[float, float, float]:
    if all(key in previous and key in current for key in (_TRANSIENT_VOXEL_KEY, _TRANSIENT_CONTEXT_KEY)):
        prev_voxel = np.asarray(previous[_TRANSIENT_VOXEL_KEY], dtype=np.uint8)
        cur_voxel = np.asarray(current[_TRANSIENT_VOXEL_KEY], dtype=np.uint8)
        prev_context = np.asarray(previous[_TRANSIENT_CONTEXT_KEY], dtype=np.uint8)
        cur_context = np.asarray(current[_TRANSIENT_CONTEXT_KEY], dtype=np.uint8)
    else:
        prev_arrays = load_snapshot(str(previous["snapshot_path"]))
        cur_arrays = load_snapshot(str(current["snapshot_path"]))
        prev_rc = np.asarray([[int(previous["row"]), int(previous["col"])]], dtype=np.int32)
        cur_rc = np.asarray([[int(current["row"]), int(current["col"])]], dtype=np.int32)
        prev_voxel = extract_seed_voxel_patch_from_snapshot(
            prev_arrays,
            seed_index=int(previous["seed_index"]),
            seed_rc=prev_rc,
            patch_size=LOCAL_PATCH_SIZE,
        )
        cur_voxel = extract_seed_voxel_patch_from_snapshot(
            cur_arrays,
            seed_index=int(current["seed_index"]),
            seed_rc=cur_rc,
            patch_size=LOCAL_PATCH_SIZE,
        )
        context_key = "vertical_class_map_xy" if str(current["context_source"]) == "vertical" else "nav_class_map_xy"
        prev_context = extract_class_patches(
            prev_arrays[context_key], prev_rc, patch_size=CONTEXT_PATCH_SIZE
        )[0]
        cur_context = extract_class_patches(
            cur_arrays[context_key], cur_rc, patch_size=CONTEXT_PATCH_SIZE
        )[0]
    if prev_voxel.shape != cur_voxel.shape:
        raise ValueError("voxel patch shape changed within one dedup group")
    voxel_change = float(np.count_nonzero(prev_voxel != cur_voxel)) / float(max(1, prev_voxel.size))
    context_change = float(np.count_nonzero(prev_context != cur_context)) / float(max(1, prev_context.size))
    prev_unknown = float(np.count_nonzero((prev_voxel == 0) | (prev_voxel == 3))) / float(max(1, prev_voxel.size))
    cur_unknown = float(np.count_nonzero((cur_voxel == 0) | (cur_voxel == 3))) / float(max(1, cur_voxel.size))
    return voxel_change, context_change, prev_unknown - cur_unknown


def deterministic_scene_split(scene_ids: Iterable[str]) -> dict[str, str]:
    scenes = sorted(set(str(value) for value in scene_ids), key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())
    if not scenes:
        return {}
    if len(scenes) == 1:
        return {scenes[0]: "train"}
    val_count = max(1, int(round(len(scenes) * 0.2)))
    val_count = min(val_count, len(scenes) - 1)
    train_count = len(scenes) - val_count
    return {scene: ("train" if index < train_count else "val") for index, scene in enumerate(scenes)}


def read_jsonl(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("dataset index rows must be JSON objects")
                rows.append(value)
    return rows


def _near_dedup(entries: list[dict[str, object]], cfg: DoorSeedLearningConfig) -> list[dict[str, object]]:
    if len(entries) <= 2:
        return list(entries)
    retained = [entries[0]]
    final = entries[-1]
    for item in entries[1:-1]:
        voxel_change, context_change, unknown_drop = input_change_metrics(retained[-1], item)
        item["change_from_previous_retained"] = {
            "voxel_change_ratio": voxel_change,
            "context_change_ratio": context_change,
            "unknown_ratio_drop": unknown_drop,
        }
        if (
            voxel_change >= float(cfg.dedup_change_ratio)
            or context_change >= float(cfg.dedup_change_ratio)
            or unknown_drop >= float(cfg.dedup_unknown_drop)
        ):
            retained.append(item)
    retained.append(final)
    return retained


def _cap_group(entries: list[dict[str, object]], maximum: int) -> list[dict[str, object]]:
    if len(entries) <= int(maximum):
        return list(entries)
    slots = int(maximum) - 2
    middle = sorted(entries[1:-1], key=lambda item: (float(item.get("observed_ratio", 0.0)), int(item["step"])))
    if slots <= 0:
        return [entries[0], entries[-1]]
    indices = np.linspace(0, len(middle) - 1, num=slots, dtype=np.int32)
    chosen = [middle[int(index)] for index in sorted(set(int(v) for v in indices))]
    return sorted([entries[0], *chosen, entries[-1]], key=lambda item: (int(item["step"]), int(item["decision_id"])))


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    target = Path(path)
    temp = target.with_name(target.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, target)
