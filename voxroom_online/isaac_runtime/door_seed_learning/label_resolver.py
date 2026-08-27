from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Mapping

import numpy as np

from voxroom_online.isaac_runtime.door_seed_learning.annotation_schema import load_annotation, validate_annotation
from voxroom_online.isaac_runtime.door_seed_learning.schema import (
    RESOLVED_LABEL_SCHEMA_VERSION,
    load_snapshot,
    read_json,
    scalar_value,
    sha256_file,
    write_json_atomic,
)


def resolve_scene_labels(
    scene_dir: str | Path,
    *,
    annotation_path: str | Path | None = None,
    output_path: str | Path | None = None,
    require_approved: bool = True,
) -> dict[str, object]:
    scene = Path(scene_dir)
    manifest_path = scene / "manifest.json"
    annotation_file = Path(annotation_path) if annotation_path is not None else scene / "annotations" / "manual_final_seed_labels.json"
    resolved_file = Path(output_path) if output_path is not None else scene / "annotations" / "resolved_seed_labels.json"
    manifest = read_json(manifest_path)
    annotation = load_annotation(annotation_file)
    final_seed_set = validate_annotation(annotation, scene, require_approved=require_approved)
    manual = annotation.label_map()
    snapshots = sorted(manifest.get("snapshots", []), key=lambda item: int(item["decision_id"]))
    if not snapshots:
        raise ValueError("collection manifest has no snapshots")
    occurrences: list[dict[str, object]] = []
    source_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    manifest_map_hash = str(manifest["map"]["map_info_hash"])
    final_id = int(manifest["final_snapshot_decision_id"])
    for record in snapshots:
        snapshot_path = scene / str(record["path"])
        if sha256_file(snapshot_path) != str(record["sha256"]):
            raise ValueError("snapshot sha256 mismatch: %s" % snapshot_path)
        arrays = load_snapshot(snapshot_path)
        if str(scalar_value(arrays, "map_info_hash", "")) != manifest_map_hash:
            raise ValueError("map_info_hash changed across collection snapshots")
        raw_seed_rc = np.asarray(arrays["raw_seed_rc"], dtype=np.int32)
        for seed_index, (row, col) in enumerate(raw_seed_rc):
            coordinate = (int(row), int(col))
            is_final = int(record["decision_id"]) == final_id
            if coordinate not in final_seed_set:
                label: int | None = 0
                source = "auto_absent_from_final_raw_seed"
            elif coordinate not in manual:
                label = None
                source = "unresolved"
            else:
                label = int(manual[coordinate])
                if is_final:
                    source = "manual_final_positive" if label == 1 else "manual_final_negative"
                else:
                    source = "propagated_from_final_manual_positive" if label == 1 else "propagated_from_final_manual_negative"
            source_counts[source] += 1
            label_counts["unresolved" if label is None else ("positive" if label == 1 else "negative")] += 1
            occurrences.append(
                {
                    "scene_uid": str(manifest["scene_uid"]),
                    "decision_id": int(record["decision_id"]),
                    "step": int(record["step"]),
                    "snapshot_path": str(record["path"]),
                    "row": int(row),
                    "col": int(col),
                    "seed_index": int(seed_index),
                    "label": label,
                    "label_source": source,
                    "is_final": bool(is_final),
                }
            )
    result = {
        "schema_version": RESOLVED_LABEL_SCHEMA_VERSION,
        "scene_uid": str(manifest["scene_uid"]),
        "manifest_sha256": sha256_file(manifest_path),
        "manual_annotation_sha256": sha256_file(annotation_file),
        "map_info_hash": manifest_map_hash,
        "counts": {
            "occurrences": len(occurrences),
            "positive": int(label_counts["positive"]),
            "negative": int(label_counts["negative"]),
            "unresolved": int(label_counts["unresolved"]),
            "auto_negative": int(source_counts["auto_absent_from_final_raw_seed"]),
            "manual_propagated_negative": int(source_counts["propagated_from_final_manual_negative"]),
            "manual_propagated_positive": int(source_counts["propagated_from_final_manual_positive"]),
            "manual_final_negative": int(source_counts["manual_final_negative"]),
            "manual_final_positive": int(source_counts["manual_final_positive"]),
        },
        "occurrences": occurrences,
    }
    write_json_atomic(resolved_file, result)
    return result


def load_resolved_labels(path: str | Path) -> dict[str, object]:
    result = read_json(path)
    if result.get("schema_version") != RESOLVED_LABEL_SCHEMA_VERSION:
        raise ValueError("unsupported resolved door seed label schema")
    return result
