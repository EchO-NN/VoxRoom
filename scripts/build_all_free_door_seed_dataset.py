#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("line %d is not an object" % line_number)
            yield row


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build an every-free-cell DoorSeed index from the approved temporal snapshots."
    )
    parser.add_argument("--source-index", required=True)
    parser.add_argument("--out-index", required=True)
    parser.add_argument("--candidate-source", choices=["vertical", "nav"], default="vertical")
    parser.add_argument("--context-source", choices=["vertical", "nav"], default="vertical")
    parser.add_argument(
        "--one-row-per-coordinate",
        action="store_true",
        help="Keep the latest temporal occurrence of every scene/free coordinate; the candidate domain is unchanged.",
    )
    args = parser.parse_args()

    source = Path(args.source_index).expanduser().resolve()
    output = Path(args.out_index).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    positive_by_scene: dict[str, set[tuple[int, int]]] = defaultdict(set)
    snapshot_records: dict[str, dict[str, Any]] = {}
    source_rows = 0
    for row in _read_jsonl(source):
        source_rows += 1
        scene_id = str(row["scene_id"])
        if int(row["label"]) == 1:
            positive_by_scene[scene_id].add((int(row["row"]), int(row["col"])))
        path = str(Path(str(row["snapshot_path"])).expanduser().resolve())
        record = {
            "scene_id": scene_id,
            "scene_uid": str(row["scene_uid"]),
            "episode_id": str(row["episode_id"]),
            "decision_id": int(row["decision_id"]),
            "step": int(row["step"]),
            "split": str(row["split"]),
            "snapshot_path": path,
        }
        previous = snapshot_records.setdefault(path, record)
        if previous != record:
            raise ValueError("inconsistent metadata for snapshot: %s" % path)

    map_key = "%s_class_map_xy" % str(args.candidate_source)
    ordered_snapshots = sorted(
        snapshot_records.items(),
        key=lambda item: (
            item[1]["scene_id"],
            item[1]["decision_id"],
            item[1]["step"],
        ),
    )
    latest_snapshot_by_scene: dict[str, np.ndarray] = {}
    if bool(args.one_row_per_coordinate):
        for snapshot_number, (snapshot_path, record) in enumerate(ordered_snapshots):
            with np.load(snapshot_path, allow_pickle=False) as arrays:
                class_map = np.asarray(arrays[map_key], dtype=np.uint8)
            scene_id = str(record["scene_id"])
            latest = latest_snapshot_by_scene.setdefault(
                scene_id, np.full(class_map.shape, -1, dtype=np.int16)
            )
            if latest.shape != class_map.shape:
                raise ValueError("map shape changed within scene: %s" % scene_id)
            latest[class_map == 1] = int(snapshot_number)
    split_counts: Counter[str] = Counter()
    split_positive: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()
    scene_positive: Counter[str] = Counter()
    temp = output.with_name(output.name + ".tmp")
    started = time.time()
    total = 0
    positive = 0
    with temp.open("w", encoding="utf-8") as handle:
        for snapshot_offset, (snapshot_path, record) in enumerate(ordered_snapshots):
            snapshot_number = snapshot_offset + 1
            with np.load(snapshot_path, allow_pickle=False) as arrays:
                if map_key not in arrays.files:
                    raise KeyError("snapshot has no %s: %s" % (map_key, snapshot_path))
                free_rc = np.argwhere(np.asarray(arrays[map_key], dtype=np.uint8) == 1)
            scene_id = str(record["scene_id"])
            if bool(args.one_row_per_coordinate):
                latest = latest_snapshot_by_scene[scene_id]
                free_rc = free_rc[
                    latest[free_rc[:, 0], free_rc[:, 1]] == int(snapshot_offset)
                ]
            split = str(record["split"])
            positives = positive_by_scene[scene_id]
            for row_value, col_value in free_rc:
                row_i, col_i = int(row_value), int(col_value)
                label = int((row_i, col_i) in positives)
                out_row = {
                    **record,
                    "row": row_i,
                    "col": col_i,
                    "seed_index": -1,
                    "label": label,
                    "label_source": "final_manual_positive_coordinate_else_negative_on_every_free_cell",
                    "group_id": "all_free:%s:%d:%d" % (scene_id, row_i, col_i),
                    "context_source": str(args.context_source),
                    "candidate_source": "%s_free_all" % str(args.candidate_source),
                    "schema_version": "voxroom_door_seed_dataset_v2_patch19_context41_all_free_ablation_v1",
                    "duplicate_count": 1,
                }
                handle.write(json.dumps(out_row, ensure_ascii=False, sort_keys=True) + "\n")
                total += 1
                positive += label
                split_counts[split] += 1
                split_positive[split] += label
                scene_counts[scene_id] += 1
                scene_positive[scene_id] += label
            print(
                "[all-free-index] %d/%d scene=%s decision=%d free=%d total=%d"
                % (
                    snapshot_number,
                    len(snapshot_records),
                    scene_id,
                    int(record["decision_id"]),
                    len(free_rc),
                    total,
                ),
                flush=True,
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, output)
    report = {
        "schema_version": "voxroom_all_free_dataset_build_report_v1",
        "created_at_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "source_index": str(source),
        "source_index_sha256": _sha256(source),
        "source_rows": source_rows,
        "snapshot_count": len(snapshot_records),
        "scene_count": len(scene_counts),
        "candidate_source": "%s_free_all" % str(args.candidate_source),
        "context_source": str(args.context_source),
        "temporal_row_policy": (
            "latest_occurrence_per_scene_coordinate"
            if bool(args.one_row_per_coordinate)
            else "every_snapshot_occurrence"
        ),
        "label_contract": "positive iff the free coordinate is a final manually accepted raw DoorSeed coordinate in the same scene; every other free coordinate is negative",
        "row_count": total,
        "positive_count": positive,
        "negative_count": total - positive,
        "split_counts": dict(sorted(split_counts.items())),
        "split_positive_counts": dict(sorted(split_positive.items())),
        "scene_counts": dict(sorted(scene_counts.items())),
        "scene_positive_counts": dict(sorted(scene_positive.items())),
        "output_index": str(output),
        "output_index_sha256": _sha256(output),
    }
    report_path = output.with_suffix(output.suffix + ".build_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
