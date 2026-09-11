#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from voxroom_online.isaac_runtime.evaluation.online_roomseg.common import (
    write_json_atomic,
)
from voxroom_online.isaac_runtime.evaluation.online_roomseg.snapshot_index import (
    load_index,
)


REQUIRED_FULL_VOXEL_KEYS = (
    "voxel_occupancy_state_zyx",
    "voxel_occupancy_log_odds_zyx",
    "voxel_sensor_range_count_zyx",
    "voxel_occupancy_z_centers_m",
    "roomseg_eval_reference_explorable_mask",
    "roomseg_eval_explored_reference_mask",
    "final_room_label_map",
)
VERTICAL_FREE_KEYS = (
    "voxel_vertical_free_xy",
    "height_profile_vertical_free_xy",
    "vertical_free_room_domain",
)
SAME_STEP_CORE_KEYS = (
    "voxel_occupancy_state_zyx",
    "voxel_vertical_free_xy",
    "roomseg_eval_explored_reference_mask",
    "final_room_label_map",
)
DEFAULT_EXPECTED_EVENTS = (
    "milestone_020",
    "milestone_040",
    "milestone_060",
    "milestone_070",
    "milestone_080",
    "milestone_090",
    "final",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(
    index_path: Path,
    expected_events: Sequence[str],
    *,
    dataset_name: str = "grscene",
) -> dict[str, Any]:
    index_path = index_path.resolve()
    index = load_index(index_path)
    episodes = [dict(value) for value in index.get("episodes", [])]
    event_counts: Counter[str] = Counter()
    missing_files: list[str] = []
    incomplete_files: list[dict[str, Any]] = []
    missing_events: list[dict[str, Any]] = []
    duplicate_steps: list[dict[str, Any]] = []
    unique_paths: set[str] = set()
    total_bytes = 0

    for episode in episodes:
        uid = str(episode.get("episode_uid"))
        snapshots = [dict(value) for value in episode.get("snapshots", [])]
        present_events = {str(value.get("coverage_event_id")) for value in snapshots}
        absent = [value for value in expected_events if value not in present_events]
        if absent:
            missing_events.append(
                {
                    "episode_uid": uid,
                    "absent_untriggered_events": absent,
                    "last_snapshot_step": episode.get("last_snapshot_step"),
                }
            )

        previous_step: int | None = None
        previous_event: str | None = None
        previous_source: Path | None = None
        for snapshot in snapshots:
            event_id = str(snapshot.get("coverage_event_id"))
            event_counts[event_id] += 1
            step = int(snapshot["step"])
            same_step_record: dict[str, Any] | None = None
            if previous_step == step:
                same_step_record = {
                    "episode_uid": uid,
                    "step": step,
                    "events": [previous_event, event_id],
                }
                duplicate_steps.append(same_step_record)
            if previous_step is not None and step < previous_step:
                raise ValueError(
                    f"snapshot steps regress for {uid}: {previous_step} -> {step}"
                )
            previous_step = step
            previous_event = event_id

            source = Path(str(snapshot["snapshot_path"]))
            if (
                same_step_record is not None
                and previous_source is not None
                and previous_source.is_file()
                and source.is_file()
            ):
                with np.load(previous_source, allow_pickle=False) as previous_payload:
                    with np.load(source, allow_pickle=False) as current_payload:
                        different = [
                            key
                            for key in SAME_STEP_CORE_KEYS
                            if key not in previous_payload.files
                            or key not in current_payload.files
                            or not np.array_equal(
                                previous_payload[key], current_payload[key]
                            )
                        ]
                same_step_record["core_inputs_equal"] = not different
                same_step_record["different_core_keys"] = different
            previous_source = source
            source_text = str(source.resolve())
            if source_text in unique_paths:
                continue
            unique_paths.add(source_text)
            if not source.is_file():
                missing_files.append(source_text)
                continue
            total_bytes += source.stat().st_size
            with np.load(source, allow_pickle=False) as payload:
                keys = set(payload.files)
            missing_keys = sorted(set(REQUIRED_FULL_VOXEL_KEYS).difference(keys))
            vertical_keys = sorted(set(VERTICAL_FREE_KEYS).intersection(keys))
            if missing_keys or not vertical_keys:
                incomplete_files.append(
                    {
                        "snapshot_path": source_text,
                        "missing_required_keys": missing_keys,
                        "vertical_free_keys": vertical_keys,
                    }
                )

    actual = sum(event_counts.values())
    nominal = len(episodes) * len(expected_events)
    return {
        "schema_version": (
            "voxroom_grscene_voxel_snapshot_audit_v1"
            if str(dataset_name) == "grscene"
            else "voxroom_voxel_snapshot_audit_v1"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_name),
        "index_path": str(index_path),
        "index_sha256": _sha256(index_path),
        "episode_count": len(episodes),
        "expected_events_per_episode": list(expected_events),
        "nominal_checkpoint_count": nominal,
        "actual_checkpoint_count": actual,
        "unique_snapshot_path_count": len(unique_paths),
        "physical_snapshot_file_count": len(unique_paths) - len(missing_files),
        "complete_full_voxel_snapshot_count": (
            len(unique_paths) - len(missing_files) - len(incomplete_files)
        ),
        "total_snapshot_bytes": total_bytes,
        "total_snapshot_gib": total_bytes / float(1024**3),
        "event_counts": dict(sorted(event_counts.items())),
        "missing_physical_files": missing_files,
        "incomplete_full_voxel_files": incomplete_files,
        "absent_untriggered_event_count": nominal - actual,
        "episodes_with_absent_untriggered_events": missing_events,
        "same_step_event_pair_count": len(duplicate_steps),
        "same_step_core_input_mismatch_count": sum(
            not bool(value.get("core_inputs_equal", False))
            for value in duplicate_steps
        ),
        "same_step_event_pairs": duplicate_steps,
        "all_triggered_snapshots_present_and_complete": bool(
            not missing_files and not incomplete_files and actual == len(unique_paths)
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit saved full-voxel room-segmentation checkpoints."
    )
    parser.add_argument("--index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-name", default="grscene")
    parser.add_argument(
        "--expected-events",
        nargs="+",
        default=list(DEFAULT_EXPECTED_EVENTS),
    )
    args = parser.parse_args(argv)
    result = audit(
        Path(args.index),
        tuple(args.expected_events),
        dataset_name=str(args.dataset_name),
    )
    write_json_atomic(Path(args.output).resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["all_triggered_snapshots_present_and_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
