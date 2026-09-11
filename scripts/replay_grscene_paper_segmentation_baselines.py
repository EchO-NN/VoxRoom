#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from voxroom_online.isaac_runtime.baselines.data_contract import load_npz_arrays
from voxroom_online.isaac_runtime.baselines.mask_io import (
    SEGMENTATION_INPUT_MODE_KEY,
    SEGMENTATION_INPUT_MODES,
    build_segmentation_domain_from_source,
)
from voxroom_online.isaac_runtime.baselines.offline.run_saved_snapshots import (
    BASELINE_CHOICES,
    make_runner,
)
from voxroom_online.isaac_runtime.comparison.metadata_gate import (
    assert_main_experiment_metadata,
)
from voxroom_online.isaac_runtime.door_seed_learning.schema import source_tree_hash
from voxroom_online.isaac_runtime.evaluation.online_roomseg.common import (
    write_json_atomic,
)
from voxroom_online.isaac_runtime.evaluation.online_roomseg.snapshot_index import (
    load_index,
)


PAPER_BASELINES = (
    "dude_incremental",
    "gomez_incremental",
    "dude_offline",
    "rose2",
    "morphological",
    "distance_transform",
    "voronoi",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _approved_uids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("approved_episode_uids")
    if not isinstance(values, list) or not values:
        raise ValueError(f"approved summary has no approved_episode_uids: {path}")
    return {str(value) for value in values}


def _prediction_path(
    output_root: Path, method: str, episode_uid: str, event_id: str
) -> Path:
    return output_root / "predictions" / method / episode_uid / (event_id + ".npz")


def _save_prediction(
    path: Path,
    *,
    label_map: np.ndarray,
    metadata: Mapping[str, Any],
    source_snapshot: Path,
    event_id: str,
    step: int,
    debug_arrays: Mapping[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            final_room_label_map=np.asarray(label_map, dtype=np.int32),
            baseline_metadata_json=np.asarray(
                json.dumps(dict(metadata), ensure_ascii=False, sort_keys=True)
            ),
            source_snapshot_path=np.asarray(str(source_snapshot)),
            coverage_event_id=np.asarray(str(event_id)),
            step=np.asarray(int(step), dtype=np.int64),
            **{key: np.asarray(value) for key, value in (debug_arrays or {}).items()
               if key.startswith("dude_")},
        )
        handle.flush()
    tmp.replace(path)


def _prediction_is_valid(
    path: Path,
    *,
    method: str,
    shape: tuple[int, int],
    segmentation_input_mode: str,
    tvars_snapshot_path: Path | None = None,
) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            labels = np.asarray(data["final_room_label_map"])
            metadata = json.loads(str(data["baseline_metadata_json"]))
        valid = (
            labels.shape == shape
            and str(metadata.get("method")) == method
            and str(metadata.get("segmentation_input_mode"))
            == str(segmentation_input_mode)
        )
        if method == "gomez_incremental":
            valid = bool(
                valid
                and metadata.get("raw_seed_source")
                == "saved_tvars_original_hough_door_seed_map"
                and metadata.get("checkpoint_virtual_laser_recomputed") is False
                and tvars_snapshot_path is not None
                and str(metadata.get("raw_seed_source_snapshot"))
                == str(tvars_snapshot_path)
            )
        return bool(valid)
    except Exception:
        return False


def _episodes_by_uid(index: Mapping[str, Any], *, label: str) -> dict[str, dict[str, Any]]:
    episodes: dict[str, dict[str, Any]] = {}
    for raw_episode in index.get("episodes", []):
        episode = dict(raw_episode)
        uid = str(episode.get("episode_uid") or "")
        if not uid:
            raise ValueError(f"{label} index contains an episode without episode_uid")
        if uid in episodes:
            raise ValueError(f"{label} index contains duplicate episode_uid: {uid}")
        episodes[uid] = episode
    return episodes


def _snapshots_by_event(
    episode: Mapping[str, Any], *, label: str
) -> dict[str, dict[str, Any]]:
    snapshots: dict[str, dict[str, Any]] = {}
    for raw_snapshot in episode.get("snapshots", []):
        snapshot = dict(raw_snapshot)
        event_id = str(snapshot.get("coverage_event_id") or "")
        if not event_id:
            raise ValueError(f"{label} snapshot has no coverage_event_id")
        if event_id in snapshots:
            raise ValueError(f"{label} has duplicate coverage_event_id: {event_id}")
        snapshots[event_id] = snapshot
    return snapshots


def _paired_tvars_snapshot(
    *,
    tvars_episodes: Mapping[str, Mapping[str, Any]],
    episode_uid: str,
    voxroom_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    tvars_episode = tvars_episodes.get(str(episode_uid))
    if tvars_episode is None:
        raise ValueError(
            "Gomez replay requires a paired TVARS episode; missing episode_uid: "
            + str(episode_uid)
        )
    event_id = str(voxroom_snapshot["coverage_event_id"])
    tvars_snapshot = _snapshots_by_event(
        tvars_episode,
        label=f"TVARS episode {episode_uid}",
    ).get(event_id)
    if tvars_snapshot is None:
        raise ValueError(
            "Gomez replay requires a paired TVARS checkpoint; missing "
            f"episode_uid={episode_uid}, coverage_event_id={event_id}"
        )
    voxroom_step = int(voxroom_snapshot["step"])
    tvars_step = int(tvars_snapshot["step"])
    if tvars_step != voxroom_step:
        raise ValueError(
            "paired VoxRoom/TVARS checkpoint step mismatch: "
            f"episode_uid={episode_uid}, coverage_event_id={event_id}, "
            f"voxroom_step={voxroom_step}, tvars_step={tvars_step}"
        )
    tvars_path = Path(str(tvars_snapshot["snapshot_path"]))
    if not tvars_path.is_file():
        raise FileNotFoundError(str(tvars_path))
    return tvars_snapshot


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_index_path = Path(args.voxroom_index).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_index = load_index(source_index_path)
    approved = _approved_uids(
        None if args.approved_summary is None else Path(args.approved_summary).resolve()
    )
    episodes = [
        dict(episode)
        for episode in source_index.get("episodes", [])
        if approved is None or str(episode.get("episode_uid")) in approved
    ]
    episodes.sort(key=lambda episode: str(episode.get("episode_uid")))
    if approved is not None:
        found = {str(episode.get("episode_uid")) for episode in episodes}
        missing = sorted(approved - found)
        if missing:
            raise ValueError(
                "approved episodes missing from VoxRoom index: " + ", ".join(missing[:10])
            )
    if args.max_scenes is not None:
        episodes = episodes[: int(args.max_scenes)]
    if not episodes:
        raise ValueError("no source episodes selected")
    segmentation_input_mode = str(args.segmentation_input_map)

    methods = [str(method) for method in args.methods]
    unknown = sorted(set(methods).difference(PAPER_BASELINES))
    if unknown:
        raise ValueError("methods are not paper segmentation baselines: %s" % unknown)

    tvars_index_path: Path | None = None
    tvars_episodes: dict[str, dict[str, Any]] = {}
    if "gomez_incremental" in methods:
        if args.tvars_index is None:
            raise ValueError(
                "--tvars-index is required for Gomez: its raw seeds must come "
                "from paired saved TVARS checkpoints"
            )
        tvars_index_path = Path(args.tvars_index).resolve()
        tvars_episodes = _episodes_by_uid(
            load_index(tvars_index_path),
            label="TVARS",
        )

    method_manifests: dict[str, dict[str, Any]] = {}
    for method in methods:
        rows: list[dict[str, Any]] = []
        for episode in episodes:
            uid = str(episode["episode_uid"])
            scene_id = str(episode.get("scene_id") or uid)
            snapshots = [dict(item) for item in episode.get("snapshots", [])]
            snapshots.sort(
                key=lambda item: (
                    int(item["step"]),
                    str(item.get("coverage_event_id")) == "final",
                )
            )
            if args.max_checkpoints_per_scene is not None:
                snapshots = snapshots[: int(args.max_checkpoints_per_scene)]
            if not snapshots:
                raise ValueError(f"episode has no snapshots: {uid}")
            expected_outputs = [
                _prediction_path(
                    output_root,
                    method,
                    uid,
                    str(snapshot["coverage_event_id"]),
                )
                for snapshot in snapshots
            ]
            paired_tvars = [
                _paired_tvars_snapshot(
                    tvars_episodes=tvars_episodes,
                    episode_uid=uid,
                    voxroom_snapshot=snapshot,
                )
                if method == "gomez_incremental"
                else None
                for snapshot in snapshots
            ]
            if bool(args.resume) and all(path.is_file() for path in expected_outputs):
                all_valid = True
                for snapshot, output_path, tvars_snapshot in zip(
                    snapshots, expected_outputs, paired_tvars
                ):
                    with np.load(Path(snapshot["snapshot_path"]), allow_pickle=False) as data:
                        shape = np.asarray(data["final_room_label_map"]).shape
                    if not _prediction_is_valid(
                        output_path,
                        method=method,
                        shape=shape,
                        segmentation_input_mode=segmentation_input_mode,
                        tvars_snapshot_path=(
                            None
                            if tvars_snapshot is None
                            else Path(str(tvars_snapshot["snapshot_path"]))
                        ),
                    ):
                        all_valid = False
                        break
                if all_valid:
                    rows.extend(
                        _manifest_row(
                            uid,
                            scene_id,
                            method,
                            snapshot,
                            output_path,
                            "resumed",
                            tvars_snapshot=tvars_snapshot,
                        )
                        for snapshot, output_path, tvars_snapshot in zip(
                            snapshots, expected_outputs, paired_tvars
                        )
                    )
                    continue

            runner = make_runner(method, args)
            runner.start_scene(uid)
            try:
                for snapshot, output_path, tvars_snapshot in zip(
                    snapshots, expected_outputs, paired_tvars
                ):
                    source_path = Path(str(snapshot["snapshot_path"]))
                    if not source_path.is_file():
                        raise FileNotFoundError(str(source_path))
                    arrays = load_npz_arrays(source_path)
                    if method == "gomez_incremental":
                        if tvars_snapshot is None:
                            raise RuntimeError("paired TVARS snapshot unexpectedly missing")
                        tvars_path = Path(str(tvars_snapshot["snapshot_path"]))
                        tvars_arrays = load_npz_arrays(tvars_path)
                        key = "tvars_original_hough_door_seed_map"
                        if key not in tvars_arrays:
                            raise KeyError(
                                f"paired TVARS checkpoint lacks {key}: {tvars_path}"
                            )
                        tvars_raw_seed = np.asarray(tvars_arrays[key], dtype=bool)
                        source_shape = np.asarray(arrays["final_room_label_map"]).shape
                        if tvars_raw_seed.shape != source_shape:
                            raise ValueError(
                                "paired TVARS raw-seed map shape differs from VoxRoom "
                                f"checkpoint: {tvars_path}"
                            )
                        arrays[key] = tvars_raw_seed
                        arrays["tvars_original_raw_seed_source_snapshot"] = np.asarray(
                            str(tvars_path)
                        )
                    arrays[SEGMENTATION_INPUT_MODE_KEY] = np.asarray(
                        segmentation_input_mode
                    )
                    _input_mask, segmentation_input_source_key = (
                        build_segmentation_domain_from_source(arrays)
                    )
                    result = runner.segment_snapshot(source_path, arrays)
                    result_metadata = {
                        **dict(result.metadata),
                        "segmentation_input_mode": segmentation_input_mode,
                        "segmentation_input_source_key": str(
                            segmentation_input_source_key
                        ),
                        "coverage_reference_used_as_segmentation_input": False,
                    }
                    if bool(args.strict_main):
                        assert_main_experiment_metadata(result_metadata, method)
                    _save_prediction(
                        output_path,
                        label_map=result.label_map,
                        metadata=result_metadata,
                        source_snapshot=source_path,
                        event_id=str(snapshot["coverage_event_id"]),
                        step=int(snapshot["step"]),
                        debug_arrays=result.debug_arrays,
                    )
                    rows.append(
                        _manifest_row(
                            uid,
                            scene_id,
                            method,
                            snapshot,
                            output_path,
                            "written",
                            tvars_snapshot=tvars_snapshot,
                        )
                    )
            finally:
                runner.end_scene()

        method_manifest = {
            "schema_version": "voxroom_paper_baseline_replay_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "method": method,
            "source_voxroom_index": str(source_index_path),
            "source_voxroom_index_sha256": _sha256(source_index_path),
            "source_tvars_index": (
                str(tvars_index_path) if method == "gomez_incremental" else None
            ),
            "source_tvars_index_sha256": (
                _sha256(tvars_index_path)
                if method == "gomez_incremental" and tvars_index_path is not None
                else None
            ),
            "approved_summary": args.approved_summary,
            "scene_count": len({str(row["episode_uid"]) for row in rows}),
            "checkpoint_count": len(rows),
            "strict_main": bool(args.strict_main),
            "segmentation_input_mode": segmentation_input_mode,
            "rows": rows,
        }
        write_json_atomic(
            output_root / "manifests" / (method + ".json"), method_manifest
        )
        method_manifests[method] = method_manifest

    combined = {
        "schema_version": "voxroom_paper_baseline_replay_set_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_voxroom_index": str(source_index_path),
        "source_voxroom_index_sha256": _sha256(source_index_path),
        "source_tvars_index": (
            str(tvars_index_path) if tvars_index_path is not None else None
        ),
        "source_tvars_index_sha256": (
            _sha256(tvars_index_path) if tvars_index_path is not None else None
        ),
        "segmentation_input_mode": segmentation_input_mode,
        "coverage_reference_used_as_segmentation_input": False,
        "methods": methods,
        "method_checkpoint_counts": {
            method: int(manifest["checkpoint_count"])
            for method, manifest in method_manifests.items()
        },
        "voxroom_repo_commit": _git_head(Path.cwd()),
        "voxroom_python_source_tree_sha256": source_tree_hash(
            Path.cwd() / "voxroom_online"
        ),
        "external_repositories": {
            "dude": _git_head(Path(args.dude_repo_root)),
            "rose2": _git_head(Path(args.rose2_ros_workspace) / "src" / "ROSE2"),
            "ipa": _git_head(
                Path(args.ipa_ros_workspace) / "src" / "ipa_coverage_planning"
            ),
        },
    }
    write_json_atomic(output_root / "replay_summary.json", combined)
    return combined


def _manifest_row(
    uid: str,
    scene_id: str,
    method: str,
    snapshot: Mapping[str, Any],
    output_path: Path,
    status: str,
    *,
    tvars_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "episode_uid": uid,
        "scene_id": scene_id,
        "method": method,
        "coverage_event_id": str(snapshot["coverage_event_id"]),
        "coverage_event_kind": snapshot.get("coverage_event_kind"),
        "coverage_ratio": snapshot.get("coverage_ratio"),
        "coverage_threshold": snapshot.get("coverage_threshold"),
        "is_last": bool(snapshot.get("is_last")),
        "step": int(snapshot["step"]),
        "source_snapshot": str(snapshot["snapshot_path"]),
        "source_tvars_raw_seed_snapshot": (
            None
            if tvars_snapshot is None
            else str(tvars_snapshot["snapshot_path"])
        ),
        "prediction_npz": str(output_path),
        "status": status,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replay every room-segmentation baseline from the TVARS paper on "
            "saved VoxRoom coverage checkpoints."
        )
    )
    parser.add_argument("--voxroom-index", required=True)
    parser.add_argument(
        "--tvars-index",
        help=(
            "Paired TVARS coverage index. Required when Gomez is selected; "
            "Gomez imports tvars_original_hough_door_seed_map from it."
        ),
    )
    parser.add_argument("--approved-summary")
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--segmentation-input-map",
        choices=SEGMENTATION_INPUT_MODES,
        default="raw_vertical_free",
        help=(
            "Raw map supplied to every replayed paper baseline. "
            "raw_nav_free_no_clearance reads voxel_nav_free_xy."
        ),
    )
    parser.add_argument(
        "--methods", nargs="+", choices=PAPER_BASELINES, default=list(PAPER_BASELINES)
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-main", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fallback-python", action="store_true", default=False)
    parser.add_argument("--allow-baseline-failure", action="store_true", default=False)
    parser.add_argument("--map-resolution-m", type=float, default=0.05)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--max-checkpoints-per-scene", type=int)
    parser.add_argument("--ros-baseline-setup")
    parser.add_argument("--ros-baseline-python")
    parser.add_argument("--dude-ws", required=True)
    parser.add_argument("--rose2-ws")
    parser.add_argument("--ipa-ws")
    parser.add_argument("--active-room-seg-root")
    parser.add_argument("--topology-checkpoint")
    parser.add_argument("--ipa-ros-workspace", required=True)
    parser.add_argument("--rose2-ros-workspace", required=True)
    parser.add_argument(
        "--rose2-launch-file", default="configs/ros/rose2_headless.launch"
    )
    parser.add_argument("--dude-repo-root", required=True)
    parser.add_argument("--dude-concavity-threshold-m", type=float, default=2.5)
    args = parser.parse_args(argv)
    if args.strict_main and args.fallback_python:
        parser.error("--strict-main cannot be combined with --fallback-python")
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
