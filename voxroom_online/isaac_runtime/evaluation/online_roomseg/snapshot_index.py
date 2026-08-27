from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .common import now_iso, parse_snapshot_step, slug, write_json_atomic
from .snapshot_io import load_snapshot_arrays


INDEX_SCHEMA_VERSION = "voxroom_online_roomseg_index_v1"


@dataclass(frozen=True)
class SnapshotRecord:
    step: int
    snapshot_path: str
    summary_json: str | None
    navigation_png: str | None
    is_last: bool
    step_source: str


def build_index(
    result_roots: Iterable[Path],
    *,
    snapshot_policy: str = "all",
    require_npz: bool = True,
    allow_missing_png: bool = False,
    validate_npz: bool = True,
) -> dict:
    roots = [Path(root) for root in result_roots]
    if snapshot_policy not in {"all", "last"}:
        raise ValueError("snapshot_policy must be all or last")
    episodes: list[dict] = []
    errors: list[str] = []
    for root in roots:
        if not root.exists():
            errors.append("result_root_missing:%s" % root)
            continue
        for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            snap_dir = scene_dir / "roomseg_snapshots"
            if not snap_dir.exists():
                continue
            try:
                episode = _index_scene_dir(
                    root,
                    scene_dir,
                    snapshot_policy=snapshot_policy,
                    require_npz=require_npz,
                    allow_missing_png=allow_missing_png,
                    validate_npz=validate_npz,
                )
                if episode is not None:
                    episodes.append(episode)
            except Exception as exc:
                errors.append("%s:%s" % (scene_dir, exc))
    if errors:
        raise ValueError("snapshot index failed: " + "; ".join(errors[:20]))
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "created_at": now_iso(),
        "result_roots": [str(root) for root in roots],
        "snapshot_policy": str(snapshot_policy),
        "episodes": episodes,
    }


def build_coverage_index(
    result_roots: Iterable[Path],
    *,
    method: str,
    snapshot_policy: str = "all",
    validate_npz: bool = True,
) -> dict:
    roots = [Path(root) for root in result_roots]
    if snapshot_policy not in {"all", "last"}:
        raise ValueError("snapshot_policy must be all or last")
    artifact_key = {
        "voxroom": "voxroom_snapshot_npz",
        "tvars_original": "tvars_original_snapshot_npz",
    }.get(str(method))
    if artifact_key is None:
        raise ValueError("coverage index method must be voxroom or tvars_original")
    episodes: list[dict] = []
    manifests: list[Path] = []
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"coverage result root is missing: {root}")
        manifests.extend(sorted(root.rglob("roomseg_coverage_eval/manifest.json")))
        if root.name == "roomseg_coverage_eval" and (root / "manifest.json").is_file():
            manifests.append(root / "manifest.json")
    unique_manifests = sorted(set(path.resolve() for path in manifests))
    for manifest_path in unique_manifests:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("schema_version") != "voxroom_roomseg_coverage_eval_v1":
            raise ValueError(f"unsupported coverage manifest: {manifest_path}")
        complete_events = [
            event
            for event in manifest.get("events", [])
            if event.get("status") == "complete"
        ]
        terminal = [event for event in complete_events if event.get("event_id") == "final"]
        if len(terminal) != 1:
            raise ValueError(f"coverage manifest must contain one complete final event: {manifest_path}")
        snapshots: list[dict] = []
        for event in complete_events:
            artifact = event.get("artifacts", {}).get(artifact_key)
            if not artifact:
                raise KeyError(
                    f"coverage event {event.get('event_id')} lacks {artifact_key}: {manifest_path}"
                )
            npz_path = Path(str(artifact))
            if not npz_path.is_file():
                raise FileNotFoundError(str(npz_path))
            if validate_npz:
                if artifact_key == "tvars_original_snapshot_npz":
                    source_artifact = event.get("artifacts", {}).get(
                        "voxroom_snapshot_npz"
                    )
                    if not source_artifact:
                        raise KeyError(
                            f"coverage event {event.get('event_id')} lacks shared VoxRoom voxel snapshot"
                        )
                    source_path = Path(str(source_artifact))
                    if not source_path.is_file():
                        raise FileNotFoundError(str(source_path))
                    source_snapshot = load_snapshot_arrays(source_path)
                    _validate_coverage_snapshot_contract(
                        source_path,
                        snapshot_shape=source_snapshot.shape,
                        event=event,
                        require_full_voxel=True,
                    )
                snapshot = load_snapshot_arrays(npz_path)
                _validate_coverage_snapshot_contract(
                    npz_path,
                    snapshot_shape=snapshot.shape,
                    event=event,
                    require_full_voxel=(
                        artifact_key == "voxroom_snapshot_npz"
                    ),
                )
            summary_path = npz_path.with_suffix(".summary.json")
            nav_path = npz_path.with_suffix(".navigation_room_masks.png")
            snapshots.append(
                {
                    "step": int(event["step"]),
                    "snapshot_path": str(npz_path),
                    "summary_json": str(summary_path) if summary_path.is_file() else None,
                    "navigation_png": str(nav_path) if nav_path.is_file() else None,
                    "is_last": event.get("event_id") == "final",
                    "step_source": "coverage_manifest",
                    "coverage_event_id": str(event["event_id"]),
                    "coverage_event_kind": str(event["event_kind"]),
                    "coverage_ratio": float(event["coverage_ratio"]),
                    "coverage_threshold": event.get("threshold"),
                }
            )
        snapshots.sort(
            key=lambda item: (
                int(item["step"]),
                item["coverage_event_id"] == "final",
            )
        )
        last = next(item for item in snapshots if bool(item["is_last"]))
        kept = [last] if snapshot_policy == "last" else snapshots
        run_dir = manifest_path.parent.parent
        scene_id = str(manifest.get("scene_id") or run_dir.name)
        run_name = run_dir.parent.name
        episode_id = manifest.get("episode_id")
        # Both methods intentionally share one episode UID. This lets one
        # approved final-step annotation and its backprojected GT evaluate the
        # paired VoxRoom and TVARS snapshots without duplicating annotations.
        uid_parts = [run_name, scene_id, run_dir.name]
        if episode_id is not None:
            uid_parts.append(str(episode_id))
        episode_uid = slug("__".join(uid_parts))
        episodes.append(
            {
                "episode_uid": episode_uid,
                "run_name": run_name,
                "scene_id": scene_id,
                "scene_dir": str(run_dir),
                "results_jsonl": None,
                "episode_id": episode_id,
                "final_reported_step": int(last["step"]),
                "last_snapshot_step": int(last["step"]),
                "last_snapshot_path": str(last["snapshot_path"]),
                "step_delta_to_reported_final": 0,
                "step_reverse_status": "coverage_manifest_final",
                "result_row_diagnostics": {},
                "coverage_manifest": str(manifest_path),
                "coverage_method": str(method),
                "snapshots": kept,
            }
        )
    if not episodes:
        raise FileNotFoundError("no roomseg_coverage_eval/manifest.json found")
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "created_at": now_iso(),
        "result_roots": [str(root) for root in roots],
        "snapshot_policy": str(snapshot_policy),
        "coverage_method": str(method),
        "episodes": episodes,
    }


def _validate_coverage_snapshot_contract(
    path: Path,
    *,
    snapshot_shape: tuple[int, int],
    event: dict,
    require_full_voxel: bool,
) -> None:
    required = {
        "roomseg_eval_reference_explorable_mask",
        "roomseg_eval_explored_reference_mask",
        "roomseg_eval_event_id",
        "roomseg_eval_event_kind",
        "roomseg_eval_coverage_ratio",
        "roomseg_eval_explored_cells",
        "roomseg_eval_total_explorable_cells",
    }
    if require_full_voxel:
        required.update(
            {
                "voxel_occupancy_state_zyx",
                "voxel_occupancy_log_odds_zyx",
                "voxel_sensor_range_count_zyx",
                "voxel_occupancy_z_centers_m",
            }
        )
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(
                f"coverage snapshot {path} is missing required arrays: {missing}"
            )
        reference = np.asarray(
            data["roomseg_eval_reference_explorable_mask"], dtype=bool
        )
        explored = np.asarray(
            data["roomseg_eval_explored_reference_mask"], dtype=bool
        )
        if reference.shape != snapshot_shape or explored.shape != snapshot_shape:
            raise ValueError("coverage reference masks do not match snapshot shape")
        if not np.any(reference):
            raise ValueError("coverage reference mask is empty")
        if np.any(explored & ~reference):
            raise ValueError("coverage explored mask lies outside fixed reference")
        explored_cells = int(
            np.asarray(data["roomseg_eval_explored_cells"]).reshape(())
        )
        total_cells = int(
            np.asarray(data["roomseg_eval_total_explorable_cells"]).reshape(())
        )
        ratio = float(
            np.asarray(data["roomseg_eval_coverage_ratio"]).reshape(())
        )
        event_id = str(np.asarray(data["roomseg_eval_event_id"]).reshape(()))
        event_kind = str(np.asarray(data["roomseg_eval_event_kind"]).reshape(()))
    measured_explored = int(np.count_nonzero(explored))
    measured_total = int(np.count_nonzero(reference))
    measured_ratio = float(measured_explored / measured_total)
    if explored_cells != measured_explored or total_cells != measured_total:
        raise ValueError("coverage snapshot cell counts do not match saved masks")
    if not np.isclose(ratio, measured_ratio, atol=1.0e-12, rtol=0.0):
        raise ValueError("coverage snapshot ratio does not match saved masks")
    if event_id != str(event.get("event_id")):
        raise ValueError("coverage snapshot event id differs from manifest")
    if event_kind != str(event.get("event_kind")):
        raise ValueError("coverage snapshot event kind differs from manifest")
    if int(event.get("explored_cells")) != measured_explored:
        raise ValueError("coverage manifest explored count differs from snapshot")
    if int(event.get("total_explorable_cells")) != measured_total:
        raise ValueError("coverage manifest total count differs from snapshot")
    if not np.isclose(
        float(event.get("coverage_ratio")),
        measured_ratio,
        atol=1.0e-12,
        rtol=0.0,
    ):
        raise ValueError("coverage manifest ratio differs from snapshot")


def write_index(index: dict, out: Path) -> None:
    write_json_atomic(Path(out), index)


def load_index(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    if index.get("schema_version") != INDEX_SCHEMA_VERSION:
        raise ValueError("unsupported index schema: %s" % index.get("schema_version"))
    return index


def _index_scene_dir(
    root: Path,
    scene_dir: Path,
    *,
    snapshot_policy: str,
    require_npz: bool,
    allow_missing_png: bool,
    validate_npz: bool,
) -> dict | None:
    snap_dir = scene_dir / "roomseg_snapshots"
    npz_paths = sorted(snap_dir.glob("roomseg_step_*.npz"))
    if require_npz and not npz_paths:
        raise FileNotFoundError("no roomseg_step_*.npz in %s" % snap_dir)
    if not npz_paths:
        return None
    snapshots: list[dict] = []
    for npz_path in npz_paths:
        summary_json = npz_path.with_suffix(".summary.json")
        nav_png = npz_path.with_suffix(".navigation_room_masks.png")
        step, step_source = parse_snapshot_step(npz_path, summary_json)
        if step is None:
            raise ValueError("could not parse step for %s" % npz_path)
        if not allow_missing_png and not nav_png.exists():
            raise FileNotFoundError("missing navigation png: %s" % nav_png)
        if validate_npz:
            load_snapshot_arrays(npz_path)
        snapshots.append(
            {
                "step": int(step),
                "snapshot_path": str(npz_path),
                "summary_json": str(summary_json) if summary_json.exists() else None,
                "navigation_png": str(nav_png) if nav_png.exists() else None,
                "is_last": False,
                "step_source": str(step_source),
            }
        )
    snapshots.sort(key=lambda item: int(item["step"]))
    results_jsonl = scene_dir / "results.jsonl"
    final_row, result_diag = _read_last_results_row(results_jsonl)
    final_reported_step = _final_step_from_row(final_row)
    last, status = _select_last_snapshot(snapshots, final_reported_step)
    for snap in snapshots:
        snap["is_last"] = bool(snap["snapshot_path"] == last["snapshot_path"])
    scene_id = scene_dir.name
    run_name = root.name
    episode_id = None if final_row is None else final_row.get("episode_id")
    uid_parts = [run_name, scene_id]
    if episode_id is not None:
        uid_parts.append(str(episode_id))
    episode_uid = slug("__".join(uid_parts))
    kept_snapshots = [last] if snapshot_policy == "last" else snapshots
    last_step = int(last["step"])
    return {
        "episode_uid": episode_uid,
        "run_name": run_name,
        "scene_id": scene_id,
        "scene_dir": str(scene_dir),
        "results_jsonl": str(results_jsonl) if results_jsonl.exists() else None,
        "episode_id": episode_id,
        "final_reported_step": None if final_reported_step is None else int(final_reported_step),
        "last_snapshot_step": last_step,
        "last_snapshot_path": str(last["snapshot_path"]),
        "step_delta_to_reported_final": None if final_reported_step is None else int(last_step - int(final_reported_step)),
        "step_reverse_status": status,
        "result_row_diagnostics": result_diag,
        "snapshots": kept_snapshots,
    }


def _read_last_results_row(path: Path) -> tuple[dict | None, dict]:
    if not path.exists():
        return None, {"row_count": 0, "episode_ids": []}
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    ids = [row.get("episode_id") for row in rows if row.get("episode_id") is not None]
    return (rows[-1] if rows else None), {"row_count": int(len(rows)), "episode_ids": ids}


def _final_step_from_row(row: dict | None) -> int | None:
    if not row:
        return None
    for key in ("steps", "num_steps", "step"):
        value = row.get(key)
        if value is not None:
            try:
                return int(value)
            except Exception:
                pass
    return None


def _select_last_snapshot(snapshots: list[dict], final_reported_step: int | None) -> tuple[dict, str]:
    if not snapshots:
        raise ValueError("no snapshots")
    if final_reported_step is not None:
        candidates = [item for item in snapshots if int(item["step"]) <= int(final_reported_step)]
        if candidates:
            last = max(candidates, key=lambda item: int(item["step"]))
            status = "exact" if int(last["step"]) == int(final_reported_step) else "within_final"
            return last, status
        nearest = min(snapshots, key=lambda item: abs(int(item["step"]) - int(final_reported_step)))
        return nearest, "nearest_after_final"
    return max(snapshots, key=lambda item: int(item["step"])), "snapshot_only"
