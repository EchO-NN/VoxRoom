from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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

