from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from voxroom_online.isaac_runtime.baselines.mask_io import enforce_room_mask_contract
from voxroom_online.isaac_runtime.comparison.metadata_gate import assert_main_experiment_metadata


DEFAULT_METHODS = ("topology_visual_active", "dude_incremental", "rose2", "morphological", "distance_transform", "voronoi")


def validate_all_baselines(
    *,
    run_root: Path,
    methods: Sequence[str] = DEFAULT_METHODS,
    fail_on_non_main: bool = True,
    require_same_snapshot_set: bool = True,
) -> dict[str, Any]:
    run_root = Path(run_root)
    scenes: list[dict[str, Any]] = []
    problems: list[str] = []
    for scene_dir in sorted(p for p in run_root.iterdir() if p.is_dir()):
        source_snaps = sorted((scene_dir / "roomseg_snapshots").glob("roomseg_step_*.npz"))
        if not source_snaps:
            continue
        source_names = {p.name for p in source_snaps}
        method_rows: dict[str, Any] = {}
        for method in methods:
            snap_dir = scene_dir / "baselines" / str(method) / "roomseg_snapshots"
            snaps = sorted(snap_dir.glob("roomseg_step_*.npz")) if snap_dir.is_dir() else []
            method_rows[str(method)] = {"snapshot_count": len(snaps)}
            if require_same_snapshot_set and {p.name for p in snaps} != source_names:
                problems.append("%s/%s snapshot set differs from source" % (scene_dir.name, method))
                continue
            for snap in snaps:
                try:
                    _validate_snapshot(snap, method=str(method), fail_on_non_main=fail_on_non_main)
                except Exception as exc:
                    problems.append("%s: %s" % (snap, exc))
                    if len(problems) >= 20:
                        break
        scenes.append({"scene_id": scene_dir.name, "source_snapshot_count": len(source_snaps), "methods": method_rows})
        if len(problems) >= 20:
            break
    report = {"run_root": str(run_root), "methods": list(methods), "scene_count": len(scenes), "scenes": scenes, "problems": problems}
    if problems:
        raise ValueError("baseline validation failed:\n" + "\n".join(problems[:20]))
    return report


def _validate_snapshot(path: Path, *, method: str, fail_on_non_main: bool) -> None:
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: np.asarray(data[k]).copy() for k in data.files}
    if str(arrays.get("baseline_name", "")) != method:
        raise ValueError("baseline_name mismatch")
    raw_meta = arrays.get("baseline_metadata_json")
    metadata = json.loads(str(raw_meta)) if raw_meta is not None else {}
    if fail_on_non_main:
        assert_main_experiment_metadata(metadata, method)
    final = np.asarray(arrays["final_room_label_map"], dtype=np.int32)
    contracted = enforce_room_mask_contract(final, arrays, clip_to_eval_domain=True)
    if not np.array_equal(final, contracted):
        raise ValueError("final_room_label_map violates label/domain contract")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate baseline snapshots before evaluator compute.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--fail-on-non-main", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-same-snapshot-set", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    report = validate_all_baselines(
        run_root=Path(args.run_root),
        methods=[str(v) for v in args.methods],
        fail_on_non_main=bool(args.fail_on_non_main),
        require_same_snapshot_set=bool(args.require_same_snapshot_set),
    )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("validated %d scene(s)" % int(report["scene_count"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

