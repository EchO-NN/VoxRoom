from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any
from typing import Sequence


def link_baseline_run_root(
    *,
    source_run_root: Path,
    baseline: str,
    linked_run_root: Path,
    copy: bool = False,
    overwrite: bool = False,
    allow_non_main: bool = False,
) -> list[Path]:
    source_run_root = Path(source_run_root)
    linked_run_root = Path(linked_run_root)
    linked_run_root.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for scene_dir in sorted(path for path in source_run_root.iterdir() if path.is_dir()):
        baseline_snapshots = scene_dir / "baselines" / baseline / "roomseg_snapshots"
        if not baseline_snapshots.is_dir():
            continue
        if not bool(allow_non_main):
            _validate_main_experiment_snapshots(baseline_snapshots, baseline=baseline)
        dst_scene = linked_run_root / scene_dir.name
        dst_snapshots = dst_scene / "roomseg_snapshots"
        dst_scene.mkdir(parents=True, exist_ok=True)
        if dst_snapshots.exists() or dst_snapshots.is_symlink():
            if not overwrite:
                created.append(dst_snapshots)
                continue
            if dst_snapshots.is_symlink() or dst_snapshots.is_file():
                dst_snapshots.unlink()
            else:
                shutil.rmtree(dst_snapshots)
        if copy:
            shutil.copytree(baseline_snapshots, dst_snapshots)
        else:
            os.symlink(os.path.relpath(baseline_snapshots, dst_scene), dst_snapshots)
        for sidecar in ("results.jsonl", "command.txt"):
            src = scene_dir / sidecar
            dst = dst_scene / sidecar
            if src.exists() and not dst.exists():
                try:
                    os.symlink(os.path.relpath(src, dst_scene), dst)
                except OSError:
                    shutil.copy2(src, dst)
        created.append(dst_snapshots)
    return created


def _validate_main_experiment_snapshots(snapshot_dir: Path, *, baseline: str) -> None:
    problems: list[str] = []
    checked = 0
    for npz_path in sorted(Path(snapshot_dir).glob("roomseg_step_*.npz")):
        checked += 1
        metadata = _load_baseline_metadata(npz_path)
        if bool(metadata.get("failed", False)):
            problems.append(f"{npz_path}: failed=true")
        runner_type = str(metadata.get("runner_type", "")).lower()
        fallback_scope = str(metadata.get("fallback_scope", "")).lower()
        allowed_usage = str(metadata.get("allowed_usage", "")).lower()
        if metadata.get("main_experiment_allowed") is False:
            problems.append(f"{npz_path}: main_experiment_allowed=false")
        if "fallback" in runner_type or "fallback" in fallback_scope or "smoke" in allowed_usage:
            problems.append(f"{npz_path}: non-main runner_type/scope/usage")
        if len(problems) >= 10:
            break
    if checked == 0:
        raise ValueError(f"no baseline snapshots found for {baseline}: {snapshot_dir}")
    if problems:
        raise ValueError(
            "refusing to link non-main baseline outputs for %s; pass --allow-non-main only for smoke/debug evaluation:\n%s"
            % (baseline, "\n".join(problems))
        )


def _load_baseline_metadata(npz_path: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(npz_path, allow_pickle=False) as data:
        if "baseline_metadata_json" not in data.files:
            return {}
        raw = data["baseline_metadata_json"]
    try:
        return dict(json.loads(str(raw)))
    except Exception:
        return {}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create evaluator-friendly method-level roots for baseline snapshots.")
    parser.add_argument("--source-run-root", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--linked-run-root", required=True)
    parser.add_argument("--copy", action="store_true", help="Copy snapshots instead of symlinking.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-non-main",
        action="store_true",
        help="Allow fallback/smoke/failure-marked outputs to be linked. Use only for debugging, not main paper metrics.",
    )
    args = parser.parse_args(argv)
    created = link_baseline_run_root(
        source_run_root=Path(args.source_run_root),
        baseline=str(args.baseline),
        linked_run_root=Path(args.linked_run_root),
        copy=bool(args.copy),
        overwrite=bool(args.overwrite),
        allow_non_main=bool(args.allow_non_main),
    )
    print("linked %d scene(s) for %s -> %s" % (len(created), args.baseline, args.linked_run_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
