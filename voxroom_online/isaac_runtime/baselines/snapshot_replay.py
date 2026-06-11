from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


_STEP_RE = re.compile(r"roomseg_step_(\d+)\.npz$")


def parse_snapshot_step(path: Path) -> int:
    match = _STEP_RE.search(Path(path).name)
    if match is None:
        raise ValueError(f"could not parse roomseg step from {path}")
    return int(match.group(1))


def iter_scene_dirs(source_run_root: Path, scene_glob: str) -> list[Path]:
    root = Path(source_run_root)
    return [
        path
        for path in sorted(root.glob(scene_glob))
        if path.is_dir() and (path / "roomseg_snapshots").is_dir()
    ]


def iter_snapshot_paths(scene_dir: Path, snapshot_glob: str = "roomseg_step_*.npz") -> list[Path]:
    return sorted((Path(scene_dir) / "roomseg_snapshots").glob(snapshot_glob), key=parse_snapshot_step)


def baseline_snapshot_output_path(output_run_root: Path, scene_id: str, baseline: str, source_snapshot: Path) -> Path:
    return Path(output_run_root) / str(scene_id) / "baselines" / str(baseline) / "roomseg_snapshots" / Path(source_snapshot).name


def limit_snapshots(paths: Iterable[Path], max_count: int | None) -> list[Path]:
    items = list(paths)
    if max_count is None or int(max_count) <= 0:
        return items
    return items[: int(max_count)]

