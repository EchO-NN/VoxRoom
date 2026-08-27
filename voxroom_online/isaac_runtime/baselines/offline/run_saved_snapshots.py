from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..data_contract import load_npz_arrays
from ..mask_io import build_metric_domain_from_source
from ..mask_io import save_baseline_snapshot_npz
from ..snapshot_replay import baseline_snapshot_output_path, iter_scene_dirs, iter_snapshot_paths, limit_snapshots
from ...comparison.metadata_gate import assert_main_experiment_metadata
from .base import BaselineResult, MissingOriginalImplementationError
from .dude_runner import DudeIncrementalRunner
from .fallback_voronoi_width_jump import segment_snapshot_arrays as segment_voronoi_width_jump
from .ipa_runner import IpaRoomSegmentationRunner
from .rose2_runner import Rose2Runner

BASELINE_CHOICES = (
    "dude_incremental",
    "rose2",
    "morphological",
    "distance_transform",
    "voronoi",
    "voronoi_width_jump",
)


class WidthJumpVoronoiRunner:
    def __init__(self, *, map_resolution_m: float | None = None) -> None:
        self.map_resolution_m = map_resolution_m

    def start_scene(self, scene_id: str) -> None:
        return None

    def end_scene(self) -> None:
        return None

    def segment_snapshot(self, snapshot_path: Path, arrays: dict[str, Any]) -> BaselineResult:
        labels, metadata, debug_arrays = segment_voronoi_width_jump(
            snapshot_path,
            arrays,
            default_resolution_m=float(self.map_resolution_m) if self.map_resolution_m is not None else 0.05,
        )
        return BaselineResult(label_map=labels, metadata=metadata, debug_arrays=debug_arrays)


def make_runner(name: str, args: argparse.Namespace):
    ros_setup = getattr(args, "ros_baseline_setup", None)
    ros_python = getattr(args, "ros_baseline_python", None)
    if name == "dude_incremental":
        return DudeIncrementalRunner(
            repo_root=Path(args.dude_repo_root) if args.dude_repo_root else None,
            concavity_threshold_m=float(args.dude_concavity_threshold_m),
            use_incremental=True,
            fallback_python=bool(args.fallback_python),
            map_resolution_m=args.map_resolution_m,
            ros_setup=ros_setup,
            ros_python=ros_python,
            dude_ws=Path(args.dude_ws) if getattr(args, "dude_ws", None) else None,
        )
    if name == "rose2":
        return Rose2Runner(
            ros_workspace=Path(args.rose2_ros_workspace) if args.rose2_ros_workspace else None,
            launch_file=str(args.rose2_launch_file),
            use_fallback_smoke=bool(args.fallback_python),
            map_resolution_m=args.map_resolution_m,
            ros_setup=ros_setup,
            ros_python=ros_python,
        )
    if name in {"morphological", "distance_transform", "voronoi"}:
        return IpaRoomSegmentationRunner(
            algorithm=name,  # type: ignore[arg-type]
            ros_workspace=Path(args.ipa_ros_workspace) if args.ipa_ros_workspace else None,
            fallback_python=bool(args.fallback_python),
            map_resolution_m=args.map_resolution_m,
            ros_setup=ros_setup,
            ros_python=ros_python,
        )
    if name == "voronoi_width_jump":
        return WidthJumpVoronoiRunner(map_resolution_m=args.map_resolution_m)
    raise ValueError(f"unsupported baseline: {name}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    strict_native = bool(getattr(args, "strict_native", False))
    if strict_native and bool(getattr(args, "fallback_python", False)):
        raise ValueError("--strict-native cannot be combined with --fallback-python")
    if strict_native and bool(getattr(args, "allow_baseline_failure", False)):
        raise ValueError("--strict-native cannot be combined with --allow-baseline-failure")
    source_run_root = Path(args.source_run_root)
    output_run_root = Path(args.output_run_root or args.source_run_root)
    rows: list[dict[str, Any]] = []
    for scene_dir in iter_scene_dirs(source_run_root, str(args.scene_glob)):
        scene_id = scene_dir.name
        snapshots = limit_snapshots(
            iter_snapshot_paths(scene_dir, str(args.snapshot_glob)),
            args.max_snapshots_per_scene,
        )
        for baseline_name in list(args.baselines):
            runner = make_runner(str(baseline_name), args)
            runner.start_scene(scene_id)
            try:
                for snapshot_path in snapshots:
                    output_npz = baseline_snapshot_output_path(output_run_root, scene_id, str(baseline_name), snapshot_path)
                    if output_npz.exists() and not bool(args.overwrite):
                        rows.append(_row(scene_id, baseline_name, snapshot_path, output_npz, "skipped_exists"))
                        continue
                    arrays = load_npz_arrays(snapshot_path)
                    try:
                        result = runner.segment_snapshot(snapshot_path, arrays)
                    except Exception as exc:
                        if not bool(args.allow_baseline_failure):
                            raise
                        result = _failure_result(arrays, baseline_name=str(baseline_name), error=exc)
                    if strict_native:
                        _validate_strict_native_result(result, arrays, method=str(baseline_name), snapshot_path=snapshot_path)
                    save_baseline_snapshot_npz(
                        source_npz_path=snapshot_path,
                        output_npz_path=output_npz,
                        baseline_label_map=result.label_map,
                        baseline_name=str(baseline_name),
                        metadata=result.metadata,
                        debug_arrays=result.debug_arrays,
                    )
                    rows.append(_row(scene_id, baseline_name, snapshot_path, output_npz, "written"))
            finally:
                runner.end_scene()
    manifest = {
        "schema_version": 2,
        "source_run_root": str(source_run_root),
        "output_run_root": str(output_run_root),
        "baselines": list(args.baselines),
        "fallback_python": bool(args.fallback_python),
        "strict_native": strict_native,
        "allow_baseline_failure": bool(getattr(args, "allow_baseline_failure", False)),
        "ros": {
            "ros_baseline_setup": getattr(args, "ros_baseline_setup", None),
            "ros_baseline_python": getattr(args, "ros_baseline_python", None),
            "dude_ws": getattr(args, "dude_ws", None),
            "rose2_ws": getattr(args, "rose2_ws", None),
            "ipa_ws": getattr(args, "ipa_ws", None),
            "active_room_seg_root": getattr(args, "active_room_seg_root", None),
            "topology_checkpoint": getattr(args, "topology_checkpoint", None),
        },
        "rows": rows,
    }
    manifest_path = output_run_root / "baseline_replay_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    provenance_out = getattr(args, "provenance_out", None)
    if provenance_out:
        Path(provenance_out).parent.mkdir(parents=True, exist_ok=True)
        Path(provenance_out).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run offline room-segmentation baselines from saved VoxRoom snapshots.")
    parser.add_argument("--source-run-root", required=True)
    parser.add_argument("--output-run-root")
    parser.add_argument("--baselines", nargs="+", choices=BASELINE_CHOICES, default=list(BASELINE_CHOICES))
    parser.add_argument("--scene-glob", default="kujiale_*")
    parser.add_argument("--snapshot-glob", default="roomseg_step_*.npz")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-snapshots-per-scene", type=int)
    parser.add_argument("--allow-baseline-failure", action="store_true", default=False)
    parser.add_argument("--fallback-python", action="store_true", default=False)
    parser.add_argument("--strict-native", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--map-resolution-m", type=float)
    parser.add_argument("--ros-baseline-setup")
    parser.add_argument("--ros-baseline-python")
    parser.add_argument("--dude-ws")
    parser.add_argument("--rose2-ws")
    parser.add_argument("--ipa-ws")
    parser.add_argument("--active-room-seg-root")
    parser.add_argument("--topology-checkpoint")
    parser.add_argument("--ipa-ros-workspace")
    parser.add_argument("--rose2-ros-workspace")
    parser.add_argument("--rose2-launch-file", default="ROSE.launch")
    parser.add_argument("--dude-repo-root")
    parser.add_argument("--dude-concavity-threshold-m", type=float, default=3.0)
    parser.add_argument("--provenance-out")
    args = parser.parse_args(argv)
    try:
        manifest = run(args)
    except MissingOriginalImplementationError as exc:
        parser.exit(1, f"offline baseline failed: {exc}\n")
    print(
        "processed %d snapshot result(s) for %d baseline(s) -> %s"
        % (len(manifest["rows"]), len(manifest["baselines"]), manifest["output_run_root"])
    )
    return 0


def _failure_result(arrays: dict[str, Any], *, baseline_name: str, error: Exception) -> BaselineResult:
    shape = np.asarray(arrays["occupancy_map"]).shape
    return BaselineResult(
        label_map=np.zeros(shape, dtype=np.int32),
        metadata={
            "method": baseline_name,
            "failed": True,
            "error": repr(error),
            "runner_type": "failure_placeholder",
            "main_experiment_allowed": False,
        },
        debug_arrays={},
    )


def _validate_strict_native_result(
    result: BaselineResult,
    arrays: dict[str, Any],
    *,
    method: str,
    snapshot_path: Path,
) -> None:
    assert_main_experiment_metadata(result.metadata, method)
    label_map = np.asarray(result.label_map, dtype=np.int32)
    domain = build_metric_domain_from_source(arrays)
    if bool(np.any(domain)) and not bool(np.any(label_map[domain] > 0)):
        if bool(result.metadata.get("valid_zero_room_output")):
            return
        raise ValueError("%s strict-native result has no positive room labels on non-empty domain: %s" % (method, snapshot_path))


def _row(scene_id: str, baseline_name: str, snapshot_path: Path, output_npz: Path, status: str) -> dict[str, Any]:
    return {
        "scene_id": str(scene_id),
        "baseline": str(baseline_name),
        "source_snapshot": str(snapshot_path),
        "output_npz": str(output_npz),
        "status": str(status),
    }


if __name__ == "__main__":
    raise SystemExit(main())
