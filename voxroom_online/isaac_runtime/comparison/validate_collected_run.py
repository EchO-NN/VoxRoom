from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from voxroom_online.isaac_runtime.baselines.mask_io import enforce_room_mask_contract


def validate_collected_run(
    *,
    run_root: Path,
    require_topology: bool = False,
    require_active_stream: bool = False,
    require_panorama: bool = False,
    fail_on_placeholder: bool = True,
    fail_on_fallback: bool = True,
) -> dict[str, Any]:
    run_root = Path(run_root)
    problems: list[str] = []
    scenes: list[dict[str, Any]] = []
    for scene_dir in sorted(p for p in run_root.iterdir() if p.is_dir()):
        source_snaps = sorted((scene_dir / "roomseg_snapshots").glob("roomseg_step_*.npz"))
        if not source_snaps:
            continue
        row: dict[str, Any] = {"scene_id": scene_dir.name, "source_snapshot_count": len(source_snaps)}
        if require_active_stream:
            manifest = scene_dir / "active_stream" / "stream_manifest.jsonl"
            if not manifest.exists():
                problems.append("%s missing active_stream/stream_manifest.jsonl" % scene_dir.name)
            row["active_stream_manifest"] = str(manifest)
        if require_panorama:
            pano_root = scene_dir / "active_stream" / "panoramas"
            views = list(pano_root.glob("step_*/view_*.npz")) if pano_root.exists() else []
            if len(views) < 12:
                problems.append("%s has fewer than 12 panorama views" % scene_dir.name)
            row["panorama_view_count"] = len(views)
        if require_topology:
            topo_dir = scene_dir / "baselines" / "topology_visual_active" / "roomseg_snapshots"
            topo_snaps = sorted(topo_dir.glob("roomseg_step_*.npz")) if topo_dir.is_dir() else []
            if {p.name for p in topo_snaps} != {p.name for p in source_snaps}:
                problems.append("%s topology snapshot set differs from source" % scene_dir.name)
            for snap in topo_snaps:
                try:
                    _validate_topology_snapshot(
                        snap,
                        fail_on_placeholder=fail_on_placeholder,
                        fail_on_fallback=fail_on_fallback,
                        require_panorama=require_panorama,
                    )
                except Exception as exc:
                    problems.append("%s: %s" % (snap, exc))
                    if len(problems) >= 20:
                        break
            row["topology_snapshot_count"] = len(topo_snaps)
        scenes.append(row)
        if len(problems) >= 20:
            break
    if not scenes:
        problems.append("no scenes with source snapshots under %s" % run_root)
    report = {"run_root": str(run_root), "scene_count": len(scenes), "scenes": scenes, "problems": problems}
    if problems:
        raise ValueError("collected run validation failed:\n" + "\n".join(problems[:20]))
    return report


def _validate_topology_snapshot(
    path: Path,
    *,
    fail_on_placeholder: bool,
    fail_on_fallback: bool,
    require_panorama: bool,
) -> None:
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: np.asarray(data[k]).copy() for k in data.files}
    if str(arrays.get("baseline_name", "")) != "topology_visual_active":
        raise ValueError("baseline_name is not topology_visual_active")
    metadata = json.loads(str(arrays.get("baseline_metadata_json", "{}")))
    if metadata.get("main_experiment_allowed") is not True:
        raise ValueError("topology main_experiment_allowed is not true")
    if metadata.get("runner_type") not in {"original_active_room_segmentation", "original_repo_adapter"}:
        raise ValueError("topology runner_type is not original")
    for key in ("door_detector_available", "detector_adapter_verified", "projection_verified"):
        if metadata.get(key) is not True:
            raise ValueError("topology %s is not true" % key)
    if metadata.get("door_detector") != "original_detr":
        raise ValueError("topology door_detector is not original_detr")
    if metadata.get("policy_control") != "never":
        raise ValueError("topology policy_control is not never")
    if require_panorama and int(metadata.get("panorama_views_saved", 0) or 0) < 12:
        raise ValueError("topology panorama_views_saved < 12")
    text = json.dumps(metadata, ensure_ascii=False).lower()
    if fail_on_placeholder and "placeholder_not_canonical" in text:
        raise ValueError("topology metadata contains placeholder_not_canonical")
    if fail_on_fallback and ("fallback" in text or "smoke" in text):
        raise ValueError("topology metadata contains fallback/smoke")
    final = np.asarray(arrays["final_room_label_map"], dtype=np.int32)
    contracted = enforce_room_mask_contract(final, arrays, clip_to_eval_domain=True)
    if not np.array_equal(final, contracted):
        raise ValueError("topology final_room_label_map violates label/domain contract")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate collected VoxRoom plus live topology baseline run.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--require-topology", action="store_true")
    parser.add_argument("--require-active-stream", action="store_true")
    parser.add_argument("--require-panorama", action="store_true")
    parser.add_argument("--fail-on-placeholder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-on-fallback", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    report = validate_collected_run(
        run_root=Path(args.run_root),
        require_topology=bool(args.require_topology),
        require_active_stream=bool(args.require_active_stream),
        require_panorama=bool(args.require_panorama),
        fail_on_placeholder=bool(args.fail_on_placeholder),
        fail_on_fallback=bool(args.fail_on_fallback),
    )
    print("validated collected run with %d scene(s)" % int(report["scene_count"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

