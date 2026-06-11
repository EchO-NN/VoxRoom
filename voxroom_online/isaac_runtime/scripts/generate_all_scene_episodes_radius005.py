from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from voxroom_online.isaac_runtime.dataset.episode_generator import select_longest_instance_astar_episode, write_jsonl
from voxroom_online.isaac_runtime.visualization.draw_map import save_map_png


def _scene_dirs(preprocessed_dir: Path, scene_ids: list[str] | None) -> list[Path]:
    out = [path for path in sorted(preprocessed_dir.glob("kujiale_*")) if path.is_dir()]
    if scene_ids:
        wanted = set(scene_ids)
        out = [path for path in out if path.name in wanted]
    return out


def _candidate_without_heavy_arrays(candidate: dict) -> dict:
    return {key: value for key, value in candidate.items() if key not in {"goal_regions_grid", "path_grid"}}


def _selection_profiles(args: argparse.Namespace) -> list[dict[str, float | str]]:
    return [
        {
            "name": "requested",
            "min_start_object_clearance_m": float(args.min_start_object_clearance_m),
            "min_start_goal_bbox_distance_m": float(args.min_start_goal_bbox_distance_m),
            "min_goal_clearance_m": float(args.min_goal_clearance_m),
            "min_planning_clearance_m": float(args.min_planning_clearance_m),
        },
        {
            "name": "relaxed_clearance_040",
            "min_start_object_clearance_m": min(float(args.min_start_object_clearance_m), 0.40),
            "min_start_goal_bbox_distance_m": min(float(args.min_start_goal_bbox_distance_m), 1.00),
            "min_goal_clearance_m": min(float(args.min_goal_clearance_m), 0.40),
            "min_planning_clearance_m": min(float(args.min_planning_clearance_m), 0.0),
        },
        {
            "name": "relaxed_clearance_020",
            "min_start_object_clearance_m": min(float(args.min_start_object_clearance_m), 0.20),
            "min_start_goal_bbox_distance_m": min(float(args.min_start_goal_bbox_distance_m), 0.50),
            "min_goal_clearance_m": 0.0,
            "min_planning_clearance_m": 0.0,
        },
        {
            "name": "no_extra_clearance",
            "min_start_object_clearance_m": 0.0,
            "min_start_goal_bbox_distance_m": 0.0,
            "min_goal_clearance_m": 0.0,
            "min_planning_clearance_m": 0.0,
        },
    ]


def _select_with_fallbacks(scene_dir: Path, args: argparse.Namespace) -> tuple[dict, list[dict], dict]:
    errors: list[dict[str, str]] = []
    for profile in _selection_profiles(args):
        try:
            episode, candidates = select_longest_instance_astar_episode(
                str(scene_dir),
                seed=int(args.seed),
                success_distance_m=float(args.success_distance_m),
                min_start_goal_distance_m=float(args.min_start_goal_distance_m),
                max_start_goal_distance_m=float(args.max_start_goal_distance_m),
                min_start_object_clearance_m=float(profile["min_start_object_clearance_m"]),
                min_start_goal_bbox_distance_m=float(profile["min_start_goal_bbox_distance_m"]),
                min_goal_clearance_m=float(profile["min_goal_clearance_m"]),
                min_planning_clearance_m=float(profile["min_planning_clearance_m"]),
                robot_spawn_height_m=float(args.robot_spawn_height_m),
                top_k=int(args.top_k),
                exclude_categories=["doorsill", "door", "door_handle"],
            )
            return episode, candidates, {"profile": profile, "fallback_errors": errors}
        except Exception as exc:
            errors.append({"profile": str(profile["name"]), "error_type": type(exc).__name__, "error": str(exc)})
    raise ValueError("No valid episode after fallback profiles: %s" % errors)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed-dir", default="data/interioragent_preprocessed_radius005")
    parser.add_argument("--out-dir", default="data/interioragent_episodes/radius005_all_scenes")
    parser.add_argument("--combined-out", default=None)
    parser.add_argument("--report-out", default=None)
    parser.add_argument("--file-list-out", default=None)
    parser.add_argument("--debug-map-dir", default=None)
    parser.add_argument("--scene-id", action="append", default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--success-distance-m", type=float, default=1.0)
    parser.add_argument("--min-start-goal-distance-m", type=float, default=0.0)
    parser.add_argument("--max-start-goal-distance-m", type=float, default=100.0)
    parser.add_argument("--min-start-object-clearance-m", type=float, default=0.80)
    parser.add_argument("--min-start-goal-bbox-distance-m", type=float, default=1.5)
    parser.add_argument("--min-goal-clearance-m", type=float, default=0.80)
    parser.add_argument("--min-planning-clearance-m", type=float, default=0.0)
    parser.add_argument("--robot-spawn-height-m", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--require-all", action="store_true", default=False)
    args = parser.parse_args(argv)

    preprocessed_dir = Path(args.preprocessed_dir)
    out_dir = Path(args.out_dir)
    combined_out = Path(args.combined_out) if args.combined_out else out_dir / "all_scenes_radius005.jsonl"
    report_out = Path(args.report_out) if args.report_out else out_dir / "all_scenes_radius005_report.json"
    file_list_out = Path(args.file_list_out) if args.file_list_out else out_dir / "episode_files.txt"
    debug_map_dir = Path(args.debug_map_dir) if args.debug_map_dir else out_dir / "debug_maps"

    if not preprocessed_dir.exists():
        print("Error: preprocessed directory not found: %s" % preprocessed_dir)
        return 2
    scenes = _scene_dirs(preprocessed_dir, args.scene_id)
    if not scenes:
        print("Error: no preprocessed kujiale_* scenes found in %s" % preprocessed_dir)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    debug_map_dir.mkdir(parents=True, exist_ok=True)
    episodes: list[dict] = []
    report_rows: list[dict] = []
    episode_files: list[str] = []
    for scene_dir in scenes:
        scene_id = scene_dir.name
        episode_path = out_dir / ("%s.jsonl" % scene_id)
        report_path = out_dir / ("%s.report.json" % scene_id)
        debug_map_path = debug_map_dir / ("%s.png" % scene_id)
        print("[episodes] %s: selecting longest A* episode" % scene_id, flush=True)
        try:
            episode, candidates, selection_debug = _select_with_fallbacks(scene_dir, args)
            episode.setdefault("metadata", {})["robot_radius_m"] = 0.05
            episode.setdefault("metadata", {})["episode_set"] = "radius005_all_scenes"
            episode.setdefault("metadata", {})["selection_profile"] = dict(selection_debug["profile"])
            episode.setdefault("metadata", {})["selection_fallback_errors"] = list(selection_debug["fallback_errors"])
            write_jsonl([episode], str(episode_path))
            occupancy = np.load(scene_dir / "occupancy.npy")
            navigable = np.load(scene_dir / "navigable.npy").astype(bool)
            selected = candidates[0]
            save_map_png(
                str(debug_map_path),
                occupancy,
                navigable,
                start=tuple(int(v) for v in selected["start_grid"]),
                goals=[tuple(int(v) for v in cell) for cell in selected["goal_regions_grid"]],
                path_cells=[tuple(int(v) for v in cell) for cell in selected["path_grid"]],
            )
            scene_report = {
                "scene_id": scene_id,
                "ok": True,
                "episode_file": str(episode_path),
                "debug_map": str(debug_map_path),
                "selected_episode_id": episode["episode_id"],
                "selected_goal_category": episode["goal_category"],
                "selected_goal_instance_id": episode["goal_instance_ids"][0],
                "selected_start_pose_world": episode["start_pose_world"],
                "selected_start_grid": episode["start_grid"],
                "selected_shortest_path_distance_m": episode["shortest_path_distance_m"],
                "selected_start_object_clearance_m": episode["start_object_clearance_m"],
                "selected_start_goal_bbox_distance_m": episode["start_goal_bbox_distance_m"],
                "selection_profile": dict(selection_debug["profile"]),
                "selection_fallback_errors": list(selection_debug["fallback_errors"]),
                "evaluated_instance_count": len(candidates),
                "top_candidates": [_candidate_without_heavy_arrays(item) for item in candidates[: max(1, int(args.top_k))]],
            }
            report_path.write_text(json.dumps(scene_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            episodes.append(episode)
            report_rows.append(scene_report)
            episode_files.append(str(episode_path))
            print("[episodes] %s: wrote %s" % (scene_id, episode_path), flush=True)
        except Exception as exc:
            scene_report = {
                "scene_id": scene_id,
                "ok": False,
                "episode_file": str(episode_path),
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
            report_path.write_text(json.dumps(scene_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            report_rows.append(scene_report)
            print("[episodes] %s: failed: %s" % (scene_id, exc), flush=True)

    write_jsonl(episodes, str(combined_out))
    report = {
        "preprocessed_dir": str(preprocessed_dir),
        "out_dir": str(out_dir),
        "combined_episode_file": str(combined_out),
        "episode_file_list": str(file_list_out),
        "scene_count": len(scenes),
        "success_count": len(episodes),
        "failure_count": len([row for row in report_rows if not row.get("ok")]),
        "robot_radius_m": 0.05,
        "rows": report_rows,
    }
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    file_list_out.write_text("\n".join(episode_files) + ("\n" if episode_files else ""), encoding="utf-8")
    print("[episodes] wrote %d/%d scenes to %s" % (len(episodes), len(scenes), out_dir), flush=True)
    if not episodes:
        return 1
    if bool(args.require_all) and len(episodes) != len(scenes):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
