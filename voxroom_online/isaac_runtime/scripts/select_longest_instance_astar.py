from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from voxroom_online.isaac_runtime.dataset.episode_generator import select_longest_instance_astar_episode, write_jsonl
from voxroom_online.isaac_runtime.visualization.draw_map import save_map_png


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed-dir", default="data/interioragent_preprocessed")
    parser.add_argument("--scene-id", default="kujiale_0031")
    parser.add_argument("--scene-dir", default=None)
    parser.add_argument("--out", default="data/interioragent_episodes/longest_instance_astar.jsonl")
    parser.add_argument("--report-out", default="data/interioragent_episodes/longest_instance_astar_report.json")
    parser.add_argument("--debug-map", default="debug/longest_instance_astar.png")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--success-distance-m", type=float, default=1.0)
    parser.add_argument("--min-start-goal-distance-m", type=float, default=0.0)
    parser.add_argument("--max-start-goal-distance-m", type=float, default=100.0)
    parser.add_argument("--min-start-object-clearance-m", type=float, default=0.80)
    parser.add_argument("--min-start-goal-bbox-distance-m", type=float, default=1.5)
    parser.add_argument("--min-goal-clearance-m", type=float, default=0.80)
    parser.add_argument("--min-planning-clearance-m", type=float, default=0.0)
    parser.add_argument("--exclude-category", action="append", default=["doorsill", "door", "door_handle"])
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args(argv)

    scene_dir = Path(args.scene_dir) if args.scene_dir else Path(args.preprocessed_dir) / args.scene_id
    episode, candidates = select_longest_instance_astar_episode(
        str(scene_dir),
        seed=args.seed,
        exclude_categories=args.exclude_category,
        success_distance_m=args.success_distance_m,
        min_start_goal_distance_m=args.min_start_goal_distance_m,
        max_start_goal_distance_m=args.max_start_goal_distance_m,
        min_start_object_clearance_m=args.min_start_object_clearance_m,
        min_start_goal_bbox_distance_m=args.min_start_goal_bbox_distance_m,
        min_goal_clearance_m=args.min_goal_clearance_m,
        min_planning_clearance_m=args.min_planning_clearance_m,
        top_k=args.top_k,
    )
    write_jsonl([episode], args.out)

    report = {
        "selected_episode_file": args.out,
        "selected_episode_id": episode["episode_id"],
        "selected_goal_category": episode["goal_category"],
        "selected_goal_instance_id": episode["goal_instance_ids"][0],
        "selected_start_pose_world": episode["start_pose_world"],
        "selected_start_grid": episode["start_grid"],
        "selected_shortest_path_distance_m": episode["shortest_path_distance_m"],
        "selected_start_object_clearance_m": episode["start_object_clearance_m"],
        "selected_start_goal_bbox_distance_m": episode["start_goal_bbox_distance_m"],
        "selected_min_goal_clearance_m": episode["metadata"]["min_goal_clearance_m"],
        "selected_min_planning_clearance_m": episode["metadata"]["min_planning_clearance_m"],
        "evaluated_instance_count": len(candidates),
        "top_candidates": [
            {key: value for key, value in candidate.items() if key not in {"goal_regions_grid", "path_grid"}}
            for candidate in candidates[: max(1, int(args.top_k))]
        ],
    }
    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.debug_map:
        occupancy = np.load(scene_dir / "occupancy.npy")
        navigable = np.load(scene_dir / "navigable.npy").astype(bool)
        selected = candidates[0]
        start = tuple(int(v) for v in selected["start_grid"])
        goals = [tuple(int(v) for v in cell) for cell in selected["goal_regions_grid"]]
        path = [tuple(int(v) for v in cell) for cell in selected["path_grid"]]
        save_map_png(args.debug_map, occupancy, navigable, start=start, goals=goals, path_cells=path)
        report["debug_map"] = args.debug_map
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
