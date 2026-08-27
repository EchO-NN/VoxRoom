from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from voxroom_online.isaac_runtime.dataset.episode_generator import generate_scene_episodes, write_jsonl


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed-dir", default="data/interioragent_preprocessed")
    parser.add_argument("--out", default="data/interioragent_episodes/train.jsonl")
    parser.add_argument("--scene-id", action="append", default=None)
    parser.add_argument("--episodes-per-scene", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--success-distance-m", type=float, default=1.0)
    parser.add_argument("--min-start-goal-distance-m", type=float, default=1.5)
    parser.add_argument("--max-start-goal-distance-m", type=float, default=25.0)
    parser.add_argument("--min-start-object-clearance-m", type=float, default=0.80)
    parser.add_argument("--min-start-goal-bbox-distance-m", type=float, default=1.5)
    parser.add_argument("--min-goal-clearance-m", type=float, default=0.0)
    parser.add_argument("--min-planning-clearance-m", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-time-s", type=float, default=300.0)
    args = parser.parse_args(argv)

    root = Path(args.preprocessed_dir)
    if not root.exists():
        print("Error: preprocessed directory not found: %s" % root, file=sys.stderr)
        return 2
    scene_dirs = [p for p in sorted(root.glob("kujiale_*")) if p.is_dir()]
    if args.scene_id:
        wanted = set(args.scene_id)
        scene_dirs = [p for p in scene_dirs if p.name in wanted]
    if not scene_dirs:
        requested = ", ".join(args.scene_id or ["<all kujiale_*>"])
        print("Error: no preprocessed scenes found in %s for %s" % (root, requested), file=sys.stderr)
        return 2
    episodes = []
    for scene_dir in scene_dirs:
        scene_eps = generate_scene_episodes(
            str(scene_dir),
            episodes_per_scene=args.episodes_per_scene,
            seed=args.seed,
            success_distance_m=args.success_distance_m,
            min_start_goal_distance_m=args.min_start_goal_distance_m,
            max_start_goal_distance_m=args.max_start_goal_distance_m,
            min_start_object_clearance_m=args.min_start_object_clearance_m,
            min_start_goal_bbox_distance_m=args.min_start_goal_bbox_distance_m,
            min_goal_clearance_m=args.min_goal_clearance_m,
            min_planning_clearance_m=args.min_planning_clearance_m,
            max_steps=args.max_steps,
            max_time_s=args.max_time_s,
        )
        print("[episodes] %s: %d" % (scene_dir.name, len(scene_eps)), flush=True)
        episodes.extend(scene_eps)
    write_jsonl(episodes, args.out)
    print("[episodes] wrote %d episodes to %s" % (len(episodes), args.out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
