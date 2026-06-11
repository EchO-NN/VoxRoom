from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from voxroom_online.isaac_runtime.dataset.episode_generator import read_jsonl
from voxroom_online.isaac_runtime.navigation.astar import GridAStarPlanner
from voxroom_online.isaac_runtime.visualization.draw_map import save_map_png


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-file", required=True)
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--out", default="debug/episode.png")
    args = parser.parse_args(argv)
    episodes = read_jsonl(args.episode_file)
    if args.episode_id:
        episode = next(ep for ep in episodes if ep["episode_id"] == args.episode_id)
    else:
        episode = episodes[args.episode_index]
    scene_dir = Path(episode["preprocessed_scene_dir"])
    occupancy = np.load(scene_dir / "occupancy.npy")
    navigable = np.load(scene_dir / "navigable.npy").astype(bool)
    planner = GridAStarPlanner(navigable, float(episode["metadata"]["map_resolution_m"]))
    start = tuple(episode["start_grid"])
    goals = [tuple(cell) for cell in episode["goal_regions_grid"]]
    path = planner.plan(start, goals).path
    save_map_png(args.out, occupancy, navigable, start=start, goals=goals, path_cells=path)
    print(json.dumps({"episode_id": episode["episode_id"], "out": args.out, "path_cells": len(path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

