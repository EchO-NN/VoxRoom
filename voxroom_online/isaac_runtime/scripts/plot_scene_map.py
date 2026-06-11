from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np

from voxroom_online.isaac_runtime.visualization.draw_map import save_map_png


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--preprocessed-dir", default="data/interioragent_preprocessed")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    scene_dir = Path(args.preprocessed_dir) / args.scene_id
    occupancy = np.load(scene_dir / "occupancy.npy")
    navigable = np.load(scene_dir / "navigable.npy").astype(bool)
    out = args.out or "debug/%s_map.png" % args.scene_id
    save_map_png(out, occupancy, navigable)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

