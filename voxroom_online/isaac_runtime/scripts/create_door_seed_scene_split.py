from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from voxroom_online.isaac_runtime.door_seed_learning.scene_split import (
    build_scene_split,
    write_scene_split,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a reproducible scene-level DoorSeed split.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--scene-glob", default="kujiale_*")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--development-fraction", type=float, default=0.5)
    parser.add_argument("--development-val-fraction", type=float, default=0.2)
    parser.add_argument("--out", required=True)
    parser.add_argument("--list-dir", default=None)
    args = parser.parse_args(argv)

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    scenes = sorted(path.name for path in dataset_root.glob(args.scene_glob) if path.is_dir())
    if not scenes:
        raise ValueError("no scenes found under %s with glob %s" % (dataset_root, args.scene_glob))
    payload = build_scene_split(
        scenes,
        seed=int(args.seed),
        development_fraction=float(args.development_fraction),
        development_val_fraction=float(args.development_val_fraction),
    )
    payload["dataset_root"] = str(dataset_root)
    payload["scene_glob"] = str(args.scene_glob)
    write_scene_split(args.out, payload)

    list_dir = Path(args.list_dir).expanduser() if args.list_dir else Path(args.out).expanduser().parent
    list_dir.mkdir(parents=True, exist_ok=True)
    roles = dict(payload["roles"])
    lists = {
        "development_collection": roles["development_collection"],
        "heldout_validation": roles["heldout_validation"],
        "train": payload["train"],
        "val": payload["val"],
        "test": payload["test"],
    }
    for name, values in lists.items():
        (list_dir / (name + ".txt")).write_text("\n".join(values) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "seed": payload["seed"], "counts": payload["counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
