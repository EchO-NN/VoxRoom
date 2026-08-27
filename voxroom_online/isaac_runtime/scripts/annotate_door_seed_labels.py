from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from voxroom_online.isaac_runtime.door_seed_learning.annotation_app import run_annotation_app


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Annotate final raw door seeds for one collected scene.")
    parser.add_argument("--scene-dir", default=None)
    parser.add_argument(
        "--collection-root",
        default=None,
        help="Collection root used by the scene sidebar; opens the first unfinished scene when --scene-dir is omitted.",
    )
    parser.add_argument("--annotator", default="annotator")
    parser.add_argument(
        "--prefill-unlabeled-reject",
        action="store_true",
        help="Persist every currently unlabeled final raw seed as reject without changing existing labels.",
    )
    args = parser.parse_args(argv)
    if args.scene_dir is None and args.collection_root is None:
        parser.error("one of --scene-dir or --collection-root is required")
    run_annotation_app(
        args.scene_dir,
        annotator=args.annotator,
        collection_root=args.collection_root,
        prefill_unlabeled_reject=args.prefill_unlabeled_reject,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
