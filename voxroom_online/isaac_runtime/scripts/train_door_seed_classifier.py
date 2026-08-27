from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from voxroom_online.isaac_runtime.door_seed_learning.model import DoorSeedModelConfig
from voxroom_online.isaac_runtime.door_seed_learning.dataset import normalize_right_angle_rotations
from voxroom_online.isaac_runtime.door_seed_learning.training import TrainingConfig, train_classifier


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a VoxRoom raw DoorSeed binary classifier.")
    parser.add_argument("--index", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--context-source", choices=["vertical", "nav"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--height-scale-m", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument(
        "--early-stopping-metric",
        choices=["validation_score", "train_loss"],
        default="validation_score",
    )
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--target-recall", type=float, default=0.98)
    parser.add_argument(
        "--threshold-selection-mode",
        choices=["fixed", "target_recall"],
        default="fixed",
        help="Validation and checkpoint operating point; defaults to a fixed threshold.",
    )
    parser.add_argument(
        "--fixed-keep-threshold",
        type=float,
        default=0.5,
        help="Accept probabilities at or above this threshold when fixed mode is selected.",
    )
    parser.add_argument("--max-pos-weight", type=float, default=10.0)
    parser.add_argument(
        "--snapshot-cache-size",
        type=int,
        default=0,
        help="Snapshots retained in RAM per split; 0 caches every snapshot referenced by that split.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--train-rotation-degrees",
        type=_parse_rotation_degrees,
        default=(0, 90, 180, 270),
        help="Comma-separated deterministic train rotations; defaults to 0,90,180,270.",
    )
    parser.add_argument(
        "--train-mirror-lr-once",
        action="store_true",
        help="Add one deterministic left-right mirrored copy of every base training sample.",
    )
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--source-root", default=str(REPO_ROOT / "voxroom_online/isaac_runtime/door_seed_learning"))
    args = parser.parse_args(argv)
    result = train_classifier(
        index_path=args.index,
        output_dir=args.out_dir,
        context_source=args.context_source,
        height_scale_m=float(args.height_scale_m),
        model_config=None if not args.local_only else {"z_count": _z_count(args.index), "use_context_branch": False},
        training_config=TrainingConfig(
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            batch_size=int(args.batch_size),
            max_epochs=int(args.max_epochs),
            early_stopping_patience=int(args.patience),
            early_stopping_metric=str(args.early_stopping_metric),
            early_stopping_min_delta=float(args.early_stopping_min_delta),
            target_recall=float(args.target_recall),
            threshold_selection_mode=str(args.threshold_selection_mode),
            fixed_keep_threshold=float(args.fixed_keep_threshold),
            max_pos_weight=float(args.max_pos_weight),
            snapshot_cache_size=int(args.snapshot_cache_size),
            seed=int(args.seed),
            device=str(args.device),
            train_rotation_degrees=tuple(args.train_rotation_degrees),
            train_mirror_lr_once=bool(args.train_mirror_lr_once),
            precision=str(args.precision),
        ),
        source_root=args.source_root,
    )
    print(result)
    return 0


def _z_count(index_path: str) -> int:
    from voxroom_online.isaac_runtime.door_seed_learning.dataset import read_index
    from voxroom_online.isaac_runtime.door_seed_learning.schema import load_snapshot

    rows = read_index(index_path)
    if not rows:
        raise ValueError("empty dataset index")
    return int(len(load_snapshot(str(rows[0]["snapshot_path"]))["z_centers_m"]))


def _parse_rotation_degrees(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
        return normalize_right_angle_rotations(parsed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
