from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from voxroom_online.isaac_runtime.door_seed_learning.dataset import DoorSeedDataset
from voxroom_online.isaac_runtime.door_seed_learning.config import DoorSeedLearningConfig
from voxroom_online.isaac_runtime.door_seed_learning.inference import DoorSeedInferenceEngine
from voxroom_online.isaac_runtime.door_seed_learning.metrics import binary_classification_metrics, ranking_metrics, recall_rejection_table
from voxroom_online.isaac_runtime.door_seed_learning.schema import load_snapshot, scalar_value, write_json_atomic


def main(argv: Sequence[str] | None = None) -> int:
    import torch
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description="Evaluate a DoorSeed checkpoint on a scene-level split.")
    parser.add_argument("--index", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    raw_checkpoint = _load_checkpoint(args.checkpoint)
    engine = DoorSeedInferenceEngine(
        DoorSeedLearningConfig(
            mode="inference",
            checkpoint_path=str(args.checkpoint),
            context_source=str(raw_checkpoint["context_source"]),
            height_scale_m=float(raw_checkpoint["height_scale_m"]),
            device=str(args.device),
            inference_batch_size=int(args.batch_size),
            fallback_to_rule_seed_on_error=False,
        ).validate()
    )
    checkpoint = engine.checkpoint
    if checkpoint is None or engine.model is None:
        raise RuntimeError("strict checkpoint loading did not produce a model")
    dataset = DoorSeedDataset(
        args.index,
        split=args.split,
        context_source=str(checkpoint["context_source"]),
        height_scale_m=float(checkpoint["height_scale_m"]),
        augment=False,
    )
    if not len(dataset):
        raise ValueError("selected evaluation split is empty")
    for snapshot_path in sorted({str(row["snapshot_path"]) for row in dataset.rows}):
        arrays = load_snapshot(snapshot_path)
        engine._validate_runtime_metadata(
            z_centers_m=arrays["z_centers_m"],
            z_min_m=float(scalar_value(arrays, "z_min_m")),
            z_resolution_m=float(scalar_value(arrays, "z_resolution_m")),
            xy_resolution_m=float(scalar_value(arrays, "resolution_m")),
            raw_seed_config_hash=str(scalar_value(arrays, "raw_seed_config_hash", "")),
            input_semantics_hash=str(scalar_value(arrays, "input_semantics_hash", "")),
        )
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=0)
    model = engine.model
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but unavailable")
    model.to(device).eval()
    labels: list[int] = []
    probabilities: list[float] = []
    scene_ids: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["voxel"].to(device), batch["context"].to(device))
            if not torch.all(torch.isfinite(logits)):
                raise RuntimeError("model produced non-finite logits")
            labels.extend(int(value) for value in batch["label"].numpy().reshape(-1))
            probabilities.extend(float(value) for value in torch.sigmoid(logits).cpu().numpy().reshape(-1))
            scene_ids.extend(str(value) for value in batch["scene_id"])
    threshold = float(checkpoint["recommended_keep_threshold"])
    aggregate_metrics = _fixed_threshold_metrics(labels, probabilities, threshold)
    per_scene = {}
    for scene_id in sorted(set(scene_ids)):
        selected = [index for index, value in enumerate(scene_ids) if value == scene_id]
        scene_labels = [labels[index] for index in selected]
        scene_probabilities = [probabilities[index] for index in selected]
        per_scene[scene_id] = {
            "sample_count": len(selected),
            "positive_count": int(sum(value == 1 for value in scene_labels)),
            "negative_count": int(sum(value == 0 for value in scene_labels)),
            **_fixed_threshold_metrics(scene_labels, scene_probabilities, threshold),
        }
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "index": str(Path(args.index).resolve()),
        "split": args.split,
        "sample_count": len(labels),
        **aggregate_metrics,
        "recall_rejection": recall_rejection_table(labels, probabilities),
        "per_scene": per_scene,
    }
    write_json_atomic(args.out, result)
    print(result)
    return 0


def _load_checkpoint(path: str):
    import torch

    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = True
    return torch.load(path, **kwargs)


def _fixed_threshold_metrics(
    labels: list[int],
    probabilities: list[float],
    threshold: float,
) -> dict[str, object]:
    classification = binary_classification_metrics(labels, probabilities, threshold)
    rejected_count = int(classification["tn"]) + int(classification["fn"])
    return {
        **classification,
        "rejected_seed_count": rejected_count,
        "correct_rejected_seed_count": int(classification["tn"]),
        "incorrect_rejected_seed_count": int(classification["fn"]),
        "rejected_seed_accuracy": (
            float(classification["tn"]) / float(rejected_count) if rejected_count else None
        ),
        **ranking_metrics(labels, probabilities),
    }


if __name__ == "__main__":
    raise SystemExit(main())
