from __future__ import annotations

import argparse
import csv
import inspect
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image

from voxroom_online.isaac_runtime.door_seed_learning.config import DoorSeedLearningConfig
from voxroom_online.isaac_runtime.door_seed_learning.inference import DoorSeedInferenceEngine
from voxroom_online.isaac_runtime.door_seed_learning.label_resolver import load_resolved_labels
from voxroom_online.isaac_runtime.door_seed_learning.metrics import binary_classification_metrics
from voxroom_online.isaac_runtime.door_seed_learning.schema import load_snapshot, read_json, scalar_value, write_json_atomic


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay a learned DoorSeed classifier on compact collection snapshots.")
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--keep-threshold", type=float, default=None)
    args = parser.parse_args(argv)
    checkpoint = _load_checkpoint(args.checkpoint)
    cfg = DoorSeedLearningConfig(
        mode="inference",
        context_source=str(checkpoint["context_source"]),
        checkpoint_path=str(args.checkpoint),
        device=str(args.device),
        inference_batch_size=int(args.batch_size),
        keep_threshold=args.keep_threshold,
        fallback_to_rule_seed_on_error=False,
        height_scale_m=float(checkpoint["height_scale_m"]),
    ).validate()
    engine = DoorSeedInferenceEngine(cfg)
    scene = Path(args.scene_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = read_json(scene / "manifest.json")
    resolved_path = scene / "annotations" / "resolved_seed_labels.json"
    resolved = load_resolved_labels(resolved_path) if resolved_path.exists() else None
    labels_by_occurrence = {}
    if resolved is not None:
        labels_by_occurrence = {
            (int(item["decision_id"]), int(item["row"]), int(item["col"])): item.get("label")
            for item in resolved.get("occurrences", [])
        }
    csv_rows: list[dict[str, object]] = []
    for record in sorted(manifest.get("snapshots", []), key=lambda item: int(item["decision_id"])):
        arrays = load_snapshot(scene / str(record["path"]))
        rc = np.asarray(arrays["raw_seed_rc"], dtype=np.int32)
        context_key = "vertical_class_map_xy" if cfg.context_source == "vertical" else "nav_class_map_xy"
        engine._validate_runtime_metadata(
            z_centers_m=arrays["z_centers_m"],
            z_min_m=float(scalar_value(arrays, "z_min_m")),
            z_resolution_m=float(scalar_value(arrays, "z_resolution_m")),
            xy_resolution_m=float(scalar_value(arrays, "resolution_m")),
            raw_seed_config_hash=str(scalar_value(arrays, "raw_seed_config_hash", "")),
            input_semantics_hash=str(scalar_value(arrays, "input_semantics_hash", "")),
        )
        probabilities = engine.predict_arrays(
            voxel_state_nzyx=arrays["raw_seed_voxel_state_nzyx"],
            z_centers_m=arrays["z_centers_m"],
            class_map_xy=arrays[context_key],
            seed_rc=rc,
        ) if len(rc) else np.empty((0,), dtype=np.float32)
        keep = probabilities >= engine.keep_threshold
        probability_map = np.full(arrays["raw_seed_mask_xy"].shape, np.nan, dtype=np.float32)
        keep_map = np.zeros_like(arrays["raw_seed_mask_xy"], dtype=bool)
        for index, (row, col) in enumerate(rc):
            probability_map[int(row), int(col)] = float(probabilities[index])
            keep_map[int(row), int(col)] = bool(keep[index])
        prefix = out / ("decision_%06d" % int(record["decision_id"]))
        _save_mask(prefix.with_suffix(".raw.png"), arrays["raw_seed_mask_xy"], (35, 110, 220))
        _save_probability(prefix.with_suffix(".probability.png"), probability_map)
        _save_mask(prefix.with_suffix(".keep.png"), keep_map, (20, 155, 70))
        _save_mask(prefix.with_suffix(".reject.png"), np.asarray(arrays["raw_seed_mask_xy"], dtype=bool) & ~keep_map, (210, 55, 45))
        if bool(record.get("is_final", False)) and resolved is not None:
            _save_final_label_comparison(
                out / "final_label_comparison.png",
                np.asarray(arrays[context_key], dtype=np.uint8),
                rc,
                keep,
                labels_by_occurrence,
                int(record["decision_id"]),
            )
        labels = []
        probs = []
        for index, (row, col) in enumerate(rc):
            label = labels_by_occurrence.get((int(record["decision_id"]), int(row), int(col)))
            if label is not None:
                labels.append(int(label))
                probs.append(float(probabilities[index]))
        metrics = binary_classification_metrics(labels, probs, engine.keep_threshold) if labels else {}
        csv_rows.append(
            {
                "decision_id": int(record["decision_id"]),
                "step": int(record["step"]),
                "seed_count_before": len(rc),
                "seed_count_after": int(np.count_nonzero(keep)),
                "true_kept": metrics.get("tp"),
                "true_deleted": metrics.get("fn"),
                "false_kept": metrics.get("fp"),
                "false_deleted": metrics.get("tn"),
                "recall": metrics.get("recall"),
                "precision": metrics.get("precision"),
                "negative_rejection": metrics.get("negative_rejection_rate"),
            }
        )
    with (out / "replay_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]) if csv_rows else ["decision_id"])
        writer.writeheader()
        writer.writerows(csv_rows)
    summary = {"scene_uid": manifest["scene_uid"], "snapshot_count": len(csv_rows), "checkpoint": str(Path(args.checkpoint).resolve())}
    write_json_atomic(out / "replay_summary.json", summary)
    print(summary)
    return 0


def _load_checkpoint(path: str):
    import torch

    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = True
    return torch.load(path, **kwargs)


def _save_mask(path: Path, mask: np.ndarray, color: tuple[int, int, int]) -> None:
    value = np.full(np.asarray(mask).shape + (3,), 245, dtype=np.uint8)
    value[np.asarray(mask, dtype=bool)] = color
    Image.fromarray(value, mode="RGB").save(path)


def _save_probability(path: Path, probability: np.ndarray) -> None:
    value = np.full(np.asarray(probability).shape + (3,), 245, dtype=np.uint8)
    finite = np.isfinite(probability)
    p = np.clip(np.nan_to_num(probability, nan=0.0), 0.0, 1.0)
    value[finite, 0] = np.asarray(255 * p[finite], dtype=np.uint8)
    value[finite, 1] = np.asarray(255 * (1.0 - np.abs(p[finite] - 0.5) * 2.0), dtype=np.uint8)
    value[finite, 2] = np.asarray(255 * (1.0 - p[finite]), dtype=np.uint8)
    Image.fromarray(value, mode="RGB").save(path)


def _save_final_label_comparison(
    path: Path,
    context_map: np.ndarray,
    seed_rc: np.ndarray,
    keep: np.ndarray,
    labels_by_occurrence: dict[tuple[int, int, int], object],
    decision_id: int,
) -> None:
    palette = np.asarray(((235, 235, 235), (252, 252, 252), (45, 45, 45)), dtype=np.uint8)
    rgb = palette[np.asarray(context_map, dtype=np.uint8)].copy()
    colors = {
        (1, True): (20, 155, 70),
        (1, False): (238, 145, 20),
        (0, True): (205, 45, 145),
        (0, False): (45, 115, 220),
    }
    for index, (row, col) in enumerate(np.asarray(seed_rc, dtype=np.int32)):
        label = labels_by_occurrence.get((int(decision_id), int(row), int(col)))
        if label is None:
            continue
        color = colors[(int(label), bool(keep[index]))]
        r0, r1 = max(0, int(row) - 2), min(rgb.shape[0], int(row) + 3)
        c0, c1 = max(0, int(col) - 2), min(rgb.shape[1], int(col) + 3)
        rgb[r0:r1, c0:c1] = color
    Image.fromarray(rgb, mode="RGB").save(path)


if __name__ == "__main__":
    raise SystemExit(main())
