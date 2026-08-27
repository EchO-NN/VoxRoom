from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the original Active_room_segmentation DETR door detector.")
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--input-npz")
    parser.add_argument("--output-json")
    parser.add_argument("--output-masks-npz")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--score", type=float, default=0.85)
    args = parser.parse_args(argv)

    repo = Path(args.repo_dir).resolve()
    os.chdir(repo)
    import sys

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from detr_door_detection.run_detr import run_detr

    if args.self_test:
        payload = {"ok": True, "repo_dir": str(repo), "loaded": "detr_door_detection.run_detr"}
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        else:
            print(json.dumps(payload, sort_keys=True))
        return 0

    if not args.input_npz or not args.output_json or not args.output_masks_npz:
        parser.error(
            "--input-npz, --output-json, and --output-masks-npz are required unless --self-test is used"
        )
    with np.load(args.input_npz, allow_pickle=False) as data:
        if "rgb_batch" not in data.files:
            raise KeyError("input npz must contain rgb_batch")
        rgb_batch = np.asarray(data["rgb_batch"])
    if rgb_batch.ndim != 4 or rgb_batch.shape[-1] < 3:
        raise ValueError("rgb_batch must be NxHxWxC, got %s" % (rgb_batch.shape,))
    frame_detections: list[list[dict]] = []
    frame_masks: list[np.ndarray] = []
    for rgb in rgb_batch:
        frame = np.asarray(rgb[:, :, :3])
        if frame.dtype != np.float32:
            frame = frame.astype(np.float32)
        if frame.max(initial=0.0) > 1.5:
            frame = frame / 255.0
        mask, _, _ = run_detr(frame)
        binary_mask = (np.asarray(mask) > 0).astype(np.uint8)
        if binary_mask.shape != frame.shape[:2]:
            raise RuntimeError(
                "original DETR mask shape mismatch: expected %s, found %s"
                % (frame.shape[:2], binary_mask.shape)
            )
        frame_masks.append(binary_mask)
        frame_detections.append(
            _mask_to_detections(binary_mask, score=float(args.score))
        )
    np.savez_compressed(
        args.output_masks_npz,
        mask_batch=np.stack(frame_masks, axis=0).astype(np.uint8, copy=False),
    )
    Path(args.output_json).write_text(
        json.dumps(
            {
                "ok": True,
                "frame_count": int(len(frame_detections)),
                "frames": frame_detections,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def _mask_to_detections(mask: np.ndarray, *, score: float) -> list[dict]:
    import cv2

    arr = np.asarray(mask)
    if arr.ndim != 2:
        return []
    binary = (arr > 0).astype(np.uint8)
    if not np.any(binary):
        return []
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    detections: list[dict] = []
    for idx in range(1, int(count)):
        x, y, w, h, area = [int(v) for v in stats[idx]]
        if area <= 0 or w <= 0 or h <= 0:
            continue
        detections.append(
            {
                "bbox_xyxy": [float(x), float(y), float(x + w), float(y + h)],
                "score": float(score),
                "class_id": 1,
                "component_id": int(idx),
                "area_px": int(area),
            }
        )
    return detections


if __name__ == "__main__":
    raise SystemExit(main())
