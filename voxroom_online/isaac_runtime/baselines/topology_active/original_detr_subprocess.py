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

    if not args.input_npz or not args.output_json:
        parser.error("--input-npz and --output-json are required unless --self-test is used")
    with np.load(args.input_npz, allow_pickle=False) as data:
        rgb = np.asarray(data["rgb"])
    if rgb.dtype != np.float32:
        rgb = rgb.astype(np.float32)
    if rgb.max(initial=0.0) > 1.5:
        rgb = rgb / 255.0
    mask, _, _ = run_detr(rgb)
    detections = _mask_to_detections(np.asarray(mask), score=float(args.score))
    Path(args.output_json).write_text(
        json.dumps({"ok": True, "detections": detections}, ensure_ascii=False, sort_keys=True) + "\n",
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
                "area_px": int(area),
            }
        )
    return detections


if __name__ == "__main__":
    raise SystemExit(main())
