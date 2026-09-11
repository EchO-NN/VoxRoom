#!/usr/bin/env python3
"""Show the newest saved VoxRoom panel in a continuously updating window."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--window-name", default="VoxRoom Kujiale 0003 - Live Recording")
    parser.add_argument("--width", type=int, default=1760)
    parser.add_argument("--height", type=int, default=1120)
    parser.add_argument("--poll-seconds", type=float, default=0.10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frames_dir = args.frames_dir.expanduser().resolve()
    frames_dir.mkdir(parents=True, exist_ok=True)
    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(args.window_name, max(320, args.width), max(240, args.height))
    last_path: Path | None = None
    while True:
        candidates = list(frames_dir.glob("voxroom_step_*.jpg"))
        if candidates:
            newest = max(candidates, key=lambda path: path.stat().st_mtime_ns)
            if newest != last_path:
                frame = cv2.imread(str(newest), cv2.IMREAD_COLOR)
                if frame is not None:
                    cv2.imshow(args.window_name, frame)
                    last_path = newest
        key = cv2.waitKey(max(1, int(round(args.poll_seconds * 1000.0)))) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break
        try:
            if cv2.getWindowProperty(args.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break
        if not candidates:
            time.sleep(max(0.01, args.poll_seconds))
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
