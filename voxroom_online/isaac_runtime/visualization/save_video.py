from __future__ import annotations

from typing import Iterable

import numpy as np


def save_video(frames: Iterable[np.ndarray], path: str, fps: int = 10) -> bool:
    try:
        import cv2
    except Exception:
        return False
    frames = list(frames)
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    for frame in frames:
        writer.write(frame[:, :, ::-1] if frame.shape[-1] == 3 else frame)
    writer.release()
    return True

