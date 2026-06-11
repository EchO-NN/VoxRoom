from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path
from typing import List, Protocol

import numpy as np
from PIL import Image


class _Viewer(Protocol):
    def show(self, rgb: np.ndarray) -> bool:
        ...

    def close(self) -> None:
        ...


class _Cv2Viewer:
    def __init__(self, window_name: str, width: int, height: int) -> None:
        import cv2

        self._cv2 = cv2
        self._window_name = str(window_name)
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._window_name, int(width), int(height))

    def show(self, rgb: np.ndarray) -> bool:
        self._cv2.imshow(self._window_name, self._cv2.cvtColor(rgb, self._cv2.COLOR_RGB2BGR))
        key = self._cv2.waitKey(1) & 0xFF
        return key not in (27, ord("q"))

    def close(self) -> None:
        try:
            self._cv2.destroyWindow(self._window_name)
        except Exception:
            pass


class _TkViewer:
    def __init__(self, window_name: str, width: int, height: int) -> None:
        import tkinter as tk
        from PIL import ImageTk

        self._tk = tk
        self._image_tk = ImageTk
        self._root = tk.Tk()
        self._root.title(str(window_name))
        self._root.geometry("%dx%d" % (int(width), int(height)))
        self._root.protocol("WM_DELETE_WINDOW", self._request_close)
        self._label = tk.Label(self._root)
        self._label.pack(fill=tk.BOTH, expand=True)
        self._photo = None
        self._closed = False
        self._root.update()

    def _request_close(self) -> None:
        self._closed = True

    def show(self, rgb: np.ndarray) -> bool:
        if self._closed:
            return False
        image = Image.fromarray(rgb, mode="RGB")
        win_w = max(1, int(self._root.winfo_width()))
        win_h = max(1, int(self._root.winfo_height()))
        image.thumbnail((win_w, win_h), Image.Resampling.BILINEAR)
        self._photo = self._image_tk.PhotoImage(image=image)
        self._label.configure(image=self._photo)
        self._root.update()
        return not self._closed

    def close(self) -> None:
        try:
            self._root.destroy()
        except Exception:
            pass


def _decode_image(payload: str) -> np.ndarray:
    raw = base64.b64decode(payload.encode("ascii"))
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)


def _write_ready() -> None:
    print(json.dumps({"type": "ready"}), flush=True)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-name", default="VoxRoom Isaac Debug")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args(argv)

    viewer: _Viewer | None = None
    errors: list[str] = []
    for factory in (_Cv2Viewer, _TkViewer):
        try:
            viewer = factory(args.window_name, int(args.width), int(args.height))
            break
        except Exception as exc:
            errors.append("%s: %s" % (factory.__name__, exc))
    if viewer is None:
        print("[voxroom-viz-worker] popup unavailable: %s" % " | ".join(errors), file=sys.stderr, flush=True)
        return 1

    _write_ready()
    for line in sys.stdin:
        try:
            req = json.loads(line)
            req_type = req.get("type")
            if req_type == "close":
                break
            if req_type == "frame_path":
                frame_path = Path(str(req["path"]))
                rgb = np.asarray(Image.open(frame_path).convert("RGB"), dtype=np.uint8)
            elif req_type == "frame":
                rgb = _decode_image(str(req["image_png_b64"]))
            else:
                continue
            if not viewer.show(rgb):
                break
        except Exception as exc:
            print("[voxroom-viz-worker] frame update failed: %s" % exc, file=sys.stderr, flush=True)
            continue
    viewer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
