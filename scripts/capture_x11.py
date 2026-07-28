#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

from PIL import ImageGrab


def field(output, label):
    match = re.search(r"^\s*{}:\s+(.+)$".format(re.escape(label)), output, re.MULTILINE)
    if match is None:
        raise RuntimeError("xwininfo did not report {}".format(label))
    return match.group(1).strip()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path, payload):
    temporary = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-id", required=True)
    parser.add_argument("--window-output", required=True)
    parser.add_argument("--desktop-output")
    parser.add_argument("--receipt-output")
    parser.add_argument("--run-id")
    parser.add_argument("--process-id", type=int)
    parser.add_argument("--capture-label")
    parser.add_argument("--step", type=int)
    parser.add_argument("--render-step", type=int)
    parser.add_argument("--frame-path")
    parser.add_argument("--frame-file")
    parser.add_argument("--frame-sha256")
    parser.add_argument("--window-file")
    args = parser.parse_args()
    receipt_values = (
        args.receipt_output,
        args.run_id,
        args.process_id,
        args.capture_label,
        args.step,
        args.render_step,
        args.frame_path,
        args.frame_file,
        args.frame_sha256,
        args.window_file,
    )
    if any(value is not None for value in receipt_values) and any(
        value is None for value in receipt_values
    ):
        raise ValueError("Capture receipt arguments must be provided together")

    window_info = subprocess.check_output(
        ["xwininfo", "-id", args.window_id],
        text=True,
    )
    if field(window_info, "Map State") != "IsViewable":
        raise RuntimeError("The target window is not viewable")

    x = int(field(window_info, "Absolute upper-left X"))
    y = int(field(window_info, "Absolute upper-left Y"))
    width = int(field(window_info, "Width"))
    height = int(field(window_info, "Height"))
    if width < 640 or height < 480:
        raise RuntimeError("Visualization window is too small: {}x{}".format(width, height))

    desktop = ImageGrab.grab()
    window = desktop.crop((x, y, x + width, y + height))
    window_output = Path(args.window_output)
    window_output.parent.mkdir(parents=True, exist_ok=True)
    window.save(window_output)

    if args.desktop_output:
        desktop_output = Path(args.desktop_output)
        desktop_output.parent.mkdir(parents=True, exist_ok=True)
        desktop.save(desktop_output)
    if args.receipt_output is not None:
        frame_path = Path(args.frame_path)
        if not frame_path.is_file():
            raise FileNotFoundError(
                "Rendered frame is missing: {}".format(frame_path)
            )
        observed_frame_sha256 = sha256(frame_path)
        if observed_frame_sha256 != args.frame_sha256:
            raise RuntimeError("Rendered frame SHA256 changed before capture")
        write_json_atomic(
            Path(args.receipt_output),
            {
                "run_id": args.run_id,
                "process_id": args.process_id,
                "window_id": args.window_id,
                "capture_label": args.capture_label,
                "step": args.step,
                "render_step": args.render_step,
                "frame_file": args.frame_file,
                "frame_sha256": observed_frame_sha256,
                "window_file": args.window_file,
                "window_sha256": sha256(window_output),
                "window_size": [width, height],
                "captured_at_unix": time.time(),
            },
        )


if __name__ == "__main__":
    main()
