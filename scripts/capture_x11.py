#!/usr/bin/env python3
import argparse
import re
import subprocess
from pathlib import Path

from PIL import ImageGrab


def field(output, label):
    match = re.search(r"^\s*{}:\s+(.+)$".format(re.escape(label)), output, re.MULTILINE)
    if match is None:
        raise RuntimeError("xwininfo did not report {}".format(label))
    return match.group(1).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-id", required=True)
    parser.add_argument("--window-output", required=True)
    parser.add_argument("--desktop-output")
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
