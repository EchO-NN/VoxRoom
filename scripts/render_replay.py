#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", default="topology_replay.mp4")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    frame_dir = run_dir / "visualization_frames"
    frames = sorted(frame_dir.glob("frame_*.png"))
    if len(frames) < 2:
        raise RuntimeError(
            "At least two dashboard frames are required, found {}".format(
                len(frames)
            )
        )
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to render the replay")
    output_path = run_dir / args.output
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(args.fps),
            "-pattern_type",
            "glob",
            "-i",
            str(frame_dir / "frame_*.png"),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ],
        check=True,
    )
    if not output_path.is_file() or output_path.stat().st_size < 1024:
        raise RuntimeError("Replay video was not created")
    report = {
        "status": "completed",
        "frame_count": len(frames),
        "fps": args.fps,
        "first_frame": frames[0].name,
        "last_frame": frames[-1].name,
        "output": output_path.name,
        "output_size": output_path.stat().st_size,
    }
    (run_dir / "replay_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
