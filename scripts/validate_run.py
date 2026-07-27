#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path

from PIL import Image, ImageStat


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_key_values(path):
    values = {}
    for line in path.read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values


def check_image(path, minimum_size):
    with Image.open(path) as image:
        image.load()
        if image.width < minimum_size[0] or image.height < minimum_size[1]:
            raise RuntimeError(
                "{} is too small: {}x{}".format(path, image.width, image.height)
            )
        if max(ImageStat.Stat(image.convert("RGB")).stddev) < 2.0:
            raise RuntimeError("{} is blank or nearly uniform".format(path))
        return [image.width, image.height]


def write_json_atomic(path, payload):
    temporary = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    paths = {
        "result": run_dir / "result.json",
        "metadata": run_dir / "run_metadata.json",
        "progress": run_dir / "progress.jsonl",
        "visualization": run_dir / "live_visualization.txt",
        "desktop": run_dir / "live_desktop.png",
        "window_first": run_dir / "window_first.png",
        "window_later": run_dir / "window_later.png",
        "runtime_log": run_dir / "runtime.log",
        "command": run_dir / "command.txt",
        "process_id": run_dir / "process_id.txt",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing run evidence: {}".format(", ".join(missing)))
    if paths["runtime_log"].stat().st_size == 0:
        raise RuntimeError("Runtime log is empty")

    result = json.loads(paths["result"].read_text())
    metadata = json.loads(paths["metadata"].read_text())
    visualization = read_key_values(paths["visualization"])
    process_id = int(paths["process_id"].read_text().strip())

    if result.get("status") != "completed":
        raise RuntimeError("Run did not complete: {}".format(result))
    if result.get("run_id") != args.run_id or metadata.get("run_id") != args.run_id:
        raise RuntimeError("Run ID mismatch between result and metadata")
    source_commit = result.get("source_commit")
    if source_commit != metadata.get("source_commit"):
        raise RuntimeError("Source commit mismatch between result and metadata")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise RuntimeError("Run artifacts do not contain a full source commit")
    if visualization.get("run_id") != args.run_id:
        raise RuntimeError("Run ID mismatch in visualization evidence")
    if visualization.get("window_title") != result.get("window_title"):
        raise RuntimeError("Window title mismatch in visualization evidence")
    if metadata.get("window_title") != result.get("window_title"):
        raise RuntimeError("Window title mismatch in run metadata")
    if result.get("process_id") != process_id or metadata.get("process_id") != process_id:
        raise RuntimeError("Process ID mismatch in run artifacts")
    if int(visualization.get("process_id", -1)) != process_id:
        raise RuntimeError("Process ID mismatch in visualization evidence")
    if result.get("requested_max_episode_steps") != args.expected_steps:
        raise RuntimeError("Requested step count changed during the run")
    if result.get("executed_steps") != args.expected_steps:
        raise RuntimeError(
            "Executed {} steps, expected {}".format(
                result.get("executed_steps"), args.expected_steps
            )
        )
    if not result.get("visualization") or visualization.get("confirmed") != "1":
        raise RuntimeError("The physical X11 visualization was not confirmed")
    if result.get("detector_device") != "cuda":
        raise RuntimeError("The verified run did not use CUDA door detection")
    if not result.get("scene_name"):
        raise RuntimeError("Result does not identify the simulated scene")
    if result.get("finished_at_unix", 0) <= result.get("started_at_unix", 0):
        raise RuntimeError("Invalid run timestamps")

    progress = [
        json.loads(line)
        for line in paths["progress"].read_text().splitlines()
        if line.strip()
    ]
    expected_sequence = list(range(1, args.expected_steps + 1))
    if [event.get("step") for event in progress] != expected_sequence:
        raise RuntimeError("Control-step sequence is not exactly 1..N")
    if [event.get("simulator_step") for event in progress] != expected_sequence:
        raise RuntimeError("Simulator-step sequence is not exactly 1..N")
    if any(event.get("run_id") != args.run_id for event in progress):
        raise RuntimeError("Progress contains a foreign run ID")
    if any(event.get("process_id") != process_id for event in progress):
        raise RuntimeError("Progress contains a foreign process ID")

    image_sizes = {
        "desktop": check_image(paths["desktop"], (1280, 720)),
        "window_first": check_image(paths["window_first"], (640, 480)),
        "window_later": check_image(paths["window_later"], (640, 480)),
    }
    image_hashes = {
        name: sha256(paths[name])
        for name in ("desktop", "window_first", "window_later")
    }
    if image_hashes["window_first"] == image_hashes["window_later"]:
        raise RuntimeError("The live window pixels did not change across control steps")

    command = paths["command"].read_text()
    if args.run_id not in command or "--detector_device cuda" not in command:
        raise RuntimeError("Command evidence is not bound to this CUDA run")

    report = {
        "status": "validated",
        "run_id": args.run_id,
        "process_id": process_id,
        "source_commit": source_commit,
        "steps": args.expected_steps,
        "image_sizes": image_sizes,
        "image_sha256": image_hashes,
        "scene_name": result["scene_name"],
    }
    write_json_atomic(run_dir / "validation.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
