#!/usr/bin/env python3
import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageChops, ImageStat
from scipy.fft import dctn

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
from run_context_contract import (
    STRICT_CAPTURE_MARKER_HEIGHT,
    STRICT_CAPTURE_MARKER_PAYLOAD_BITS,
    STRICT_CAPTURE_MARKER_SYNC,
    STRICT_CAPTURE_MARKER_WIDTH,
    STRICT_CAPTURE_MARKER_X,
    STRICT_CAPTURE_MARKER_Y,
    STRICT_LIVE_CANVAS_SIZE,
    STRICT_VISUAL_CAPTURE_STEPS,
    episode_contract_sha256,
    strict_capture_marker_bits,
)
from topology_contract import (
    crossing_evidence_survives,
    surviving_crossing_count,
)


EXPECTED_GIBSON_CONTEXT = {
    "source": "official_gibson_habitat_trainval",
    "archive_sha256": (
        "b8280c7fec1175794656bf274a94caa77a980e32df8bce6995aad41f35b910fe"
    ),
    "archive_size": 10833075327,
    "archive_entry_count": 984,
    "pointnav_tree_sha256": (
        "4da848fa38be405123092f3ba74c3e7503acd3ccc150614f5a5eceeae006966e"
    ),
    "pointnav_file_count": 75,
    "pointnav_required_scene_count": 86,
    "available_scene_count": 492,
}
DASHBOARD_PHASH_MIN_DISTANCE = 12


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


def dashboard_panel_boxes(size):
    width, height = size
    left = int(width * 0.03)
    right = int(width * 0.98)
    top = int(height * 0.07)
    bottom = int(height * 0.92)
    middle_x = (left + right) // 2
    middle_y = (top + bottom) // 2
    return {
        "rgb": (left, top, middle_x, middle_y),
        "map": (middle_x, top, right, middle_y),
        "topology": (left, middle_y, middle_x, bottom),
        "status": (middle_x, middle_y, right, bottom),
    }


def dashboard_panel_signature(path):
    with Image.open(path) as image:
        dashboard = image.convert("RGB")
    return {
        name: dashboard_panel_perceptual_hash(dashboard.crop(box))
        for name, box in dashboard_panel_boxes(dashboard.size).items()
    }


def dashboard_panel_perceptual_hash(panel):
    pixels = np.asarray(
        panel.convert("L").resize(
            (64, 64),
            Image.Resampling.LANCZOS,
        ),
        dtype=np.float32,
    )
    coefficients = dctn(pixels, norm="ortho")[:16, :16]
    threshold = float(np.median(coefficients.reshape(-1)[1:]))
    bits = coefficients > threshold
    return np.packbits(bits.reshape(-1)).tobytes().hex()


def perceptual_hash_distance(first, second):
    if len(first) != len(second):
        raise RuntimeError("Perceptual hashes have inconsistent lengths")
    return sum(
        bin(left ^ right).count("1")
        for left, right in zip(bytes.fromhex(first), bytes.fromhex(second))
    )


def require_distinct_panel_signatures(signatures, label):
    for first_index, first in enumerate(signatures):
        for second in signatures[first_index + 1 :]:
            distances = {
                name: perceptual_hash_distance(first[name], second[name])
                for name in first
            }
            if max(distances.values()) < DASHBOARD_PHASH_MIN_DISTANCE:
                raise RuntimeError(
                    "{} contain scale-equivalent frozen panels".format(label)
                )


def decode_capture_marker(path, canvas_size=None):
    with Image.open(path) as source:
        image = source.convert("L")
    if canvas_size is None:
        canvas_width, canvas_height = image.size
    else:
        canvas_width, canvas_height = map(int, canvas_size)
        if image.width < canvas_width or image.height < canvas_height:
            raise RuntimeError(
                "{} is smaller than its declared dashboard canvas".format(path)
            )
        image = image.crop((0, 0, canvas_width, canvas_height))
    bit_count = len(STRICT_CAPTURE_MARKER_SYNC) + (
        STRICT_CAPTURE_MARKER_PAYLOAD_BITS
    )
    bit_width = STRICT_CAPTURE_MARKER_WIDTH / bit_count
    y0 = int(
        (1.0 - STRICT_CAPTURE_MARKER_Y - 0.72 * STRICT_CAPTURE_MARKER_HEIGHT)
        * canvas_height
    )
    y1 = int(
        (1.0 - STRICT_CAPTURE_MARKER_Y - 0.28 * STRICT_CAPTURE_MARKER_HEIGHT)
        * canvas_height
    )
    if y1 <= y0:
        raise RuntimeError("Capture marker sampling height collapsed")
    bits = []
    for index in range(bit_count):
        center_x = (
            STRICT_CAPTURE_MARKER_X + (index + 0.5) * bit_width
        ) * canvas_width
        half_sample_width = max(1.0, 0.18 * bit_width * canvas_width)
        x0 = int(center_x - half_sample_width)
        x1 = int(center_x + half_sample_width)
        if x1 <= x0:
            raise RuntimeError("Capture marker sampling width collapsed")
        value = float(np.asarray(image.crop((x0, y0, x1, y1))).mean())
        bits.append(1 if value < 128.0 else 0)
    return tuple(bits)


def check_capture_marker(path, run_id, step, canvas_size=None):
    expected = strict_capture_marker_bits(run_id, step)
    observed = decode_capture_marker(path, canvas_size=canvas_size)
    if observed != expected:
        raise RuntimeError(
            "{} is not pixel-bound to run {} step {}".format(
                path,
                run_id,
                step,
            )
        )
    return "".join(str(bit) for bit in observed)


def validate_capture_receipt(
    receipt_path,
    *,
    expected,
    run_dir,
    expected_window_size,
):
    receipt = json.loads(receipt_path.read_text())
    captured_at_unix = receipt.pop("captured_at_unix", None)
    if (
        not isinstance(captured_at_unix, (int, float))
        or not 0 < float(captured_at_unix) <= time.time() + 5.0
    ):
        raise RuntimeError("Capture receipt has an invalid timestamp")
    window_path = run_dir / expected["window_file"]
    frame_path = run_dir / expected["frame_file"]
    expected_receipt = {
        **expected,
        "window_sha256": sha256(window_path),
        "window_size": list(expected_window_size),
    }
    if receipt != expected_receipt:
        raise RuntimeError("Capture receipt identity or pixels changed")
    if sha256(frame_path) != expected["frame_sha256"]:
        raise RuntimeError("Capture receipt rendered-frame SHA256 changed")
    return {
        **receipt,
        "captured_at_unix": float(captured_at_unix),
    }


def check_dashboard_nonblank(path):
    with Image.open(path) as image:
        dashboard = image.convert("RGB")
    stddev = {}
    for name, box in dashboard_panel_boxes(dashboard.size).items():
        panel_stddev = max(ImageStat.Stat(dashboard.crop(box)).stddev)
        if panel_stddev < 2.0:
            raise RuntimeError("{} dashboard panel is blank".format(name))
        stddev[name] = panel_stddev
    return stddev


def check_dashboard_panels(first_path, terminal_path):
    with Image.open(first_path) as first_image:
        first = first_image.convert("RGB")
    with Image.open(terminal_path) as terminal_image:
        terminal = terminal_image.convert("RGB")
    first_boxes = dashboard_panel_boxes(first.size)
    terminal_boxes = dashboard_panel_boxes(terminal.size)
    metrics = {}
    for name in first_boxes:
        first_panel = first.crop(first_boxes[name]).resize(
            (640, 360),
            Image.Resampling.BILINEAR,
        )
        terminal_panel = terminal.crop(terminal_boxes[name]).resize(
            (640, 360),
            Image.Resampling.BILINEAR,
        )
        terminal_stddev = max(ImageStat.Stat(terminal_panel).stddev)
        difference_mean = max(
            ImageStat.Stat(ImageChops.difference(first_panel, terminal_panel)).mean
        )
        perceptual_distance = perceptual_hash_distance(
            dashboard_panel_perceptual_hash(first_panel),
            dashboard_panel_perceptual_hash(terminal_panel),
        )
        if terminal_stddev < 2.0:
            raise RuntimeError("{} dashboard panel is blank".format(name))
        if perceptual_distance < DASHBOARD_PHASH_MIN_DISTANCE:
            raise RuntimeError("{} dashboard panel did not change".format(name))
        metrics[name] = {
            "terminal_stddev": terminal_stddev,
            "first_to_terminal_difference_mean": difference_mean,
            "first_to_terminal_perceptual_hash_distance": perceptual_distance,
        }
    return metrics


def write_json_atomic(path, payload):
    temporary = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def require_fail_fast_sources(repository_root):
    checked_paths = [
        repository_root / "explorable_with_door_detection.py",
        repository_root / "run_context_contract.py",
        repository_root / "topology_contract.py",
        repository_root / "topomap_construction.py",
        repository_root / "frontier_detection.py",
        repository_root / "door_detection.py",
        repository_root / "env" / "habitat" / "__init__.py",
        repository_root / "env" / "habitat" / "exploration_env.py",
        repository_root / "env" / "habitat" / "hough_door_detection.py",
    ]
    violations = []
    for path in checked_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            catches_exception = (
                node.type is None
                or (
                    isinstance(node.type, ast.Name)
                    and node.type.id in {"Exception", "BaseException"}
                )
            )
            if catches_exception:
                violations.append("{}:{}".format(path, node.lineno))
    if violations:
        raise RuntimeError(
            "Broad exception recovery remains active: {}".format(violations)
        )
    return [str(path.relative_to(repository_root)) for path in checked_paths]


def detect_run_context(run_dir, required=False):
    manifest_exists = (run_dir / "input_manifest.json").is_file()
    dataset_exists = (run_dir / "input_dataset.json.gz").is_file()
    if manifest_exists != dataset_exists:
        raise RuntimeError("Run context evidence is only partially present")
    if required and not manifest_exists:
        raise FileNotFoundError("Required run context evidence is missing")
    return manifest_exists


def expected_visualization_frames(frame_every_steps, last_render_step):
    if frame_every_steps < 1:
        raise ValueError("Visualization frame interval must be positive")
    return [
        "visualization_frames/frame_{:06d}.png".format(step)
        for step in range(
            frame_every_steps,
            last_render_step + 1,
            frame_every_steps,
        )
    ]


def expected_physical_checkpoint_steps(executed_steps, checkpoint_every_steps):
    if checkpoint_every_steps < 1:
        raise ValueError("Physical-window checkpoint interval must be positive")
    return list(
        range(
            checkpoint_every_steps,
            executed_steps + 1,
            checkpoint_every_steps,
        )
    )


def collect_artifact_hashes(run_dir):
    hashes = {}
    sizes = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("Run evidence contains a symlink: {}".format(path))
        if not path.is_file():
            continue
        relative_path = path.relative_to(run_dir).as_posix()
        if relative_path == "validation.json":
            continue
        hashes[relative_path] = sha256(path)
        sizes[relative_path] = path.stat().st_size
    return hashes, sizes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--require-topology-transition", action="store_true")
    parser.add_argument("--allow-early-completion", action="store_true")
    parser.add_argument("--require-run-context", action="store_true")
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[1]
    validator_commit = subprocess.check_output(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        text=True,
        timeout=30,
    ).strip()
    validator_changes = subprocess.check_output(
        ["git", "-C", str(repository_root), "status", "--porcelain"],
        text=True,
        timeout=30,
    ).strip()
    if validator_changes:
        raise RuntimeError("Refusing to validate from a dirty source tree")

    run_dir = Path(args.run_dir).resolve()
    context_mode = detect_run_context(run_dir, required=args.require_run_context)
    strict_topology = bool(args.require_topology_transition or context_mode)
    early_completion = bool(args.allow_early_completion or context_mode)
    paths = {
        "result": run_dir / "result.json",
        "metadata": run_dir / "run_metadata.json",
        "progress": run_dir / "progress.jsonl",
        "visualization": run_dir / "live_visualization.txt",
        "desktop": run_dir / "live_desktop.png",
        "window_first": run_dir / "window_first.png",
        "window_later": run_dir / "window_later.png",
        "runtime_log": run_dir / "runtime.log",
        "runtime_install": run_dir / "runtime_install.json",
        "command": run_dir / "command.txt",
        "process_id": run_dir / "process_id.txt",
        "topology": run_dir / "topology_final.json",
        "topology_events": run_dir / "topology_events.jsonl",
        "visualization_manifest": run_dir / "visualization_manifest.json",
        "visualization_final": run_dir / "visualization_final.png",
        "room_mask_final": run_dir / "room_mask_final.png",
        "room_labels_final": run_dir / "room_labels_final.npz",
        "replay_manifest": run_dir / "replay_manifest.json",
        "replay": run_dir / "topology_replay.mp4",
    }
    if context_mode:
        paths.update(
            {
                "window_mid": run_dir / "window_mid.png",
                "window_terminal": run_dir / "window_terminal.png",
                "desktop_terminal": run_dir / "live_desktop_terminal.png",
                "terminal_ready": run_dir / "terminal_capture_ready.json",
                "terminal_receipt": run_dir / "terminal_capture_receipt.json",
                "terminal_ack": run_dir / "terminal_capture_ack.txt",
                "input_manifest": run_dir / "input_manifest.json",
                "input_dataset": run_dir / "input_dataset.json.gz",
            }
        )
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing run evidence: {}".format(", ".join(missing)))
    if paths["runtime_log"].stat().st_size == 0:
        raise RuntimeError("Runtime log is empty")
    subprocess.run(
        [sys.executable, str(repository_root / "scripts" / "validate_install.py")],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
    )
    current_install = json.loads(
        (repository_root / "reproduction_install.json").read_text()
    )
    runtime_install = json.loads(paths["runtime_install"].read_text())
    if runtime_install != current_install:
        raise RuntimeError("Runtime dependency closure changed during the run")

    result = json.loads(paths["result"].read_text())
    metadata = json.loads(paths["metadata"].read_text())
    visualization = read_key_values(paths["visualization"])
    topology = json.loads(paths["topology"].read_text())
    visualization_manifest = json.loads(paths["visualization_manifest"].read_text())
    replay_manifest = json.loads(paths["replay_manifest"].read_text())
    process_id = int(paths["process_id"].read_text().strip())
    checked_fail_fast_sources = require_fail_fast_sources(repository_root)
    command = paths["command"].read_text()
    recorded_context = (
        metadata.get("run_context") is not None
        or result.get("run_context") is not None
        or metadata.get("task_config") == "tasks/pointnav_gibson_visual.yaml"
        or result.get("task_config") == "tasks/pointnav_gibson_visual.yaml"
        or topology.get("task_config") == "tasks/pointnav_gibson_visual.yaml"
        or "--run_context_manifest" in command
        or "--run_context_dataset" in command
    )
    if recorded_context != context_mode:
        raise RuntimeError("Recorded run context and input evidence disagree")

    if result.get("status") != "completed":
        raise RuntimeError("Run did not complete: {}".format(result))
    if result.get("run_id") != args.run_id or metadata.get("run_id") != args.run_id:
        raise RuntimeError("Run ID mismatch between result and metadata")
    source_commit = result.get("source_commit")
    if source_commit != metadata.get("source_commit"):
        raise RuntimeError("Source commit mismatch between result and metadata")
    runtime_install_sha256 = sha256(paths["runtime_install"])
    if (
        metadata.get("runtime_install_sha256") != runtime_install_sha256
        or result.get("runtime_install_sha256") != runtime_install_sha256
        or topology.get("runtime_install_sha256") != runtime_install_sha256
        or runtime_install.get("active_room_commit") != source_commit
    ):
        raise RuntimeError("Runtime dependency evidence is inconsistent")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise RuntimeError("Run artifacts do not contain a full source commit")
    if context_mode and source_commit != validator_commit:
        raise RuntimeError(
            "Strict Gibson runs require the exact validator source commit"
        )
    if not context_mode and subprocess.run(
        [
            "git",
            "-C",
            str(repository_root),
            "merge-base",
            "--is-ancestor",
            source_commit,
            validator_commit,
        ],
        check=False,
        timeout=30,
    ).returncode != 0:
        raise RuntimeError(
            "Run source commit is not an ancestor of the validator commit"
        )
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
    x11_client_window_id = metadata.get("x11_client_window_id")
    if (
        not isinstance(x11_client_window_id, str)
        or not x11_client_window_id.startswith("0x")
        or result.get("x11_client_window_id") != x11_client_window_id
        or visualization.get("window_id") != x11_client_window_id
    ):
        raise RuntimeError("X11 client window identity is inconsistent")
    display = visualization.get("display", "")
    expected_socket = (
        "/tmp/.X11-unix/X{}".format(display[1:])
        if display.startswith(":") and display[1:].isdigit()
        else None
    )
    if (
        visualization.get("physical_session_confirmed") != "1"
        or expected_socket is None
        or visualization.get("x11_socket") != expected_socket
        or not visualization.get("session_id")
        or int(visualization.get("session_vtnr", 0)) < 1
        or int(visualization.get("xserver_pid", 0)) < 1
        or not visualization.get("xauthority", "").endswith(
            "/gdm/Xauthority"
        )
    ):
        raise RuntimeError("Physical seat0 X11 identity is incomplete")
    recorded_window_size = [
        int(visualization.get("window_width", 0)),
        int(visualization.get("window_height", 0)),
    ]
    window_geometry_checks = int(
        visualization.get("window_geometry_checks", 0)
    )
    last_window_geometry_unix = float(
        visualization.get("last_window_geometry_unix", 0)
    )
    if context_mode and (
        visualization.get("window_locked") != "1"
        or recorded_window_size[0] < STRICT_LIVE_CANVAS_SIZE[0]
        or recorded_window_size[1] < STRICT_LIVE_CANVAS_SIZE[1]
    ):
        raise RuntimeError("Strict live window was not locked at the large size")
    if result.get("requested_max_episode_steps") != args.expected_steps:
        raise RuntimeError("Requested step count changed during the run")
    executed_steps = int(result.get("executed_steps", 0))
    if early_completion:
        if not 0 < executed_steps <= args.expected_steps:
            raise RuntimeError("Executed steps exceed the requested maximum")
        if result.get("pad_episode_to_max_steps"):
            raise RuntimeError("Early-completion run enabled episode padding")
        if result.get("episode_tail_steps") != 0:
            raise RuntimeError("Early-completion run contains episode tail steps")
        if result.get("topology_exploration_steps") != executed_steps:
            raise RuntimeError("Run continued after topology exploration completed")
    else:
        if executed_steps != args.expected_steps:
            raise RuntimeError(
                "Executed {} steps, expected {}".format(
                    executed_steps,
                    args.expected_steps,
                )
            )
        if not result.get("pad_episode_to_max_steps"):
            raise RuntimeError("Exact-step run did not enable episode padding")
    if not result.get("visualization") or visualization.get("confirmed") != "1":
        raise RuntimeError("The physical X11 visualization was not confirmed")
    if result.get("detector_device") != "cuda":
        raise RuntimeError("The verified run did not use CUDA door detection")
    if not result.get("scene_name"):
        raise RuntimeError("Result does not identify the simulated scene")
    actual_episode_id = str(result.get("episode_id", ""))
    actual_scene_id = str(metadata.get("actual_scene_id", ""))
    if (
        not actual_episode_id
        or metadata.get("actual_episode_id") != actual_episode_id
        or result.get("scene_name") != actual_scene_id
    ):
        raise RuntimeError("Habitat episode identity is inconsistent")
    actual_episode_contract_sha256 = result.get("episode_contract_sha256")
    if (
        not isinstance(actual_episode_contract_sha256, str)
        or len(actual_episode_contract_sha256) != 64
        or metadata.get("actual_episode_contract_sha256")
        != actual_episode_contract_sha256
        or topology.get("actual_episode_contract_sha256")
        != actual_episode_contract_sha256
    ):
        raise RuntimeError("Habitat episode contract identity is inconsistent")
    if result.get("finished_at_unix", 0) <= result.get("started_at_unix", 0):
        raise RuntimeError("Invalid run timestamps")
    completion_reason = result.get("completion_reason")
    if completion_reason != "topology_exploration_completed":
        raise RuntimeError("Run does not have the natural topology completion reason")
    if topology.get("completion_reason") != completion_reason:
        raise RuntimeError("Topology completion reason differs from the result")
    window_viewable_checks = int(visualization.get("window_viewable_checks", 0))
    if window_viewable_checks < 3:
        raise RuntimeError("The live window was not monitored throughout the run")
    last_window_viewable_unix = float(
        visualization.get("last_window_viewable_unix", 0)
    )
    if last_window_viewable_unix < result["finished_at_unix"] - 5.0:
        raise RuntimeError("The final live-window check is stale")
    physical_session_checks = int(
        visualization.get("physical_session_checks", 0)
    )
    last_physical_session_check_unix = float(
        visualization.get("last_physical_session_check_unix", 0)
    )
    if (
        physical_session_checks < window_viewable_checks
        or last_physical_session_check_unix
        < result["finished_at_unix"] - 5.0
    ):
        raise RuntimeError("The physical seat0 session was not monitored")
    if context_mode and (
        window_geometry_checks != window_viewable_checks
        or last_window_geometry_unix < result["finished_at_unix"] - 5.0
    ):
        raise RuntimeError("The locked live-window geometry was not monitored")
    if topology.get("run_id") != args.run_id:
        raise RuntimeError("Topology artifact has a foreign run ID")
    if topology.get("process_id") != process_id:
        raise RuntimeError("Topology artifact has a foreign process ID")
    if topology.get("source_commit") != source_commit:
        raise RuntimeError("Topology artifact source commit mismatch")
    if (
        topology.get("actual_episode_id") != actual_episode_id
        or topology.get("actual_scene_id") != actual_scene_id
    ):
        raise RuntimeError("Topology artifact episode identity mismatch")
    snapshot = topology.get("snapshot", {})
    if result.get("topology_room_count") != snapshot.get("room_count"):
        raise RuntimeError("Topology room count mismatch")
    if result.get("topology_edge_count") != snapshot.get("edge_count"):
        raise RuntimeError("Topology edge count mismatch")
    if result.get("topology_transition_count") != topology.get("transition_count"):
        raise RuntimeError("Topology transition count mismatch")
    if result.get("door_crossing_count") != topology.get("door_crossing_count"):
        raise RuntimeError("Door crossing count mismatch")
    if result.get("surviving_door_crossing_count") != topology.get(
        "surviving_door_crossing_count"
    ):
        raise RuntimeError("Surviving door crossing count mismatch")
    if strict_topology:
        if topology.get("transition_count", 0) < 1:
            raise RuntimeError("No room transition was confirmed")
        if topology.get("door_crossing_count", 0) < 1:
            raise RuntimeError("No door crossing was confirmed")
        if topology.get("status") != "cross_room_verified":
            raise RuntimeError("Topology status is not cross_room_verified")
        if not topology.get("requirement_met"):
            raise RuntimeError("Strict topology requirement was not met")

    progress = [
        json.loads(line)
        for line in paths["progress"].read_text().splitlines()
        if line.strip()
    ]
    expected_sequence = list(range(1, executed_steps + 1))
    if [event.get("step") for event in progress] != expected_sequence:
        raise RuntimeError("Control-step sequence is not exactly 1..N")
    if [event.get("simulator_step") for event in progress] != expected_sequence:
        raise RuntimeError("Simulator-step sequence is not exactly 1..N")
    if any(event.get("run_id") != args.run_id for event in progress):
        raise RuntimeError("Progress contains a foreign run ID")
    if any(event.get("process_id") != process_id for event in progress):
        raise RuntimeError("Progress contains a foreign process ID")
    if any(str(event.get("episode_id", "")) != actual_episode_id for event in progress):
        raise RuntimeError("Progress contains a foreign episode ID")
    if any(
        event.get("episode_contract_sha256")
        != actual_episode_contract_sha256
        for event in progress
    ):
        raise RuntimeError("Progress contains a foreign episode contract")
    if any(not event.get("phase") for event in progress):
        raise RuntimeError("Progress contains a step without a runtime phase")
    if any(not event.get("action") for event in progress):
        raise RuntimeError("Progress contains a step without an action label")
    if any("topology_room_count" not in event for event in progress):
        raise RuntimeError("Progress is missing topology state")
    runtime_timing = result.get("runtime_timing", {})
    required_timing_stages = {
        "door_filter",
        "frontier_primary",
        "frontier_primary_wavefront",
        "topology_check",
    }
    if not required_timing_stages.issubset(runtime_timing):
        raise RuntimeError(
            "Missing runtime timing stages: {}".format(
                sorted(required_timing_stages - set(runtime_timing))
            )
        )
    for stage in required_timing_stages:
        stats = runtime_timing[stage]
        if stats.get("count", 0) < 1 or stats.get("total_seconds", -1) < 0:
            raise RuntimeError("Invalid runtime timing for {}".format(stage))

    topology_events = [
        json.loads(line)
        for line in paths["topology_events"].read_text().splitlines()
        if line.strip()
    ]
    if not topology_events:
        raise RuntimeError("Topology event stream is empty")
    if [event.get("sequence") for event in topology_events] != list(
        range(1, len(topology_events) + 1)
    ):
        raise RuntimeError("Topology event sequence is not contiguous")
    if any(event.get("run_id") != args.run_id for event in topology_events):
        raise RuntimeError("Topology events contain a foreign run ID")
    if any(event.get("process_id") != process_id for event in topology_events):
        raise RuntimeError("Topology events contain a foreign process ID")
    if any(
        not 0 <= int(event.get("step", -1)) <= executed_steps
        for event in topology_events
    ):
        raise RuntimeError("Topology event step is outside the episode")
    event_types = {event.get("event_type") for event in topology_events}
    required_event_types = {
        "topology_initialized",
        "exploration_started",
        "topology_exploration_completed",
        "episode_control_completed",
        "run_completed",
    }
    if not required_event_types.issubset(event_types):
        raise RuntimeError(
            "Missing topology events: {}".format(
                sorted(required_event_types - event_types)
            )
        )
    if "room_scan_profile" not in event_types:
        raise RuntimeError("Topology events do not contain room scan profiles")
    if "exploration_aborted" in event_types:
        raise RuntimeError("Topology event stream contains an aborted exploration")
    completion_event_types = [
        "topology_exploration_completed",
        "episode_control_completed",
        "run_completed",
    ]
    completion_events = {
        event_type: [
            event
            for event in topology_events
            if event.get("event_type") == event_type
        ]
        for event_type in completion_event_types
    }
    if any(len(events) != 1 for events in completion_events.values()):
        raise RuntimeError("Completion events are not unique")
    completion_sequences = [
        completion_events[event_type][0]["sequence"]
        for event_type in completion_event_types
    ]
    if completion_sequences != sorted(completion_sequences):
        raise RuntimeError("Completion events are out of order")
    if topology_events[-1].get("event_type") != "run_completed":
        raise RuntimeError("Run completion is not the final topology event")
    observed_event_summary = {
        "event_count": len(topology_events),
        "event_counts": dict(
            sorted(Counter(event["event_type"] for event in topology_events).items())
        ),
        "event_file": paths["topology_events"].name,
    }
    if topology.get("events") != observed_event_summary:
        raise RuntimeError("Topology event summary differs from the event stream")
    if visualization_manifest.get("events") != observed_event_summary:
        raise RuntimeError("Visualization event summary differs from the event stream")
    if visualization_manifest.get("topology") != snapshot:
        raise RuntimeError("Visualization topology differs from the final topology")
    with np.load(paths["room_labels_final"], allow_pickle=False) as room_data:
        if set(room_data.files) != {"room_labels", "occupied", "explored"}:
            raise RuntimeError("Room-label artifact has unexpected arrays")
        room_labels = room_data["room_labels"]
        room_occupied = room_data["occupied"]
        room_explored = room_data["explored"]
    if (
        room_labels.ndim != 2
        or room_occupied.shape != room_labels.shape
        or room_explored.shape != room_labels.shape
    ):
        raise RuntimeError("Room-label artifact arrays have inconsistent shapes")
    if (
        not np.issubdtype(room_labels.dtype, np.integer)
        or np.any(room_labels < 0)
    ):
        raise RuntimeError("Room-label artifact contains invalid labels")
    observed_room_labels = [
        int(label) for label in np.unique(room_labels) if label > 0
    ]
    expected_room_labels = list(range(1, int(snapshot["room_count"]) + 1))
    if observed_room_labels != expected_room_labels:
        raise RuntimeError(
            "Room-label artifact does not cover every topology room"
        )
    observed_room_pixel_counts = {
        str(label): int(np.count_nonzero(room_labels == label))
        for label in observed_room_labels
    }
    if (
        visualization_manifest.get("room_label_ids") != observed_room_labels
        or visualization_manifest.get("room_pixel_counts")
        != observed_room_pixel_counts
        or visualization_manifest.get("room_labels_file")
        != paths["room_labels_final"].name
        or visualization_manifest.get("room_labels_sha256")
        != sha256(paths["room_labels_final"])
    ):
        raise RuntimeError("Room-label manifest does not match its raw arrays")
    if (
        visualization_manifest.get("capture_marker_scheme")
        != "sha256_run_step_v1"
        or visualization_manifest.get("capture_identity_sha256")
        != hashlib.sha256(args.run_id.encode("utf-8")).hexdigest()
    ):
        raise RuntimeError("Visualization capture-marker identity changed")
    topology_completion_event = completion_events[
        "topology_exploration_completed"
    ][0]
    if (
        int(topology_completion_event["step"])
        != int(result.get("topology_exploration_steps", -1))
        or topology.get("topology_exploration_complete_step")
        != result.get("topology_exploration_steps")
    ):
        raise RuntimeError("Topology completion step is inconsistent")
    run_completion_payload = completion_events["run_completed"][0].get(
        "payload",
        {},
    )
    if run_completion_payload.get("completion_reason") != completion_reason:
        raise RuntimeError("Run completion event has the wrong reason")
    episode_control_event = completion_events["episode_control_completed"][0]
    run_completion_event = completion_events["run_completed"][0]
    if (
        int(episode_control_event["step"]) != executed_steps
        or int(
            episode_control_event.get("payload", {}).get(
                "executed_steps",
                -1,
            )
        )
        != executed_steps
        or int(run_completion_event["step"]) != executed_steps
        or int(run_completion_payload.get("executed_steps", -1))
        != executed_steps
    ):
        raise RuntimeError("Control/run completion events have the wrong step")
    if early_completion and int(topology_completion_event["step"]) != executed_steps:
        raise RuntimeError("Early-completion topology event has the wrong step")
    if strict_topology:
        crossing_events = [
            event
            for event in topology_events
            if event.get("event_type") == "door_crossing_confirmed"
        ]
        if not crossing_events or any(
            event.get("payload", {}).get("method") != "trajectory_geometry"
            for event in crossing_events
        ):
            raise RuntimeError(
                "Door crossing was not confirmed by trajectory geometry"
            )
        for event in crossing_events:
            segments = (
                event.get("payload", {})
                .get("evidence", {})
                .get("segments", [])
            )
            if not segments or any(
                not segment.get("confirmed") for segment in segments
            ):
                raise RuntimeError(
                    "Door crossing geometry contains an unconfirmed segment"
                )
        transition_events = [
            event
            for event in topology_events
            if event.get("event_type") == "room_transition_confirmed"
        ]
        if not transition_events or any(
            event.get("payload", {}).get("confirmation_method")
            != "trajectory_geometry"
            for event in transition_events
        ):
            raise RuntimeError(
                "Room transition lacks trajectory geometry confirmation"
            )
        crossing_evidence = [
            event["payload"]["evidence"]
            for event in crossing_events
        ]
        observed_surviving_crossings = surviving_crossing_count(
            crossing_evidence,
            snapshot,
        )
        if (
            observed_surviving_crossings < 1
            or topology.get("surviving_door_crossing_count")
            != observed_surviving_crossings
            or result.get("surviving_door_crossing_count")
            != observed_surviving_crossings
        ):
            raise RuntimeError(
                "No confirmed door crossing survives in the final topology"
            )
        surviving_events = [
            event
            for event in crossing_events
            if crossing_evidence_survives(
                event["payload"]["evidence"],
                snapshot,
            )
        ]
        if not any(
            transition.get("step") == crossing.get("step")
            and transition.get("payload", {}).get("evidence")
            == crossing.get("payload", {}).get("evidence")
            for crossing in surviving_events
            for transition in transition_events
        ):
            raise RuntimeError(
                "No surviving crossing has a matching room transition"
            )

    image_sizes = {
        "desktop": check_image(paths["desktop"], (1280, 720)),
        "window_first": check_image(paths["window_first"], (640, 480)),
        "window_later": check_image(paths["window_later"], (640, 480)),
        "visualization_final": check_image(
            paths["visualization_final"],
            (960, 540),
        ),
        "room_mask_final": check_image(
            paths["room_mask_final"],
            (320, 320),
        ),
    }
    if context_mode:
        image_sizes["window_mid"] = check_image(
            paths["window_mid"],
            (640, 480),
        )
        image_sizes["window_terminal"] = check_image(
            paths["window_terminal"],
            (640, 480),
        )
        image_sizes["desktop_terminal"] = check_image(
            paths["desktop_terminal"],
            (1280, 720),
        )
        strict_capture_names = [
            "window_first",
            "window_later",
            "window_mid",
            "window_terminal",
        ]
        strict_window_size = image_sizes["window_first"]
        if strict_window_size != recorded_window_size or any(
            image_sizes[name] != strict_window_size
            for name in strict_capture_names
        ):
            raise RuntimeError(
                "Strict physical-window captures changed dimensions"
            )
    image_hashes = {
        name: sha256(paths[name])
        for name in (
            "desktop",
            "window_first",
            "window_later",
            "visualization_final",
            "room_mask_final",
            *(
                (
                    "window_mid",
                    "window_terminal",
                    "desktop_terminal",
                )
                if context_mode
                else ()
            ),
        )
    }
    live_capture_names = [
        "window_first",
        "window_later",
    ]
    if context_mode:
        live_capture_names.extend(["window_mid", "window_terminal"])
    live_capture_panel_signatures = {
        name: dashboard_panel_signature(paths[name])
        for name in live_capture_names
    }
    live_capture_panel_stddev = {
        name: check_dashboard_nonblank(paths[name])
        for name in live_capture_names
    }
    panel_metrics = None
    rendered_stage_steps = {}
    rendered_stage_frame_sha256 = {}
    rendered_stage_frame_panel_phash = {}
    capture_receipts = {}
    checkpoint_steps = []
    checkpoint_hashes = []
    checkpoint_panel_signatures = []
    checkpoint_render_frame_hashes = []
    checkpoint_render_frame_panel_phash = []
    if context_mode:
        if executed_steps < max(
            step for _, step in STRICT_VISUAL_CAPTURE_STEPS
        ):
            raise RuntimeError(
                "Strict run ended before all rendered-stage captures"
            )
        stage_dir = run_dir / "window_stages"
        expected_stage_files = set()
        for stage_name, stage_step in STRICT_VISUAL_CAPTURE_STEPS:
            ready_path = stage_dir / "ready_{}.json".format(stage_name)
            ack_path = stage_dir / "ack_{}.txt".format(stage_name)
            receipt_path = stage_dir / "receipt_{}.json".format(stage_name)
            if (
                not ready_path.is_file()
                or not ack_path.is_file()
                or not receipt_path.is_file()
            ):
                raise RuntimeError("Rendered-stage capture handshake is missing")
            ready = json.loads(ready_path.read_text())
            frame_file = "visualization_frames/frame_{:06d}.png".format(
                stage_step
            )
            frame_path = run_dir / frame_file
            frame_sha256 = sha256(frame_path)
            if check_image(frame_path, (1600, 900)) != [1600, 900]:
                raise RuntimeError(
                    "Rendered-stage frame does not use the fixed canvas"
                )
            if ready != {
                "run_id": args.run_id,
                "process_id": process_id,
                "window_id": x11_client_window_id,
                "stage": stage_name,
                "step": stage_step,
                "render_step": stage_step,
                "frame_file": frame_file,
                "frame_sha256": frame_sha256,
            }:
                raise RuntimeError(
                    "Rendered-stage capture request identity mismatch"
                )
            if ack_path.read_text().strip() != args.run_id:
                raise RuntimeError(
                    "Rendered-stage capture acknowledgement mismatch"
                )
            if int(visualization.get("{}_step".format(stage_name), -1)) != stage_step:
                raise RuntimeError(
                    "Rendered-stage capture report has the wrong step"
                )
            progress_event = progress[stage_step - 1]
            if (
                progress_event.get("step") != stage_step
                or progress_event.get("run_id") != args.run_id
                or progress_event.get("process_id") != process_id
            ):
                raise RuntimeError(
                    "Rendered-stage capture is not bound to progress"
                )
            rendered_stage_steps[stage_name] = stage_step
            rendered_stage_frame_sha256[stage_name] = frame_sha256
            rendered_stage_frame_panel_phash[stage_name] = (
                dashboard_panel_signature(frame_path)
            )
            window_file = "window_{}.png".format(stage_name)
            check_capture_marker(
                run_dir / window_file,
                args.run_id,
                stage_step,
                canvas_size=STRICT_LIVE_CANVAS_SIZE,
            )
            check_capture_marker(
                frame_path,
                args.run_id,
                stage_step,
            )
            capture_receipts[stage_name] = validate_capture_receipt(
                receipt_path,
                expected={
                    "run_id": args.run_id,
                    "process_id": process_id,
                    "window_id": x11_client_window_id,
                    "capture_label": stage_name,
                    "step": stage_step,
                    "render_step": stage_step,
                    "frame_file": frame_file,
                    "frame_sha256": frame_sha256,
                    "window_file": window_file,
                },
                run_dir=run_dir,
                expected_window_size=strict_window_size,
            )
            expected_stage_files.update(
                {
                    ready_path.name,
                    ack_path.name,
                    receipt_path.name,
                }
            )
        actual_stage_files = (
            {
                path.name
                for path in stage_dir.iterdir()
                if path.is_file()
            }
            if stage_dir.is_dir()
            else set()
        )
        if actual_stage_files != expected_stage_files:
            raise RuntimeError(
                "Rendered-stage capture file inventory is not exact"
            )
        if len(set(rendered_stage_frame_sha256.values())) != len(
            STRICT_VISUAL_CAPTURE_STEPS
        ):
            raise RuntimeError("Rendered-stage frame artifacts contain frozen pixels")
        require_distinct_panel_signatures(
            list(rendered_stage_frame_panel_phash.values()),
            "Rendered-stage frame artifacts",
        )
        panel_metrics = check_dashboard_panels(
            run_dir / "visualization_frames" / "frame_000005.png",
            paths["visualization_final"],
        )
        terminal_ready = json.loads(paths["terminal_ready"].read_text())
        terminal_frame_sha256 = sha256(paths["visualization_final"])
        if terminal_ready != {
            "run_id": args.run_id,
            "process_id": process_id,
            "window_id": x11_client_window_id,
            "step": executed_steps,
            "render_step": executed_steps,
            "frame_file": paths["visualization_final"].name,
            "frame_sha256": terminal_frame_sha256,
            "completion_reason": completion_reason,
        }:
            raise RuntimeError("Terminal capture request identity mismatch")
        if paths["terminal_ack"].read_text().strip() != args.run_id:
            raise RuntimeError("Terminal capture acknowledgement mismatch")
        if visualization.get("terminal_captured") != "1":
            raise RuntimeError("Terminal physical-window capture was not confirmed")
        if int(visualization.get("terminal_step", -1)) != executed_steps:
            raise RuntimeError("Terminal physical-window capture has the wrong step")
        check_capture_marker(
            paths["window_terminal"],
            args.run_id,
            executed_steps,
            canvas_size=STRICT_LIVE_CANVAS_SIZE,
        )
        check_capture_marker(
            paths["visualization_final"],
            args.run_id,
            executed_steps,
        )
        capture_receipts["terminal"] = validate_capture_receipt(
            paths["terminal_receipt"],
            expected={
                "run_id": args.run_id,
                "process_id": process_id,
                "window_id": x11_client_window_id,
                "capture_label": "terminal",
                "step": executed_steps,
                "render_step": executed_steps,
                "frame_file": paths["visualization_final"].name,
                "frame_sha256": terminal_frame_sha256,
                "window_file": paths["window_terminal"].name,
            },
            run_dir=run_dir,
            expected_window_size=strict_window_size,
        )
        checkpoint_every_steps = int(
            visualization.get("checkpoint_every_steps", 0)
        )
        checkpoint_steps = [
            int(value)
            for value in visualization.get("checkpoint_steps", "").split(",")
            if value
        ]
        checkpoint_count = int(visualization.get("checkpoint_count", -1))
        if checkpoint_every_steps != 100:
            raise RuntimeError("Strict physical-window checkpoint interval changed")
        expected_checkpoint_steps = expected_physical_checkpoint_steps(
            executed_steps,
            checkpoint_every_steps,
        )
        if (
            checkpoint_count != len(expected_checkpoint_steps)
            or checkpoint_steps != expected_checkpoint_steps
        ):
            raise RuntimeError(
                "Physical-window checkpoints are not exactly every 100 steps"
            )
        checkpoint_dir = run_dir / "window_checkpoints"
        checkpoint_paths = [
            checkpoint_dir / "step_{:06d}.png".format(step)
            for step in checkpoint_steps
        ]
        checkpoint_hashes = []
        for checkpoint_path in checkpoint_paths:
            checkpoint_size = check_image(checkpoint_path, (640, 480))
            if checkpoint_size != strict_window_size:
                raise RuntimeError(
                    "Physical-window checkpoint changed locked dimensions"
                )
            checkpoint_hashes.append(sha256(checkpoint_path))
            checkpoint_panel_signatures.append(
                dashboard_panel_signature(checkpoint_path)
            )
        for step in checkpoint_steps:
            checkpoint_label = "{:06d}".format(step)
            ready_path = (
                run_dir
                / "window_checkpoints"
                / "ready_{}.json".format(checkpoint_label)
            )
            ack_path = (
                run_dir
                / "window_checkpoints"
                / "ack_{}.txt".format(checkpoint_label)
            )
            receipt_path = (
                run_dir
                / "window_checkpoints"
                / "receipt_{}.json".format(checkpoint_label)
            )
            if (
                not ready_path.is_file()
                or not ack_path.is_file()
                or not receipt_path.is_file()
            ):
                raise RuntimeError("Physical-window checkpoint handshake is missing")
            ready = json.loads(ready_path.read_text())
            frame_file = "visualization_frames/frame_{}.png".format(
                checkpoint_label
            )
            frame_path = run_dir / frame_file
            frame_sha256 = sha256(frame_path)
            if check_image(frame_path, (1600, 900)) != [1600, 900]:
                raise RuntimeError(
                    "Checkpoint frame does not use the fixed canvas"
                )
            if ready != {
                "run_id": args.run_id,
                "process_id": process_id,
                "window_id": x11_client_window_id,
                "step": step,
                "render_step": step,
                "frame_file": frame_file,
                "frame_sha256": frame_sha256,
            }:
                raise RuntimeError(
                    "Physical-window checkpoint request identity mismatch"
                )
            if ack_path.read_text().strip() != args.run_id:
                raise RuntimeError(
                    "Physical-window checkpoint acknowledgement mismatch"
                )
            checkpoint_render_frame_hashes.append(frame_sha256)
            checkpoint_render_frame_panel_phash.append(
                dashboard_panel_signature(frame_path)
            )
            window_file = "window_checkpoints/step_{}.png".format(
                checkpoint_label
            )
            check_capture_marker(
                run_dir / window_file,
                args.run_id,
                step,
                canvas_size=STRICT_LIVE_CANVAS_SIZE,
            )
            check_capture_marker(frame_path, args.run_id, step)
            capture_receipts["checkpoint_{}".format(checkpoint_label)] = (
                validate_capture_receipt(
                    receipt_path,
                    expected={
                        "run_id": args.run_id,
                        "process_id": process_id,
                        "window_id": x11_client_window_id,
                        "capture_label": "checkpoint_{}".format(
                            checkpoint_label
                        ),
                        "step": step,
                        "render_step": step,
                        "frame_file": frame_file,
                        "frame_sha256": frame_sha256,
                        "window_file": window_file,
                    },
                    run_dir=run_dir,
                    expected_window_size=strict_window_size,
                )
            )
        expected_checkpoint_files = {
            "{}_{:06d}.{}".format(prefix, step, suffix)
            for step in checkpoint_steps
            for prefix, suffix in (
                ("ready", "json"),
                ("ack", "txt"),
                ("receipt", "json"),
                ("step", "png"),
            )
        }
        actual_checkpoint_files = (
            {
                path.name
                for path in checkpoint_dir.iterdir()
                if path.is_file()
            }
            if checkpoint_dir.is_dir()
            else set()
        )
        if actual_checkpoint_files != expected_checkpoint_files:
            raise RuntimeError(
                "Physical-window checkpoint file inventory is not exact"
            )
        if len(set(checkpoint_hashes)) != len(checkpoint_hashes):
            raise RuntimeError("Physical-window checkpoints contain frozen pixels")
        if len(set(checkpoint_render_frame_hashes)) != len(
            checkpoint_render_frame_hashes
        ):
            raise RuntimeError(
                "Physical-window checkpoint render frames contain frozen pixels"
            )
        require_distinct_panel_signatures(
            checkpoint_render_frame_panel_phash,
            "Physical-window checkpoint render frames",
        )
        require_distinct_panel_signatures(
            list(live_capture_panel_signatures.values())
            + checkpoint_panel_signatures,
            "Physical X11 capture sequence",
        )
    frame_count = int(visualization_manifest.get("frame_count", 0))
    frame_every_steps = int(
        visualization_manifest.get("frame_every_steps", 0)
    )
    if frame_every_steps < 1:
        raise RuntimeError("Visualization frame interval is invalid")
    if context_mode and frame_every_steps != 5:
        raise RuntimeError("Strict visualization frame interval changed")
    if frame_every_steps != metadata.get("visualization_frame_every_steps"):
        raise RuntimeError("Visualization frame interval changed during the run")
    if visualization_manifest.get("frame_size") != [1600, 900]:
        raise RuntimeError("Visualization frames do not use the fixed canvas")
    if visualization_manifest.get("final_image_size") != [1920, 1080]:
        raise RuntimeError("Final visualization does not use the fixed canvas")
    if image_sizes["visualization_final"] != [1920, 1080]:
        raise RuntimeError("Final visualization dimensions are inconsistent")
    if (
        visualization_manifest.get("final_image")
        != paths["visualization_final"].name
        or visualization_manifest.get("final_image_sha256")
        != image_hashes["visualization_final"]
    ):
        raise RuntimeError("Final visualization identity is inconsistent")
    if (
        visualization_manifest.get("room_mask_image")
        != paths["room_mask_final"].name
        or visualization_manifest.get("room_mask_image_sha256")
        != image_hashes["room_mask_final"]
        or visualization_manifest.get("room_mask_image_size")
        != image_sizes["room_mask_final"]
    ):
        raise RuntimeError("Colored room-mask image identity is inconsistent")
    last_render_step = int(visualization_manifest.get("last_render_step", -1))
    if last_render_step != executed_steps:
        raise RuntimeError("Final dashboard render is not bound to run completion")
    expected_frame_names = expected_visualization_frames(
        frame_every_steps,
        last_render_step,
    )
    if visualization_manifest.get("frames") != expected_frame_names:
        raise RuntimeError("Visualization frame sequence is incomplete")
    if frame_count != len(expected_frame_names) or frame_count < 2:
        raise RuntimeError("Visualization frame count is invalid")
    frame_paths = [run_dir / relative_path for relative_path in expected_frame_names]
    if any(
        not frame.is_file() for frame in frame_paths
    ):
        raise RuntimeError("Visualization frame manifest is incomplete")
    for frame_path in frame_paths:
        if check_image(frame_path, (1600, 900)) != [1600, 900]:
            raise RuntimeError(
                "Visualization frame does not use the fixed canvas"
            )
        frame_step = int(frame_path.stem.rsplit("_", 1)[1])
        check_capture_marker(frame_path, args.run_id, frame_step)
    if replay_manifest.get("frame_count") != frame_count:
        raise RuntimeError("Replay frame count mismatch")
    if paths["replay"].stat().st_size < 1024:
        raise RuntimeError("Topology replay is empty")
    if "\nmoved to another room\n" in paths["runtime_log"].read_text():
        raise RuntimeError("Runtime contains the old unconditional transition claim")

    if args.run_id not in command or "--detector_device cuda" not in command:
        raise RuntimeError("Command evidence is not bound to this CUDA run")
    run_context = None
    if context_mode:
        manifest = json.loads(paths["input_manifest"].read_text())
        dataset_digest = sha256(paths["input_dataset"])
        if manifest.get("status") != "prepared" or manifest.get("format_version") != 1:
            raise RuntimeError("Run input manifest is incomplete")
        if any(
            manifest.get(key) != value
            for key, value in EXPECTED_GIBSON_CONTEXT.items()
        ):
            raise RuntimeError("Run input manifest does not match pinned Gibson data")
        if dataset_digest != manifest.get("output_dataset_sha256"):
            raise RuntimeError("Run input dataset SHA256 mismatch")
        with gzip.open(paths["input_dataset"], "rt", encoding="utf-8") as stream:
            input_episodes = json.load(stream).get("episodes", [])
        if len(input_episodes) != 1:
            raise RuntimeError("Run input dataset must contain one episode")
        input_episode_contract_sha256 = episode_contract_sha256(
            input_episodes[0]
        )
        if (
            input_episode_contract_sha256
            != manifest.get("selected_episode_contract_sha256")
        ):
            raise RuntimeError("Run input episode contract SHA256 mismatch")
        run_context = metadata.get("run_context")
        if not run_context or result.get("run_context") != run_context:
            raise RuntimeError("Run context differs between metadata and result")
        expected_context = {
            "manifest_file": paths["input_manifest"].name,
            "manifest_sha256": sha256(paths["input_manifest"]),
            "dataset_file": paths["input_dataset"].name,
            "dataset_sha256": dataset_digest,
            "scene_id": manifest["scene_id"],
            "episode_id": manifest["selected_episode_id"],
            "episode_contract_sha256": input_episode_contract_sha256,
            "scene_sha256": manifest["scene_sha256"],
            "navmesh_sha256": manifest["navmesh_sha256"],
            "archive_sha256": manifest["archive_sha256"],
            "pointnav_tree_sha256": manifest["pointnav_tree_sha256"],
            "geodesic_distance": manifest[
                "selected_episode_geodesic_distance"
            ],
        }
        if run_context != expected_context:
            raise RuntimeError("Recorded run context does not match input evidence")
        if result.get("scene_name") != manifest.get("scene_id"):
            raise RuntimeError("Executed scene differs from run context")
        if actual_episode_id != manifest.get("selected_episode_id"):
            raise RuntimeError("Executed episode differs from run context")
        if actual_episode_contract_sha256 != input_episode_contract_sha256:
            raise RuntimeError("Executed episode contract differs from run context")
        if metadata.get("require_topology_transition") is not True:
            raise RuntimeError("Strict run metadata disabled topology validation")
        if result.get("require_topology_transition") is not True:
            raise RuntimeError("Strict run result disabled topology validation")
        if metadata.get("pad_episode_to_max_steps") is not False:
            raise RuntimeError("Strict run metadata enabled episode-tail padding")
        if result.get("pad_episode_to_max_steps") is not False:
            raise RuntimeError("Strict run result enabled episode-tail padding")
        if (
            metadata.get("task_config") != "tasks/pointnav_gibson_visual.yaml"
            or result.get("task_config") != "tasks/pointnav_gibson_visual.yaml"
            or topology.get("task_config") != "tasks/pointnav_gibson_visual.yaml"
        ):
            raise RuntimeError("Strict run did not use the pinned Gibson task")
        if (
            metadata.get("split") != "val"
            or result.get("split") != "val"
            or topology.get("split") != "val"
        ):
            raise RuntimeError("Strict run did not use the val split")
        required_command_fragments = {
            "--task_config tasks/pointnav_gibson_visual.yaml",
            "--split val",
            "--run_context_manifest",
            "--run_context_dataset",
            "--require_topology_transition 1",
            "--pad_episode_to_max_steps 0",
        }
        if any(fragment not in command for fragment in required_command_fragments):
            raise RuntimeError("Command is not bound to the strict run context")

    artifact_hashes, artifact_sizes = collect_artifact_hashes(run_dir)
    repeated_hashes, repeated_sizes = collect_artifact_hashes(run_dir)
    if (
        artifact_hashes != repeated_hashes
        or artifact_sizes != repeated_sizes
    ):
        raise RuntimeError("Run evidence changed while computing its hash closure")
    artifact_closure_sha256 = hashlib.sha256(
        json.dumps(
            {
                "sha256": artifact_hashes,
                "size": artifact_sizes,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    validation_path = run_dir / "validation.json"
    if validation_path.is_file():
        previous_validation = json.loads(validation_path.read_text())
        if (
            previous_validation.get("artifact_sha256") != artifact_hashes
            or previous_validation.get("artifact_size") != artifact_sizes
            or previous_validation.get("artifact_closure_sha256")
            != artifact_closure_sha256
        ):
            raise RuntimeError("Run evidence changed after its prior validation")

    report = {
        "status": "validated",
        "run_id": args.run_id,
        "process_id": process_id,
        "source_commit": source_commit,
        "validator_commit": validator_commit,
        "requested_max_steps": args.expected_steps,
        "steps": executed_steps,
        "early_completion": early_completion,
        "image_sizes": image_sizes,
        "image_sha256": image_hashes,
        "live_capture_panel_phash": live_capture_panel_signatures,
        "live_capture_panel_stddev": live_capture_panel_stddev,
        "dashboard_panel_metrics": panel_metrics,
        "rendered_stage_steps": rendered_stage_steps,
        "rendered_stage_frame_sha256": rendered_stage_frame_sha256,
        "rendered_stage_frame_panel_phash": rendered_stage_frame_panel_phash,
        "physical_window_checkpoint_steps": checkpoint_steps,
        "physical_window_checkpoint_sha256": checkpoint_hashes,
        "physical_window_checkpoint_panel_sha256": (
            checkpoint_panel_signatures
        ),
        "physical_window_checkpoint_render_frame_sha256": (
            checkpoint_render_frame_hashes
        ),
        "physical_window_checkpoint_render_frame_panel_phash": (
            checkpoint_render_frame_panel_phash
        ),
        "scene_name": result["scene_name"],
        "topology_status": topology["status"],
        "topology_room_count": snapshot["room_count"],
        "topology_edge_count": snapshot["edge_count"],
        "topology_transition_count": topology["transition_count"],
        "door_crossing_count": topology["door_crossing_count"],
        "surviving_door_crossing_count": topology[
            "surviving_door_crossing_count"
        ],
        "topology_event_count": len(topology_events),
        "room_label_ids": observed_room_labels,
        "room_pixel_counts": observed_room_pixel_counts,
        "visualization_frame_count": frame_count,
        "window_viewable_checks": window_viewable_checks,
        "locked_window_size": recorded_window_size,
        "window_geometry_checks": window_geometry_checks,
        "capture_receipts": capture_receipts,
        "physical_session_checks": physical_session_checks,
        "replay": paths["replay"].name,
        "runtime_timing": runtime_timing,
        "run_context": run_context,
        "runtime_install": runtime_install,
        "artifact_sha256": artifact_hashes,
        "artifact_size": artifact_sizes,
        "artifact_closure_sha256": artifact_closure_sha256,
        "strict_guard_violations": 0,
        "fail_fast_sources": checked_fail_fast_sources,
        "crossing_evidence": (
            crossing_events[0]["payload"]["evidence"]
            if strict_topology
            else None
        ),
    }
    write_json_atomic(validation_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
