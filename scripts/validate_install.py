#!/usr/bin/env python3
import gzip
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import habitat
import habitat_sim
import igraph
import matplotlib
import numpy
import numpy as np
import seaborn
import torch
import torchvision
import treelib


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_ROOM_UPSTREAM_COMMIT = "a941e3a5a1c8e16920a158fa0a9198c95be9c978"
SG_NAV_COMMIT = "d56863c96dea311aaa67fb0d39a1a8ccc3f0487f"
DETR_COMMIT = "29901c51d7fe8712168b8d0d64351170bc0f83e0"
DOOR_WEIGHTS_SHA256 = "d971e3b760421eb29665c1ca986854ce8ec57ddcc50209fcc77125c5e3cef7ec"
DETR_DIR = ROOT / "third_party" / "detr"
WEIGHTS_PATH = (
    ROOT
    / "detr_door_detection"
    / "train_params"
    / "detr_resnet50_4"
    / "final_doors_dataset"
    / "model.pth"
)
DATASET_PATH = (
    ROOT
    / "data"
    / "datasets"
    / "pointnav"
    / "habitat-test-scenes"
    / "v1"
    / "train"
    / "train.json.gz"
)
SCENE_DIR = ROOT / "data" / "scene_datasets" / "habitat-test-scenes"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repo, *args):
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
    ).strip()


def require_equal(name, actual, expected):
    if actual != expected:
        raise RuntimeError(
            "{} mismatch: {!r}, expected {!r}".format(name, actual, expected)
        )


def main():
    required = [
        ROOT / ".git",
        DETR_DIR / "hubconf.py",
        WEIGHTS_PATH,
        DATASET_PATH,
        SCENE_DIR,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing runtime assets: {}".format(", ".join(missing)))

    active_room_commit = git_output(ROOT, "rev-parse", "HEAD")
    detr_commit = git_output(DETR_DIR, "rev-parse", "HEAD")
    upstream_is_ancestor = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "merge-base",
            "--is-ancestor",
            ACTIVE_ROOM_UPSTREAM_COMMIT,
            active_room_commit,
        ],
        check=False,
    )
    if upstream_is_ancestor.returncode != 0:
        raise RuntimeError(
            "Active Room upstream commit is not an ancestor of the reproduction"
        )
    if git_output(ROOT, "status", "--porcelain"):
        raise RuntimeError("Active Room reproduction checkout is dirty")
    require_equal("DETR commit", detr_commit, DETR_COMMIT)
    if git_output(DETR_DIR, "status", "--porcelain"):
        raise RuntimeError("Pinned DETR checkout is dirty")

    require_equal("Python", "{}.{}".format(sys.version_info.major, sys.version_info.minor), "3.9")
    require_equal("Habitat-Lab", habitat.__version__, "0.2.1")
    require_equal("Habitat-Sim", habitat_sim.__version__, "0.2.4")
    require_equal("NumPy", numpy.__version__, "1.26.4")
    require_equal("PyTorch", torch.__version__, "2.7.1+cu128")
    require_equal("TorchVision", torchvision.__version__, "0.22.1+cu128")
    require_equal("PyTorch CUDA", torch.version.cuda, "12.8")
    require_equal("igraph", igraph.__version__, "0.11.9")
    require_equal("Matplotlib", matplotlib.__version__, "3.9.4")
    require_equal("Seaborn", seaborn.__version__, "0.13.2")
    require_equal("TreeLib", importlib.metadata.version("treelib"), "1.7.1")
    require_equal("door checkpoint SHA256", sha256(WEIGHTS_PATH), DOOR_WEIGHTS_SHA256)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    habitat_root = Path(habitat.__file__).resolve().parents[1]
    sg_nav_root = habitat_root.parent
    require_equal("SG-Nav commit", git_output(sg_nav_root, "rev-parse", "HEAD"), SG_NAV_COMMIT)
    habitat_diff = subprocess.run(
        ["git", "-C", str(sg_nav_root), "diff", "--quiet", "--", "habitat-lab"],
        check=False,
    )
    if habitat_diff.returncode != 0:
        raise RuntimeError("The Habitat-Lab source used by this environment is dirty")

    with gzip.open(DATASET_PATH, "rt", encoding="utf-8") as dataset_file:
        dataset = json.load(dataset_file)
    episodes = dataset.get("episodes", [])
    if len(episodes) != 10000:
        raise RuntimeError("Unexpected Habitat test episode count: {}".format(len(episodes)))
    scene_ids = sorted({episode["scene_id"] for episode in episodes})
    missing_scenes = [
        scene_id for scene_id in scene_ids if not (ROOT / scene_id).is_file()
    ]
    if missing_scenes:
        raise FileNotFoundError("Dataset references missing scenes: {}".format(missing_scenes))

    os.environ["ACTIVE_ROOM_DETR_DIR"] = str(DETR_DIR)
    os.environ["ACTIVE_ROOM_DETECTOR_DEVICE"] = "cuda"
    sys.path.insert(0, str(ROOT))
    from detr_door_detection.run_detr import run_detr

    door_mask, visualization, vertical_mask = run_detr(
        np.zeros((256, 256, 3), dtype=np.float32),
        device="cuda",
    )
    if door_mask.shape != (256, 256) or vertical_mask.shape != (256, 256):
        raise RuntimeError("Door detector returned invalid mask dimensions")
    if visualization.shape != (256, 256, 3):
        raise RuntimeError("Door detector returned invalid visualization dimensions")

    report = {
        "active_room_commit": active_room_commit,
        "active_room_upstream_commit": ACTIVE_ROOM_UPSTREAM_COMMIT,
        "cuda_available": True,
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "dataset_episode_count": len(episodes),
        "dataset_scenes": scene_ids,
        "detr_commit": detr_commit,
        "door_detector_cuda_inference": True,
        "door_weights_sha256": DOOR_WEIGHTS_SHA256,
        "habitat_path": str(Path(habitat.__file__).resolve()),
        "habitat_sim_version": habitat_sim.__version__,
        "igraph_version": igraph.__version__,
        "numpy_version": numpy.__version__,
        "python": str(Path(sys.executable).resolve()),
        "seaborn_version": seaborn.__version__,
        "sg_nav_commit": SG_NAV_COMMIT,
        "torch_version": torch.__version__,
        "treelib_path": str(Path(treelib.__file__).resolve()),
    }
    output = ROOT / "reproduction_install.json"
    temporary = output.with_name(".{}.{}.tmp".format(output.name, os.getpid()))
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
