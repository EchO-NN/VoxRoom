from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class OriginalDetrAdapterStatus:
    repo_dir: str
    repo_exists: bool
    git_head: str | None
    checkpoint_path: str
    checkpoint_exists: bool
    checkpoint_sha256: str | None
    source_candidates_present: list[str]
    detector_adapter_verified: bool
    reason: str
    self_test_returncode: int | None = None
    self_test_stdout: str = ""
    self_test_stderr: str = ""


def inspect_original_detr_adapter(
    *,
    repo_dir: Path | str | None = None,
    checkpoint: Path | str | None = None,
    run_self_test: bool = False,
) -> OriginalDetrAdapterStatus:
    repo_raw = str(repo_dir or os.environ.get("ACTIVE_ROOM_SEG_ROOT", "") or "")
    ckpt_raw = str(checkpoint or os.environ.get("TOPOLOGY_DOOR_DETR_CHECKPOINT", "") or "")
    if not ckpt_raw and repo_raw:
        candidate = (
            Path(repo_raw).expanduser()
            / "detr_door_detection"
            / "train_params"
            / "detr_resnet50_4"
            / "final_doors_dataset"
            / "model.pth"
        )
        if candidate.exists():
            ckpt_raw = str(candidate)
    repo = Path(repo_raw).expanduser() if repo_raw else None
    ckpt = Path(ckpt_raw).expanduser() if ckpt_raw else None
    candidates = [
        repo / "detect.py",
        repo / "detr_door_detection" / "run_detr.py",
        repo / "detr" / "models" / "detr.py",
        repo / "models" / "detr.py",
        repo / "src" / "models" / "detr.py",
    ] if repo is not None else []
    present = [str(path) for path in candidates if path.exists()]
    checkpoint_exists = bool(ckpt is not None and ckpt.exists() and ckpt.is_file())
    repo_exists = bool(repo is not None and repo.exists())
    source_ok = bool(repo_exists and present)
    self_test_returncode: int | None = None
    self_test_stdout = ""
    self_test_stderr = ""
    verified = bool(source_ok and checkpoint_exists)
    reason = "adapter_import_not_wired"
    if not repo_exists:
        reason = "active_room_segmentation_repo_missing"
    elif not present:
        reason = "detr_source_candidate_missing"
    elif not checkpoint_exists:
        reason = "checkpoint_missing"
    elif run_self_test:
        proc = _run_self_test(repo=repo, checkpoint=ckpt)
        self_test_returncode = proc.returncode
        self_test_stdout = proc.stdout[-4000:]
        self_test_stderr = proc.stderr[-4000:]
        verified = proc.returncode == 0
        reason = "verified" if verified else "self_test_failed"
    elif verified:
        reason = "source_and_checkpoint_present_self_test_not_run"
    return OriginalDetrAdapterStatus(
        repo_dir="" if repo is None else str(repo),
        repo_exists=repo_exists,
        git_head=_git_head(repo) if repo_exists and repo is not None else None,
        checkpoint_path="" if ckpt is None else str(ckpt),
        checkpoint_exists=checkpoint_exists,
        checkpoint_sha256=_sha256(ckpt) if checkpoint_exists and ckpt is not None else None,
        source_candidates_present=present,
        detector_adapter_verified=verified,
        reason=reason,
        self_test_returncode=self_test_returncode,
        self_test_stdout=self_test_stdout,
        self_test_stderr=self_test_stderr,
    )


def require_verified_original_detr(
    *,
    repo_dir: Path | str | None = None,
    checkpoint: Path | str | None = None,
) -> OriginalDetrAdapterStatus:
    status = inspect_original_detr_adapter(repo_dir=repo_dir, checkpoint=checkpoint, run_self_test=True)
    if not status.detector_adapter_verified:
        raise RuntimeError("original DETR adapter is not verified: %s" % asdict(status))
    return status


def _git_head(path: Path) -> str | None:
    try:
        head = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    return head.strip() or None


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        return None


def make_original_detr_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("PYTHONHOME", None)

    repo_root = Path(__file__).resolve().parents[4]
    keep_pythonpath: list[str] = [str(repo_root)]
    for entry in str(env.get("PYTHONPATH", "")).split(os.pathsep):
        if not entry:
            continue
        lowered = entry.lower()
        if "isaac-sim" in lowered or "omni.isaac" in lowered or "pip_prebundle" in lowered:
            continue
        if entry not in keep_pythonpath:
            keep_pythonpath.append(entry)
    env["PYTHONPATH"] = os.pathsep.join(keep_pythonpath)
    if not env.get("DETR_TORCH_HUB_REPO"):
        candidate = Path(env.get("VOXROOM_REPO_ROOT", Path.cwd())) / "external_baselines" / "facebookresearch_detr"
        if candidate.exists():
            env["DETR_TORCH_HUB_REPO"] = str(candidate)
    return env


def _run_self_test(*, repo: Path | None, checkpoint: Path | None) -> subprocess.CompletedProcess[str]:
    _ = checkpoint
    if repo is None:
        return subprocess.CompletedProcess(args=[], returncode=2, stdout="", stderr="repo missing")
    python_cmd = os.environ.get("TOPOLOGY_DOOR_DETR_PYTHON", "python")
    env = make_original_detr_subprocess_env()
    cmd = shlex.split(python_cmd) + [
        "-m",
        "voxroom_online.isaac_runtime.baselines.topology_active.original_detr_subprocess",
        "--repo-dir",
        str(repo),
        "--self-test",
    ]
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, timeout=180, check=False)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect the original Active_room_segmentation DETR adapter contract.")
    parser.add_argument("--repo-dir")
    parser.add_argument("--checkpoint")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    status = inspect_original_detr_adapter(repo_dir=args.repo_dir, checkpoint=args.checkpoint, run_self_test=bool(args.strict))
    print(json.dumps(asdict(status), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if (not args.strict or status.detector_adapter_verified) else 1


if __name__ == "__main__":
    raise SystemExit(main())
