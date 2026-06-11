from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def file_sha256(path: Path) -> str | None:
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_payload(*, repo_root: Path, external_paths: dict[str, Path], checkpoint: Path | None = None) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "repo_root": str(repo_root),
        "repo_commit": git_commit(repo_root),
        "external_commits": {name: git_commit(path) for name, path in external_paths.items()},
        "topology_checkpoint": None
        if checkpoint is None
        else {"path": str(checkpoint), "sha256": file_sha256(checkpoint), "exists": Path(checkpoint).exists()},
    }


def write_environment_lock(path: Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

