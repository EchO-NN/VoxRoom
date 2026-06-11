from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

from .ipa_runner import IPA_ALGORITHM_IDS, ORIGINAL_ACTION, ORIGINAL_PACKAGE, _ipa_package_root


def inspect_ipa_room_segmentation(workspace: Path | str | None) -> dict[str, Any]:
    root = None if workspace is None else Path(workspace).expanduser()
    package_root = None if root is None else _ipa_package_root(root)
    action_path = None if package_root is None else package_root.parent / "ipa_building_msgs" / "action" / "MapSegmentation.action"
    cfg_path = None if package_root is None else package_root / "cfg" / "RoomSegmentation.cfg"
    sources = {
        "morphological": None if package_root is None else package_root / "common" / "src" / "morphological_segmentation.cpp",
        "distance_transform": None if package_root is None else package_root / "common" / "src" / "distance_segmentation.cpp",
        "voronoi": None if package_root is None else package_root / "common" / "src" / "voronoi_segmentation.cpp",
    }
    return {
        "workspace": None if root is None else str(root),
        "package": ORIGINAL_PACKAGE,
        "package_root": None if package_root is None else str(package_root),
        "git_head": _git_head(package_root.parent if package_root is not None else root),
        "action": ORIGINAL_ACTION,
        "action_path": None if action_path is None else str(action_path),
        "action_exists": bool(action_path and action_path.exists()),
        "cfg_path": None if cfg_path is None else str(cfg_path),
        "cfg_exists": bool(cfg_path and cfg_path.exists()),
        "algorithm_ids": dict(IPA_ALGORITHM_IDS),
        "sources": {name: {"path": None if path is None else str(path), "exists": bool(path and path.exists())} for name, path in sources.items()},
    }


def _git_head(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect ipa_room_segmentation original ROS action contract.")
    parser.add_argument("--workspace")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    report = inspect_ipa_room_segmentation(args.workspace)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not args.strict:
        return 0
    ok = bool(report["git_head"] and report["action_exists"] and report["cfg_exists"] and all(v["exists"] for v in report["sources"].values()))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
