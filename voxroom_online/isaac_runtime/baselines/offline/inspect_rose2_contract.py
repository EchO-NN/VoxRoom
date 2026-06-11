from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .rose2_runner import ORIGINAL_ROSE2_CONFIRMED_CONTRACT, _discover_original_commit


def inspect_rose2_contract(workspace: Path | str | None) -> dict:
    root = None if workspace is None else Path(workspace).expanduser()
    rose_root = None
    if root is not None:
        for candidate in (root, root / "src" / "ROSE2"):
            if (candidate / "launch" / "ROSE.launch").exists() or (candidate / "ROSE.launch").exists():
                rose_root = candidate
                break
    return {
        "workspace": None if root is None else str(root),
        "rose2_root": None if rose_root is None else str(rose_root),
        "git_head": _discover_original_commit(root),
        "confirmed_contract": ORIGINAL_ROSE2_CONFIRMED_CONTRACT,
        "launch_exists": bool(rose_root and ((rose_root / "launch" / "ROSE.launch").exists() or (rose_root / "ROSE.launch").exists())),
        "rooms_output": "/rooms",
        "features_output": "/features_ROSE2",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect ROSE2 original ROS topic/service contract.")
    parser.add_argument("--workspace")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    report = inspect_rose2_contract(args.workspace)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if (not args.strict or (report["git_head"] and report["launch_exists"])) else 1


if __name__ == "__main__":
    raise SystemExit(main())
