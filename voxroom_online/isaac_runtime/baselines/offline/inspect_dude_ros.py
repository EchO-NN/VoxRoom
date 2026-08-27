from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .dude_runner import inspect_original_dude_repo


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect Incremental_DuDe_ROS source/build contract.")
    parser.add_argument("--repo-root")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    status = inspect_original_dude_repo(None if args.repo_root is None else Path(args.repo_root))
    payload = status.__dict__
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    ok = bool(status.source_tree_present and status.catkin_executable_present and status.git_head)
    return 0 if (not args.strict or ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
