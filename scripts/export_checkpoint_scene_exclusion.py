#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import torch

from voxroom_online.isaac_runtime.evaluation.online_roomseg.common import write_json_atomic


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export train/validation scene IDs embedded in a door-seed checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--scene-prefix", default="")
    parser.add_argument("--exclude-substring", action="append", default=[])
    args = parser.parse_args(argv)

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    training = {str(value) for value in checkpoint.get("training_scene_ids", [])}
    validation = {str(value) for value in checkpoint.get("validation_scene_ids", [])}

    def selected(value: str) -> bool:
        return bool(value.startswith(str(args.scene_prefix))) and not any(
            token in value for token in args.exclude_substring
        )

    training = {value for value in training if selected(value)}
    validation = {value for value in validation if selected(value)}
    excluded = sorted(training | validation)
    payload = {
        "schema_version": "voxroom_checkpoint_training_scene_exclusion_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset_name),
        "policy": "exclude every scene used in checkpoint training or validation",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "training_scene_ids": sorted(training),
        "validation_scene_ids": sorted(validation),
        "excluded_scene_count": len(excluded),
        "excluded_scene_ids": excluded,
    }
    write_json_atomic(Path(args.output).resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
