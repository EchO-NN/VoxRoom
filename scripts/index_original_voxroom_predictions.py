#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Mapping


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _approved(annotation_dir: Path, uid: str) -> bool:
    path = annotation_dir / uid / "last_step.annotation.json"
    if not path.is_file():
        return False
    raw = json.loads(path.read_text(encoding="utf-8"))
    review = raw.get("review", {})
    return isinstance(review, Mapping) and str(review.get("status")) == "approved"


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an identity prediction index for the original saved VoxRoom outputs.")
    parser.add_argument("--source-index", required=True)
    parser.add_argument("--annotation-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    source_path = Path(args.source_index).expanduser().resolve()
    annotation_dir = Path(args.annotation_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    episodes = []
    processed = 0
    for episode in source.get("episodes", []):
        uid = str(episode["episode_uid"])
        if not _approved(annotation_dir, uid):
            continue
        copied = dict(episode)
        copied_snapshots = []
        for snapshot in episode.get("snapshots", []):
            record = dict(snapshot)
            record["source_snapshot_path"] = str(snapshot["snapshot_path"])
            copied_snapshots.append(record)
            processed += 1
        copied["snapshots"] = copied_snapshots
        episodes.append(copied)
    manifest = {
        "schema_version": "voxroom_door_seed_ablation_identity_index_v1",
        "variant": "production_full_model_original",
        "candidate_policy": "original_online_runtime_output",
        "source_history_limitation": None,
        "index": str(source_path),
        "index_sha256": _sha256(source_path),
        "annotation_dir": str(annotation_dir),
        "checkpoint": None,
        "checkpoint_sha256": None,
        "learning_config": None,
        "episode_count": len(episodes),
        "snapshot_count": processed,
        "processed": processed,
        "started_at_unix": time.time(),
        "finished_at_unix": time.time(),
        "elapsed_seconds": 0.0,
        "complete": True,
        "episodes": episodes,
    }
    temp = out_dir / "prediction_index.json.tmp"
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, out_dir / "prediction_index.json")
    print(json.dumps({"episode_count": len(episodes), "snapshot_count": processed}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
