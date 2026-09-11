#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from voxroom_online.isaac_runtime.evaluation.online_roomseg.annotation_schema import (
    RoomsegAnnotation,
    load_annotation,
    save_annotation_atomic,
    snapshot_sha256,
    validate_annotation,
)
from voxroom_online.isaac_runtime.evaluation.online_roomseg.common import write_json_atomic
from voxroom_online.isaac_runtime.evaluation.online_roomseg.snapshot_index import load_index
from voxroom_online.isaac_runtime.evaluation.online_roomseg.snapshot_io import load_snapshot_arrays


REFERENCE_KEYS = (
    "reference_explorable_mask",
    "resolution_m",
    "total_explorable_cells",
    "static_explorable_mask",
    "static_navigable_full_mask",
    "static_execution_navigable_mask",
    "static_map_resolution_m",
    "static_map_bounds_xyxy_m",
    "runtime_map_bounds_xyxy_m",
)


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _reference_hashes(path: Path) -> dict[str, str]:
    with np.load(path, allow_pickle=False) as payload:
        missing = sorted(set(REFERENCE_KEYS).difference(payload.files))
        if missing:
            raise KeyError(f"scene reference is missing arrays {missing}: {path}")
        return {key: _array_sha256(np.asarray(payload[key])) for key in REFERENCE_KEYS}


def _source_references(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(root.rglob("scene_reference.npz")):
        relative = path.relative_to(root)
        if len(relative.parts) < 2:
            continue
        scene_id = str(relative.parts[0])
        previous = result.get(scene_id)
        if previous is not None:
            if _reference_hashes(previous) != _reference_hashes(path):
                raise ValueError(f"source scene has inconsistent references: {scene_id}")
            continue
        result[scene_id] = path
    return result


def _approved_annotations(root: Path) -> dict[str, tuple[Path, RoomsegAnnotation]]:
    result: dict[str, tuple[Path, RoomsegAnnotation]] = {}
    for path in sorted(root.glob("*/last_step.annotation.json")):
        annotation = load_annotation(path)
        if annotation.review.status != "approved":
            continue
        previous = result.get(annotation.scene_id)
        if previous is not None:
            raise ValueError(f"multiple approved annotations for scene {annotation.scene_id}")
        result[annotation.scene_id] = (path, annotation)
    return result


def _target_reference(episode: Mapping[str, Any]) -> Path:
    path = Path(str(episode["scene_dir"])) / "roomseg_coverage_eval" / "scene_reference.npz"
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return path


def transfer(args: argparse.Namespace) -> dict[str, Any]:
    source_annotation_dir = Path(args.source_annotation_dir).resolve()
    source_reference_root = Path(args.source_reference_root).resolve()
    target_index_path = Path(args.target_index).resolve()
    target_annotation_dir = Path(args.target_annotation_dir).resolve()
    target_annotation_dir.mkdir(parents=True, exist_ok=True)

    sources = _approved_annotations(source_annotation_dir)
    source_references = _source_references(source_reference_root)
    target_index = load_index(target_index_path)
    rows: list[dict[str, Any]] = []

    for episode in target_index.get("episodes", []):
        scene_id = str(episode["scene_id"])
        source_item = sources.get(scene_id)
        if source_item is None:
            continue
        source_path, source = source_item
        source_reference = source_references.get(scene_id)
        if source_reference is None:
            # A shared annotation directory may contain several datasets while
            # source_reference_root intentionally selects only one of them.
            continue
        target_reference = _target_reference(episode)
        source_hashes = _reference_hashes(source_reference)
        target_hashes = _reference_hashes(target_reference)
        if source_hashes != target_hashes:
            different = sorted(
                key for key in REFERENCE_KEYS if source_hashes[key] != target_hashes[key]
            )
            raise ValueError(
                f"static map alignment differs for {scene_id}: {different}"
            )

        target_snapshot = load_snapshot_arrays(Path(str(episode["last_snapshot_path"])))
        if source.shape != tuple(target_snapshot.shape):
            raise ValueError(f"snapshot shape differs for {scene_id}")
        if source.segmentation_domain_key != target_snapshot.segmentation_domain_key:
            raise ValueError(f"segmentation domain source differs for {scene_id}")
        if source.output_domain_key != target_snapshot.domain_key:
            raise ValueError(f"output domain source differs for {scene_id}")

        uid = str(episode["episode_uid"])
        out_path = target_annotation_dir / uid / "last_step.annotation.json"
        if out_path.exists() and not bool(args.overwrite):
            raise FileExistsError(str(out_path))
        target = replace(
            source,
            episode_uid=uid,
            run_name=str(episode.get("run_name", "")),
            scene_id=scene_id,
            episode_id=(
                None
                if episode.get("episode_id") is None
                else str(episode.get("episode_id"))
            ),
            last_step=int(episode["last_snapshot_step"]),
            snapshot_path=str(target_snapshot.path),
            navigation_png=next(
                (
                    item.get("navigation_png")
                    for item in episode.get("snapshots", [])
                    if bool(item.get("is_last"))
                ),
                None,
            ),
            snapshot_sha256=snapshot_sha256(target_snapshot.path),
            shape=tuple(target_snapshot.shape),
            generated_gt={},
        )
        validate_annotation(target, target_snapshot)
        save_annotation_atomic(target, out_path)
        rows.append(
            {
                "scene_id": scene_id,
                "source_episode_uid": source.episode_uid,
                "target_episode_uid": uid,
                "source_annotation": str(source_path),
                "target_annotation": str(out_path),
                "source_reference": str(source_reference),
                "target_reference": str(target_reference),
                "reference_array_sha256": source_hashes,
                "target_snapshot_sha256": target.snapshot_sha256,
            }
        )

    rows.sort(key=lambda row: str(row["scene_id"]))
    manifest = {
        "schema_version": "voxroom_static_aligned_annotation_transfer_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "alignment_contract": "all static/reference map arrays are byte-identical",
        "source_annotation_dir": str(source_annotation_dir),
        "source_reference_root": str(source_reference_root),
        "target_index": str(target_index_path),
        "target_annotation_dir": str(target_annotation_dir),
        "approved_scene_count": len(rows),
        "approved_episode_uids": [str(row["target_episode_uid"]) for row in rows],
        "rows": rows,
    }
    approved_uids = set(str(value) for value in manifest["approved_episode_uids"])
    if args.approved_index_out:
        approved_index_path = Path(args.approved_index_out).resolve()
        approved_index = {
            **dict(target_index),
            "episodes": [
                episode
                for episode in target_index.get("episodes", [])
                if str(episode.get("episode_uid")) in approved_uids
            ],
            "selection": {
                "policy": "approved_static_aligned_annotations_only",
                "source_transfer_manifest": str(Path(args.manifest_out).resolve()),
            },
        }
        write_json_atomic(approved_index_path, approved_index)
        manifest["approved_index"] = str(approved_index_path)
    write_json_atomic(Path(args.manifest_out).resolve(), manifest)
    write_json_atomic(
        Path(args.approved_summary_out).resolve(),
        {
            "schema_version": "voxroom_approved_episode_selection_v1",
            "created_at": manifest["created_at"],
            "approved_scene_count": len(rows),
            "approved_episode_uids": manifest["approved_episode_uids"],
            "source_transfer_manifest": str(Path(args.manifest_out).resolve()),
        },
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Transfer approved room-segmentation annotations only when source and "
            "target scene reference grids are byte-identical."
        )
    )
    parser.add_argument("--source-annotation-dir", required=True)
    parser.add_argument("--source-reference-root", required=True)
    parser.add_argument("--target-index", required=True)
    parser.add_argument("--target-annotation-dir", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--approved-summary-out", required=True)
    parser.add_argument("--approved-index-out")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = transfer(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
