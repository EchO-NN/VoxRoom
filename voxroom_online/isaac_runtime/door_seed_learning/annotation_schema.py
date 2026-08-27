from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np

from voxroom_online.isaac_runtime.door_seed_learning.schema import (
    COORDINATE_CONVENTION,
    MANUAL_ANNOTATION_SCHEMA_VERSION,
    load_snapshot,
    read_json,
    scalar_value,
    sha256_file,
    write_json_atomic,
)


@dataclass(frozen=True)
class SeedLabel:
    row: int
    col: int
    label: int
    updated_at: str

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "SeedLabel":
        label = int(data["label"])
        if label not in {0, 1}:
            raise ValueError("manual seed label must be 0 or 1")
        return cls(
            row=int(data["row"]),
            col=int(data["col"]),
            label=label,
            updated_at=str(data.get("updated_at", "")),
        )

    def to_dict(self) -> dict[str, object]:
        return {"row": self.row, "col": self.col, "label": self.label, "updated_at": self.updated_at}


@dataclass(frozen=True)
class AnnotationReview:
    status: str = "draft"
    approved_by: str | None = None
    approved_at: str | None = None
    notes: str = ""

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "AnnotationReview":
        raw = dict(data or {})
        status = str(raw.get("status", "draft"))
        if status not in {"draft", "approved"}:
            raise ValueError("annotation review status must be draft or approved")
        return cls(
            status=status,
            approved_by=None if raw.get("approved_by") is None else str(raw.get("approved_by")),
            approved_at=None if raw.get("approved_at") is None else str(raw.get("approved_at")),
            notes=str(raw.get("notes", "")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ManualFinalSeedAnnotation:
    schema_version: str
    scene_uid: str
    final_snapshot_path: str
    final_snapshot_sha256: str
    map_info_hash: str
    coordinate_convention: str = COORDINATE_CONVENTION
    labels: tuple[SeedLabel, ...] = field(default_factory=tuple)
    review: AnnotationReview = field(default_factory=AnnotationReview)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "ManualFinalSeedAnnotation":
        if data.get("schema_version") != MANUAL_ANNOTATION_SCHEMA_VERSION:
            raise ValueError("unsupported manual door seed annotation schema")
        labels = tuple(SeedLabel.from_mapping(item) for item in data.get("labels", []))
        coordinates = [(item.row, item.col) for item in labels]
        if len(set(coordinates)) != len(coordinates):
            raise ValueError("manual seed annotation contains duplicate coordinates")
        return cls(
            schema_version=str(data["schema_version"]),
            scene_uid=str(data["scene_uid"]),
            final_snapshot_path=str(data["final_snapshot_path"]),
            final_snapshot_sha256=str(data["final_snapshot_sha256"]),
            map_info_hash=str(data["map_info_hash"]),
            coordinate_convention=str(data.get("coordinate_convention", COORDINATE_CONVENTION)),
            labels=labels,
            review=AnnotationReview.from_mapping(data.get("review")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scene_uid": self.scene_uid,
            "final_snapshot_path": self.final_snapshot_path,
            "final_snapshot_sha256": self.final_snapshot_sha256,
            "map_info_hash": self.map_info_hash,
            "coordinate_convention": self.coordinate_convention,
            "labels": [item.to_dict() for item in sorted(self.labels, key=lambda item: (item.row, item.col))],
            "review": self.review.to_dict(),
        }

    def label_map(self) -> dict[tuple[int, int], int]:
        return {(item.row, item.col): item.label for item in self.labels}

    def with_label(self, row: int, col: int, label: int) -> "ManualFinalSeedAnnotation":
        if int(label) not in {0, 1}:
            raise ValueError("manual seed label must be 0 or 1")
        values = {(item.row, item.col): item for item in self.labels}
        values[(int(row), int(col))] = SeedLabel(int(row), int(col), int(label), now_iso())
        return replace(
            self,
            labels=tuple(sorted(values.values(), key=lambda item: (item.row, item.col))),
            review=replace(self.review, status="draft", approved_by=None, approved_at=None),
        )

    def without_label(self, row: int, col: int) -> "ManualFinalSeedAnnotation":
        labels = tuple(item for item in self.labels if (item.row, item.col) != (int(row), int(col)))
        return replace(self, labels=labels, review=replace(self.review, status="draft", approved_by=None, approved_at=None))


def create_initial_annotation(scene_dir: str | Path) -> ManualFinalSeedAnnotation:
    scene = Path(scene_dir)
    manifest = read_json(scene / "manifest.json")
    final_record = _final_record(manifest)
    snapshot_path = scene / str(final_record["path"])
    arrays = load_snapshot(snapshot_path)
    return ManualFinalSeedAnnotation(
        schema_version=MANUAL_ANNOTATION_SCHEMA_VERSION,
        scene_uid=str(manifest["scene_uid"]),
        final_snapshot_path=str(final_record["path"]),
        final_snapshot_sha256=sha256_file(snapshot_path),
        map_info_hash=str(scalar_value(arrays, "map_info_hash", "")),
    )


def load_annotation(path: str | Path) -> ManualFinalSeedAnnotation:
    return ManualFinalSeedAnnotation.from_mapping(read_json(path))


def save_annotation(annotation: ManualFinalSeedAnnotation, path: str | Path) -> None:
    write_json_atomic(path, annotation.to_dict())


def validate_annotation(
    annotation: ManualFinalSeedAnnotation,
    scene_dir: str | Path,
    *,
    require_approved: bool = False,
    snapshot_arrays: Mapping[str, object] | None = None,
) -> set[tuple[int, int]]:
    scene = Path(scene_dir)
    manifest = read_json(scene / "manifest.json")
    if annotation.scene_uid != str(manifest["scene_uid"]):
        raise ValueError("manual annotation scene_uid mismatch")
    final_record = _final_record(manifest)
    if annotation.final_snapshot_path != str(final_record["path"]):
        raise ValueError("manual annotation points to a different final snapshot")
    snapshot_path = scene / annotation.final_snapshot_path
    if sha256_file(snapshot_path) != annotation.final_snapshot_sha256:
        raise ValueError("manual annotation final_snapshot_sha256 mismatch")
    arrays = load_snapshot(snapshot_path) if snapshot_arrays is None else snapshot_arrays
    map_hash = str(scalar_value(arrays, "map_info_hash", ""))
    if map_hash != annotation.map_info_hash or map_hash != str(manifest["map"]["map_info_hash"]):
        raise ValueError("manual annotation map_info_hash mismatch")
    final_seed_set = {tuple(int(v) for v in rc) for rc in np.asarray(arrays["raw_seed_rc"], dtype=np.int32)}
    labels = annotation.label_map()
    extra = set(labels) - final_seed_set
    if extra:
        raise ValueError("manual annotation contains coordinates absent from final raw seeds")
    if annotation.review.status == "approved" or require_approved:
        if annotation.review.status != "approved":
            raise ValueError("manual annotation is not approved")
        if set(labels) != final_seed_set:
            raise ValueError("approved manual annotation must label every final raw seed")
    return final_seed_set


def approve_annotation(
    annotation: ManualFinalSeedAnnotation,
    scene_dir: str | Path,
    *,
    approved_by: str,
    notes: str = "",
) -> ManualFinalSeedAnnotation:
    final_seed_set = validate_annotation(annotation, scene_dir)
    if set(annotation.label_map()) != final_seed_set:
        raise ValueError("cannot approve while final raw seeds remain unlabeled")
    return replace(
        annotation,
        review=AnnotationReview(
            status="approved",
            approved_by=str(approved_by),
            approved_at=now_iso(),
            notes=str(notes),
        ),
    )


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _final_record(manifest: Mapping[str, object]) -> Mapping[str, object]:
    finals = [item for item in manifest.get("snapshots", []) if bool(item.get("is_final", False))]
    if len(finals) != 1:
        raise ValueError("collection manifest must contain exactly one final snapshot")
    if int(manifest.get("final_snapshot_decision_id", -1)) != int(finals[0]["decision_id"]):
        raise ValueError("collection manifest final snapshot id is inconsistent")
    return finals[0]
