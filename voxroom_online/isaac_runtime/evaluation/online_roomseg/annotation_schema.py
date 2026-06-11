from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .common import now_iso, read_json, write_json_atomic


ANNOTATION_SCHEMA_VERSION = "voxroom_online_roomseg_annotation_v1"
VALID_REVIEW_STATUS = {"draft", "approved", "rejected"}


@dataclass(frozen=True)
class LineAnnotation:
    id: str
    p0_rc: tuple[int, int]
    p1_rc: tuple[int, int]
    width_cells: int
    kind: str = "separator"
    comment: str = ""

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "LineAnnotation":
        return cls(
            id=str(data.get("id", "")),
            p0_rc=_rc_tuple(data.get("p0_rc", (0, 0))),
            p1_rc=_rc_tuple(data.get("p1_rc", (0, 0))),
            width_cells=max(1, int(data.get("width_cells", 1))),
            kind=str(data.get("kind", "separator")),
            comment=str(data.get("comment", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "p0_rc": [int(self.p0_rc[0]), int(self.p0_rc[1])],
            "p1_rc": [int(self.p1_rc[0]), int(self.p1_rc[1])],
            "width_cells": int(self.width_cells),
            "kind": self.kind,
            "comment": self.comment,
        }


@dataclass(frozen=True)
class MergeGroup:
    id: str
    component_ids: tuple[int, ...]
    comment: str = ""

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "MergeGroup":
        return cls(
            id=str(data.get("id", "")),
            component_ids=tuple(int(v) for v in data.get("component_ids", [])),
            comment=str(data.get("comment", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "component_ids": [int(v) for v in self.component_ids], "comment": self.comment}


@dataclass(frozen=True)
class ReviewState:
    status: str = "draft"
    approved_by: str | None = None
    approved_at: str | None = None
    notes: str = ""

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "ReviewState":
        raw = dict(data or {})
        status = str(raw.get("status", "draft"))
        if status not in VALID_REVIEW_STATUS:
            raise ValueError("invalid review status: %s" % status)
        return cls(
            status=status,
            approved_by=None if raw.get("approved_by") is None else str(raw.get("approved_by")),
            approved_at=None if raw.get("approved_at") is None else str(raw.get("approved_at")),
            notes=str(raw.get("notes", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class RoomsegAnnotation:
    schema_version: str
    episode_uid: str
    run_name: str
    scene_id: str
    episode_id: str | None
    last_step: int
    snapshot_path: str
    navigation_png: str | None
    snapshot_sha256: str
    shape: tuple[int, int]
    coordinate_convention: str = "row_col_image_origin_upper_left"
    domain_key: str = "navigation_free_room_domain"
    line_width_cells_default: int = 3
    preclose_radius_cells: int = 1
    min_room_area_cells: int = 1
    split_lines: tuple[LineAnnotation, ...] = field(default_factory=tuple)
    merge_groups: tuple[MergeGroup, ...] = field(default_factory=tuple)
    generated_gt: dict[str, Any] = field(default_factory=dict)
    review: ReviewState = field(default_factory=ReviewState)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RoomsegAnnotation":
        if data.get("schema_version") != ANNOTATION_SCHEMA_VERSION:
            raise ValueError("unsupported annotation schema: %s" % data.get("schema_version"))
        return cls(
            schema_version=str(data["schema_version"]),
            episode_uid=str(data["episode_uid"]),
            run_name=str(data.get("run_name", "")),
            scene_id=str(data.get("scene_id", "")),
            episode_id=None if data.get("episode_id") is None else str(data.get("episode_id")),
            last_step=int(data["last_step"]),
            snapshot_path=str(data["snapshot_path"]),
            navigation_png=None if data.get("navigation_png") is None else str(data.get("navigation_png")),
            snapshot_sha256=str(data.get("snapshot_sha256", "")),
            shape=_shape_tuple(data["shape"]),
            coordinate_convention=str(data.get("coordinate_convention", "row_col_image_origin_upper_left")),
            domain_key=str(data.get("domain_key", "navigation_free_room_domain")),
            line_width_cells_default=max(1, int(data.get("line_width_cells_default", 3))),
            preclose_radius_cells=max(0, int(data.get("preclose_radius_cells", 1))),
            min_room_area_cells=max(1, int(data.get("min_room_area_cells", 1))),
            split_lines=tuple(LineAnnotation.from_mapping(item) for item in data.get("split_lines", [])),
            merge_groups=tuple(MergeGroup.from_mapping(item) for item in data.get("merge_groups", [])),
            generated_gt=dict(data.get("generated_gt", {}) or {}),
            review=ReviewState.from_mapping(data.get("review")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_uid": self.episode_uid,
            "run_name": self.run_name,
            "scene_id": self.scene_id,
            "episode_id": self.episode_id,
            "last_step": int(self.last_step),
            "snapshot_path": self.snapshot_path,
            "navigation_png": self.navigation_png,
            "snapshot_sha256": self.snapshot_sha256,
            "shape": [int(self.shape[0]), int(self.shape[1])],
            "coordinate_convention": self.coordinate_convention,
            "domain_key": self.domain_key,
            "line_width_cells_default": int(self.line_width_cells_default),
            "preclose_radius_cells": int(self.preclose_radius_cells),
            "min_room_area_cells": int(self.min_room_area_cells),
            "split_lines": [line.to_dict() for line in self.split_lines],
            "merge_groups": [group.to_dict() for group in self.merge_groups],
            "generated_gt": dict(self.generated_gt),
            "review": self.review.to_dict(),
        }


def snapshot_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_annotation(path: Path) -> RoomsegAnnotation:
    return RoomsegAnnotation.from_mapping(read_json(Path(path)))


def save_annotation_atomic(annotation: RoomsegAnnotation, path: Path) -> None:
    write_json_atomic(Path(path), annotation.to_dict())


def validate_annotation(annotation: RoomsegAnnotation, snapshot_arrays, *, allow_snapshot_changed: bool = False) -> list[str]:
    warnings: list[str] = []
    if annotation.shape != tuple(snapshot_arrays.shape):
        raise ValueError("annotation shape does not match snapshot shape")
    current_sha = snapshot_sha256(Path(annotation.snapshot_path))
    if annotation.snapshot_sha256 and current_sha != annotation.snapshot_sha256:
        if not allow_snapshot_changed:
            raise ValueError("annotation snapshot_sha256 mismatch")
        warnings.append("snapshot_sha256_changed")
    h, w = annotation.shape
    for line in annotation.split_lines:
        if int(line.width_cells) < 1:
            raise ValueError("line width must be >= 1")
        for rc in (line.p0_rc, line.p1_rc):
            if not (0 <= int(rc[0]) < h and 0 <= int(rc[1]) < w):
                warnings.append("line_endpoint_out_of_bounds:%s" % line.id)
    if annotation.review.status not in VALID_REVIEW_STATUS:
        raise ValueError("invalid annotation review status")
    return warnings


def approved_or_raise(annotation: RoomsegAnnotation) -> None:
    if annotation.review.status != "approved":
        raise ValueError("annotation is not approved: %s" % annotation.review.status)


def make_initial_annotation(
    *,
    episode: Mapping[str, Any],
    snapshot_arrays,
    line_width_cells: int,
    preclose_radius_cells: int,
    min_room_area_cells: int = 1,
) -> RoomsegAnnotation:
    return RoomsegAnnotation(
        schema_version=ANNOTATION_SCHEMA_VERSION,
        episode_uid=str(episode["episode_uid"]),
        run_name=str(episode.get("run_name", "")),
        scene_id=str(episode.get("scene_id", "")),
        episode_id=None if episode.get("episode_id") is None else str(episode.get("episode_id")),
        last_step=int(episode["last_snapshot_step"]),
        snapshot_path=str(snapshot_arrays.path),
        navigation_png=episode.get("last_navigation_png") or _last_navigation_png(episode),
        snapshot_sha256=snapshot_sha256(Path(snapshot_arrays.path)),
        shape=tuple(snapshot_arrays.shape),
        domain_key=str(snapshot_arrays.domain_key),
        line_width_cells_default=max(1, int(line_width_cells)),
        preclose_radius_cells=max(0, int(preclose_radius_cells)),
        min_room_area_cells=max(1, int(min_room_area_cells)),
        review=ReviewState(status="draft", approved_at=None),
    )


def approved_review(approved_by: str = "annotator", notes: str = "") -> ReviewState:
    return ReviewState(status="approved", approved_by=approved_by, approved_at=now_iso(), notes=notes)


def _rc_tuple(value: Any) -> tuple[int, int]:
    seq = list(value)
    if len(seq) != 2:
        raise ValueError("expected [row, col]")
    return int(seq[0]), int(seq[1])


def _shape_tuple(value: Any) -> tuple[int, int]:
    seq = list(value)
    if len(seq) != 2:
        raise ValueError("expected [height, width]")
    return int(seq[0]), int(seq[1])


def _last_navigation_png(episode: Mapping[str, Any]) -> str | None:
    last_path = str(episode.get("last_snapshot_path", ""))
    for snap in episode.get("snapshots", []):
        if str(snap.get("snapshot_path")) == last_path:
            return snap.get("navigation_png")
    return None

