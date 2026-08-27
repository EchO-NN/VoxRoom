from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .common import now_iso, read_json, write_json_atomic


CSR_SCHEMA_VERSION = "voxroom_online_roomseg_csr_v1"
CSR_RUBRIC_VERSION = "paper_online_roomseg_csr_v1"


@dataclass(frozen=True)
class CsrReview:
    episode_uid: str
    step: int
    snapshot_path: str
    csr: int
    reviewer: str = "annotator"
    reviewed_at: str = ""
    rubric_version: str = CSR_RUBRIC_VERSION
    notes: str = ""

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CsrReview":
        if data.get("schema_version") != CSR_SCHEMA_VERSION:
            raise ValueError("unsupported CSR schema: %s" % data.get("schema_version"))
        csr = int(data.get("csr"))
        if csr not in {0, 1}:
            raise ValueError("csr must be 0 or 1")
        return cls(
            episode_uid=str(data["episode_uid"]),
            step=int(data["step"]),
            snapshot_path=str(data.get("snapshot_path", "")),
            csr=csr,
            reviewer=str(data.get("reviewer", "annotator")),
            reviewed_at=str(data.get("reviewed_at") or now_iso()),
            rubric_version=str(data.get("rubric_version", CSR_RUBRIC_VERSION)),
            notes=str(data.get("notes", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        reviewed_at = self.reviewed_at or now_iso()
        return {
            "schema_version": CSR_SCHEMA_VERSION,
            "episode_uid": self.episode_uid,
            "step": int(self.step),
            "snapshot_path": self.snapshot_path,
            "csr": int(self.csr),
            "reviewer": self.reviewer,
            "reviewed_at": reviewed_at,
            "rubric_version": self.rubric_version,
            "notes": self.notes,
        }


def csr_path(csr_dir: Path, episode_uid: str, snapshot_stem: str) -> Path:
    return Path(csr_dir) / str(episode_uid) / f"{snapshot_stem}.csr.json"


def load_csr(path: Path) -> CsrReview:
    return CsrReview.from_mapping(read_json(Path(path)))


def save_csr_atomic(review: CsrReview, path: Path) -> None:
    write_json_atomic(Path(path), review.to_dict())
