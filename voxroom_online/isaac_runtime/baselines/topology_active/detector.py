from __future__ import annotations

import os
import json
import shlex
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..data_contract import MapInfo
from .schema import DoorPointCandidate


@dataclass(frozen=True)
class DoorDetection2D:
    bbox_xyxy: tuple[float, float, float, float]
    score: float
    class_id: int | None = None
    mask: np.ndarray | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int | None = None
    height: int | None = None


class DoorDetector:
    name = "base"
    available = False
    uses_oracle_semantics = False

    def detect(self, rgb: np.ndarray) -> list[DoorDetection2D]:
        raise NotImplementedError


class DisabledDoorDetector(DoorDetector):
    name = "disabled"
    available = True

    def detect(self, rgb: np.ndarray) -> list[DoorDetection2D]:
        _ = rgb
        return []


class OriginalDetrDoorDetector(DoorDetector):
    name = "original_detr"

    def __init__(
        self,
        *,
        repo_dir: str | Path | None = None,
        checkpoint: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        repo_raw = str(repo_dir or os.environ.get("ACTIVE_ROOM_SEG_ROOT", "") or "")
        self.repo_dir = None if not repo_raw else Path(repo_raw).expanduser()
        configured_checkpoint = checkpoint if checkpoint is not None else checkpoint_path
        self.checkpoint_path = str(configured_checkpoint or os.environ.get("TOPOLOGY_DOOR_DETR_CHECKPOINT") or "")
        from .original_detr_adapter import inspect_original_detr_adapter

        status = inspect_original_detr_adapter(repo_dir=self.repo_dir, checkpoint=self.checkpoint_path, run_self_test=True)
        self.active_repo_commit = status.git_head
        self.checkpoint_sha256 = status.checkpoint_sha256
        self.detector_adapter_verified = bool(status.detector_adapter_verified)
        self.adapter_status = status
        self.available = bool(status.checkpoint_exists and status.repo_exists)

    def detect(self, rgb: np.ndarray) -> list[DoorDetection2D]:
        return self.detect_batch([rgb])[0]

    def detect_batch(self, rgb_frames: list[np.ndarray]) -> list[list[DoorDetection2D]]:
        if not self.available:
            raise RuntimeError("original DETR door detector is unavailable")
        if not self.detector_adapter_verified:
            raise RuntimeError("original DETR door detector adapter is not verified: %s" % (self.adapter_status,))
        if not rgb_frames:
            raise ValueError("original DETR batch must contain at least one RGB frame")
        batch = np.stack([np.asarray(rgb) for rgb in rgb_frames], axis=0)
        with tempfile.TemporaryDirectory(prefix="voxroom_original_detr_") as tmp:
            tmp_path = Path(tmp)
            input_npz = tmp_path / "input_rgb.npz"
            output_json = tmp_path / "detections.json"
            output_masks_npz = tmp_path / "door_masks.npz"
            np.savez_compressed(input_npz, rgb_batch=batch)
            python_cmd = os.environ.get("TOPOLOGY_DOOR_DETR_PYTHON", "python")
            from .original_detr_adapter import make_original_detr_subprocess_env

            env = make_original_detr_subprocess_env()
            cmd = shlex.split(python_cmd) + [
                "-m",
                "voxroom_online.isaac_runtime.baselines.topology_active.original_detr_subprocess",
                "--repo-dir",
                str(self.repo_dir),
                "--input-npz",
                str(input_npz),
                "--output-json",
                str(output_json),
                "--output-masks-npz",
                str(output_masks_npz),
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, timeout=180, check=False)
            if proc.returncode != 0:
                raise RuntimeError("original DETR subprocess failed:\nstdout=%s\nstderr=%s" % (proc.stdout[-4000:], proc.stderr[-4000:]))
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            if not output_masks_npz.is_file():
                raise RuntimeError("original DETR subprocess did not write its raw mask batch")
            with np.load(output_masks_npz, allow_pickle=False) as mask_data:
                if "mask_batch" not in mask_data.files:
                    raise RuntimeError("original DETR mask output is missing mask_batch")
                mask_batch = np.asarray(mask_data["mask_batch"], dtype=np.uint8).copy()
        frame_payloads = payload.get("frames")
        if not isinstance(frame_payloads, list) or len(frame_payloads) != len(rgb_frames):
            raise RuntimeError(
                "original DETR batch output count mismatch: expected %d, found %s"
                % (len(rgb_frames), None if not isinstance(frame_payloads, list) else len(frame_payloads))
            )
        if mask_batch.shape != batch.shape[:3]:
            raise RuntimeError(
                "original DETR mask batch shape mismatch: expected %s, found %s"
                % (batch.shape[:3], mask_batch.shape)
            )
        results: list[list[DoorDetection2D]] = []
        for frame_index, frame_rows in enumerate(frame_payloads):
            if not isinstance(frame_rows, list):
                raise RuntimeError("original DETR batch frame payload is not a list")
            component_count, component_labels = cv2.connectedComponents(
                (mask_batch[frame_index] > 0).astype(np.uint8),
                connectivity=8,
            )
            detections = []
            for row in frame_rows:
                bbox = tuple(float(v) for v in row["bbox_xyxy"])
                component_id = int(row["component_id"])
                if component_id <= 0 or component_id >= int(component_count):
                    raise RuntimeError(
                        "original DETR component id %d is outside [1, %d)"
                        % (component_id, int(component_count))
                    )
                detections.append(
                    DoorDetection2D(
                        bbox_xyxy=bbox,
                        score=float(row.get("score", 0.0)),
                        class_id=row.get("class_id"),
                        mask=(component_labels == component_id),
                    )
                )
            results.append(detections)
        return results


class IsaacSemanticDoorDetectorDebug(DoorDetector):
    name = "isaac_semantic_debug"
    available = False
    uses_oracle_semantics = True

    def detect(self, rgb: np.ndarray) -> list[DoorDetection2D]:
        _ = rgb
        return []


def make_door_detector(name: str) -> DoorDetector:
    key = str(name).strip().lower()
    if key == "disabled":
        return DisabledDoorDetector()
    if key == "original_detr":
        return OriginalDetrDoorDetector()
    if key == "isaac_semantic_debug":
        return IsaacSemanticDoorDetectorDebug()
    raise ValueError(f"unknown door detector: {name}")


def camera_intrinsics_from_mapping(value: Any) -> CameraIntrinsics | None:
    if value is None:
        return None
    if isinstance(value, CameraIntrinsics):
        return value
    if isinstance(value, dict):
        try:
            return CameraIntrinsics(
                fx=float(value.get("fx", value.get("f_x"))),
                fy=float(value.get("fy", value.get("f_y"))),
                cx=float(value.get("cx", value.get("c_x"))),
                cy=float(value.get("cy", value.get("c_y"))),
                width=None if value.get("width") is None else int(value.get("width")),
                height=None if value.get("height") is None else int(value.get("height")),
            )
        except Exception:
            return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (3, 3):
        return CameraIntrinsics(fx=float(arr[0, 0]), fy=float(arr[1, 1]), cx=float(arr[0, 2]), cy=float(arr[1, 2]))
    flat = arr.reshape(-1)
    if flat.size >= 4:
        return CameraIntrinsics(fx=float(flat[0]), fy=float(flat[1]), cx=float(flat[2]), cy=float(flat[3]))
    return None


def project_door_bbox_to_grid(
    detection: DoorDetection2D,
    depth: np.ndarray,
    camera_intrinsics: Any,
    camera_pose_world: np.ndarray,
    map_info: MapInfo,
) -> DoorPointCandidate | None:
    candidate, _status = project_door_bbox_to_grid_with_status(
        detection=detection,
        depth=depth,
        camera_intrinsics=camera_intrinsics,
        camera_pose_world=camera_pose_world,
        map_info=map_info,
    )
    return candidate


def project_door_bbox_to_grid_with_status(
    detection: DoorDetection2D,
    depth: np.ndarray,
    camera_intrinsics: Any,
    camera_pose_world: np.ndarray,
    map_info: MapInfo,
) -> tuple[DoorPointCandidate | None, str]:
    from .depth_projection import project_door_bbox_to_grid_rc_with_status

    attempt = project_door_bbox_to_grid_rc_with_status(
        detection=detection,
        depth=depth,
        camera_intrinsics=camera_intrinsics,
        camera_pose_world=camera_pose_world,
        map_info=map_info,
    )
    result = attempt.result
    if result is None:
        return None, str(attempt.status)
    return (
        DoorPointCandidate(
            rc=result.rc,
            source="vision",
            score=float(detection.score),
            metadata={
                "projection_status": "pinhole_depth_projected",
                "depth_m": float(result.depth_m),
                "sample_count": int(result.sample_count),
                "world_xyz": [float(v) for v in result.world_xyz],
            },
        ),
        "ok",
    )
