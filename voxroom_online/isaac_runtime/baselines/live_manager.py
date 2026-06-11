from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .topology_active.adapter import ActiveRoomSegmentationBaseline
from .topology_active.detector import make_door_detector
from .tvars_original.adapter import BASELINE_NAME as TVARS_ORIGINAL_BASELINE_NAME
from .tvars_original.adapter import TVARSOriginalIsaacBaseline


class LiveBaselineManager:
    def __init__(self, args: Any, scene_id: str, run_dir: Path) -> None:
        self.args = args
        self.scene_id = str(scene_id)
        self.run_dir = Path(run_dir)
        self.enabled = str(getattr(args, "live_roomseg_baseline", "none")).strip().lower() != "none"
        self.impl: ActiveRoomSegmentationBaseline | TVARSOriginalIsaacBaseline | None = None
        self.baseline_name = str(getattr(args, "live_roomseg_baseline", "none")).strip().lower()
        if not self.enabled:
            return
        baseline = self.baseline_name
        if baseline not in {"topology_visual_active", TVARS_ORIGINAL_BASELINE_NAME}:
            raise ValueError(f"unsupported live roomseg baseline: {baseline}")
        if str(getattr(args, "live_baseline_policy_control", "never")) != "never":
            raise ValueError("live baseline policy control must be 'never'")
        output_dir = getattr(args, "live_baseline_output_dir", None)
        if output_dir is None:
            output_dir = self.run_dir / "baselines" / baseline
        if baseline == "topology_visual_active":
            self.impl = ActiveRoomSegmentationBaseline(
                output_dir=Path(output_dir),
                detector=make_door_detector(str(getattr(args, "live_baseline_door_detector", "original_detr"))),
                panorama_views=int(getattr(args, "live_baseline_panorama_views", 12)),
                policy_control="never",
                save_stream=bool(getattr(args, "live_baseline_save_stream", False)),
                save_every_snapshot=bool(getattr(args, "live_baseline_save_every_snapshot", False)),
            )
        else:
            self.impl = TVARSOriginalIsaacBaseline(
                output_dir=Path(output_dir),
                save_stream=bool(getattr(args, "live_baseline_save_stream", False)),
                save_every_snapshot=bool(getattr(args, "live_baseline_save_every_snapshot", False)),
            )

    def on_episode_start(self, episode_metadata: Mapping[str, Any]) -> None:
        if self.impl is not None:
            self.impl.on_episode_start(episode_metadata)

    def on_step(
        self,
        *,
        step: int,
        obs: Mapping[str, Any] | None,
        sgnav_obs: Mapping[str, Any] | None = None,
        map_state: Mapping[str, Any],
        mapper: Any | None = None,
        room_segmenter: Any | None = None,
        frontier_map: np.ndarray | None = None,
        selected_frontier_center_rc: tuple[int, int] | None = None,
        camera_intrinsics: Any | None = None,
    ) -> None:
        _ = room_segmenter
        if self.impl is None:
            return
        self.impl.update(
            step=int(step),
            obs=obs,
            sgnav_obs=sgnav_obs,
            map_state=map_state,
            mapper=mapper,
            room_segmenter=room_segmenter,
            frontier_map=frontier_map,
            selected_frontier_center_rc=selected_frontier_center_rc,
            camera_intrinsics=camera_intrinsics,
        )

    def on_snapshot_saved(
        self,
        *,
        step: int,
        source_snapshot_npz: Path,
        source_summary_json: Path | None = None,
    ) -> Path | None:
        _ = source_summary_json
        if self.impl is None:
            return None
        return self.impl.save_snapshot_like_voxroom(
            source_snapshot_npz=source_snapshot_npz,
            step=int(step),
            source_summary_json=source_summary_json,
        )

    def on_episode_end(self) -> None:
        if self.impl is not None:
            self.impl.on_episode_end()
