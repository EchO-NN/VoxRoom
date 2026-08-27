from __future__ import annotations

from dataclasses import replace
from typing import Mapping

import numpy as np

from voxroom_online.isaac_runtime.door_seed_learning.config import DoorSeedLearningConfig
from voxroom_online.isaac_runtime.door_seed_learning.schema import config_hash
from voxroom_online.isaac_runtime.door_seed_learning.stage_extractor import DoorSeedStageResult
from voxroom_online.isaac_runtime.evaluation.online_roomseg.hough_raw_seed import (
    reconstruct_hough_raw_door_seed,
)


def yaw_deg_from_map_state(map_state: Mapping[str, object]) -> float:
    pose = map_state.get("pose")
    if pose is None:
        return 0.0
    matrix = np.asarray(pose, dtype=np.float64)
    if matrix.shape == (4, 4):
        return float(np.degrees(np.arctan2(float(matrix[1, 0]), float(matrix[0, 0]))))
    flat = matrix.reshape(-1)
    if flat.size >= 4:
        return float(np.degrees(float(flat[3])))
    if flat.size >= 3:
        return float(np.degrees(float(flat[2])))
    return 0.0


class VoxroomTvarsRawSeedAccumulator:
    """Combine current VoxRoom/TVARS raw seeds and retain a final label superset.

    Non-final snapshots contain the union produced at that decision.  The final
    snapshot contains the union of every candidate seen during the episode, so
    one final annotation can be propagated safely back to all prior snapshots.
    TVARS range jumps and line pairing are always computed on Vertical Free and
    its vertical-wall/unknown complement, never on navigation free space.
    """

    def __init__(self, config: DoorSeedLearningConfig | Mapping[str, object]) -> None:
        self.config = (
            config
            if isinstance(config, DoorSeedLearningConfig)
            else DoorSeedLearningConfig.from_mapping(config)
        )
        self._voxroom_history: np.ndarray | None = None
        self._tvars_history: np.ndarray | None = None
        self.decision_count = 0

    def combine(
        self,
        stage: DoorSeedStageResult,
        *,
        agent_rc: tuple[int, int],
        yaw_deg: float,
        resolution_m: float,
        final: bool = False,
        use_history: bool = False,
    ) -> DoorSeedStageResult:
        shape = tuple(int(value) for value in stage.map_shape)
        current_voxroom = np.asarray(
            stage.voxroom_raw_seed_mask_xy
            if stage.voxroom_raw_seed_mask_xy is not None
            else stage.raw_seed_mask_xy,
            dtype=bool,
        ).copy()
        if current_voxroom.shape != shape:
            raise ValueError("VoxRoom raw seed mask does not match the stage map")

        vertical_free = np.asarray(stage.evidence.vertical_free_xy, dtype=bool)
        vertical_wall = np.asarray(stage.evidence.wall_xy, dtype=bool)
        if vertical_free.shape != shape or vertical_wall.shape != shape:
            raise ValueError("Vertical Free layers do not match the stage map")
        vertical_unknown = ~(vertical_free | vertical_wall)
        reconstructed = reconstruct_hough_raw_door_seed(
            obstacle_mask=vertical_wall,
            unknown_mask=vertical_unknown,
            agent_rc=(int(agent_rc[0]), int(agent_rc[1])),
            yaw_deg=float(yaw_deg),
            resolution_m=float(resolution_m),
            seed_width_cells=int(self.config.tvars_seed_width_cells),
        )
        outside = np.asarray(stage.outside_boundary_mask_xy, dtype=bool)
        current_voxroom &= ~outside
        current_tvars = np.asarray(reconstructed.raw_seed_mask, dtype=bool) & ~outside

        if self._voxroom_history is None:
            self._voxroom_history = np.zeros(shape, dtype=bool)
            self._tvars_history = np.zeros(shape, dtype=bool)
        if self._voxroom_history.shape != shape or self._tvars_history is None or self._tvars_history.shape != shape:
            raise ValueError("raw seed map shape changed during one episode")
        self._voxroom_history |= current_voxroom
        self._tvars_history |= current_tvars
        self.decision_count += 1

        current_union = current_voxroom | current_tvars
        persistent_union = self._voxroom_history | self._tvars_history
        use_persistent_final = bool(final and self.config.persistent_final_raw_seed_union)
        use_persistent = bool(use_history or use_persistent_final)
        selected_union = persistent_union if use_persistent else current_union
        seed_hash = config_hash(
            {
                "algorithm": "voxroom_tvars_vertical_raw_seed_union_v1",
                "voxroom_raw_seed_config_hash": str(stage.raw_seed_config_hash),
                "tvars_algorithm": "original_360deg_range_jump_then_prefilter_pairing",
                "tvars_input": "voxel_vertical_free_wall_unknown",
                "tvars_seed_width_cells": int(self.config.tvars_seed_width_cells),
                "persistent_final_raw_seed_union": bool(
                    self.config.persistent_final_raw_seed_union
                ),
            }
        )
        semantics_hash = config_hash(
            {
                "base_input_semantics_hash": str(stage.input_semantics_hash),
                "raw_seed_source": "voxroom_tvars_vertical_union",
                "nonfinal_union_scope": (
                    "episode_history"
                    if self.config.save_full_voxel_milestones
                    else "current_decision"
                ),
                "final_union_scope": (
                    "all_episode_decisions"
                    if self.config.persistent_final_raw_seed_union
                    else "current_decision"
                ),
                "vertical_unknown": "not_vertical_free_or_vertical_wall",
                "outside_boundary_removed": True,
            }
        )
        debug = {
            **dict(stage.debug),
            "raw_seed_source": "voxroom_tvars_vertical_union",
            "raw_seed_union_scope": "episode_history" if use_persistent else "current_decision",
            "raw_seed_union_decision_count": int(self.decision_count),
            "voxroom_raw_seed_cells_current": int(np.count_nonzero(current_voxroom)),
            "tvars_vertical_raw_seed_cells_current": int(np.count_nonzero(current_tvars)),
            "raw_seed_cells_current_union": int(np.count_nonzero(current_union)),
            "voxroom_raw_seed_cells_history": int(np.count_nonzero(self._voxroom_history)),
            "tvars_vertical_raw_seed_cells_history": int(np.count_nonzero(self._tvars_history)),
            "raw_seed_cells_history_union": int(np.count_nonzero(persistent_union)),
            "tvars_vertical_raw_seed_debug": dict(reconstructed.debug),
            "voxroom_raw_seed_mask": current_voxroom.copy(),
            "tvars_vertical_raw_seed_mask": current_tvars.copy(),
            "raw_seed_union_mask": selected_union.copy(),
        }
        return replace(
            stage,
            raw_seed_mask_xy=selected_union,
            raw_seed_config_hash=seed_hash,
            input_semantics_hash=semantics_hash,
            debug=debug,
            voxroom_raw_seed_mask_xy=current_voxroom,
            tvars_vertical_raw_seed_mask_xy=current_tvars,
            voxroom_raw_seed_history_mask_xy=self._voxroom_history.copy(),
            tvars_vertical_raw_seed_history_mask_xy=self._tvars_history.copy(),
        )
