from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .topology_active.adapter import ActiveRoomSegmentationBaseline
from .topology_active.detector import make_door_detector
from .tvars_original.adapter import BASELINE_NAME as TVARS_ORIGINAL_BASELINE_NAME
from .tvars_original.adapter import (
    ORIGINAL_POLICY_CONTROL,
    OriginalFMMNavigationStep,
    OriginalPolicyDecision,
    TVARSOriginalIsaacBaseline,
)


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
        policy_control = str(
            getattr(args, "live_baseline_policy_control", "never")
        ).strip().lower()
        if baseline == "topology_visual_active" and policy_control != "never":
            raise ValueError(
                "topology_visual_active policy control must remain 'never'"
            )
        if baseline == TVARS_ORIGINAL_BASELINE_NAME and policy_control not in {
            "never",
            ORIGINAL_POLICY_CONTROL,
        }:
            raise ValueError(
                "tvars_original_isaac policy control must be 'never' or '%s'"
                % ORIGINAL_POLICY_CONTROL
            )
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
            robot_radius_m = float(getattr(args, "robot_radius_m", 0.0))
            runtime_margin_m = float(
                getattr(args, "runtime_planning_clearance_m", 0.0)
            )
            required_clearance_m = max(
                0.05,
                robot_radius_m + runtime_margin_m,
            )
            self.impl = TVARSOriginalIsaacBaseline(
                output_dir=Path(output_dir),
                policy_control=policy_control,
                panorama_views=int(
                    getattr(args, "live_baseline_panorama_views", 12)
                ),
                save_stream=bool(getattr(args, "live_baseline_save_stream", False)),
                save_every_snapshot=bool(getattr(args, "live_baseline_save_every_snapshot", False)),
                fmm_obstacle_inflation_m=required_clearance_m,
                fmm_preferred_clearance_m=max(
                    required_clearance_m,
                    2.0 * robot_radius_m + runtime_margin_m,
                ),
                fmm_clearance_speed_floor=0.25,
                fmm_collision_inflation_m=0.0,
            )

    @property
    def original_policy_control_enabled(self) -> bool:
        return bool(
            isinstance(self.impl, TVARSOriginalIsaacBaseline)
            and self.impl.policy_control == ORIGINAL_POLICY_CONTROL
        )

    @property
    def scan_required(self) -> bool:
        if not self.original_policy_control_enabled:
            return False
        assert isinstance(self.impl, TVARSOriginalIsaacBaseline)
        return bool(self.impl.scan_required)

    @property
    def scan_views_collected(self) -> int:
        if not self.original_policy_control_enabled:
            return 0
        assert isinstance(self.impl, TVARSOriginalIsaacBaseline)
        return int(self.impl.scan_views_collected)

    @property
    def scan_views_remaining(self) -> int:
        if not self.original_policy_control_enabled:
            return 0
        assert isinstance(self.impl, TVARSOriginalIsaacBaseline)
        return int(self.impl.scan_views_remaining)

    @property
    def policy_update_required(self) -> bool:
        if not self.original_policy_control_enabled:
            return False
        assert isinstance(self.impl, TVARSOriginalIsaacBaseline)
        return bool(self.impl.policy_update_required)

    @property
    def policy_trace_path(self) -> Path | None:
        if not self.original_policy_control_enabled:
            return None
        assert isinstance(self.impl, TVARSOriginalIsaacBaseline)
        return self.impl.policy_trace_path

    @property
    def original_fmm_plan_count(self) -> int:
        if not self.original_policy_control_enabled:
            return 0
        assert isinstance(self.impl, TVARSOriginalIsaacBaseline)
        return int(self.impl.fmm_plan_count)

    @property
    def original_collision_update_count(self) -> int:
        if not self.original_policy_control_enabled:
            return 0
        assert isinstance(self.impl, TVARSOriginalIsaacBaseline)
        return int(self.impl.collision_update_count)

    @property
    def original_terminal_frontier_target_count(self) -> int:
        if not self.original_policy_control_enabled:
            return 0
        assert isinstance(self.impl, TVARSOriginalIsaacBaseline)
        return int(self.impl.terminal_frontier_target_count)

    @property
    def original_reached_global_frontier_target_count(self) -> int:
        if not self.original_policy_control_enabled:
            return 0
        assert isinstance(self.impl, TVARSOriginalIsaacBaseline)
        return int(self.impl.reached_global_frontier_target_count)

    def observe_scan_frame(
        self,
        *,
        obs: Mapping[str, Any],
        map_state: Mapping[str, Any],
        mapper: Any | None,
        camera_intrinsics: Any,
    ) -> None:
        if not self.original_policy_control_enabled:
            raise RuntimeError(
                "scan observation requires tvars_original_isaac original_topology control"
            )
        assert isinstance(self.impl, TVARSOriginalIsaacBaseline)
        self.impl.observe_scan_frame(
            obs=obs,
            map_state=map_state,
            mapper=mapper,
            camera_intrinsics=camera_intrinsics,
        )

    def policy_decision(self) -> OriginalPolicyDecision:
        if not self.original_policy_control_enabled:
            raise RuntimeError(
                "policy decision requires tvars_original_isaac original_topology control"
            )
        assert isinstance(self.impl, TVARSOriginalIsaacBaseline)
        return self.impl.policy_decision()

    def plan_original_fmm_short_term_goal(
        self,
        *,
        map_state: Mapping[str, Any],
        mapper: Any | None = None,
    ) -> OriginalFMMNavigationStep:
        if not self.original_policy_control_enabled:
            raise RuntimeError(
                "FMM navigation requires tvars_original_isaac "
                "original_topology control"
            )
        assert isinstance(self.impl, TVARSOriginalIsaacBaseline)
        return self.impl.plan_fmm_short_term_goal(
            map_state=map_state,
            mapper=mapper,
        )

    def refresh_original_snapshot_partition(
        self,
        *,
        step: int,
        map_state: Mapping[str, Any],
    ) -> None:
        if not self.original_policy_control_enabled:
            raise RuntimeError(
                "snapshot refresh requires tvars_original_isaac "
                "original_topology control"
            )
        assert isinstance(self.impl, TVARSOriginalIsaacBaseline)
        self.impl.refresh_snapshot_partition(
            step=int(step),
            map_state=map_state,
        )

    def observe_policy_pose(
        self,
        *,
        step: int,
        current_rc: tuple[int, int],
    ) -> None:
        if not self.original_policy_control_enabled:
            raise RuntimeError(
                "policy pose observation requires tvars_original_isaac original_topology control"
            )
        assert isinstance(self.impl, TVARSOriginalIsaacBaseline)
        self.impl.observe_policy_pose(
            step=int(step),
            current_rc=current_rc,
        )

    def mark_policy_target_reached(
        self,
        *,
        step: int,
        current_rc: tuple[int, int],
        reached_via: str = "radius",
    ) -> None:
        if not self.original_policy_control_enabled:
            raise RuntimeError(
                "policy target arrival requires tvars_original_isaac original_topology control"
            )
        assert isinstance(self.impl, TVARSOriginalIsaacBaseline)
        self.impl.mark_policy_target_reached(
            step=int(step),
            current_rc=current_rc,
            reached_via=str(reached_via),
        )

    def mark_policy_target_unreachable(
        self,
        *,
        step: int,
        current_rc: tuple[int, int],
        failure_reason: str,
    ) -> str:
        if not self.original_policy_control_enabled:
            raise RuntimeError(
                "policy target rejection requires tvars_original_isaac "
                "original_topology control"
            )
        assert isinstance(self.impl, TVARSOriginalIsaacBaseline)
        return self.impl.mark_policy_target_unreachable(
            step=int(step),
            current_rc=current_rc,
            failure_reason=str(failure_reason),
        )

    def record_original_collision(
        self,
        *,
        step: int,
        current_rc: tuple[int, int],
        collision_rc: tuple[int, int],
        intended_rc: tuple[int, int],
    ) -> None:
        if not self.original_policy_control_enabled:
            raise RuntimeError(
                "collision feedback requires tvars_original_isaac original_topology control"
            )
        assert isinstance(self.impl, TVARSOriginalIsaacBaseline)
        self.impl.record_policy_collision(
            step=int(step),
            current_rc=current_rc,
            collision_rc=collision_rc,
            intended_rc=intended_rc,
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
        output_dir: Path | None = None,
        excluded_source_keys: tuple[str, ...] | None = None,
    ) -> Path | None:
        _ = source_summary_json
        if self.impl is None:
            return None
        return self.impl.save_snapshot_like_voxroom(
            source_snapshot_npz=source_snapshot_npz,
            step=int(step),
            source_summary_json=source_summary_json,
            output_dir=output_dir,
            excluded_source_keys=excluded_source_keys,
        )

    def visualization_payload(self) -> dict[str, Any] | None:
        if not isinstance(self.impl, TVARSOriginalIsaacBaseline):
            return None
        preview = self.impl.latest_preview_rgb
        if preview is None:
            return None
        preview_array = np.asarray(preview, dtype=np.uint8)
        if preview_array.ndim != 3 or preview_array.shape[2] != 3:
            raise RuntimeError(
                "TVARS live preview must be an HxWx3 uint8 image"
            )
        return {
            "baseline_name": TVARS_ORIGINAL_BASELINE_NAME,
            "preview_rgb": preview_array.copy(),
            "metadata": dict(self.impl.latest_metadata),
        }

    def on_episode_end(self) -> None:
        if self.impl is not None:
            self.impl.on_episode_end()
