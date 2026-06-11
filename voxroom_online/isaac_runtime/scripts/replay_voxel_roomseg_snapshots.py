from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import re
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image

from voxroom_online.isaac_runtime.config import get_nested, load_config
from voxroom_online.isaac_runtime.debug.roomseg_layer_dump import save_roomseg_layer_dump
from voxroom_online.isaac_runtime.mapping.coordinate_transform import MapInfo
from voxroom_online.isaac_runtime.mapping.voxel_occupancy_door_wall_roomseg import VoxelOccupancyDoorWallRoomSegmenter
from voxroom_online.isaac_runtime.mapping.voxel_occupancy_grid import (
    VOXEL_FREE,
    VOXEL_OCCUPIED,
    VOXEL_UNKNOWN,
    VoxelOccupancyGrid3D,
    VoxelOccupancyGridConfig,
)


REPLAY_NPZ_KEYS = (
    "occupancy_map",
    "observed_free_mask",
    "obstacle_mask",
    "unknown_mask",
    "navigation_free_room_domain",
    "vertical_free_room_domain",
    "voxel_nav_free_xy",
    "voxel_door_seed_no_clearance_free_xy",
    "voxel_nav_occupied_xy",
    "voxel_nav_unknown_xy",
    "voxel_nav_observed_xy",
    "voxel_occupancy_state_zyx",
    "voxel_occupancy_log_odds_zyx",
    "voxel_sensor_range_count_zyx",
    "voxel_occupancy_z_centers_m",
    "voxel_occupancy_z_min_m",
    "voxel_occupancy_z_max_m",
    "voxel_occupancy_z_resolution_m",
    "voxel_occupancy_active_z_min_m",
    "voxel_occupancy_active_z_max_m",
    "voxel_occupancy_ceiling_height_estimate_m",
    "voxel_occupancy_ceiling_estimate_status",
    "voxel_ceiling_estimate_status",
    "voxel_nav_occupied_endpoint_count_xy",
    "voxel_nav_free_ray_count_xy",
    "voxel_roomseg_memory_before_json",
    "voxel_roomseg_memory_after_json",
    "voxel_door_memory_before_roomseg_json",
    "voxel_door_memory_after_roomseg_json",
    "voxel_separator_memory_before_roomseg_json",
    "voxel_separator_memory_after_roomseg_json",
    "final_room_label_map",
    "voxel_final_room_label_map",
    "accepted_separators",
    "rejected_separators",
    "voxel_door_seed_mask",
    "voxel_door_raw_seed_mask",
    "voxel_door_seed_component_id_map",
    "voxel_door_seed_cluster_id_map",
    "voxel_door_seed_line_primitive_id_map",
    "voxel_door_seed_line_primitive_mask",
    "voxel_door_first_upper_occupied_z_xy",
    "voxel_door_upper_non_ceiling_actual_occupied_count_xy",
    "voxel_door_upper_ceiling_occupied_count_xy",
    "voxel_door_ceiling_like_reject_mask",
    "voxel_door_turn_z_above_lintel_max_mask",
    "voxel_door_first_occ_too_high_mask",
    "voxel_door_extensible_primitive_mask",
    "voxel_door_rejected_primitive_mask",
    "voxel_door_extension_trials_map",
    "voxel_door_extension_reject_reason_id_map",
    "voxel_door_centerline_mask",
    "voxel_door_cut_mask",
    "voxel_door_visual_only_mask",
    "voxel_door_partition_cut_candidate_mask",
    "voxel_door_partition_cut_mask",
    "voxel_door_partition_cut_rejected_mask",
    "voxel_door_topology_effective_cut_mask",
    "voxel_door_partition_reject_reason_id_map",
    "voxel_current_door_cut_mask",
    "voxel_current_door_topology_effective_mask",
    "voxel_stable_door_cut_mask",
    "voxel_door_stable_cut_mask",
    "voxel_stable_door_visual_mask",
    "voxel_door_memory_observed_decay_band_mask",
    "voxel_door_memory_unobserved_track_mask",
    "voxel_door_memory_contradiction_mask",
    "voxel_step1_wall_gap_fill_map",
    "voxel_wall_after_step1_map",
    "voxel_step2_extension_candidate_map",
    "voxel_step2_extension_separator_map",
    "voxel_step2_rejected_extension_map",
    "voxel_final_separator_map",
    "voxel_wall_xy",
    "voxel_vertical_free_xy",
    "voxel_unknown_xy",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay voxel room segmentation from saved roomseg snapshot NPZ files.")
    parser.add_argument("--snapshot", default=None)
    parser.add_argument("--snapshot-dir", default=None)
    parser.add_argument("--snapshot-root", default=None, help="Recursively replay every roomseg snapshot under this root.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", default="voxroom_online/isaac_runtime/configs/voxroom_online.isaac_runtime.yaml")
    parser.add_argument("--resolution-m", type=float, default=None)
    parser.add_argument("--max-snapshots", type=int, default=0)
    parser.add_argument("--reset-memory-per-snapshot", action="store_true")
    parser.add_argument("--mode", choices=["visualize", "stateless", "stateful"], default="stateless")
    parser.add_argument("--memory-source", choices=["none", "saved-before", "saved-after"], default="none")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--recompute-state-from-logodds", action="store_true")
    parser.add_argument("--save-overlay", action="store_true", default=True)
    parser.add_argument("--no-save-overlay", dest="save_overlay", action="store_false")
    parser.add_argument("--save-layer-grid", action="store_true")
    parser.add_argument("--save-compact-npz", action="store_true", default=True)
    parser.add_argument("--no-save-compact-npz", dest="save_compact_npz", action="store_false")
    parser.add_argument("--mask-only", action="store_true", help="Write only colored navigation room mask PNGs.")
    parser.add_argument("--save-debug-mask", action="store_true", default=True)
    parser.add_argument("--no-save-debug-mask", dest="save_debug_mask", action="store_false")
    parser.add_argument("--debug-mask-dilation-cells", type=int, default=1)
    parser.add_argument("--flat-output", action="store_true", help="Do not preserve source scene directories when using --snapshot-root.")
    args = parser.parse_args(argv)
    if bool(args.mask_only):
        args.save_overlay = False
        args.save_layer_grid = False
        args.save_compact_npz = False

    snapshot_path = None if args.snapshot is None else Path(args.snapshot).expanduser()
    snapshot_dir = None if args.snapshot_dir is None else Path(args.snapshot_dir).expanduser()
    snapshot_root = None if args.snapshot_root is None else Path(args.snapshot_root).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    source_count = sum(path is not None for path in (snapshot_path, snapshot_dir, snapshot_root))
    if source_count != 1:
        raise SystemExit("must pass exactly one of --snapshot, --snapshot-dir, or --snapshot-root")
    if snapshot_path is not None:
        npz_paths = [snapshot_path]
    elif snapshot_dir is not None:
        npz_paths = sorted(snapshot_dir.glob("roomseg_step_*.npz"), key=_snapshot_sort_key)
    elif snapshot_root is not None:
        npz_paths = sorted(snapshot_root.glob("**/roomseg_snapshots/roomseg_step_*.npz"), key=_snapshot_root_sort_key)
    else:
        npz_paths = []
    if args.max_snapshots and int(args.max_snapshots) > 0:
        npz_paths = npz_paths[: int(args.max_snapshots)]
    if not npz_paths:
        raise SystemExit("no roomseg_step_*.npz files found")

    manifest: dict[str, object] = {
        "snapshot": None if snapshot_path is None else str(snapshot_path),
        "snapshot_dir": None if snapshot_dir is None else str(snapshot_dir),
        "snapshot_root": None if snapshot_root is None else str(snapshot_root),
        "out_dir": str(out_dir),
        "config": str(Path(args.config)),
        "snapshot_count": int(len(npz_paths)),
        "processed": 0,
        "errors": [],
        "mode": str(args.mode),
        "memory_source": str(args.memory_source),
        "jobs": int(args.jobs),
        "reset_memory_per_snapshot": bool(args.reset_memory_per_snapshot),
        "started_at_unix": time.time(),
        "steps": [],
    }
    worker_kwargs = {
        "out_dir": str(out_dir),
        "config_path": str(args.config),
        "resolution_m": args.resolution_m,
        "mode": str(args.mode),
        "memory_source": str(args.memory_source),
        "reset_memory_per_snapshot": bool(args.reset_memory_per_snapshot),
        "save_overlay": bool(args.save_overlay),
        "save_layer_grid": bool(args.save_layer_grid),
        "save_compact_npz": bool(args.save_compact_npz),
        "mask_only": bool(args.mask_only),
        "recompute_state_from_logodds": bool(args.recompute_state_from_logodds),
        "snapshot_count": int(len(npz_paths)),
        "snapshot_root": None if snapshot_root is None else str(snapshot_root),
        "preserve_relative_paths": bool(snapshot_root is not None and not args.flat_output),
        "save_debug_mask": bool(args.save_debug_mask),
        "debug_mask_dilation_cells": int(args.debug_mask_dilation_cells),
    }
    jobs = max(1, int(args.jobs))
    if jobs > 1 and str(args.mode) in {"visualize", "stateless"}:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = [
                executor.submit(_process_snapshot_replay, str(path), **worker_kwargs)
                for path in npz_paths
            ]
            for future in as_completed(futures):
                result = future.result()
                _record_replay_manifest_entry(manifest, result)
    else:
        if jobs > 1 and str(args.mode) == "stateful":
            manifest["parallel_disabled_reason"] = "stateful_replay_runs_serially"
        for npz_path in npz_paths:
            result = _process_snapshot_replay(str(npz_path), **worker_kwargs)
            _record_replay_manifest_entry(manifest, result)

    manifest["finished_at_unix"] = time.time()
    if not bool(args.mask_only):
        (out_dir / "manifest.json").write_text(json.dumps(_json_ready(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "processed": manifest["processed"], "errors": len(manifest["errors"])}, ensure_ascii=False))
    return 0 if not manifest["errors"] else 2


def _record_replay_manifest_entry(manifest: dict[str, object], result: Mapping[str, object]) -> None:
    if result.get("error") is not None:
        err = {"step": int(result.get("step", -1)), "input": str(result.get("input", "")), "error": str(result.get("error"))}
        manifest["errors"].append(err)  # type: ignore[index,union-attr]
        print("[replay] ERROR step=%06d %s" % (int(err["step"]), str(err["error"])), flush=True)
        return
    manifest["processed"] = int(manifest["processed"]) + 1
    manifest["steps"].append(dict(result))  # type: ignore[index,union-attr]
    print(
        "[replay] step=%06d mode=%s rooms old=%s new=%s labeled_changed=%d out=%s"
        % (
            int(result.get("step", -1)),
            str(result.get("replay_mode", "")),
            result.get("original_room_count"),
            result.get("replay_room_count"),
            int(result.get("label_presence_changed_cells", 0) or 0),
            result.get("navigation_room_masks_png"),
        ),
        flush=True,
    )


def _process_snapshot_replay(
    npz_path_str: str,
    *,
    out_dir: str,
    config_path: str,
    resolution_m: float | None,
    mode: str,
    memory_source: str,
    reset_memory_per_snapshot: bool,
    save_overlay: bool,
    save_layer_grid: bool,
    save_compact_npz: bool,
    mask_only: bool,
    recompute_state_from_logodds: bool,
    snapshot_count: int,
    snapshot_root: str | None,
    preserve_relative_paths: bool,
    save_debug_mask: bool,
    debug_mask_dilation_cells: int,
) -> dict[str, object]:
    npz_path = Path(npz_path_str).expanduser()
    actual_out_dir = _output_dir_for_snapshot(
        Path(out_dir).expanduser(),
        npz_path,
        snapshot_root=None if snapshot_root is None else Path(snapshot_root).expanduser(),
        preserve_relative_paths=bool(preserve_relative_paths),
    )
    actual_out_dir.mkdir(parents=True, exist_ok=True)
    step = _infer_step(npz_path)
    started = time.perf_counter()
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            arrays = {name: np.asarray(data[name]).copy() for name in data.files}
        shape = _snapshot_shape(npz_path)
        cfg = load_config(config_path)
        roomseg_cfg = dict(get_nested(cfg, "mapping.room_segmentation", {}) or {})
        voxel_grid_cfg = dict(roomseg_cfg.get("voxel_grid", {}) or {})
        res_m = float(resolution_m or get_nested(cfg, "mapping.online_resolution_m", 0.05))
        map_info = MapInfo(
            resolution_m=float(res_m),
            min_x=0.0,
            max_x=float(shape[1]) * float(res_m),
            min_y=0.0,
            max_y=float(shape[0]) * float(res_m),
            width=int(shape[1]),
            height=int(shape[0]),
        )
        occupancy_map = _bool_array(arrays, "occupancy_map", shape)
        observed_free_mask = _bool_array(arrays, "observed_free_mask", shape)
        obstacle_mask = _bool_array(arrays, "obstacle_mask", shape)
        unknown_mask = _bool_array(arrays, "unknown_mask", shape)
        exact = {
            "voxel_log_odds_loaded_exact": False,
            "voxel_sensor_range_loaded_exact": False,
            "voxel_memory_loaded": False,
        }
        if str(mode) == "visualize":
            replay_labels = np.asarray(
                arrays.get("voxel_final_room_label_map", arrays.get("final_room_label_map", np.zeros(shape, dtype=np.int32))),
                dtype=np.int32,
            )
            replay_separator = np.asarray(
                arrays.get("voxel_final_separator_map", arrays.get("accepted_separators", np.zeros(shape, dtype=bool))),
                dtype=bool,
            )
            room_debug = {name: np.asarray(value) for name, value in arrays.items() if np.asarray(value).ndim <= 3}
            room_debug["algorithm"] = "voxel_replay_visualize"
            room_debug["backend"] = "voxel_replay_visualize"
            room_debug["final_room_label_map"] = replay_labels
            room_debug["voxel_final_room_label_map"] = replay_labels
            room_debug["accepted_separators"] = replay_separator
            room_debug["voxel_final_separator_map"] = replay_separator
        else:
            voxel_grid = _voxel_grid_from_snapshot(
                arrays,
                map_info,
                voxel_grid_cfg,
                recompute_state_from_logodds=bool(recompute_state_from_logodds),
            )
            exact["voxel_log_odds_loaded_exact"] = "voxel_occupancy_log_odds_zyx" in arrays
            exact["voxel_sensor_range_loaded_exact"] = "voxel_sensor_range_count_zyx" in arrays
            segmenter = VoxelOccupancyDoorWallRoomSegmenter(config=roomseg_cfg, map_info=map_info)
            if str(mode) == "stateful" or str(memory_source) != "none":
                memory_state = _load_memory_state_from_snapshot(arrays, str(memory_source))
                if str(mode) == "stateful" and memory_state is None:
                    raise RuntimeError("stateful replay requested but snapshot has no saved memory for %s" % str(memory_source))
                if memory_state is not None:
                    segmenter.import_replay_state(memory_state)
                    exact["voxel_memory_loaded"] = True
            elif bool(reset_memory_per_snapshot):
                segmenter = VoxelOccupancyDoorWallRoomSegmenter(config=roomseg_cfg, map_info=map_info)
            segmenter.update(
                occupancy_map=occupancy_map,
                observed_free_mask=observed_free_mask,
                obstacle_mask=obstacle_mask,
                unknown_mask=unknown_mask,
                voxel_grid=voxel_grid,
                step=int(step),
            )
            result = segmenter.last_result
            if result is None:
                raise RuntimeError("segmenter did not produce a result")
            replay_labels = np.asarray(result.room_label_map, dtype=np.int32)
            replay_separator = np.asarray(result.separator_map, dtype=bool)
            room_debug = dict(segmenter.last_debug)
            room_debug["final_room_label_map"] = replay_labels
            room_debug["voxel_final_room_label_map"] = replay_labels
            room_debug["accepted_separators"] = replay_separator
            room_debug["voxel_final_separator_map"] = replay_separator
        frontier_map = arrays.get("frontier_map")
        selected_frontier = _members_from_mask(arrays.get("selected_frontier_mask"))
        selected_center = arrays.get("selected_frontier_center_rc")
        agent_rc = arrays.get("agent_rc")
        dump = save_roomseg_layer_dump(
            out_dir=str(actual_out_dir),
            step=int(step),
            room_debug=room_debug,
            occupancy_map=occupancy_map,
            observed_free_mask=observed_free_mask,
            obstacle_mask=obstacle_mask,
            unknown_mask=unknown_mask,
            frontier_map=None if frontier_map is None else np.asarray(frontier_map, dtype=bool),
            selected_frontier_members=selected_frontier,
            selected_frontier_center_rc=None if selected_center is None else np.asarray(selected_center, dtype=np.int32).reshape(-1)[:2],
            agent_rc=None if agent_rc is None else np.asarray(agent_rc, dtype=np.int32).reshape(-1)[:2],
            max_saves=max(10000, int(snapshot_count) + 10),
            save_npz=bool(save_compact_npz),
            save_png=True,
            save_summary_json=not bool(mask_only),
            save_overlay_png=bool(save_overlay),
            save_layers_png=bool(save_layer_grid),
            save_navigation_room_masks_png=True,
            npz_keys=REPLAY_NPZ_KEYS,
            extra_npz_arrays=None,
            include_selected_frontier_sector=False,
        )
        comparison = _compare_with_original(arrays, replay_labels, replay_separator)
        stable_diff = _stable_door_diff(arrays, room_debug, shape)
        comparison.update(stable_diff)
        diff_paths = {}
        if not bool(mask_only):
            diff_paths = _write_replay_diff_images(arrays, replay_labels, replay_separator, room_debug, shape, actual_out_dir, int(step))
        wall_debug_mask_png = None
        door_debug_mask_png = None
        if bool(save_debug_mask):
            wall_debug_mask_path = actual_out_dir / ("roomseg_step_%06d.wall_debug_overlay.png" % int(step))
            _render_room_mask_wall_overlay(
                replay_labels,
                room_debug,
                arrays,
                shape,
                dilation_cells=int(debug_mask_dilation_cells),
            ).save(wall_debug_mask_path)
            wall_debug_mask_png = str(wall_debug_mask_path)

            door_debug_mask_path = actual_out_dir / ("roomseg_step_%06d.door_seed_debug_overlay.png" % int(step))
            _render_room_mask_door_overlay(
                replay_labels,
                room_debug,
                arrays,
                shape,
                dilation_cells=int(debug_mask_dilation_cells),
            ).save(door_debug_mask_path)
            door_debug_mask_png = str(door_debug_mask_path)
        summary_path = Path(dump["paths"]["summary_json"])
        summary = dict(dump.get("summary", {}))
        summary.update(
            {
                "input_snapshot": str(npz_path),
                "replay_mode": str(mode),
                "replay_memory_source": str(memory_source),
                "voxel_replay_exact_input": bool(exact["voxel_log_odds_loaded_exact"] and exact["voxel_sensor_range_loaded_exact"]),
                **exact,
                "replay_comparison": comparison,
                **comparison,
                "replay_diff_paths": diff_paths,
                "wall_debug_mask_png": wall_debug_mask_png,
                "door_debug_mask_png": door_debug_mask_png,
                "replay_runtime_ms": float((time.perf_counter() - started) * 1000.0),
            }
        )
        if not bool(mask_only):
            summary_path.write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {
            "step": int(step),
            "input": str(npz_path),
            "summary": None if bool(mask_only) else str(summary_path),
            "navigation_room_masks_png": str(dump["paths"]["navigation_room_masks_png"]),
            "wall_debug_mask_png": wall_debug_mask_png,
            "door_debug_mask_png": door_debug_mask_png,
            "overlay_png": str(dump["paths"]["overlay_png"]) if bool(save_overlay) else None,
            "runtime_ms": float(summary["replay_runtime_ms"]),
            "replay_mode": str(mode),
            "replay_memory_source": str(memory_source),
            **exact,
            **comparison,
            **diff_paths,
        }
    except Exception as exc:
        return {"step": int(step), "input": str(npz_path), "error": repr(exc)}


def _snapshot_shape(path: Path) -> tuple[int, int]:
    with np.load(path, allow_pickle=False) as data:
        if "observed_free_mask" in data:
            arr = np.asarray(data["observed_free_mask"])
        elif "final_room_label_map" in data:
            arr = np.asarray(data["final_room_label_map"])
        else:
            raise ValueError("cannot infer snapshot shape from %s" % path)
    if arr.ndim != 2:
        raise ValueError("snapshot shape source must be 2D")
    return int(arr.shape[0]), int(arr.shape[1])


def _voxel_grid_from_snapshot(
    arrays: Mapping[str, np.ndarray],
    map_info: MapInfo,
    voxel_grid_cfg: Mapping[str, object],
    *,
    recompute_state_from_logodds: bool = False,
) -> VoxelOccupancyGrid3D:
    if "voxel_occupancy_state_zyx" not in arrays:
        raise KeyError("snapshot missing voxel_occupancy_state_zyx")
    state = np.asarray(arrays["voxel_occupancy_state_zyx"], dtype=np.uint8).copy()
    if state.ndim != 3:
        raise ValueError("voxel_occupancy_state_zyx must have shape [Z,H,W]")
    exact_log_odds = "voxel_occupancy_log_odds_zyx" in arrays
    exact_sensor = "voxel_sensor_range_count_zyx" in arrays
    z_min = _scalar(arrays, "voxel_occupancy_z_min_m", float(voxel_grid_cfg.get("z_min_m", 0.0)))
    z_max = _scalar(arrays, "voxel_occupancy_z_max_m", float(voxel_grid_cfg.get("z_max_m", 3.2)))
    z_res = _scalar(arrays, "voxel_occupancy_z_resolution_m", float(voxel_grid_cfg.get("z_resolution_m", 0.05)))
    active_z_min = _scalar(arrays, "voxel_occupancy_active_z_min_m", float(voxel_grid_cfg.get("active_z_min_m", 0.10)))
    active_z_max = _scalar(
        arrays,
        "voxel_occupancy_active_z_max_m",
        float(voxel_grid_cfg.get("active_z_max_cap_m", voxel_grid_cfg.get("z_max_m", 3.2))),
    )
    ceiling = _scalar(arrays, "voxel_occupancy_ceiling_height_estimate_m", math.nan)
    cfg = VoxelOccupancyGridConfig.from_mapping(
        voxel_grid_cfg,
        z_min_m=z_min,
        z_max_m=z_max,
        z_resolution_m=z_res,
        active_z_min_m=active_z_min,
    )
    if exact_log_odds:
        log_odds = np.asarray(arrays["voxel_occupancy_log_odds_zyx"], dtype=np.int16).copy()
        if log_odds.shape != state.shape:
            raise ValueError("voxel_occupancy_log_odds_zyx must match state shape")
    else:
        log_odds = _log_odds_from_state(state, cfg)
    grid = VoxelOccupancyGrid3D(
        log_odds=log_odds,
        state=state,
        sensor_range_count=np.zeros_like(state, dtype=np.uint8),
        floor_frustum_seen_count_xy=np.asarray(
            arrays.get("voxel_floor_frustum_seen_count_xy", np.zeros(state.shape[1:], dtype=np.uint8)),
            dtype=np.uint8,
        ).copy(),
        outside_score_xy=np.asarray(
            arrays.get("voxel_outside_score_xy", np.zeros(state.shape[1:], dtype=np.uint8)),
            dtype=np.uint8,
        ).copy(),
        outside_xy=np.asarray(
            arrays.get("voxel_outside_xy", np.zeros(state.shape[1:], dtype=bool)),
            dtype=bool,
        ).copy(),
        z_min_m=float(z_min),
        z_max_m=float(z_max),
        z_resolution_m=float(z_res),
        map_info=map_info,
        config=cfg,
        active_z_min_m=float(active_z_min),
        active_z_max_m=float(active_z_max),
        ceiling_height_m=None if not np.isfinite(ceiling) else float(ceiling),
        ceiling_estimate_status="snapshot",
    )
    if exact_sensor:
        sensor = np.asarray(arrays["voxel_sensor_range_count_zyx"], dtype=np.uint8).copy()
        if sensor.shape != state.shape:
            raise ValueError("voxel_sensor_range_count_zyx must match state shape")
        grid.sensor_range_count = sensor
    else:
        grid.sensor_range_count = _reconstruct_sensor_range_count(arrays, grid)
    if bool(recompute_state_from_logodds):
        grid.refresh_state()
    return grid


def _reconstruct_sensor_range_count(arrays: Mapping[str, np.ndarray], grid: VoxelOccupancyGrid3D) -> np.ndarray:
    sensor = np.zeros_like(grid.state, dtype=np.uint8)
    active_idx = grid.active_z_indices()
    if active_idx.size == 0:
        return sensor
    active_state = np.asarray(grid.state[active_idx], dtype=np.uint8)
    shape = active_state.shape[1:]
    target_unknown = np.asarray(arrays.get("voxel_sensor_in_range_unknown_count_xy", np.zeros(shape, dtype=np.uint16)), dtype=np.uint16)
    target_total = np.asarray(arrays.get("voxel_sensor_range_count_xy", target_unknown), dtype=np.uint16)
    unknown_bin = active_state == int(VOXEL_UNKNOWN)
    if np.any(target_unknown):
        unknown_rank = np.cumsum(unknown_bin, axis=0, dtype=np.uint16)
        sensor_active = unknown_bin & (unknown_rank <= np.minimum(target_unknown, unknown_rank[-1])[None, :, :])
    else:
        sensor_active = np.zeros_like(active_state, dtype=bool)
    remaining = np.maximum(target_total.astype(np.int32) - np.sum(sensor_active, axis=0, dtype=np.int32), 0).astype(np.uint16)
    non_unknown = ~unknown_bin
    if np.any(remaining):
        known_rank = np.cumsum(non_unknown, axis=0, dtype=np.uint16)
        sensor_active |= non_unknown & (known_rank <= np.minimum(remaining, known_rank[-1])[None, :, :])
    sensor[active_idx] = sensor_active.astype(np.uint8)
    return sensor


def _log_odds_from_state(state: np.ndarray, cfg: VoxelOccupancyGridConfig) -> np.ndarray:
    log_odds = np.zeros_like(state, dtype=np.int16)
    log_odds[state == int(VOXEL_FREE)] = int(cfg.free_logodds_threshold)
    log_odds[state == int(VOXEL_OCCUPIED)] = int(cfg.occupied_logodds_threshold)
    return log_odds


def _compare_with_original(arrays: Mapping[str, np.ndarray], replay_labels: np.ndarray, replay_separator: np.ndarray) -> dict[str, object]:
    original_labels = np.asarray(arrays.get("final_room_label_map", np.zeros_like(replay_labels)), dtype=np.int32)
    replay = np.asarray(replay_labels, dtype=np.int32)
    original_presence = original_labels > 0
    replay_presence = replay > 0
    original_separator = np.asarray(arrays.get("accepted_separators", np.zeros_like(replay_presence)), dtype=bool)
    replay_sep = np.asarray(replay_separator, dtype=bool)
    return {
        "original_room_count": _room_count(original_labels),
        "replay_room_count": _room_count(replay),
        "original_labeled_cells": int(np.count_nonzero(original_presence)),
        "replay_labeled_cells": int(np.count_nonzero(replay_presence)),
        "label_presence_changed_cells": int(np.count_nonzero(original_presence ^ replay_presence)),
        "label_id_changed_labeled_overlap_cells": int(np.count_nonzero((original_labels != replay) & original_presence & replay_presence)),
        "original_separator_cells": int(np.count_nonzero(original_separator)),
        "replay_separator_cells": int(np.count_nonzero(replay_sep)),
        "separator_presence_changed_cells": int(np.count_nonzero(original_separator ^ replay_sep)),
    }


def _load_memory_state_from_snapshot(arrays: Mapping[str, np.ndarray], memory_source: str) -> dict[str, object] | None:
    source = str(memory_source or "none").strip().lower()
    if source == "none":
        return None
    if source not in {"saved-before", "saved-after"}:
        raise ValueError("unsupported replay memory source: %s" % memory_source)
    suffix = "before" if source == "saved-before" else "after"
    full_key = "voxel_roomseg_memory_%s_json" % suffix
    full_state = _json_scalar(arrays, full_key)
    if isinstance(full_state, Mapping):
        return dict(full_state)
    door_state = _json_scalar(arrays, "voxel_door_memory_%s_roomseg_json" % suffix)
    separator_state = _json_scalar(arrays, "voxel_separator_memory_%s_roomseg_json" % suffix)
    if isinstance(door_state, Mapping) or isinstance(separator_state, Mapping):
        return {
            "schema_version": 1,
            "door_memory": dict(door_state) if isinstance(door_state, Mapping) else {},
            "separator_memory": dict(separator_state) if isinstance(separator_state, Mapping) else {},
        }
    return None


def _json_scalar(arrays: Mapping[str, np.ndarray], key: str) -> object | None:
    text = _string_scalar(arrays, key)
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("snapshot key %s does not contain valid JSON" % key) from exc


def _string_scalar(arrays: Mapping[str, np.ndarray], key: str) -> str | None:
    if key not in arrays:
        return None
    arr = np.asarray(arrays[key])
    if arr.size == 0:
        return None
    value = arr.reshape(-1)[0]
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value)
    text = text.strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return text


def _stable_door_diff(arrays: Mapping[str, np.ndarray], room_debug: Mapping[str, object], shape: tuple[int, int]) -> dict[str, object]:
    out: dict[str, object] = {}
    keys = (
        "voxel_current_door_cut_mask",
        "voxel_current_door_topology_effective_mask",
        "voxel_stable_door_cut_mask",
        "voxel_door_stable_cut_mask",
        "voxel_stable_door_visual_mask",
        "voxel_door_partition_cut_mask",
        "voxel_door_topology_effective_cut_mask",
        "voxel_final_separator_map",
    )
    for key in keys:
        original = _bool_like(arrays.get(key), shape)
        replay = _bool_like(room_debug.get(key), shape)
        out["%s_original_cells" % key] = int(np.count_nonzero(original))
        out["%s_replay_cells" % key] = int(np.count_nonzero(replay))
        out["%s_changed_cells" % key] = int(np.count_nonzero(original ^ replay))
    return out


def _output_dir_for_snapshot(
    out_root: Path,
    npz_path: Path,
    *,
    snapshot_root: Path | None,
    preserve_relative_paths: bool,
) -> Path:
    if not bool(preserve_relative_paths) or snapshot_root is None:
        return out_root
    source_dir = npz_path.parent.parent if npz_path.parent.name == "roomseg_snapshots" else npz_path.parent
    try:
        rel = source_dir.resolve().relative_to(snapshot_root.resolve())
    except ValueError:
        rel = Path(_sanitize_path_component(source_dir.name))
    return out_root / rel / "roomseg_snapshots"


def _render_room_mask_wall_overlay(
    labels: np.ndarray,
    room_debug: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
    shape: tuple[int, int],
    *,
    dilation_cells: int,
) -> Image.Image:
    rgb = _room_label_rgb(labels, shape)
    wall = _mask_union(
        (
            "voxel_wall_xy",
            "voxel_wall_after_step1_map",
            "voxel_final_separator_map",
            "accepted_separators",
        ),
        room_debug,
        arrays,
        shape,
    )
    wall_extension = _mask_union(
        (
            "voxel_step1_wall_gap_fill_map",
            "voxel_step2_extension_candidate_map",
            "voxel_step2_extension_separator_map",
            "wall_extension_boundary_mask",
        ),
        room_debug,
        arrays,
        shape,
    )

    _paint_mask(rgb, wall, (255, 35, 35), dilation_cells=dilation_cells)
    _paint_mask(rgb, wall_extension, (190, 70, 255), dilation_cells=dilation_cells)
    return Image.fromarray(rgb, mode="RGB")


def _render_room_mask_door_overlay(
    labels: np.ndarray,
    room_debug: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
    shape: tuple[int, int],
    *,
    dilation_cells: int,
) -> Image.Image:
    rgb = _room_label_rgb(labels, shape)
    door_seed = _mask_union(
        (
            "voxel_door_raw_seed_mask",
            "voxel_door_seed_mask",
            "voxel_door_seed_line_primitive_mask",
            "voxel_door_extensible_primitive_mask",
        ),
        room_debug,
        arrays,
        shape,
    )
    door_seed_extension = _mask_union(
        (
            "voxel_door_extension_trials_map",
            "voxel_door_centerline_mask",
            "voxel_door_cut_mask",
            "voxel_door_partition_cut_candidate_mask",
            "voxel_door_partition_cut_mask",
            "voxel_door_topology_effective_cut_mask",
            "voxel_current_door_cut_mask",
            "voxel_current_door_topology_effective_mask",
            "voxel_stable_door_cut_mask",
            "voxel_door_stable_cut_mask",
            "partial_door_extension_cut_mask",
            "door_completion_boundary_mask",
        ),
        room_debug,
        arrays,
        shape,
    )

    _paint_mask(rgb, door_seed_extension, (35, 230, 85), dilation_cells=dilation_cells)
    _paint_mask(rgb, door_seed, (45, 120, 255), dilation_cells=dilation_cells)
    return Image.fromarray(rgb, mode="RGB")


def _room_label_rgb(labels: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    label_map = np.asarray(labels, dtype=np.int32)
    rgb = np.zeros(shape + (3,), dtype=np.uint8)
    rgb[:, :] = (17, 18, 22)
    for label in sorted(int(v) for v in np.unique(label_map) if int(v) > 0):
        rgb[label_map == label] = _label_color(label)
    return rgb


def _mask_union(
    keys: Sequence[str],
    room_debug: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
    shape: tuple[int, int],
) -> np.ndarray:
    debug_out = np.zeros(shape, dtype=bool)
    found_debug_layer = False
    for key in keys:
        debug_layer = _optional_bool_like(room_debug.get(key), shape)
        if debug_layer is None:
            continue
        debug_out |= debug_layer
        found_debug_layer = True
    if bool(found_debug_layer):
        return debug_out

    out = np.zeros(shape, dtype=bool)
    for key in keys:
        out |= _bool_like(arrays.get(key), shape)
    return out


def _optional_bool_like(value: object, shape: tuple[int, int]) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=bool)
    except Exception:
        return None
    if arr.shape != shape:
        return None
    return arr


def _paint_mask(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], *, dilation_cells: int) -> None:
    visible = _dilate_mask(mask, iterations=max(0, int(dilation_cells)))
    rgb[visible] = color


def _dilate_mask(mask: np.ndarray, *, iterations: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    for _ in range(max(0, int(iterations))):
        padded = np.pad(out, ((1, 1), (1, 1)), mode="constant", constant_values=False)
        grown = np.zeros_like(out, dtype=bool)
        for dr in range(3):
            for dc in range(3):
                grown |= padded[dr : dr + out.shape[0], dc : dc + out.shape[1]]
        out = grown
    return out


def _label_color(label: int) -> tuple[int, int, int]:
    return (
        80 + ((int(label) * 53) % 140),
        90 + ((int(label) * 97) % 125),
        105 + ((int(label) * 193) % 115),
    )


def _sanitize_path_component(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "snapshot"


def _write_replay_diff_images(
    arrays: Mapping[str, np.ndarray],
    replay_labels: np.ndarray,
    replay_separator: np.ndarray,
    room_debug: Mapping[str, object],
    shape: tuple[int, int],
    out_dir: Path,
    step: int,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    label_original = np.asarray(arrays.get("final_room_label_map", np.zeros(shape, dtype=np.int32)), dtype=np.int32)
    label_replay = np.asarray(replay_labels, dtype=np.int32)
    label_path = out_dir / ("roomseg_step_%06d.replay_label_diff.png" % int(step))
    _render_label_presence_diff(label_original, label_replay).save(label_path)
    paths["replay_label_diff_png"] = str(label_path)

    separator_original = _bool_like(arrays.get("accepted_separators"), shape)
    separator_replay = _bool_like(replay_separator, shape)
    separator_path = out_dir / ("roomseg_step_%06d.replay_separator_diff.png" % int(step))
    _render_bool_diff(separator_original, separator_replay).save(separator_path)
    paths["replay_separator_diff_png"] = str(separator_path)

    stable_original = _bool_like(arrays.get("voxel_stable_door_cut_mask"), shape) | _bool_like(arrays.get("voxel_door_stable_cut_mask"), shape)
    stable_replay = _bool_like(room_debug.get("voxel_stable_door_cut_mask"), shape) | _bool_like(room_debug.get("voxel_door_stable_cut_mask"), shape)
    stable_path = out_dir / ("roomseg_step_%06d.replay_stable_door_diff.png" % int(step))
    _render_bool_diff(stable_original, stable_replay).save(stable_path)
    paths["replay_stable_door_diff_png"] = str(stable_path)
    return paths


def _render_label_presence_diff(original_labels: np.ndarray, replay_labels: np.ndarray) -> Image.Image:
    original = np.asarray(original_labels, dtype=np.int32) > 0
    replay = np.asarray(replay_labels, dtype=np.int32) > 0
    rgb = np.zeros(original.shape + (3,), dtype=np.uint8)
    rgb[original & replay] = (210, 210, 210)
    rgb[original & ~replay] = (255, 80, 80)
    rgb[~original & replay] = (80, 150, 255)
    changed_id = (np.asarray(original_labels, dtype=np.int32) != np.asarray(replay_labels, dtype=np.int32)) & original & replay
    rgb[changed_id] = (255, 220, 60)
    return Image.fromarray(rgb, mode="RGB")


def _render_bool_diff(original_mask: np.ndarray, replay_mask: np.ndarray) -> Image.Image:
    original = np.asarray(original_mask, dtype=bool)
    replay = np.asarray(replay_mask, dtype=bool)
    rgb = np.zeros(original.shape + (3,), dtype=np.uint8)
    rgb[original & replay] = (230, 230, 230)
    rgb[original & ~replay] = (255, 80, 80)
    rgb[~original & replay] = (80, 150, 255)
    return Image.fromarray(rgb, mode="RGB")


def _bool_like(value: object, shape: tuple[int, int]) -> np.ndarray:
    if value is None:
        return np.zeros(shape, dtype=bool)
    arr = np.asarray(value)
    if arr.shape != shape:
        return np.zeros(shape, dtype=bool)
    return np.asarray(arr, dtype=bool)


def _room_count(labels: np.ndarray) -> int:
    values = np.unique(np.asarray(labels, dtype=np.int32))
    return int(np.count_nonzero(values > 0))


def _members_from_mask(mask: object, max_members: int = 20000) -> list[tuple[int, int]] | None:
    if mask is None:
        return None
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 2 or not np.any(arr):
        return None
    coords = np.argwhere(arr)
    if coords.shape[0] > int(max_members):
        coords = coords[: int(max_members)]
    return [(int(r), int(c)) for r, c in coords]


def _bool_array(arrays: Mapping[str, np.ndarray], key: str, shape: tuple[int, int]) -> np.ndarray:
    value = arrays.get(key)
    if value is None:
        return np.zeros(shape, dtype=bool)
    arr = np.asarray(value, dtype=bool)
    if arr.shape != shape:
        raise ValueError("%s shape %s does not match expected %s" % (key, arr.shape, shape))
    return arr


def _scalar(arrays: Mapping[str, np.ndarray], key: str, default: float) -> float:
    if key not in arrays:
        return float(default)
    arr = np.asarray(arrays[key])
    if arr.size == 0:
        return float(default)
    value = float(arr.reshape(-1)[0])
    return float(default) if not np.isfinite(value) and np.isfinite(default) else value


def _infer_step(path: Path) -> int:
    match = re.search(r"roomseg_step_(\d+)", path.name)
    return int(match.group(1)) if match else 0


def _snapshot_sort_key(path: Path) -> tuple[int, str]:
    return _infer_step(path), path.name


def _snapshot_root_sort_key(path: Path) -> tuple[str, int, str]:
    scene_dir = path.parent.parent if path.parent.name == "roomseg_snapshots" else path.parent
    return str(scene_dir), _infer_step(path), path.name


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
