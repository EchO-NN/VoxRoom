from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from voxroom_online.isaac_runtime.mapping.coordinate_transform import MapInfo
from voxroom_online.isaac_runtime.mapping.ceiling_height_estimator import CeilingHeightEstimator
from voxroom_online.isaac_runtime.mapping.voxel_occupancy_door_wall_roomseg import (
    VoxelOccupancyDoorWallRoomSegConfig,
    VoxelOccupancyDoorWallRoomSegmenter,
    run_voxel_occupancy_door_wall_roomseg,
)
from voxroom_online.isaac_runtime.mapping.voxel_occupancy_grid import (
    VoxelOccupancyGrid3D,
    VoxelOccupancyGridConfig,
)


def _bool_layer(npz: np.lib.npyio.NpzFile, key: str, shape: tuple[int, int], fallback: str | None = None) -> np.ndarray:
    if key in npz.files:
        return np.asarray(npz[key], dtype=bool)
    if fallback is not None and fallback in npz.files:
        return np.asarray(npz[fallback], dtype=bool)
    return np.zeros(shape, dtype=bool)


def _grid_from_snapshot(
    npz: np.lib.npyio.NpzFile,
    *,
    grid_cfg: VoxelOccupancyGridConfig,
    ceiling_estimator_cfg: dict[str, object],
    resolution_m: float,
) -> VoxelOccupancyGrid3D:
    state = np.asarray(npz["voxel_occupancy_state_zyx"], dtype=np.uint8)
    height, width = int(state.shape[1]), int(state.shape[2])
    shape = (height, width)
    ceiling_height_m = (
        float(npz["voxel_occupancy_ceiling_height_estimate_m"])
        if "voxel_occupancy_ceiling_height_estimate_m" in npz.files
        else None
    )
    ceiling_status = (
        str(npz["voxel_occupancy_ceiling_estimate_status"])
        if "voxel_occupancy_ceiling_estimate_status" in npz.files
        else "unavailable"
    )
    active_z_max_m = float(npz["voxel_occupancy_active_z_max_m"])
    ceiling_source = str(ceiling_estimator_cfg.get("source", "voxel_occupied_layer_peak") or "").strip().lower()
    if ceiling_source in {"voxel_occupied_layer_peak", "occupied_layer_peak", "occupied_z_layer_peak"}:
        if "voxel_occupancy_z_centers_m" in npz.files:
            z_centers_m = np.asarray(npz["voxel_occupancy_z_centers_m"], dtype=np.float32)
        else:
            z_min_m = float(npz["voxel_occupancy_z_min_m"])
            z_resolution_m = float(npz["voxel_occupancy_z_resolution_m"])
            z_centers_m = z_min_m + (np.arange(int(state.shape[0]), dtype=np.float32) + 0.5) * z_resolution_m
        estimator = CeilingHeightEstimator(
            ceiling_estimator_cfg,
            active_z_min_m=float(npz["voxel_occupancy_active_z_min_m"]),
            storage_z_max_m=float(npz["voxel_occupancy_z_max_m"]),
            active_z_max_fallback_m=float(getattr(grid_cfg, "active_z_max_fallback_m", 2.80)),
            active_z_max_ceiling_ratio=float(getattr(grid_cfg, "active_z_max_ceiling_ratio", 0.85)),
            active_z_max_cap_m=float(getattr(grid_cfg, "active_z_max_cap_m", 2.80)),
        )
        estimate = estimator.update_from_occupied_layers(state, z_centers_m)
        ceiling_height_m = estimate.height_m
        active_z_max_m = float(estimate.active_z_max_m)
        ceiling_status = "occupied_layer_peak" if ceiling_height_m is not None else "fallback"
    return VoxelOccupancyGrid3D(
        log_odds=np.asarray(npz["voxel_occupancy_log_odds_zyx"], dtype=np.int16),
        state=state,
        sensor_range_count=np.asarray(npz["voxel_sensor_range_count_zyx"], dtype=np.uint8),
        floor_frustum_seen_count_xy=np.asarray(
            npz["voxel_floor_frustum_seen_count_xy"] if "voxel_floor_frustum_seen_count_xy" in npz.files else np.zeros(shape, dtype=np.uint16),
            dtype=np.uint16,
        ),
        outside_score_xy=np.asarray(
            npz["voxel_outside_score_xy"] if "voxel_outside_score_xy" in npz.files else np.zeros(shape, dtype=np.uint8),
            dtype=np.uint8,
        ),
        outside_xy=np.asarray(
            npz["voxel_outside_xy"] if "voxel_outside_xy" in npz.files else np.zeros(shape, dtype=bool),
            dtype=bool,
        ),
        z_min_m=float(npz["voxel_occupancy_z_min_m"]),
        z_max_m=float(npz["voxel_occupancy_z_max_m"]),
        z_resolution_m=float(npz["voxel_occupancy_z_resolution_m"]),
        map_info=MapInfo(
            resolution_m=float(resolution_m),
            min_x=0.0,
            max_x=float(width) * float(resolution_m),
            min_y=0.0,
            max_y=float(height) * float(resolution_m),
            width=width,
            height=height,
        ),
        config=grid_cfg,
        active_z_min_m=float(npz["voxel_occupancy_active_z_min_m"]),
        active_z_max_m=active_z_max_m,
        ceiling_height_m=ceiling_height_m,
        ceiling_estimate_status=ceiling_status,
    )


def _base_nav_image(free: np.ndarray, occupied: np.ndarray, unknown: np.ndarray) -> np.ndarray:
    image = np.full((*free.shape, 3), 205, dtype=np.uint8)
    image[unknown] = (205, 205, 205)
    image[free] = (255, 255, 255)
    image[occupied] = (25, 25, 25)
    return image


def _render_snapshot(
    snapshot_path: Path,
    *,
    output_scene_dir: Path,
    npz_output_scene_dir: Path | None = None,
    roomseg_cfg: VoxelOccupancyDoorWallRoomSegConfig,
    grid_cfg_raw: dict[str, object],
    ceiling_estimator_cfg: dict[str, object],
    resolution_m: float,
    segmenter: VoxelOccupancyDoorWallRoomSegmenter | None = None,
    skip_existing: bool = False,
) -> tuple[int, int]:
    stem = snapshot_path.stem
    check_path = output_scene_dir / f"{stem}.check_red_cut.png"
    raw_seed_path = output_scene_dir / f"{stem}.raw_seed.png"
    if skip_existing and check_path.exists() and raw_seed_path.exists():
        return -1, -1

    npz = np.load(snapshot_path, allow_pickle=True)
    shape = tuple(np.asarray(npz["voxel_nav_free_xy"]).shape)
    grid_cfg = VoxelOccupancyGridConfig.from_mapping(
        {
            **grid_cfg_raw,
            "z_min_m": float(npz["voxel_occupancy_z_min_m"]),
            "z_max_m": float(npz["voxel_occupancy_z_max_m"]),
            "z_resolution_m": float(npz["voxel_occupancy_z_resolution_m"]),
            "active_z_min_m": float(npz["voxel_occupancy_active_z_min_m"]),
        }
    )
    grid = _grid_from_snapshot(npz, grid_cfg=grid_cfg, ceiling_estimator_cfg=ceiling_estimator_cfg, resolution_m=resolution_m)

    no_clearance_nav_free = _bool_layer(npz, "voxel_nav_free_xy", shape, "observed_free_mask")
    nav_occupied = _bool_layer(npz, "voxel_nav_occupied_xy", shape, "obstacle_mask")
    unknown = _bool_layer(npz, "voxel_nav_unknown_xy", shape, "unknown_mask")
    observed_free = _bool_layer(npz, "observed_free_mask", shape, "voxel_nav_free_xy")
    occupancy = _bool_layer(npz, "occupancy_map", shape, "voxel_nav_occupied_xy")
    obstacle = _bool_layer(npz, "obstacle_mask", shape, "voxel_nav_occupied_xy")

    step = _step_from_snapshot_name(snapshot_path.name)
    if segmenter is None:
        result = run_voxel_occupancy_door_wall_roomseg(
            occupancy_map=occupancy,
            observed_free_mask=observed_free,
            obstacle_mask=obstacle,
            unknown_mask=unknown,
            voxel_grid=grid,
            navigation_free_mask=no_clearance_nav_free,
            navigation_obstacle_mask=nav_occupied,
            door_seed_no_clearance_free_mask=no_clearance_nav_free,
            resolution_m=float(resolution_m),
            config=roomseg_cfg,
            step=step,
        )
    else:
        segmenter.update(
            occupancy,
            observed_free,
            obstacle,
            unknown,
            step=step,
            voxel_grid=grid,
            navigation_free_mask=no_clearance_nav_free,
            navigation_obstacle_mask=nav_occupied,
            door_seed_no_clearance_free_mask=no_clearance_nav_free,
        )
        result = segmenter.last_result
        if result is None:
            raise RuntimeError(f"stateful segmenter did not produce a result for {snapshot_path}")

    layers = result.layers
    raw_seed = np.asarray(layers.get("voxel_door_raw_seed_mask", layers.get("voxel_door_seed_mask", np.zeros(shape, dtype=bool))), dtype=bool)
    current_cut = np.asarray(
        layers.get("voxel_current_door_cut_mask", layers.get("voxel_door_current_cut_mask", np.zeros(shape, dtype=bool))),
        dtype=bool,
    )
    final_cut = np.asarray(layers.get("voxel_door_cut_mask", np.zeros(shape, dtype=bool)), dtype=bool)
    door_cut = current_cut | final_cut

    after_free = no_clearance_nav_free & ~door_cut
    after_occupied = nav_occupied | door_cut
    after_unknown = unknown & ~after_free & ~after_occupied

    check_image = _base_nav_image(after_free, after_occupied, after_unknown)
    check_image[door_cut] = (235, 35, 35)

    raw_seed_image = _base_nav_image(no_clearance_nav_free, nav_occupied, unknown)
    raw_seed_image[raw_seed] = (0, 120, 255)

    Image.fromarray(check_image).save(check_path)
    Image.fromarray(raw_seed_image).save(raw_seed_path)
    if npz_output_scene_dir is not None:
        _save_stateful_replay_npz(
            snapshot_path,
            source_npz=npz,
            result=result,
            grid=grid,
            output_scene_dir=npz_output_scene_dir,
            resolution_m=float(resolution_m),
        )
    return int(np.count_nonzero(raw_seed)), int(np.count_nonzero(door_cut))


def _save_stateful_replay_npz(
    snapshot_path: Path,
    *,
    source_npz: np.lib.npyio.NpzFile,
    result,
    grid: VoxelOccupancyGrid3D,
    output_scene_dir: Path,
    resolution_m: float,
) -> None:
    out_dir = output_scene_dir / "roomseg_snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for key in source_npz.files:
        arr = np.asarray(source_npz[key])
        if arr.dtype.hasobject:
            continue
        arrays[str(key)] = arr.copy()
    for key, value in dict(getattr(result, "debug", {}) or {}).items():
        arr = np.asarray(value)
        if arr.dtype.hasobject:
            continue
        arrays[str(key)] = arr.copy()
    for key, value in dict(result.layers).items():
        arr = np.asarray(value)
        if arr.dtype.hasobject:
            continue
        arrays[str(key)] = arr.copy()
    labels = np.asarray(result.room_label_map, dtype=np.int32)
    separators = np.asarray(result.separator_map, dtype=bool)
    arrays["final_room_label_map"] = labels
    arrays["voxel_final_room_label_map"] = labels
    arrays["accepted_separators"] = separators
    arrays["voxel_final_separator_map"] = separators
    arrays["stateful_replay_source_snapshot_path"] = np.asarray(str(snapshot_path))
    arrays["stateful_replay_mode"] = np.asarray("stateful_by_scene_current_config")
    arrays["map_resolution_m"] = np.asarray(float(resolution_m), dtype=np.float32)
    arrays["map_width_cells"] = np.asarray(int(labels.shape[1]), dtype=np.int32)
    arrays["map_height_cells"] = np.asarray(int(labels.shape[0]), dtype=np.int32)
    ceiling_height = getattr(grid, "ceiling_height_m", None)
    arrays["voxel_occupancy_ceiling_height_estimate_m"] = np.asarray(
        np.nan if ceiling_height is None else float(ceiling_height),
        dtype=np.float32,
    )
    arrays["voxel_occupancy_active_z_max_m"] = np.asarray(float(getattr(grid, "active_z_max_m", np.nan)), dtype=np.float32)
    arrays["voxel_occupancy_ceiling_estimate_status"] = np.asarray(str(getattr(grid, "ceiling_estimate_status", "unknown")))
    np.savez_compressed(out_dir / snapshot_path.name, **arrays)


def _load_render_config(
    config_path: Path,
    *,
    disable_door_line_regression_gate: bool = False,
) -> tuple[VoxelOccupancyDoorWallRoomSegConfig, dict[str, object], dict[str, object], float]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_root = yaml.safe_load(f)
    roomseg_cfg_raw = dict(cfg_root.get("mapping", {}).get("room_segmentation", {}) or {})
    resolution_m = float(roomseg_cfg_raw.get("resolution_m", 0.05) or 0.05)
    roomseg_cfg = VoxelOccupancyDoorWallRoomSegConfig.from_mapping(roomseg_cfg_raw, resolution_m=resolution_m)
    if bool(disable_door_line_regression_gate):
        roomseg_cfg.door.primitive_require_line_fit_quality = False
    grid_cfg_raw = dict(roomseg_cfg_raw.get("voxel_grid", {}) or {})
    height_profile_cfg_raw = dict(roomseg_cfg_raw.get("height_profile", {}) or {})
    ceiling_estimator_cfg = dict(height_profile_cfg_raw.get("ceiling_estimator", {}) or {})
    return roomseg_cfg, grid_cfg_raw, ceiling_estimator_cfg, resolution_m


def _render_snapshot_worker(args: tuple[str, str, str | None, str, bool, bool]) -> tuple[str, str, int, int]:
    snapshot_str, output_root_str, npz_output_root_str, config_str, skip_existing, disable_line_gate = args
    snapshot = Path(snapshot_str)
    output_root = Path(output_root_str)
    npz_output_root = None if npz_output_root_str is None else Path(npz_output_root_str)
    roomseg_cfg, grid_cfg_raw, ceiling_estimator_cfg, resolution_m = _load_render_config(
        Path(config_str),
        disable_door_line_regression_gate=bool(disable_line_gate),
    )
    scene = snapshot.parents[1].name
    output_scene_dir = output_root / scene
    output_scene_dir.mkdir(parents=True, exist_ok=True)
    seed_cells, cut_cells = _render_snapshot(
        snapshot,
        output_scene_dir=output_scene_dir,
        npz_output_scene_dir=None if npz_output_root is None else npz_output_root / scene,
        roomseg_cfg=roomseg_cfg,
        grid_cfg_raw=grid_cfg_raw,
        ceiling_estimator_cfg=ceiling_estimator_cfg,
        resolution_m=resolution_m,
        skip_existing=bool(skip_existing),
    )
    return scene, snapshot.name, seed_cells, cut_cells


def _scene_snapshot_paths(scene_dir: Path) -> list[Path]:
    return sorted(scene_dir.glob("roomseg_snapshots/roomseg_step_*.npz"))


def _render_scene_stateful(
    scene_dir: Path,
    *,
    output_root: Path,
    npz_output_root: Path | None,
    config_path: Path,
    skip_existing: bool,
    disable_door_line_regression_gate: bool,
) -> list[tuple[str, str, int, int]]:
    roomseg_cfg, grid_cfg_raw, ceiling_estimator_cfg, resolution_m = _load_render_config(
        config_path,
        disable_door_line_regression_gate=bool(disable_door_line_regression_gate),
    )
    segmenter = VoxelOccupancyDoorWallRoomSegmenter(config=roomseg_cfg)
    scene = scene_dir.name
    output_scene_dir = output_root / scene
    output_scene_dir.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, str, int, int]] = []
    for snapshot in _scene_snapshot_paths(scene_dir):
        seed_cells, cut_cells = _render_snapshot(
            snapshot,
            output_scene_dir=output_scene_dir,
            npz_output_scene_dir=None if npz_output_root is None else npz_output_root / scene,
            roomseg_cfg=roomseg_cfg,
            grid_cfg_raw=grid_cfg_raw,
            ceiling_estimator_cfg=ceiling_estimator_cfg,
            resolution_m=resolution_m,
            segmenter=segmenter,
            skip_existing=skip_existing,
        )
        rows.append((scene, snapshot.name, seed_cells, cut_cells))
    return rows


def _render_scene_stateful_worker(args: tuple[str, str, str | None, str, bool, bool]) -> list[tuple[str, str, int, int]]:
    scene_str, output_root_str, npz_output_root_str, config_str, skip_existing, disable_line_gate = args
    return _render_scene_stateful(
        Path(scene_str),
        output_root=Path(output_root_str),
        npz_output_root=None if npz_output_root_str is None else Path(npz_output_root_str),
        config_path=Path(config_str),
        skip_existing=bool(skip_existing),
        disable_door_line_regression_gate=bool(disable_line_gate),
    )


def _step_from_snapshot_name(name: str) -> int:
    parts = Path(name).stem.split("_")
    for idx, part in enumerate(parts):
        if part == "step" and idx + 1 < len(parts):
            try:
                return int(parts[idx + 1])
            except ValueError:
                return 0
        if part.startswith("step"):
            try:
                return int(part.removeprefix("step").split(".")[0])
            except ValueError:
                return 0
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="final")
    parser.add_argument("--output-root", default="对比实验/outputs/final_door_seed_rerun_nav_after_door_completion_all_snapshots")
    parser.add_argument("--npz-output-root", default=None, help="Also save current stateful roomseg NPZ snapshots under this run root.")
    parser.add_argument("--config", default="configs/voxroom_online.yaml")
    parser.add_argument(
        "--disable-door-line-regression-gate",
        action="store_true",
        help="For ablations: strict door seed line primitives skip correlation and orthogonal variance checks only.",
    )
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--stateful-by-scene",
        action="store_true",
        help="Process each scene in snapshot order with one segmenter instance so door memory is preserved.",
    )
    parser.add_argument("--scene", action="append", default=None, help="Limit rendering to one or more scene names.")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    npz_output_root = None if args.npz_output_root is None else Path(args.npz_output_root)
    if args.clean and output_root.exists():
        for path in output_root.rglob("*.png"):
            path.unlink()
    if args.clean and npz_output_root is not None and npz_output_root.exists():
        for path in npz_output_root.rglob("*.npz"):
            path.unlink()
    output_root.mkdir(parents=True, exist_ok=True)
    if npz_output_root is not None:
        npz_output_root.mkdir(parents=True, exist_ok=True)

    scene_filter = set(args.scene or [])
    snapshots = sorted(
        snapshot
        for snapshot in input_root.glob("*/roomseg_snapshots/roomseg_step_*.npz")
        if not scene_filter or snapshot.parents[1].name in scene_filter
    )
    workers = max(1, int(args.workers))
    mode = "stateful_by_scene" if args.stateful_by_scene else "stateless_snapshot"
    print(
        f"rendering {len(snapshots)} snapshots workers={workers} mode={mode} "
        f"disable_door_line_regression_gate={bool(args.disable_door_line_regression_gate)}"
    )
    if args.stateful_by_scene:
        scene_dirs = sorted({snapshot.parents[1] for snapshot in snapshots})
        if workers == 1:
            done = 0
            for scene_dir in scene_dirs:
                for scene, name, seed_cells, cut_cells in _render_scene_stateful(
                    scene_dir,
                    output_root=output_root,
                    npz_output_root=npz_output_root,
                    config_path=Path(args.config),
                    skip_existing=bool(args.skip_existing),
                    disable_door_line_regression_gate=bool(args.disable_door_line_regression_gate),
                ):
                    done += 1
                    status = "skip" if seed_cells < 0 else f"seed={seed_cells} cut={cut_cells}"
                    print(f"[{done:04d}/{len(snapshots):04d}] {scene}/{name} {status}", flush=True)
            return
        jobs = [
            (
                str(scene_dir),
                str(output_root),
                None if npz_output_root is None else str(npz_output_root),
                str(Path(args.config)),
                bool(args.skip_existing),
                bool(args.disable_door_line_regression_gate),
            )
            for scene_dir in scene_dirs
        ]
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_render_scene_stateful_worker, job) for job in jobs]
            for future in as_completed(futures):
                for scene, name, seed_cells, cut_cells in future.result():
                    done += 1
                    status = "skip" if seed_cells < 0 else f"seed={seed_cells} cut={cut_cells}"
                    print(f"[{done:04d}/{len(snapshots):04d}] {scene}/{name} {status}", flush=True)
        return
    if workers == 1:
        roomseg_cfg, grid_cfg_raw, ceiling_estimator_cfg, resolution_m = _load_render_config(
            Path(args.config),
            disable_door_line_regression_gate=bool(args.disable_door_line_regression_gate),
        )
        for idx, snapshot in enumerate(snapshots, start=1):
            scene = snapshot.parents[1].name
            output_scene_dir = output_root / scene
            output_scene_dir.mkdir(parents=True, exist_ok=True)
            seed_cells, cut_cells = _render_snapshot(
                snapshot,
                output_scene_dir=output_scene_dir,
                npz_output_scene_dir=None if npz_output_root is None else npz_output_root / scene,
                roomseg_cfg=roomseg_cfg,
                grid_cfg_raw=grid_cfg_raw,
                ceiling_estimator_cfg=ceiling_estimator_cfg,
                resolution_m=resolution_m,
                skip_existing=bool(args.skip_existing),
            )
            status = "skip" if seed_cells < 0 else f"seed={seed_cells} cut={cut_cells}"
            print(f"[{idx:04d}/{len(snapshots):04d}] {scene}/{snapshot.name} {status}", flush=True)
    else:
        jobs = [
            (
                str(snapshot),
                str(output_root),
                None if npz_output_root is None else str(npz_output_root),
                str(Path(args.config)),
                bool(args.skip_existing),
                bool(args.disable_door_line_regression_gate),
            )
            for snapshot in snapshots
        ]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_render_snapshot_worker, job) for job in jobs]
            for idx, future in enumerate(as_completed(futures), start=1):
                scene, name, seed_cells, cut_cells = future.result()
                status = "skip" if seed_cells < 0 else f"seed={seed_cells} cut={cut_cells}"
                print(f"[{idx:04d}/{len(snapshots):04d}] {scene}/{name} {status}", flush=True)


if __name__ == "__main__":
    main()
