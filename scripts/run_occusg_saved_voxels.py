#!/usr/bin/env python3
"""Official OccuSG geometry replay; saved-state adapter, not RGB-D reconstruction.

Run one scene per process to reset *all* upstream incremental state. Only the
occupancy volume, its coordinates, and saved odometry enter the predictor.
Ground truth is opened only by the separate evaluate command after replay.
"""
from __future__ import annotations

import argparse
from array import array
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

UPSTREAM = Path("/work/upstream")
PIPELINE = UPSTREAM / "src/scene_graph_ros/config/scene_graph_pipeline_params_mp3d.yaml"
DUDE_CONFIG = UPSTREAM / "src/incremental_dude_ros2/incremental_dude_ros2/config/inc_dude_params.yaml"
DATA = Path("/media/echo/data/voxroom_roomseg_evaluation")
DATASETS = {"interioragent": DATA / "interioragent_all_available_gt_20260828",
            "grscene": DATA / "grscene_all_available_gt_20260817"}
METHOD = "occusg_official_saved_voxel_replay"
INPUT_MODE = "saved_voxels"
INPUT_REFERENCE_ROOT = None
VERTICAL_MODES = ("raw_vertical_free", "raw_vertical_free_cropped")


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(dataset, output):
    excluded = set().union(*(set(json.loads((root / "training_scene_exclusion_manifest.json").read_text())["excluded_scene_ids"])
                             for root in DATASETS.values()))
    source = DATASETS[dataset] / "index_voxroom.json"
    index = json.loads(source.read_text())
    selected = []
    for episode in index["episodes"]:
        uid = episode["episode_uid"]
        annotation = DATASETS[dataset] / "annotations" / uid / "last_step.annotation.json"
        if episode["scene_id"] in excluded or not annotation.exists():
            continue
        if json.loads(annotation.read_text())["review"]["status"] != "approved":
            continue
        events = episode["snapshots"]
        assert len({s["coverage_event_id"] for s in events}) == len(events)
        assert all(Path(s["snapshot_path"]).is_file() for s in events)
        assert all(int(a["step"]) <= int(b["step"]) for a, b in zip(events, events[1:]))
        selected.append(episode)
    assert selected and len({e["scene_id"] for e in selected}) == len(selected)
    save_json(output / "inputs/index.json", {**index, "episodes": selected})
    provenance_path = output / "inputs/provenance.json"
    history = []
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text())
        assert previous["method"] == METHOD, "Refusing to mix different input experiments in one output directory"
        assert previous["source_index_sha256"] == sha(source)
        history = previous.get("implementation_history", [])
        if previous["adapter_sha256"] != sha(__file__):
            history.append({"adapter_sha256": previous["adapter_sha256"], "superseded_time": time.time()})
    save_json(provenance_path, {
        "method": METHOD, "dataset": dataset, "source_index_sha256": sha(source),
        "scenes": len(selected), "checkpoints": sum(len(e["snapshots"]) for e in selected),
        "excluded_training_and_validation_scene_ids": sorted(excluded),
        "upstream_commit": subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip(),
        "projection_submodule_commit": subprocess.check_output(["git", "-C", str(UPSTREAM / "src/mapconversion"), "rev-parse", "HEAD"], text=True).strip(),
        "official_parameter_files": {str(p): sha(p) for p in (PIPELINE, DUDE_CONFIG)},
        "adapter_sha256": sha(__file__),
        "implementation_history": history,
        "input_keys": ["voxel_occupancy_state_zyx", "voxel_occupancy_z_centers_m", "map_resolution_m", "map_origin_x_m", "map_origin_y_m", "demo_pose_world"],
        "scope": "Official OctoMap-to-mapUAV, incremental DUDE, tracking, and geometric room materialization on saved occupancy checkpoints. NOT a full RGB-D / OctoMap / object-detection reproduction.",
        "adaptations": ["Known free/occupied states become OctoMap leaves; unknown is absent; reject undefined/conflict states.",
                        "Full saved height range; no VoxRoom vertical/nav map, seed, prediction or GT enters the algorithm.",
                        "One native map update per saved checkpoint, in temporal order. Intermediate sensor frames are unavailable.",
                        "ROS wall time for transport; official geometric parameters are unchanged.",
                        "Official graph callbacks and maintenance/pipeline ticks are invoked synchronously once per saved checkpoint; real-time frequencies cannot be reproduced from sparse snapshots.",
                        "Only saved robot poses and map navigation support are available for grounding; no object detections are synthesized.",
                        "Metrics use the existing held-out TVARS-overlap P/R, per-checkpoint F1, Hungarian IoU sum divided by GT room count, 0.5 m2 filtering; NOT the paper's matched-only IoU protocol."],
    })
    if INPUT_MODE in VERTICAL_MODES:
        provenance = json.loads(provenance_path.read_text())
        provenance.update({
            "input_mode": INPUT_MODE,
            "input_reference_root": str(INPUT_REFERENCE_ROOT),
            "input_keys": sorted(VERTICAL_INPUT_KEYS) + ["map_*", "demo_pose_world"],
            "scope": "Input-only ablation: original saved Vertical Free OccupancyGrid replaces mapUAV projection; official OccuSG DUDE, tracking, update guard and room materialization are unchanged.",
            "adaptations": [
                "Bypass OctoMap/map_conversion; publish the saved raw Vertical Free + wall/unknown grid directly to /mapUAV.",
                "Assert every grid cell and map geometry equal to corrected DUDE input, before publication and after receiving our ROS message.",
                "No 3D reprojection, Nav Free input, clearance change, door seeds, predicted labels or GT/reference masks enter the algorithm.",
                "One update per saved checkpoint and the same odometry/graph callback order as the original OccuSG run.",
                "Official DUDE/tracker/guard/scene-graph parameters unchanged; only map_conversion is bypassed.",
                "Same held-out scenes and overlap P/R, per-checkpoint F1, Hungarian IoU/nGT, 0.5 m2 evaluation filter.",
            ],
            "backend_sha256": {str(p): sha(p) for p in (
                UPSTREAM / "install/incremental_dude_ros2/lib/incremental_dude_ros2/inc_dude",
                UPSTREAM / "src/scene_graph_ros/scene_graph_ros/scene_graph_region.py")},
        })
        if INPUT_MODE == "raw_vertical_free_cropped":
            provenance["scope"] = "Unknown-border control: crop only all-unknown outer rows/columns from the same raw Vertical Free grid; preserve all observed cells, interior unknown, resolution and world coordinates. Official OccuSG backend unchanged."
            provenance["adaptations"][1] = "Check full input against historical DUDE grid, then crop to the tight bounding rectangle of all known cells; reconstruct the full input exactly for audit."
            provenance["adaptations"].append("Shift input map origin by the crop offset; rasterize published world polygons back onto the original full source grid for unchanged evaluation.")
        save_json(provenance_path, provenance)
    return selected


def encode_snapshot(path, work):
    # Do not load all npz fields: many contain main-method results and GT domains.
    with np.load(path, allow_pickle=False) as d:
        state = d["voxel_occupancy_state_zyx"]
        heights = d["voxel_occupancy_z_centers_m"]
        resolution = float(d["map_resolution_m"])
        origin = [float(d["map_origin_x_m"]), float(d["map_origin_y_m"])]
        pose = d["demo_pose_world"].astype(float).tolist()
    assert abs(resolution - .05) < 1e-7, resolution
    assert state.ndim == 3 and state.shape[0] == len(heights)
    counts = np.bincount(state.ravel(), minlength=4)
    if counts[3:].sum():
        raise ValueError(f"Unspecified/conflicting voxel states: {counts.tolist()}")
    z, y, x = np.nonzero(state)
    # Snap sub-micrometre float32 serialization error only, never shift a cell.
    canonical_origin = [round(v / .05) * .05 for v in origin]
    if max(abs(a-b) for a, b in zip(origin, canonical_origin)) > 1e-5:
        raise ValueError(f"Source grid is not aligned to the official 0.05 m OctoMap grid: {origin}")
    records = np.empty((len(x), 4), dtype="<f4")
    records[:, 0] = canonical_origin[0] + (x + .5) * .05
    records[:, 1] = canonical_origin[1] + (y + .5) * .05
    records[:, 2] = heights[z]
    records[:, 3] = state[z, y, x]
    records.tofile(work / "known_voxels.bin")
    subprocess.run(["/work/adapter/snapshot_to_octomap", ".05", str(work / "known_voxels.bin"), str(work / "octomap.data")], check=True)
    return {"shape": list(state.shape[1:]), "resolution": .05, "origin": origin, "pose": pose,
            "state_counts": counts.tolist(), "z_range": [float(heights.min() - .025), float(heights.max() + .025)]}


VERTICAL_INPUT_KEYS = {
    "occupancy_map", "observed_free_mask", "obstacle_mask", "voxel_vertical_free_xy",
    "voxel_wall_xy", "structural_wall_clean", "roomseg_sanitized_wall",
}


def encode_vertical_snapshot(path, reference):
    from voxroom_online.isaac_runtime.baselines.mask_io import SEGMENTATION_INPUT_MODE_KEY
    from voxroom_online.isaac_runtime.baselines.ros_grid_io import snapshot_to_ros_occupancy_grid
    from voxroom_online.isaac_runtime.baselines.data_contract import resolve_map_info
    with np.load(path, allow_pickle=False) as source:
        arrays = {k: source[k] for k in source.files if k in VERTICAL_INPUT_KEYS or k.startswith("map_")}
        pose = source["demo_pose_world"].astype(float).tolist()
    free = np.asarray(arrays["voxel_vertical_free_xy"], dtype=bool)
    if free.ndim != 2 or not free.any():
        raise ValueError("Missing/nonempty raw voxel_vertical_free_xy required; no fallback")
    arrays[SEGMENTATION_INPUT_MODE_KEY] = np.asarray("raw_vertical_free")
    grid = snapshot_to_ros_occupancy_grid(arrays)
    np.testing.assert_array_equal(grid == 0, free)
    info = resolve_map_info(snapshot_arrays=arrays).to_metadata()
    with np.load(reference, allow_pickle=False) as old:
        np.testing.assert_array_equal(grid, old["dude_ros_occupancy_grid"])
        old_info = json.loads(str(old["baseline_metadata_json"]))["map_info"]
    for field in ("resolution_m", "min_x", "min_y", "max_x", "max_y", "width", "height"):
        assert info[field] == old_info[field], (field, info, old_info)
    assert abs(info["resolution_m"] - .05) < 1e-7
    assert grid.shape == (info["height"], info["width"])
    return grid, {"shape": list(grid.shape), "resolution": info["resolution_m"],
                  "origin": [info["min_x"], info["min_y"]], "pose": pose,
                  "input_mode": "raw_vertical_free", "input_identical_to_dude": True,
                  "input_grid_sha256": hashlib.sha256(grid.tobytes()).hexdigest(),
                  "reference_path": str(reference)}


def assert_received_vertical_grid(message, expected, geometry):
    actual = np.asarray(message.data, dtype=np.int8).reshape(message.info.height, message.info.width)
    np.testing.assert_array_equal(actual, expected)
    assert [message.info.origin.position.x, message.info.origin.position.y] == geometry["origin"]
    # ROS OccupancyGrid resolution is a float32 field.
    assert message.info.resolution == float(np.float32(geometry["resolution"]))
    assert message.info.origin.orientation.w == 1.0


def crop_unknown_border(grid, geometry):
    """Remove only empty exterior padding; never fill holes or drop known cells."""
    grid = np.asarray(grid, dtype=np.int8)
    assert list(grid.shape) == geometry["shape"]
    yy, xx = np.nonzero(grid >= 0)
    if not len(yy):
        raise ValueError("Cannot crop an entirely unknown map")
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    cropped = grid[y0:y1, x0:x1].copy()
    restored = np.full(grid.shape, -1, dtype=np.int8)
    restored[y0:y1, x0:x1] = cropped
    np.testing.assert_array_equal(restored, grid)
    resolution = geometry["resolution"]
    origin = [geometry["origin"][0] + x0*resolution, geometry["origin"][1] + y0*resolution]
    # Check world coordinates of all observed cells, not just the bounding box.
    np.testing.assert_allclose(origin[0] + (xx-x0+.5)*resolution,
                               geometry["origin"][0] + (xx+.5)*resolution, rtol=0, atol=1e-10)
    np.testing.assert_allclose(origin[1] + (yy-y0+.5)*resolution,
                               geometry["origin"][1] + (yy+.5)*resolution, rtol=0, atol=1e-10)
    return cropped, {**geometry, "shape": list(cropped.shape), "origin": origin,
                     "source_shape": list(grid.shape), "source_origin": list(geometry["origin"]),
                     "crop_bounds_yxyx": [y0, x0, y1, x1],
                     "input_mode": "raw_vertical_free_cropped", "input_identical_to_dude": False,
                     "full_grid_reconstruction_identical_to_dude": True,
                     "full_input_grid_sha256": hashlib.sha256(grid.tobytes()).hexdigest(),
                     "input_grid_sha256": hashlib.sha256(cropped.tobytes()).hexdigest(),
                     "removed_unknown_cells": int(grid.size-cropped.size), "removed_known_cells": 0,
                     "unknown_ratio_before_crop": float(np.mean(grid < 0)),
                     "unknown_ratio_after_crop": float(np.mean(cropped < 0))}


def rasterize(polygons, shape, origin, resolution):
    # Evaluate published world polygons at original source cell centres; no flip.
    import shapely
    labels = np.zeros(shape, dtype=np.int32)
    for idx, points in enumerate(polygons, 1):
        if len(points) < 3:
            continue
        polygon = shapely.Polygon(points)
        if not polygon.is_valid:
            # The official published vertices remain saved for auditing.
            polygon = shapely.make_valid(polygon)
        xmin, ymin, xmax, ymax = polygon.bounds
        x0, x1 = max(0, int(np.floor((xmin-origin[0])/resolution))), min(shape[1], int(np.ceil((xmax-origin[0])/resolution))+1)
        y0, y1 = max(0, int(np.floor((ymin-origin[1])/resolution))), min(shape[0], int(np.ceil((ymax-origin[1])/resolution))+1)
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        inside = shapely.intersects_xy(polygon, origin[0] + (xx+.5)*resolution, origin[1] + (yy+.5)*resolution)
        target = labels[y0:y1, x0:x1]
        # Shared boundaries get the first official ID in sorted order.
        target[inside & (target == 0)] = idx
    return labels


def run_scene(episode, output):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
    from rcl_interfaces.srv import GetParameters
    from rclpy.parameter import parameter_value_to_python
    import yaml
    from octomap_msgs.msg import Octomap
    from nav_msgs.msg import OccupancyGrid, Odometry
    from incremental_dude_msgs.msg import Region2DArray
    from scene_graph_ros.scene_graph_region import SceneGraphOrchestrator
    from scene_graph_core.representation import NodeType

    output.mkdir(parents=True, exist_ok=True)
    if (output / "complete.json").exists():
        raise FileExistsError(f"Completed scene must not be restarted: {output}")
    processes, logs = [], []
    rclpy.init(args=["--ros-args", "--params-file", str(PIPELINE), "-p", "use_sim_time:=false",
                      "-p", "export_json_on_shutdown:=false"])
    node = Node("snapshot_replay_adapter")
    graph = SceneGraphOrchestrator()
    maps, regions = [], []
    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
    map_sub = node.create_subscription(OccupancyGrid, "/mapUAV", maps.append, qos)
    region_sub = node.create_subscription(Region2DArray, "/dude/regions", regions.append, qos)
    vertical = INPUT_MODE in VERTICAL_MODES
    publisher = node.create_publisher(OccupancyGrid if vertical else Octomap,
                                      "/mapUAV" if vertical else "/octomap", qos)
    def wait_for(predicate, timeout=900):
        started = time.monotonic()
        while not predicate():
            for process in processes:
                if process.poll() is not None:
                    raise RuntimeError(f"Official ROS node exited {process.returncode}")
            if time.monotonic() - started > timeout:
                raise TimeoutError("No corresponding official map / region update")
            rclpy.spin_once(node, timeout_sec=.1)
    try:
        commands = [
            [str(UPSTREAM / "install/mapconversion/lib/mapconversion/map_conversion_oct_node"), "--ros-args", "-r", "__node:=map_conversion_node", "--params-file", str(PIPELINE), "-p", "use_sim_time:=false"],
            [str(UPSTREAM / "install/incremental_dude_ros2/lib/incremental_dude_ros2/inc_dude"), "--ros-args", "--params-file", str(DUDE_CONFIG), "-p", "use_sim_time:=false"],
        ]
        if vertical:
            commands = commands[1:]  # Keep the identical official DUDE command.
        save_json(output / "commands.json", commands)
        for i, command in enumerate(commands):
            log = (output / f"native_{i}.log").open("w")
            logs.append(log)
            processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
        wait_for(lambda: publisher.get_subscription_count() >= 1 and node.count_publishers("/dude/regions") >= 1 and node.count_subscribers("/mapUAV") >= 3, 45)
        # Verify *effective* ROS parameters, not just the YAML on disk.
        pipeline = yaml.safe_load(PIPELINE.read_text())
        expected_dude = yaml.safe_load(DUDE_CONFIG.read_text())["/**"]["ros__parameters"]
        expected_dude = {**expected_dude, "use_sim_time": False}
        map_keys = ("minimum_z", "minimum_occupancy", "partial_map_updates", "max_slope_ugv", "slope_estimation_size", "map_frame")
        expected_map = {k: pipeline["map_conversion_node"]["ros__parameters"][k] for k in map_keys}
        actual_parameters = {}
        parameter_nodes = [("incremental_decomposer", expected_dude)]
        if not vertical:
            parameter_nodes.append(("map_conversion_node", expected_map))
        for name, expected in parameter_nodes:
            client = node.create_client(GetParameters, f"/{name}/get_parameters")
            if not client.wait_for_service(timeout_sec=15):
                raise RuntimeError(f"No parameter services for {name}")
            request = GetParameters.Request()
            request.names = list(expected)
            future = client.call_async(request)
            wait_for(future.done, 30)
            actual = {k: parameter_value_to_python(v) for k, v in zip(expected, future.result().values)}
            assert actual == expected, (name, actual, expected)
            actual_parameters[name] = actual
        graph_keys = ("materialize_observed_regions", "nav_region_boundary_epsilon_m", "nav_region_enable_neighbor_tiebreak", "fs_cell_stride_cells", "fs_min_free_cell_count", "fs_navigation_connectivity", "pose_distance_threshold", "pose_time_threshold", "pose_window_size")
        actual_parameters["scene_graph_region"] = {k: graph.get_parameter(k).value for k in graph_keys}
        assert actual_parameters["scene_graph_region"] == {k: pipeline["scene_graph_region"]["ros__parameters"][k] for k in graph_keys}
        save_json(output / "effective_parameters.json", actual_parameters)
        for record in episode["snapshots"]:
            start = time.monotonic()
            event = record["coverage_event_id"]
            target = output / event
            target.mkdir(exist_ok=True)
            save_json(output / "status.json", {"status": "running", "event": event, "time": time.time()})
            if vertical:
                reference = INPUT_REFERENCE_ROOT / episode["episode_uid"] / (event + ".npz")
                input_grid, geometry = encode_vertical_snapshot(Path(record["snapshot_path"]), reference)
                if INPUT_MODE == "raw_vertical_free_cropped":
                    input_grid, geometry = crop_unknown_border(input_grid, geometry)
            else:
                geometry = encode_snapshot(Path(record["snapshot_path"]), output)
            old_maps, old_regions = len(maps), len(regions)
            msg = OccupancyGrid() if vertical else Octomap()
            msg.header.frame_id = "map"
            msg.header.stamp = node.get_clock().now().to_msg()
            if vertical:
                msg.info.resolution = geometry["resolution"]
                msg.info.height, msg.info.width = input_grid.shape
                msg.info.origin.position.x, msg.info.origin.position.y = geometry["origin"]
                msg.info.origin.orientation.w = 1.0
                msg.data = array("b", input_grid.ravel().tobytes())
            else:
                msg.binary = False
                msg.id = "OcTree"
                msg.resolution = .05
                msg.data = array("b", (output / "octomap.data").read_bytes())
            publisher.publish(msg)
            wait_for(lambda: len(maps) > old_maps and len(regions) > old_regions)
            grid, tracked = maps[-1], regions[-1]
            if grid.header.stamp != tracked.header.stamp:
                raise AssertionError("Map/regions stamps mismatch: refusing stale checkpoint association")
            if vertical:
                assert grid.header.stamp == msg.header.stamp
                assert_received_vertical_grid(grid, input_grid, geometry)
            # Run official room-grounding callbacks, not a reimplementation.
            odom = Odometry()
            odom.header = grid.header
            odom.child_frame_id = "base_link"
            px, py, pz, yaw = geometry["pose"]
            odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = px, py, pz
            odom.pose.pose.orientation.z = float(np.sin(yaw/2))
            odom.pose.pose.orientation.w = float(np.cos(yaw/2))
            graph._odom_callback(odom)
            graph._pose_flush_tick()
            graph._map_callback(grid)
            graph._stable_regions_callback(tracked)
            graph._maintenance_tick()
            graph._pipeline_tick()
            room_nodes = sorted(graph.sg.query.find_nodes_by_type(NodeType.ROOM), key=lambda room: room.id)
            room_data = [{"id": int(room.id), "attributes": room.attributes} for room in room_nodes]
            polys = [[(float(p["x"]), float(p["y"])) for p in room.attributes.get("polygon", [])] for room in room_nodes]
            raw = sorted(tracked.regions, key=lambda r: r.id)
            region_data = [{"id": int(r.id), "area": float(r.area), "polygon": [[p.x, p.y] for p in r.polygon.points], "adjacent_ids": list(r.adjacent_ids)} for r in raw]
            raster_shape = geometry.get("source_shape", geometry["shape"])
            raster_origin = geometry.get("source_origin", geometry["origin"])
            room_labels = rasterize(polys, raster_shape, raster_origin, .05)
            region_labels = rasterize([r["polygon"] for r in region_data], raster_shape, raster_origin, .05)
            np.savez_compressed(target / "prediction.npz", room_label_map=room_labels, tracked_region_label_map=region_labels,
                                map_uav=np.asarray(grid.data, dtype=np.int8).reshape(grid.info.height, grid.info.width),
                                map_uav_origin_xy=[grid.info.origin.position.x, grid.info.origin.position.y], map_uav_resolution=grid.info.resolution)
            save_json(target / "metadata.json", {"source": record, "geometry": geometry, "rooms": room_data,
                                                   "tracked_regions": region_data, "seconds": time.monotonic()-start})
            print(json.dumps({"scene": episode["scene_id"], "event": event, "rooms": len(room_nodes), "regions": len(raw), "seconds": time.monotonic()-start}), flush=True)
        save_json(output / "complete.json", {"status": "complete", "checkpoints": len(episode["snapshots"]), "time": time.time()})
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()
        graph.destroy_node()
        node.destroy_node()
        rclpy.shutdown()


def evaluate(dataset, output, reuse_completed=False):
    from voxroom_online.isaac_runtime.evaluation.online_roomseg.snapshot_io import load_snapshot_arrays
    from voxroom_online.isaac_runtime.evaluation.online_roomseg.step_backprojection import backproject_final_gt_to_snapshot
    from voxroom_online.isaac_runtime.evaluation.online_roomseg.metrics import prepare_metric_label_maps, compute_snapshot_metrics
    episodes = json.loads((output / "inputs/index.json").read_text())["episodes"]
    metrics_path = output / "per_checkpoint_metrics.csv"
    rows = []
    if reuse_completed and metrics_path.exists():
        with metrics_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
    cached = {(r["episode_uid"], r["coverage_event_id"]) for r in rows}
    assert len(cached) == len(rows), "duplicate cached checkpoint metrics"
    for episode in episodes:
        uid = episode["episode_uid"]
        complete = output / "replay" / uid / "complete.json"
        if not complete.exists():
            continue
        gt = np.load(DATASETS[dataset] / "final_gt" / uid / "last_step.gt_labels.npy")
        for record in episode["snapshots"]:
            if (uid, record["coverage_event_id"]) in cached:
                continue
            source = load_snapshot_arrays(Path(record["snapshot_path"]))
            step_gt = backproject_final_gt_to_snapshot(gt, source, episode_uid=uid, source_final_step=int(episode["last_snapshot_step"])).label_map
            pred_path = output / "replay" / uid / record["coverage_event_id"] / "prediction.npz"
            with np.load(pred_path) as pred:
                labels = pred["room_label_map"]
            prepared = prepare_metric_label_maps(step_gt, labels, min_room_area_cells=200)
            metric = compute_snapshot_metrics(prepared.gt, prepared.pred)
            p, r = float(metric["precision"]), float(metric["recall"])
            rows.append({"dataset": dataset, "method": METHOD, "scene_id": episode["scene_id"], "episode_uid": uid,
                         "coverage_event_id": record["coverage_event_id"], "step": record["step"], "precision": p, "recall": r,
                         "f1": 2*p*r/(p+r) if p+r else 0., "miou_room": float(metric["miou_room"]),
                         "n_gt": int(metric["n_gt"]), "n_pred": int(metric["n_pred"]),
                         "metric_domain_pixels": int(np.count_nonzero(prepared.metric_domain)),
                         "source_snapshot_path": record["snapshot_path"], "prediction_path": str(pred_path)})
    if rows:
        temporary = metrics_path.with_suffix(".csv.tmp")
        with temporary.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(metrics_path)
    return rows


def main():
    global INPUT_MODE, INPUT_REFERENCE_ROOT, METHOD
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "scene", "batch", "evaluate"])
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--input-mode", choices=["saved_voxels", *VERTICAL_MODES], default="saved_voxels")
    parser.add_argument("--input-reference-root", type=Path)
    args = parser.parse_args()
    INPUT_MODE, INPUT_REFERENCE_ROOT = args.input_mode, args.input_reference_root
    if INPUT_MODE in VERTICAL_MODES:
        if INPUT_REFERENCE_ROOT is None or not INPUT_REFERENCE_ROOT.is_dir():
            parser.error("raw_vertical_free requires an existing --input-reference-root")
        METHOD = "occusg_official_" + INPUT_MODE
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare":
        prepare(args.dataset, args.output)
    elif args.command == "evaluate":
        evaluate(args.dataset, args.output)
    elif args.command == "scene":
        episodes = json.loads((args.output / "inputs/index.json").read_text())["episodes"]
        episode = episodes[args.scene_index]
        if args.smoke:
            episode = {**episode, "snapshots": episode["snapshots"][:2]}
        run_scene(episode, args.output / "replay" / episode["episode_uid"])
    else:
        episodes = prepare(args.dataset, args.output)
        try:
            for i, episode in enumerate(episodes):
                if (args.output / "replay" / episode["episode_uid"] / "complete.json").exists():
                    continue
                save_json(args.output / "status.json", {"status": "running", "scene_index": i, "scene": episode["scene_id"], "total_scenes": len(episodes), "time": time.time()})
                child = [sys.executable, "-u", __file__, "scene", "--dataset", args.dataset, "--output", str(args.output), "--scene-index", str(i), "--input-mode", INPUT_MODE]
                if INPUT_REFERENCE_ROOT is not None:
                    child.extend(["--input-reference-root", str(INPUT_REFERENCE_ROOT)])
                subprocess.run(child, check=True)
                evaluate(args.dataset, args.output, reuse_completed=True)
            rows = evaluate(args.dataset, args.output, reuse_completed=True)
            assert len(rows) == sum(len(e["snapshots"]) for e in episodes)
            save_json(args.output / "status.json", {"status": "complete", "checkpoints": len(rows), "time": time.time()})
        except Exception as exc:
            save_json(args.output / "status.json", {"status": "failed", "error": repr(exc), "time": time.time()})
            raise


if __name__ == "__main__":
    main()
