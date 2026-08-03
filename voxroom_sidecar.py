import json
import os
from pathlib import Path
import select
import subprocess
import time

import numpy as np
from PIL import Image


REQUIRED_INFO_KEYS = (
    "voxroom_depth_m",
    "voxroom_intrinsics_fx_fy_cx_cy",
    "voxroom_intrinsics_width_height",
    "voxroom_base_pose_world_xyzyaw",
    "voxroom_camera_transform_world",
    "voxroom_geometry_contract",
)


def load_navigation_projection(path, expected_step, expected_shape):
    projection_path = Path(path).resolve()
    if not projection_path.is_file():
        raise FileNotFoundError(
            "VoxRoom navigation projection is missing: {}".format(projection_path)
        )
    if expected_shape is not None:
        expected_shape = tuple(int(value) for value in expected_shape)
        if len(expected_shape) != 2 or min(expected_shape) < 1:
            raise ValueError(
                "Active Room navigation shape is invalid: {}".format(expected_shape)
            )
    with np.load(projection_path, allow_pickle=False) as payload:
        required = {
            "format_version",
            "step",
            "shape_hw",
            "free_bits",
            "occupied_bits",
            "observed_bits",
            "unknown_bits",
            "resolution_m",
            "bounds_xyxy_m",
            "source",
        }
        missing = sorted(required.difference(payload.files))
        if missing:
            raise KeyError(
                "VoxRoom navigation projection is missing keys: {}".format(missing)
            )
        format_version = int(np.asarray(payload["format_version"]).reshape(()))
        step = int(np.asarray(payload["step"]).reshape(()))
        shape = tuple(
            int(value)
            for value in np.asarray(payload["shape_hw"], dtype=np.int64).reshape(-1)
        )
        resolution_m = float(np.asarray(payload["resolution_m"]).reshape(()))
        bounds = np.asarray(payload["bounds_xyxy_m"], dtype=np.float64).reshape(-1)
        source = str(np.asarray(payload["source"]).reshape(()))
        if format_version != 1:
            raise RuntimeError(
                "Unsupported VoxRoom navigation projection version {}".format(
                    format_version
                )
            )
        if step != int(expected_step):
            raise RuntimeError(
                "VoxRoom navigation projection step {} differs from expected {}".format(
                    step,
                    int(expected_step),
                )
            )
        if expected_shape is not None and shape != expected_shape:
            raise RuntimeError(
                "VoxRoom navigation shape {} differs from Active Room {}".format(
                    shape,
                    expected_shape,
                )
            )
        if not np.isfinite(resolution_m) or resolution_m <= 0.0:
            raise RuntimeError("VoxRoom navigation resolution is invalid")
        if bounds.size != 4 or not np.all(np.isfinite(bounds)):
            raise RuntimeError("VoxRoom navigation bounds are invalid")
        expected_bounds = np.asarray(
            [
                -0.5 * shape[1] * resolution_m,
                -0.5 * shape[0] * resolution_m,
                0.5 * shape[1] * resolution_m,
                0.5 * shape[0] * resolution_m,
            ],
            dtype=np.float64,
        )
        if not np.allclose(bounds, expected_bounds, atol=1.0e-6, rtol=0.0):
            raise RuntimeError(
                "VoxRoom navigation bounds {} differ from the centered Active Room map {}".format(
                    bounds.tolist(),
                    expected_bounds.tolist(),
                )
            )
        if source != "mapper.last_voxel_navigation_projection":
            raise RuntimeError(
                "Unexpected VoxRoom navigation source: {}".format(source)
            )
        cell_count = int(shape[0] * shape[1])
        masks = {}
        for name in ("free", "occupied", "observed", "unknown"):
            bits = np.asarray(payload[name + "_bits"], dtype=np.uint8).reshape(-1)
            mask = np.unpackbits(bits, bitorder="little", count=cell_count)
            masks[name] = mask.astype(bool, copy=False).reshape(shape)

    if np.any(masks["free"] & masks["occupied"]):
        raise RuntimeError("VoxRoom navigation marks cells both free and occupied")
    if np.any(masks["unknown"] & (masks["free"] | masks["occupied"])):
        raise RuntimeError("VoxRoom navigation unknown cells overlap known cells")
    if np.any((masks["free"] | masks["occupied"]) & ~masks["observed"]):
        raise RuntimeError("VoxRoom navigation has known cells outside observed")
    if not np.array_equal(masks["unknown"], ~masks["observed"]):
        raise RuntimeError("VoxRoom navigation unknown mask differs from inverse observed")
    return {
        "step": step,
        "resolution_m": resolution_m,
        "bounds_xyxy_m": bounds,
        "free": np.flipud(masks["free"]).copy(),
        "occupied": np.flipud(masks["occupied"]).copy(),
        "observed": np.flipud(masks["observed"]).copy(),
        "unknown": np.flipud(masks["unknown"]).copy(),
        "source": source,
    }


def attach_voxroom_projection(info, navigation):
    if "gt_map" not in info or "gt_exp" not in info:
        raise RuntimeError(
            "Active Room native obstacle and explored maps are required"
        )
    expected_shape = np.asarray(info["gt_map"]).shape
    if np.asarray(info["gt_exp"]).shape != expected_shape:
        raise RuntimeError("Active Room obstacle and explored maps have different shapes")
    for name in ("free", "occupied", "observed", "unknown"):
        if np.asarray(navigation[name]).shape != expected_shape:
            raise RuntimeError(
                "VoxRoom {} map has shape {}, expected {}".format(
                    name,
                    np.asarray(navigation[name]).shape,
                    expected_shape,
                )
            )
    info["voxroom_navigation_free"] = np.asarray(
        navigation["free"],
        dtype=np.uint8,
    )
    info["voxroom_navigation_occupied"] = np.asarray(
        navigation["occupied"],
        dtype=np.uint8,
    )
    info["voxroom_navigation_observed"] = np.asarray(
        navigation["observed"],
        dtype=np.uint8,
    )
    info["voxroom_navigation_unknown"] = np.asarray(
        navigation["unknown"],
        dtype=np.uint8,
    )
    base_pose = np.asarray(
        info["voxroom_base_pose_world_xyzyaw"],
        dtype=np.float64,
    ).reshape(-1)
    if base_pose.size != 4 or not np.all(np.isfinite(base_pose)):
        raise RuntimeError("VoxRoom base pose is invalid")
    bounds = np.asarray(
        navigation["bounds_xyxy_m"],
        dtype=np.float64,
    ).reshape(4)
    resolution_m = float(navigation["resolution_m"])
    agent_cell = np.asarray(
        [
            int(np.floor((base_pose[1] - bounds[1]) / resolution_m)),
            int(np.floor((base_pose[0] - bounds[0]) / resolution_m)),
        ],
        dtype=np.int32,
    )
    if not (
        0 <= int(agent_cell[0]) < int(expected_shape[0])
        and 0 <= int(agent_cell[1]) < int(expected_shape[1])
    ):
        raise RuntimeError(
            "VoxRoom agent cell {} is outside navigation shape {}".format(
                agent_cell.tolist(),
                expected_shape,
            )
        )
    info["voxroom_navigation_agent_cell"] = agent_cell
    info["voxroom_navigation_map_source"] = (
        "voxroom_last_voxel_navigation_projection"
    )
    info["voxroom_navigation_map_step"] = int(navigation["step"])


class VoxRoomSidecarClient:
    def __init__(
        self,
        *,
        voxroom_root,
        config_path,
        run_dir,
        map_size_m,
        map_resolution_m,
        roomseg_every_steps,
        visualization_every_steps,
        response_timeout_seconds,
        scene_id="",
        episode_id="",
        coverage_eval=False,
        coverage_milestones="20,40,60,80,100",
    ):
        self.voxroom_root = Path(voxroom_root).expanduser().resolve()
        self.config_path = Path(config_path).expanduser().resolve()
        self.run_dir = Path(run_dir).resolve()
        self.output_dir = self.run_dir / "voxroom"
        self.frame_dir = self.run_dir / "voxroom_bridge_frames"
        self.stderr_path = self.run_dir / "voxroom_worker.log"
        self.response_timeout_seconds = float(response_timeout_seconds)
        self.coverage_eval = bool(coverage_eval)
        self.coverage_milestones = str(coverage_milestones)
        self.map_resolution_m = float(map_resolution_m)
        if not np.isfinite(self.map_resolution_m) or self.map_resolution_m <= 0.0:
            raise ValueError("Active Room map resolution must be positive")
        self.last_step = -1
        self.last_simulator_step = -1
        self.latest_response = None
        self.latest_navigation = None
        self.final_result = None
        self.closed = False

        launch_script = self.voxroom_root / "scripts" / "run_voxroom_isaac_env.sh"
        worker_script = (
            self.voxroom_root
            / "voxroom_online"
            / "isaac_runtime"
            / "integration"
            / "habitat_voxroom_worker.py"
        )
        if not launch_script.is_file():
            raise FileNotFoundError("VoxRoom launch script is missing: {}".format(launch_script))
        if not worker_script.is_file():
            raise FileNotFoundError("VoxRoom Habitat worker is missing: {}".format(worker_script))
        if not self.config_path.is_file():
            raise FileNotFoundError("VoxRoom config is missing: {}".format(self.config_path))
        if self.output_dir.exists():
            raise FileExistsError("VoxRoom output directory already exists: {}".format(self.output_dir))
        self.frame_dir.mkdir(parents=True, exist_ok=False)
        self.stderr_stream = self.stderr_path.open("w", encoding="utf-8")
        command = [
            str(launch_script),
            str(worker_script),
            "--repo-root",
            str(self.voxroom_root),
            "--config",
            str(self.config_path),
            "--output-dir",
            str(self.output_dir),
            "--map-size-m",
            str(float(map_size_m)),
            "--roomseg-every-steps",
            str(int(roomseg_every_steps)),
            "--visualization-every-steps",
            str(int(visualization_every_steps)),
            "--scene-id",
            str(scene_id),
            "--episode-id",
            str(episode_id),
        ]
        if self.coverage_eval:
            command.extend(
                [
                    "--coverage-eval",
                    "--coverage-milestones",
                    self.coverage_milestones,
                ]
            )
        self.process = subprocess.Popen(
            command,
            cwd=str(self.voxroom_root),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_stream,
            text=True,
            bufsize=1,
        )
        ready = self._read_response()
        if ready != {"status": "ready"}:
            raise RuntimeError("VoxRoom worker returned an invalid ready message: {}".format(ready))

    def update(self, step, simulator_step, info, detected_doors):
        if self.closed:
            raise RuntimeError("VoxRoom sidecar is already closed")
        step = int(step)
        simulator_step = int(simulator_step)
        if simulator_step == self.last_simulator_step:
            if self.latest_navigation is None:
                raise RuntimeError("VoxRoom sidecar lost its navigation projection")
            attach_voxroom_projection(info, self.latest_navigation)
            return self.latest_response
        if simulator_step != self.last_simulator_step + 1:
            raise RuntimeError(
                "Habitat simulator step discontinuity: expected {}, received {}".format(
                    self.last_simulator_step + 1,
                    simulator_step,
                )
            )
        missing = [key for key in REQUIRED_INFO_KEYS if key not in info]
        if missing:
            raise KeyError("Habitat info is missing VoxRoom fields: {}".format(missing))
        if info["voxroom_geometry_contract"] != (
            "habitat_world_axis_sensor_se3_to_voxroom_flu_v2"
        ):
            raise RuntimeError(
                "Unexpected Habitat-to-VoxRoom geometry contract: {}".format(
                    info["voxroom_geometry_contract"]
                )
            )
        if step != simulator_step:
            raise RuntimeError(
                "Active Room action step {} differs from Habitat simulator step {}".format(
                    step,
                    simulator_step,
                )
            )
        coverage_arrays = {}
        if self.coverage_eval:
            required_coverage = (
                "gt_map",
                "gt_exp",
                "gt_explorable",
                "gt_explorable_reference_source",
                "gt_explorable_floor_height_m",
                "gt_explorable_floor_slice_eps_m",
                "gt_explorable_floor_island_index",
                "gt_explorable_source_cells",
                "gt_explorable_transformed_cells",
            )
            missing_coverage = [
                key for key in required_coverage if key not in info
            ]
            if missing_coverage:
                raise KeyError(
                    "Habitat info is missing coverage fields: {}".format(
                        missing_coverage
                    )
                )
            occupied = np.asarray(info["gt_map"], dtype=bool)
            explored = np.asarray(info["gt_exp"], dtype=bool)
            explorable = np.asarray(info["gt_explorable"], dtype=bool)
            if occupied.shape != explored.shape or occupied.shape != explorable.shape:
                raise RuntimeError("Habitat coverage maps do not share one shape")
            reference_source = str(info["gt_explorable_reference_source"])
            if reference_source != "habitat_pathfinder_start_floor_island_topdown":
                raise RuntimeError(
                    "Unexpected Habitat floor reference source: {}".format(
                        reference_source
                    )
                )
            transformed_cells = int(info["gt_explorable_transformed_cells"])
            if transformed_cells != int(np.count_nonzero(explorable)):
                raise RuntimeError(
                    "Habitat floor reference cell count differs from its metadata"
                )
            door_segments = _door_segments_rc(detected_doors)
            coverage_arrays = {
                "habitat_map_shape_hw": np.asarray(occupied.shape, dtype=np.int32),
                "habitat_explorable_bits": np.packbits(
                    explorable.reshape(-1), bitorder="little"
                ),
                "habitat_explored_bits": np.packbits(
                    explored.reshape(-1), bitorder="little"
                ),
                "habitat_occupied_bits": np.packbits(
                    occupied.reshape(-1), bitorder="little"
                ),
                "habitat_map_resolution_m": np.asarray(
                    self.map_resolution_m, dtype=np.float64
                ),
                "habitat_floor_reference_source": np.asarray(reference_source),
                "habitat_floor_height_m": np.asarray(
                    info["gt_explorable_floor_height_m"], dtype=np.float64
                ),
                "habitat_floor_slice_eps_m": np.asarray(
                    info["gt_explorable_floor_slice_eps_m"], dtype=np.float64
                ),
                "habitat_floor_island_index": np.asarray(
                    info["gt_explorable_floor_island_index"], dtype=np.int32
                ),
                "habitat_floor_source_cells": np.asarray(
                    info["gt_explorable_source_cells"], dtype=np.int64
                ),
                "habitat_floor_transformed_cells": np.asarray(
                    transformed_cells, dtype=np.int64
                ),
                "tvars_door_segments_rc": np.asarray(
                    door_segments, dtype=np.int32
                ).reshape(-1, 4),
            }
        frame_path = self.frame_dir / "frame_{:06d}.npz".format(simulator_step)
        with frame_path.open("wb") as stream:
            np.savez(
                stream,
                step=np.asarray(simulator_step, dtype=np.int64),
                depth_m=np.asarray(info["voxroom_depth_m"], dtype=np.float32),
                intrinsics_fx_fy_cx_cy=np.asarray(
                    info["voxroom_intrinsics_fx_fy_cx_cy"],
                    dtype=np.float64,
                ),
                intrinsics_width_height=np.asarray(
                    info["voxroom_intrinsics_width_height"],
                    dtype=np.int32,
                ),
                base_pose_world_xyzyaw=np.asarray(
                    info["voxroom_base_pose_world_xyzyaw"],
                    dtype=np.float64,
                ),
                camera_transform_world=np.asarray(
                    info["voxroom_camera_transform_world"],
                    dtype=np.float64,
                ),
                **coverage_arrays
            )
            stream.flush()
            os.fsync(stream.fileno())
        self._send({"op": "update", "frame_path": str(frame_path)})
        response = self._read_response()
        if response.get("status") != "updated":
            raise RuntimeError("VoxRoom worker update failed: {}".format(response))
        if int(response.get("step", -1)) != simulator_step:
            raise RuntimeError("VoxRoom worker acknowledged the wrong step: {}".format(response))
        navigation = load_navigation_projection(
            response.get("navigation_projection_path", ""),
            expected_step=simulator_step,
            expected_shape=(
                np.asarray(info["gt_map"]).shape
                if "gt_map" in info
                else None
            ),
        )
        attach_voxroom_projection(info, navigation)
        frame_path.unlink()
        self.last_step = step
        self.last_simulator_step = simulator_step
        self.latest_response = response
        self.latest_navigation = navigation
        return response

    def latest_visualization(self):
        if self.latest_response is None:
            return None
        path = Path(self.latest_response["visualization_path"])
        if not path.is_file():
            raise FileNotFoundError("VoxRoom visualization is missing: {}".format(path))
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB")).copy()

    def close(self, detected_doors):
        if self.closed:
            return self.final_result
        request = {"op": "close"}
        if self.coverage_eval:
            request["tvars_door_segments_rc"] = _door_segments_rc(
                detected_doors
            )
        self._send(request)
        result = self._read_response()
        if result.get("status") != "completed":
            raise RuntimeError("VoxRoom worker did not complete: {}".format(result))
        if int(result.get("last_step", -1)) != self.last_simulator_step:
            raise RuntimeError("VoxRoom worker terminal step does not match Habitat")
        if int(result.get("processed_frames", -1)) != self.last_simulator_step + 1:
            raise RuntimeError("VoxRoom worker frame count does not match Habitat")
        self.process.stdin.close()
        return_code = self.process.wait(timeout=self.response_timeout_seconds)
        if return_code != 0:
            raise RuntimeError("VoxRoom worker exited with code {}".format(return_code))
        self.stderr_stream.close()
        self.closed = True
        self.final_result = result
        self.latest_response = {
            **(self.latest_response or {}),
            "visualization_path": result["visualization_path"],
        }
        return result

    def _send(self, payload):
        if self.process.poll() is not None:
            raise RuntimeError(
                "VoxRoom worker exited before request with code {}".format(
                    self.process.returncode
                )
            )
        self.process.stdin.write(json.dumps(payload, sort_keys=True) + "\n")
        self.process.stdin.flush()

    def _read_response(self):
        ready, _, _ = select.select(
            [self.process.stdout],
            [],
            [],
            self.response_timeout_seconds,
        )
        if not ready:
            raise TimeoutError(
                "Timed out after {:.1f}s waiting for VoxRoom worker".format(
                    self.response_timeout_seconds
                )
            )
        line = self.process.stdout.readline()
        if not line:
            return_code = self.process.poll()
            raise RuntimeError(
                "VoxRoom worker closed stdout unexpectedly; return code {}".format(
                    return_code
                )
            )
        response = json.loads(line)
        if response.get("status") == "error":
            raise RuntimeError(
                "VoxRoom worker {}: {}".format(
                    response.get("error_type", "error"),
                    response.get("error", ""),
                )
            )
        return response


def _door_segments_rc(detected_doors):
    if detected_doors is None:
        raise TypeError("TVARS accepted door collection is required")
    segments = []
    for door in list(detected_doors):
        start = np.asarray(door["start"], dtype=np.int32).reshape(-1)
        end = np.asarray(door["end"], dtype=np.int32).reshape(-1)
        if start.size != 2 or end.size != 2:
            raise ValueError("TVARS accepted door endpoints must be 2D")
        segments.append(
            [int(start[1]), int(start[0]), int(end[1]), int(end[0])]
        )
    return segments
