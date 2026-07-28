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


def apply_navigation_projection(info, navigation):
    has_obstacle_map = "gt_map" in info
    has_explored_map = "gt_exp" in info
    if has_obstacle_map != has_explored_map:
        raise RuntimeError(
            "Active Room info contains only one half of its navigation map"
        )
    expected_shape = (
        np.asarray(info["gt_map"]).shape
        if has_obstacle_map
        else np.asarray(navigation["free"]).shape
    )
    if has_explored_map and np.asarray(info["gt_exp"]).shape != expected_shape:
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
    outside = (
        navigation["observed"]
        & ~navigation["free"]
        & ~navigation["occupied"]
    )
    info["gt_map"] = np.asarray(
        navigation["occupied"] | outside,
        dtype=np.float32,
    )
    info["gt_exp"] = np.asarray(navigation["observed"], dtype=np.float32)
    info["voxroom_navigation_free"] = np.asarray(
        navigation["free"],
        dtype=np.uint8,
    )
    info["voxroom_navigation_unknown"] = np.asarray(
        navigation["unknown"],
        dtype=np.uint8,
    )
    info["navigation_map_source"] = "voxroom_last_voxel_navigation_projection"
    info["navigation_map_step"] = int(navigation["step"])


class VoxRoomSidecarClient:
    def __init__(
        self,
        *,
        voxroom_root,
        config_path,
        run_dir,
        map_size_m,
        roomseg_every_steps,
        visualization_every_steps,
        response_timeout_seconds,
    ):
        self.voxroom_root = Path(voxroom_root).expanduser().resolve()
        self.config_path = Path(config_path).expanduser().resolve()
        self.run_dir = Path(run_dir).resolve()
        self.output_dir = self.run_dir / "voxroom"
        self.frame_dir = self.run_dir / "voxroom_bridge_frames"
        self.stderr_path = self.run_dir / "voxroom_worker.log"
        self.response_timeout_seconds = float(response_timeout_seconds)
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
        ]
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

    def update(self, step, simulator_step, info):
        if self.closed:
            raise RuntimeError("VoxRoom sidecar is already closed")
        step = int(step)
        simulator_step = int(simulator_step)
        if simulator_step == self.last_simulator_step:
            if self.latest_navigation is None:
                raise RuntimeError("VoxRoom sidecar lost its navigation projection")
            apply_navigation_projection(info, self.latest_navigation)
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
        if step != simulator_step:
            raise RuntimeError(
                "Active Room action step {} differs from Habitat simulator step {}".format(
                    step,
                    simulator_step,
                )
            )
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
        apply_navigation_projection(info, navigation)
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

    def close(self):
        if self.closed:
            return self.final_result
        self._send({"op": "close"})
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
