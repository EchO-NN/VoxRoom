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
        frame_path.unlink()
        self.last_step = step
        self.last_simulator_step = simulator_step
        self.latest_response = response
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
