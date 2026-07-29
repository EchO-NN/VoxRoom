import time
from collections import deque
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import uuid
import cv2


os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import env.habitat.utils.pose as pu
import env.habitat.utils.visualizations as vu
import logging
from arguments import get_args
from env import make_vec_envs
from utils.storage import GlobalRolloutStorage, FIFOMemory
from utils.optimization import get_optimizer
from model import RL_Policy, Local_IL_Policy, Neural_SLAM_Module
from door_detection import Door_detection
import algo
from frontier_detection import Frontier_detection
import sys
import matplotlib
from matplotlib import pyplot as plt
from action_generation import action_generator
from topomap_construction import Topomap_construction
# from hough_door_detection import hough_detection
from env.habitat.hough_door_detection import convert_2_laser
from detr_door_detection.run_detr import run_detr
from time import perf_counter, time
from visualization import RuntimeDashboard, TopologyEventRecorder
from voxroom_sidecar import VoxRoomSidecarClient
from run_context_contract import (
    STRICT_LIVE_FIGURE_DPI,
    STRICT_LIVE_FIGURE_SIZE_INCHES,
    STRICT_VISUAL_CAPTURE_STEPS,
    episode_contract_sha256,
)
from topology_contract import surviving_crossing_count

def get_local_map_boundaries(agent_loc, local_sizes, full_sizes):
    loc_r, loc_c = agent_loc
    local_w, local_h = local_sizes
    full_w, full_h = full_sizes

    if args.global_downscaling > 1:
        gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
        gx2, gy2 = gx1 + local_w, gy1 + local_h
        if gx1 < 0:
            gx1, gx2 = 0, local_w
        if gx2 > full_w:
            gx1, gx2 = full_w - local_w, full_w

        if gy1 < 0:
            gy1, gy2 = 0, local_h
        if gy2 > full_h:
            gy1, gy2 = full_h - local_h, full_h
    else:
        gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h
    #print('new boundary {}'.format([gx1, gx2, gy1, gy2]))
    return [gx1, gx2, gy1, gy2]


def as_numpy(value):
    if not torch.is_tensor(value):
        raise TypeError("expected the Tensor contract emitted by VecPyTorch")
    return value.detach().cpu().numpy()


def write_json_atomic(path, payload):
    path = Path(path)
    temporary_path = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    with temporary_path.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary_path, path)


def bind_x11_client_window(figure, window_title):
    manager = figure.canvas.manager
    window = getattr(manager, "window", None)
    if window is None or not hasattr(window, "winfo_id"):
        raise RuntimeError("The live dashboard is not backed by a Tk X11 window")
    manager.set_window_title(window_title)
    figure.canvas.draw()
    manager.show()
    window.update_idletasks()
    locked_width = int(window.winfo_width())
    locked_height = int(window.winfo_height())
    if locked_width <= 1 or locked_height <= 1:
        raise RuntimeError("Tk did not realize the live dashboard dimensions")
    window.geometry("{}x{}".format(locked_width, locked_height))
    window.minsize(locked_width, locked_height)
    window.maxsize(locked_width, locked_height)
    window.resizable(False, False)
    window.update()
    if (
        int(window.winfo_width()) != locked_width
        or int(window.winfo_height()) != locked_height
    ):
        raise RuntimeError("Tk did not preserve the locked dashboard dimensions")
    inner_window_id = int(window.winfo_id())
    if inner_window_id <= 0:
        raise RuntimeError("Tk did not expose a valid X11 client window ID")
    inner_window_id_hex = "0x{:x}".format(inner_window_id)
    window_tree = subprocess.check_output(
        ["xwininfo", "-id", inner_window_id_hex, "-tree"],
        text=True,
        timeout=10,
    )
    parent_match = re.search(
        r"Parent window id: (0x[0-9a-fA-F]+)",
        window_tree,
    )
    if parent_match is None:
        raise RuntimeError("Unable to resolve the Tk X11 client window")
    window_id_hex = parent_match.group(1).lower()
    subprocess.run(
        [
            "xprop",
            "-id",
            window_id_hex,
            "-f",
            "_NET_WM_PID",
            "32c",
            "-set",
            "_NET_WM_PID",
            str(os.getpid()),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=10,
    )
    pid_property = subprocess.check_output(
        ["xprop", "-id", window_id_hex, "_NET_WM_PID"],
        text=True,
        timeout=10,
    )
    try:
        observed_pid = int(pid_property.rsplit(" = ", 1)[1].strip())
    except (IndexError, ValueError) as error:
        raise RuntimeError("Unable to parse the X11 client PID property") from error
    if observed_pid != os.getpid():
        raise RuntimeError("The live X11 client window is not owned by this process")
    window_properties = subprocess.check_output(
        ["xprop", "-id", window_id_hex, "WM_NAME"],
        text=True,
        timeout=10,
    )
    if window_properties.strip() != 'WM_NAME(STRING) = "{}"'.format(window_title):
        raise RuntimeError("The live X11 client window has the wrong title")
    window_info = subprocess.check_output(
        ["xwininfo", "-id", window_id_hex],
        text=True,
        timeout=10,
    )
    if "Map State: IsViewable" not in window_info:
        raise RuntimeError("The live X11 client window is not viewable")
    if str(window.wm_title()) != window_title:
        raise RuntimeError("The live X11 client window title changed during binding")
    return window_id_hex


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_run_context(run_dir):
    manifest_argument = bool(args.run_context_manifest)
    dataset_argument = bool(args.run_context_dataset)
    if manifest_argument != dataset_argument:
        raise RuntimeError(
            "Run context manifest and dataset must be provided together"
        )
    if not manifest_argument:
        return None

    manifest_path = Path(args.run_context_manifest)
    dataset_path = Path(args.run_context_dataset)
    if (
        manifest_path.parent != run_dir
        or dataset_path.parent != run_dir
        or manifest_path.name != "input_manifest.json"
        or dataset_path.name != "input_dataset.json.gz"
    ):
        raise RuntimeError("Run context files are not scoped to the run directory")
    if not manifest_path.is_file() or not dataset_path.is_file():
        raise FileNotFoundError("Run context files are missing")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1 or manifest.get("status") != "prepared":
        raise RuntimeError("Unsupported or incomplete run context manifest")
    dataset_digest = sha256(dataset_path)
    if dataset_digest != manifest.get("output_dataset_sha256"):
        raise RuntimeError("Run context dataset SHA256 mismatch")
    repository_root = Path(__file__).resolve().parent
    source_dataset = Path(manifest["output_dataset"]).resolve()
    prepared_root = (
        repository_root / "data" / "gibson-visual-runs"
    ).resolve()
    if (
        source_dataset.parent.parent != prepared_root
        or source_dataset.name != "input_dataset.json.gz"
    ):
        raise RuntimeError("Prepared dataset resolves outside its run scope")
    if (
        not source_dataset.is_file()
        or sha256(source_dataset) != dataset_digest
    ):
        raise RuntimeError("Prepared source dataset differs from the run copy")
    scene_path = Path(manifest["scene_path"]).resolve()
    navmesh_path = Path(manifest["navmesh_path"]).resolve()
    expected_scene_root = (
        repository_root / "data" / "scene_datasets" / "gibson"
    ).resolve()
    if (
        scene_path.parent != expected_scene_root
        or navmesh_path.parent != expected_scene_root
        or scene_path.stem != manifest.get("scene_id")
        or scene_path.suffix != ".glb"
        or navmesh_path != scene_path.with_suffix(".navmesh")
    ):
        raise RuntimeError("Run context assets resolve outside the Gibson scene root")
    if sha256(scene_path) != manifest.get("scene_sha256"):
        raise RuntimeError("Active Gibson scene differs from run context")
    if sha256(navmesh_path) != manifest.get("navmesh_sha256"):
        raise RuntimeError("Active Gibson navmesh differs from run context")

    with gzip.open(dataset_path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    episodes = payload.get("episodes", [])
    if len(episodes) != 1:
        raise RuntimeError("Run context must contain exactly one episode")
    episode = episodes[0]
    if str(episode.get("episode_id")) != manifest.get("selected_episode_id"):
        raise RuntimeError("Run context episode ID mismatch")
    if Path(episode.get("scene_id", "")).stem != manifest.get("scene_id"):
        raise RuntimeError("Run context scene ID mismatch")
    episode_digest = episode_contract_sha256(episode)
    if episode_digest != manifest.get("selected_episode_contract_sha256"):
        raise RuntimeError("Run context episode contract SHA256 mismatch")

    return {
        "manifest_file": manifest_path.name,
        "manifest_sha256": sha256(manifest_path),
        "dataset_file": dataset_path.name,
        "dataset_sha256": dataset_digest,
        "scene_id": manifest["scene_id"],
        "episode_id": manifest["selected_episode_id"],
        "episode_contract_sha256": episode_digest,
        "scene_sha256": manifest["scene_sha256"],
        "navmesh_sha256": manifest["navmesh_sha256"],
        "archive_sha256": manifest["archive_sha256"],
        "pointnav_tree_sha256": manifest["pointnav_tree_sha256"],
        "geodesic_distance": manifest["selected_episode_geodesic_distance"],
    }


explorable_threshold = 3.15
args = get_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

map_size = args.map_size_cm // args.map_resolution  # 480 pix
full_w, full_h = map_size, map_size
local_w, local_h = int(full_w / args.global_downscaling), \
                   int(full_h / args.global_downscaling)  # 240 pix

initial_device = torch.device("cuda:0" if args.cuda else "cpu")
full_pose = torch.zeros(1, 3, device=initial_device).float()
full_pose[:, :2] = args.map_size_cm / 100.0 / 2.0  # 12, 12, 0
locs = full_pose.cpu().numpy()
r, c = locs[0, 1], locs[0, 0]
loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                int(c * 100.0 / args.map_resolution)]  # 240
[gx1, gx2, gy1, gy2] = get_local_map_boundaries((loc_r, loc_c),
                                                (local_w, local_h),
                                                (full_w, full_h))
origins = [gy1 * args.map_resolution / 100.0,
           gx1 * args.map_resolution / 100.0, 0.]
start_x_gt, start_y_gt, start_o_gt = [args.map_size_cm / 100.0 / 2.0,
                                      args.map_size_cm / 100.0 / 2.0, 0.]  # 12,12,0
gt_pos = [start_x_gt - gy1 * args.map_resolution / 100.0,
          start_y_gt - gx1 * args.map_resolution / 100.0,
          start_o_gt]
x, y, o = gt_pos
x, y = x * 100.0 / 5.0, map_size//2 - y * 100.0 / 5.0
door_location = []
detected_door_list = []
raw_detect_list= []
bot_last_loc = []  # [x,y] global and pix
take_name_flag = True
scene_name = None
# cantwell
"""prior_door_list_store = [{'start': [238, 255], 'end': [228, 244]}, {'start': [252, 296], 'end': [261, 306]},
                   {'start': [266, 307], 'end': [276, 298]}, {'start': [154, 180], 'end': [143, 190]},
                   {'start': [128, 178], 'end': [138, 188]}, {'start': [107, 156], 'end': [95, 146]},
                   {'start': [187, 224], 'end': [205, 216]}, {'start': [172, 236], 'end': [160, 252]}]"""
# {'start': [187, 224], 'end': [205, 216]}, {'start': [172, 236], 'end': [160, 252]}

# dryville
"""prior_door_list = [{'start': [202, 257], 'end': [215, 263]}, {'start': [183, 272], 'end': [196, 277]},
                   {'start': [167, 243], 'end': [180, 248]}, {'start': [159, 261], 'end': [172, 266]},
                   {'start': [221, 269], 'end': [215, 282]}]"""  #

# eastville
"""prior_door_list_store = [{'start': [182, 199], 'end': [194, 209]}, {'start': [184, 324], 'end': [193, 314]},
                   {'start': [292, 266], 'end': [276, 287]}, {'start': [272, 331], 'end': [263, 343]},
                   {'start': [249, 334], 'end': [259, 343]}]"""


# prior_door_list_store.clear()  # this will lead to neartest frontier strategy


action_count = 0
cov_ratio = 0
cov_area = 0
cov_ratio_list = []
cov_area_list = []
step_list = []
log_num = 0
last_goal = None
last_episode_action_count = 0
last_episode_cov_ratio = 0.0
last_episode_cov_area = 0.0
#frontier_failed = []

established_graph = None  # if not using, equals to None
route_map = np.zeros((960, 960))
stage1_door_map = np.zeros((960, 960))  # door detection stage1
stage2_door_map = np.zeros((960, 960))  # door detection stage2
stage3_door_map = np.zeros((960, 960))  # door detection stage3


def main():
    #log_dir = "{}/models/{}/".format(args.dump_location, args.exp_name)
    #dump_dir = "{}/dump/{}/".format(args.dump_location, args.exp_name)

    #if not os.path.exists(log_dir):
    #    os.makedirs(log_dir)

    #if not os.path.exists("{}/images/".format(dump_dir)):
    #    os.makedirs("{}/images/".format(dump_dir))

    #logging.basicConfig(
    #    filename=log_dir + 'train.log',
    #    level=logging.INFO)
    #print("Dumping at {}".format(log_dir))
    print(args)
    logging.info(args)
    args.run_id = args.run_id or str(uuid.uuid4())
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.jsonl"
    if progress_path.exists():
        raise FileExistsError("Progress file already exists: {}".format(progress_path))
    source_root = Path(__file__).resolve().parent
    source_commit = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        text=True,
        timeout=30,
    ).strip()
    source_changes = subprocess.check_output(
        ["git", "-C", str(source_root), "status", "--porcelain"],
        text=True,
        timeout=30,
    ).strip()
    if source_changes:
        raise RuntimeError("Refusing to run from a dirty source tree")
    run_context = load_run_context(run_dir)
    if (
        args.task_config == "tasks/pointnav_gibson_visual.yaml"
        and run_context is None
    ):
        raise RuntimeError("The Gibson visual task requires a run context")
    runtime_install_path = run_dir / "runtime_install.json"
    if not runtime_install_path.is_file():
        raise FileNotFoundError("Runtime installation evidence is missing")
    runtime_install_sha256 = sha256(runtime_install_path)
    if run_context is not None:
        if args.task_config != "tasks/pointnav_gibson_visual.yaml":
            raise RuntimeError("Gibson run context requires the pinned task config")
        if args.split != "val":
            raise RuntimeError("Gibson run context requires the val split")
        if not args.require_topology_transition:
            raise RuntimeError("Gibson run context requires topology transitions")
        if args.pad_episode_to_max_steps:
            raise RuntimeError("Gibson run context forbids episode-tail padding")
        if not args.visualize:
            raise RuntimeError("Gibson run context requires the live dashboard")
        if args.visualization_frame_every_steps != 5:
            raise RuntimeError("Gibson run context requires five-step frame cadence")
        if args.window_checkpoint_every_steps != 100:
            raise RuntimeError(
                "Gibson run context requires 100-step X11 checkpoints"
            )
    started_at = time()
    run_metadata = {
        "run_id": args.run_id,
        "process_id": os.getpid(),
        "started_at_unix": started_at,
        "requested_max_episode_steps": int(args.max_episode_length),
        "source_commit": source_commit,
        "detector_device": args.detector_device,
        "detr_source_dir": args.detr_source_dir,
        "visualization": bool(args.visualize),
        "window_title": args.window_title,
        "visualization_frame_every_steps": int(
            args.visualization_frame_every_steps
        ),
        "window_checkpoint_every_steps": int(
            args.window_checkpoint_every_steps
        ),
        "require_topology_transition": bool(args.require_topology_transition),
        "pad_episode_to_max_steps": bool(args.pad_episode_to_max_steps),
        "task_config": args.task_config,
        "split": args.split,
        "runtime_install_sha256": runtime_install_sha256,
        "run_context": run_context,
    }

    event_recorder = TopologyEventRecorder(
        run_dir / "topology_events.jsonl",
        args.run_id,
        os.getpid(),
    )
    topology_provider = {"snapshot": None}
    runtime_state = {
        "phase": "initializing",
        "action": "none",
        "topology_exploration_complete_step": None,
        "completion_reason": None,
        "last_dashboard_info": None,
        "last_dashboard_absolute_locs": None,
        "voxroom_result": None,
    }
    voxroom_sidecar = None
    runtime_timings = {}
    confirmed_crossing_evidence = []

    def accumulate_timing(name, duration):
        duration = float(duration)
        stats = runtime_timings.setdefault(
            str(name),
            {
                "count": 0,
                "total_seconds": 0.0,
                "max_seconds": 0.0,
                "last_seconds": 0.0,
            },
        )
        stats["count"] += 1
        stats["total_seconds"] += duration
        stats["max_seconds"] = max(stats["max_seconds"], duration)
        stats["last_seconds"] = duration

    def timing_summary():
        return {
            name: {
                **stats,
                "mean_seconds": (
                    stats["total_seconds"] / stats["count"]
                    if stats["count"]
                    else 0.0
                ),
            }
            for name, stats in sorted(runtime_timings.items())
        }

    def record_event(event_type, **payload):
        event = event_recorder.record(
            event_type,
            action_count,
            **payload,
        )
        if event_type == "door_crossing_confirmed":
            confirmed_crossing_evidence.append(
                json.loads(json.dumps(payload["evidence"]))
            )
        return event

    def set_runtime_phase(phase, action="none"):
        runtime_state["phase"] = str(phase)
        runtime_state["action"] = str(action)

    def append_progress(step, info):
        voxroom_response = None
        if voxroom_sidecar is not None:
            voxroom_started_at = perf_counter()
            voxroom_response = voxroom_sidecar.update(
                int(step),
                int(info["time"]),
                info,
            )
            accumulate_timing(
                "voxroom_sidecar",
                perf_counter() - voxroom_started_at,
            )
            if info.get("voxroom_navigation_map_source") != (
                "voxroom_last_voxel_navigation_projection"
            ):
                raise RuntimeError(
                    "VoxRoom sidecar did not publish its parallel navigation map"
                )
            if int(info.get("voxroom_navigation_map_step", -1)) != int(step):
                raise RuntimeError(
                    "VoxRoom sidecar published a stale parallel navigation map"
                )
        if info.get("active_room_map_source") != (
            "active_room_native_depth_projection"
        ):
            raise RuntimeError(
                "Active Room exploration did not retain its native depth map"
            )
        topology_snapshot = (
            topology_provider["snapshot"]()
            if topology_provider["snapshot"] is not None
            else {
                "current_node_id": 0,
                "room_count": 1,
                "edge_count": 0,
            }
        )
        event = {
            "run_id": args.run_id,
            "process_id": os.getpid(),
            "step": int(step),
            "simulator_step": int(info["time"]),
            "scene_name": str(info.get("scene_name", "")),
            "episode_id": str(info.get("episode_id", "")),
            "episode_contract_sha256": str(
                info.get("episode_contract_sha256", "")
            ),
            "explored_ratio": float(info.get("exp_ratio") or 0.0),
            "explored_reward": float(info.get("exp_reward") or 0.0),
            "phase": runtime_state["phase"],
            "action": runtime_state["action"],
            "topology_current_node_id": int(
                topology_snapshot.get("current_node_id", 0)
            ),
            "topology_room_count": int(topology_snapshot.get("room_count", 0)),
            "topology_edge_count": int(topology_snapshot.get("edge_count", 0)),
            "voxroom_step": (
                None if voxroom_response is None else int(voxroom_response["step"])
            ),
            "voxroom_room_count": (
                None
                if voxroom_response is None
                else int(voxroom_response["room_count"])
            ),
            "navigation_map_source": info.get("active_room_map_source"),
            "navigation_planner_source": "active_room_original_fmm",
            "navigation_pose_source": "active_room_accumulated_pose",
            "voxroom_map_source": info.get("voxroom_navigation_map_source"),
            "timestamp_unix": time(),
        }
        with progress_path.open("a", encoding="utf-8") as progress_file:
            progress_file.write(json.dumps(event, sort_keys=True) + "\n")
            progress_file.flush()
            os.fsync(progress_file.fileno())

    live_figure = None

    # Logging and loss variables
    num_scenes = args.num_processes
    num_episodes = int(args.num_episodes)
    device = args.device = torch.device("cuda:0" if args.cuda else "cpu")
    policy_loss = 0

    best_cost = 100000
    costs = deque(maxlen=1000)
    exp_costs = deque(maxlen=1000)
    pose_costs = deque(maxlen=1000)

    g_masks = torch.ones(num_scenes).float().to(device)
    l_masks = torch.zeros(num_scenes).float().to(device)

    best_local_loss = np.inf
    best_g_reward = -np.inf

    if args.eval:
        traj_lengths = args.max_episode_length // args.num_local_steps
        explored_area_log = np.zeros((num_scenes, num_episodes, traj_lengths))
        explored_ratio_log = np.zeros((num_scenes, num_episodes, traj_lengths))

    g_episode_rewards = deque(maxlen=1000)

    l_action_losses = deque(maxlen=1000)

    g_value_losses = deque(maxlen=1000)
    g_action_losses = deque(maxlen=1000)
    g_dist_entropies = deque(maxlen=1000)

    per_step_g_rewards = deque(maxlen=1000)

    g_process_rewards = np.zeros((num_scenes))

    # Starting environments
    torch.set_num_threads(1)
    envs = make_vec_envs(args)
    obs, infos = envs.reset()
    if args.voxroom_sidecar:
        voxroom_sidecar = VoxRoomSidecarClient(
            voxroom_root=args.voxroom_root,
            config_path=args.voxroom_config,
            run_dir=run_dir,
            map_size_m=args.voxroom_map_size_m,
            roomseg_every_steps=args.voxroom_roomseg_every_steps,
            visualization_every_steps=args.voxroom_visualization_every_steps,
            response_timeout_seconds=args.voxroom_response_timeout_seconds,
        )
        voxroom_sidecar.update(
            0,
            int(infos[0]["time"]),
            infos[0],
        )
        if infos[0].get("voxroom_navigation_map_source") != (
            "voxroom_last_voxel_navigation_projection"
        ):
            raise RuntimeError(
                "Initial VoxRoom sidecar map was not attached in parallel"
            )
    if infos[0].get("active_room_map_source") != (
        "active_room_native_depth_projection"
    ):
        raise RuntimeError("Initial Active Room map is not its native depth map")
    actual_episode_id = str(infos[0].get("episode_id", ""))
    actual_episode_contract_sha256 = str(
        infos[0].get("episode_contract_sha256", "")
    )
    actual_scene_id = Path(infos[0].get("scene_name", "")).stem
    if (
        not actual_episode_id
        or not actual_scene_id
        or len(actual_episode_contract_sha256) != 64
    ):
        raise RuntimeError("Habitat did not expose the active episode identity")
    if run_context is not None and (
        actual_episode_id != run_context["episode_id"]
        or actual_scene_id != run_context["scene_id"]
        or actual_episode_contract_sha256
        != run_context["episode_contract_sha256"]
    ):
        raise RuntimeError("Habitat loaded an episode outside the run context")
    if run_context is not None and load_run_context(run_dir) != run_context:
        raise RuntimeError("Run context changed while Habitat loaded the episode")
    dashboard = None
    x11_client_window_id = None
    if args.visualize or args.print_images:
        live_figure = plt.figure(
            num=args.window_title,
            figsize=(
                STRICT_LIVE_FIGURE_SIZE_INCHES
                if run_context is not None
                else (16.0, 9.0)
            ),
            dpi=(
                STRICT_LIVE_FIGURE_DPI
                if run_context is not None
                else 100
            ),
        )
        dashboard = RuntimeDashboard(
            live_figure,
            run_dir,
            capture_identity=args.run_id,
            frame_every_steps=args.visualization_frame_every_steps,
            refresh_seconds=args.visualization_refresh_seconds,
        )
        x11_client_window_id = bind_x11_client_window(
            live_figure,
            args.window_title,
        )
    run_metadata.update(
        {
            "actual_episode_id": actual_episode_id,
            "actual_episode_contract_sha256": actual_episode_contract_sha256,
            "actual_scene_id": actual_scene_id,
            "x11_client_window_id": x11_client_window_id,
            "voxroom_sidecar": bool(voxroom_sidecar is not None),
            "voxroom_root": args.voxroom_root if args.voxroom_sidecar else None,
            "voxroom_config": args.voxroom_config if args.voxroom_sidecar else None,
            "exploration_navigation_source": (
                "active_room_native_depth_projection"
            ),
            "voxroom_mapping_source": (
                "voxroom_last_voxel_navigation_projection"
                if args.voxroom_sidecar
                else None
            ),
        }
    )
    write_json_atomic(run_dir / "run_metadata.json", run_metadata)

    # Initialize map variables
    ### Full map consists of 4 channels containing the following:
    ### 1. Obstacle Map
    ### 2. Exploread Area
    ### 3. Current Agent Location
    ### 4. Past Agent Locations

    torch.set_grad_enabled(False)
    for scene_idx in range(1):
        scene_idx += 0

        # print('pp_list {}'.format(prior_door_list))
        # print('start_next_scene')
        frontier_detector = Frontier_detection(
            args.map_size_cm // args.map_resolution,
            vision_range=args.vision_range,
        )
        topo = Topomap_construction(
            map_size=args.map_size_cm // args.map_resolution,
            vision_range=args.vision_range,
            event_callback=lambda event_type, payload: record_event(
                event_type,
                **payload,
            ),
        )
        topology_provider["snapshot"] = topo.snapshot
        if established_graph:
            topo.use_exist_topomap(established_graph)
        # Calculating full and local map sizes
        map_size = args.map_size_cm // args.map_resolution
        full_w, full_h = map_size, map_size
        local_w, local_h = int(full_w / args.global_downscaling), \
                           int(full_h / args.global_downscaling)

        # Initializing full and local map
        full_map = torch.zeros(num_scenes, 4, full_w, full_h).float().to(device)
        local_map = torch.zeros(num_scenes, 4, local_w, local_h).float().to(device)

        # Initial full and local pose
        full_pose = torch.zeros(num_scenes, 3).float().to(device)
        local_pose = torch.zeros(num_scenes, 3).float().to(device)

        # Origin of local map
        origins = np.zeros((num_scenes, 3))

        # Local Map Boundaries
        lmb = np.zeros((num_scenes, 4)).astype(int)

        ### Planner pose inputs has 7 dimensions
        ### 1-3 store continuous global agent location
        ### 4-7 store local map boundaries
        planner_pose_inputs = np.zeros((num_scenes, 7))

        def init_map_and_pose():
            full_map.fill_(0.)
            full_pose.fill_(0.)
            full_pose[:, :2] = args.map_size_cm / 100.0 / 2.0

            locs = full_pose.cpu().numpy()
            planner_pose_inputs[:, :3] = locs
            for e in range(num_scenes):
                r, c = locs[e, 1], locs[e, 0]
                loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                int(c * 100.0 / args.map_resolution)]  # position on the grid(center of local map, unit: grid)

                full_map[e, 2:, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

                lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                                  (local_w, local_h),
                                                  (full_w, full_h))

                planner_pose_inputs[e, 3:] = lmb[e]
                origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                              lmb[e][0] * args.map_resolution / 100.0,
                              0.]  # position of the bottom left (origin) of local map(unit: m)

            for e in range(num_scenes):
                local_map[e] = full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]]
                local_pose[e] = full_pose[e] - \
                                torch.from_numpy(origins[e]).to(device).float()  # position of the robot in local view

        init_map_and_pose()
        keep_exploring = True
        locs = np.array([args.map_size_cm / 100.0 / 4.0, args.map_size_cm / 100.0 / 4.0, 0])  # origins[0] [y,x,o]
        panoramic_obstacle_map = torch.zeros(num_scenes, 4, full_w, full_h).float().to(device)
        trajectory_xy = deque(maxlen=args.max_episode_length + 1)

        def require_x11_capture(ready_path, ack_path, payload, timeout_message):
            if ready_path.is_file():
                if json.loads(ready_path.read_text(encoding="utf-8")) != payload:
                    raise RuntimeError("X11 capture request identity changed")
            else:
                write_json_atomic(ready_path, payload)
            capture_deadline = time() + 120.0
            while not ack_path.is_file():
                if time() >= capture_deadline:
                    raise TimeoutError(timeout_message)
                live_figure.canvas.flush_events()
                plt.pause(0.05)
            if ack_path.read_text(encoding="utf-8").strip() != args.run_id:
                raise RuntimeError("X11 capture acknowledgement mismatch")

        def render_dashboard(info, absolute_locs, goal_xy=None, frontiers=None):
            if dashboard is None:
                return
            runtime_state["last_dashboard_info"] = info
            runtime_state["last_dashboard_absolute_locs"] = np.asarray(
                absolute_locs
            ).copy()
            heading_degrees = float(absolute_locs[2])
            agent_xy = (
                float(absolute_locs[1] * 100.0 / args.map_resolution),
                float(absolute_locs[0] * 100.0 / args.map_resolution),
            )
            if not trajectory_xy or np.linalg.norm(
                np.asarray(agent_xy) - np.asarray(trajectory_xy[-1])
            ) > 0.05:
                trajectory_xy.append(agent_xy)
            occupied = np.asarray(info["gt_map"]).transpose()
            explored = np.asarray(info["gt_exp"]).transpose()
            topology_snapshot = topo.snapshot()
            room_labels = topo.room_label_map(occupied.shape)
            room_labels = dashboard.navigation_clipped_room_labels(
                occupied,
                explored,
                room_labels,
            )
            explored_ratio = float(info.get("exp_ratio") or 0.0)
            explored_area = float(info.get("exp_reward") or 0.0) * 50.0
            dashboard.render(
                step=action_count,
                rgb=info["door_detection"],
                occupied=occupied,
                explored=explored,
                room_labels=room_labels,
                agent_xy=agent_xy,
                heading_degrees=heading_degrees,
                goal_xy=goal_xy,
                doors=detected_door_list,
                raw_doors=raw_detect_list,
                frontiers=frontiers or [],
                trajectory_xy=list(trajectory_xy),
                topology_snapshot=topology_snapshot,
                coverage_history=list(cov_ratio_list),
                explored_ratio=explored_ratio,
                explored_area=explored_area,
                status={
                    "phase": runtime_state["phase"],
                    "action": runtime_state["action"],
                    "scene_name": scene_name,
                    "door_count": len(detected_door_list),
                    "raw_door_count": len(raw_detect_list),
                    "frontier_count": len(frontiers or []),
                },
                events=event_recorder.latest_events(),
                voxroom_image=(
                    None
                    if voxroom_sidecar is None
                    else voxroom_sidecar.latest_visualization()
                ),
            )
            if run_context is not None:
                for stage_name, stage_step in STRICT_VISUAL_CAPTURE_STEPS:
                    if action_count != stage_step:
                        continue
                    if dashboard.last_render_step != stage_step:
                        raise RuntimeError(
                            "Strict visual stage is not bound to its rendered step"
                        )
                    frame_path = dashboard.frame_dir / (
                        "frame_{:06d}.png".format(stage_step)
                    )
                    if (
                        dashboard.last_saved_step != stage_step
                        or not frame_path.is_file()
                    ):
                        raise RuntimeError(
                            "Strict visual stage has no rendered frame artifact"
                        )
                    stage_dir = run_dir / "window_stages"
                    stage_dir.mkdir(parents=True, exist_ok=True)
                    require_x11_capture(
                        stage_dir / "ready_{}.json".format(stage_name),
                        stage_dir / "ack_{}.txt".format(stage_name),
                        {
                            "run_id": args.run_id,
                            "process_id": os.getpid(),
                            "window_id": x11_client_window_id,
                            "stage": stage_name,
                            "step": stage_step,
                            "render_step": dashboard.last_render_step,
                            "frame_file": frame_path.relative_to(
                                run_dir
                            ).as_posix(),
                            "frame_sha256": sha256(frame_path),
                        },
                        "Timed out waiting for the {} X11 stage capture".format(
                            stage_name
                        ),
                    )
            if (
                run_context is not None
                and action_count > 0
                and action_count % args.window_checkpoint_every_steps == 0
            ):
                checkpoint_dir = run_dir / "window_checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_label = "{:06d}".format(action_count)
                ready_path = checkpoint_dir / (
                    "ready_{}.json".format(checkpoint_label)
                )
                ack_path = checkpoint_dir / (
                    "ack_{}.txt".format(checkpoint_label)
                )
                frame_path = dashboard.frame_dir / (
                    "frame_{}.png".format(checkpoint_label)
                )
                if (
                    dashboard.last_render_step != action_count
                    or dashboard.last_saved_step != action_count
                    or not frame_path.is_file()
                ):
                    raise RuntimeError(
                        "Periodic X11 checkpoint has no rendered frame artifact"
                    )
                require_x11_capture(
                    ready_path,
                    ack_path,
                    {
                        "run_id": args.run_id,
                        "process_id": os.getpid(),
                        "window_id": x11_client_window_id,
                        "step": action_count,
                        "render_step": dashboard.last_render_step,
                        "frame_file": frame_path.relative_to(
                            run_dir
                        ).as_posix(),
                        "frame_sha256": sha256(frame_path),
                    },
                    "Timed out waiting for X11 checkpoint capture",
                )

        def generate_12_parts(agent_pose):
            x = agent_pose[1]  # x and y unit is m and presented in local frame
            y = agent_pose[0]
            angle = agent_pose[2]  # the angle unit is degree
            angle_list = [np.deg2rad(angle + i * 30) for i in range(12)]  # in the list, the unit becomes rad
            goal_list = []
            for content in angle_list:
                dx = 3 * np.cos(content)  # 3 * np.cos(content)
                dy = 3 * np.sin(content)  # 3 * np.sin(content)
                goal_list.append([int((x + dx) * 100 / 5), int((y + dy) * 100 / 5)])
            return goal_list

        def take_action(action, locs, first_flag=True, long_term_goal=None, room_search_flag = False):
            global action_count
            global cov_ratio
            global cov_ratio_list
            global cov_area
            global cov_area_list
            global step_list
            global detected_door_list
            global raw_detect_list
            global bot_last_loc
            global take_name_flag
            global scene_name
            global log_num
            #global frontier_failed
            global route_map
            global last_goal
            global stage1_door_map
            global stage2_door_map
            global stage3_door_map
            global last_episode_action_count
            global last_episode_cov_ratio
            global last_episode_cov_area

            if action == 4:
                runtime_state["action"] = "scan_turn_right"
            else:
                runtime_state["action"] = {
                    0: "turn_left",
                    1: "turn_right",
                    2: "forward",
                }.get(int(action), "unknown")

            """if action != 4:
                action_count += 1
            else:
                action_count += 12"""
            if action != 4:
                kernel = np.ones((3, 3), np.uint8)
                obs, rew, done, infos = envs.step(torch.tensor([action]))  #
                if take_name_flag:
                    scene_name = infos[0]['scene_name']
                    scene_name = scene_name.split('/')
                    scene_name = scene_name[-1]
                    scene_name = scene_name[:-4]

                    take_name_flag = False
                completed_step = action_count + 1
                action_count = completed_step
                append_progress(completed_step, infos[0])
                if bool(np.asarray(done).reshape(-1)[0]):  # means this round should be over
                    last_episode_action_count = completed_step
                    last_episode_cov_ratio = float(infos[0]["exp_ratio"])
                    last_episode_cov_area = float(infos[0]["exp_reward"]) * 50.0
                    record_event(
                        "episode_done",
                        simulator_step=int(infos[0]["time"]),
                        explored_ratio=last_episode_cov_ratio,
                    )
                    return np.array([None]), np.array([None]), np.array([None]), False
                # print(infos[0]['sensor_pose'])
                # print(locs)
                # locs = locs + infos[0]['sensor_pose']
                locs = pu.get_new_pose(locs, infos[0]['sensor_pose'])
                absolute_locs = locs + origins[0]
                long_term_goal.reverse()  # [y, x]
                global_goal = np.array(long_term_goal) + origins[0][:2] * 100 / 5  # [y,x]
                r, c = absolute_locs[1], absolute_locs[0]
                loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                int(c * 100.0 / args.map_resolution)]
                global_loc_xy_pix = [round(r * 100.0 / args.map_resolution),
                                     round(c * 100.0 / args.map_resolution)]
                route_map[global_loc_xy_pix[1], global_loc_xy_pix[0]] = action_count

                lmb[0] = get_local_map_boundaries((loc_r, loc_c),
                                                  (local_w, local_h),
                                                  (full_w, full_h))
                planner_pose_inputs[0, 3:] = lmb[0]
                planner_pose_inputs[0, :3] = absolute_locs
                [gx1, gx2, gy1, gy2] = lmb[0]
                origins[0] = [lmb[0][2] * args.map_resolution / 100.0,
                              lmb[0][0] * args.map_resolution / 100.0, 0.]
                goal = global_goal - origins[0][:2] * 100 / 5
                #goal = goal.astype(int).tolist()
                goal = np.round(goal).astype(int).tolist()
                goal.reverse()  # [x, y]
                locs = absolute_locs - origins[0]

                bot_last_loc.append(np.array([loc_r, loc_c]))

                # print('origin {}'.format(origins[0]))
                # print('absolute_locs {}'.format(absolute_locs))
                # print('locs {}'.format(locs))
                # goal_list = generate_12_parts(locs)
                # print(goal_list)
                angle = np.deg2rad(locs[2])
                dx = np.sin(angle)
                dy = np.cos(angle)

                #obs_show = obs[0].cpu().detach().numpy().astype(int)  # this two line is the obs from observation
                #obs_show = obs_show.transpose(1, 2, 0)

                obs_show = infos[0]['door_detection']  # this obs is from detr network after door detection

                # obs_show = run_detr(obs_show / 255)  # after door detection
                exp_ratio = infos[0]['exp_ratio']
                exp_area = infos[0]['exp_reward']*50.0  # convert to m2
                time_cost_current = time() - t_start
                #step_list.append(action_count)
                step_list.append(time_cost_current)
                cov_ratio_list.append(exp_ratio)
                cov_area_list.append(exp_area)
                """if exp_ratio:
                    cov_ratio = exp_ratio"""
                # print('exp ratio {}'.format(cov_ratio))

                # obs_show = obs_show.transpose(2,1,0)
                gt_map = infos[0]['gt_map']  # ['gt_map']
                gt_exp = infos[0]['gt_exp']
                gt_map_door = gt_map.copy()
                gt_exp_door = gt_exp.copy()
                # gt_map = gt_map[::-1, ::-1]
                # gt_map = gt_map[::-1,:]  # flip vertically
                # gt_map = gt_map[:, ::-1]  # flip horizontally
                gt_map_local_grid = np.rint(gt_map[gx1:gx2, gy1:gy2])
                gt_exp_local_grid = np.rint(gt_exp[gx1:gx2, gy1:gy2])
                gt_map = gt_map.transpose()
                gt_exp = gt_exp.transpose()
                # ---------------------------------------------
                """planner_pose_inputs_copy = planner_pose_inputs.copy()
                planner_inputs = [{} for e in range(num_scenes)]
                for e, p_input in enumerate(planner_inputs):
                    p_input['goal'] = goal
                    p_input['map_pred'] = gt_map_local_grid
                    p_input['exp_pred'] = gt_exp_local_grid
                    p_input['pose_pred'] = planner_pose_inputs[0]
                    p_input['mid_out'] = True
                path_list = []
                for _ in range(150):
                    output = envs.get_short_term_goal(planner_inputs)
                    # print(output)
                    dist = as_numpy(output[0][0])
                    stg = as_numpy(output[0][1:])
                    stg_x, stg_y = stg
                    # global_stg = np.array([stg_y, stg_x]) + np.array([origins[0][0] * 100 / 5, origins[0][1] * 100 / 5])
                    # global_stg = np.array([stg_y, stg_x]) + np.array([origins[0][0] * 100 / 5, origins[0][1] * 100 / 5])
                    path_list.append([int(stg_x + origins[0][1] * 100 / 5), int(stg_y + origins[0][0] * 100 / 5)])
                    # print(total_dist)
                    dist2goal = pu.get_l2_distance(stg_x, goal[0], stg_y, goal[1])
                    # if dist2goal < 1:
                    #    break

                    planner_pose_inputs[0, :2] = stg_y * 5 / 100 + origins[0][0], stg_x * 5 / 100 + origins[0][1]
                    for e, p_input in enumerate(planner_inputs):
                        p_input['goal'] = goal
                        p_input['map_pred'] = gt_map_local_grid
                        p_input['exp_pred'] = gt_exp_local_grid
                        p_input['pose_pred'] = planner_pose_inputs[0]
                        p_input['mid_out'] = True
                planner_pose_inputs[0, :2] = planner_pose_inputs_copy[0, :2]"""
                # ---------------------------------------------------

                planner_inputs = [{} for e in range(num_scenes)]

                if room_search_flag:

                    for door_detected in detected_door_list:
                        door_start = door_detected['start']
                        door_end = door_detected['end']
                        gt_map_door = cv2.line(gt_map_door, (door_start[1], door_start[0]), (door_end[1], door_end[0]), 1, thickness = 1)
                        gt_exp_door = cv2.line(gt_exp_door, (door_start[1], door_start[0]), (door_end[1], door_end[0]),
                                               0, thickness=1)
                    gt_map_local_grid = np.rint(gt_map_door[gx1:gx2, gy1:gy2])
                    gt_exp_local_grid = np.rint(gt_exp_door[gx1:gx2, gy1:gy2])
                    #plt.imshow(gt_map_local_grid)
                    #plt.show()
                for e, p_input in enumerate(planner_inputs):
                    p_input['goal'] = goal
                    p_input['map_pred'] = gt_map_local_grid
                    p_input['exp_pred'] = gt_exp_local_grid
                    p_input['pose_pred'] = planner_pose_inputs[0]
                    p_input['mid_out'] = True
                output = envs.get_short_term_goal(planner_inputs)
                stg = as_numpy(output[0][1:])

                # stg = stg.tolist()  # under local frame
                stg_x, stg_y = stg
                """plt.imshow(gt_map_local_grid.transpose())
                plt.plot(locs[1] * 100 / 5, locs[0] * 100 / 5, 'o', color='pink')
                plt.plot(goal[0], goal[1], 'o', color='red')

                for i in path_list:
                    plt.plot(i[0]-origins[0][1]*100/5, i[1]-origins[0][0]*100/5, 'o', color = 'plum')
                plt.plot(stg_x, stg_y, 'o', color='blue')
                plt.arrow(locs[1] * 100 / 5, locs[0] * 100 / 5, dx * 8, dy * (8 * 1.25), head_width=8,
                          head_length=8 * 1.25,
                          length_includes_head=True, fc='Red', ec='Red', alpha=0.9)
                plt.plot(locs[1] * 100 / 5, locs[0] * 100 / 5, 'o', color = 'white')
                plt.show()
                plt.clf()"""
                # stg_x = stg_x + origins[0][1]*100/args.map_resolution
                # stg_y = stg_y + origins[0][0]*100/args.map_resolution
                stg = [stg_x * args.map_resolution / 100, stg_y * args.map_resolution / 100]
                render_dashboard(
                    infos[0],
                    absolute_locs,
                    goal_xy=(float(global_goal[1]), float(global_goal[0])),
                )

                return locs, stg, goal, False
            else:
                action = 1  # 12 turn_right makes up on scan motion
                for turn_num in range(12):
                    obs, rew, done, infos = envs.step(torch.tensor([action]))  #
                    if take_name_flag:
                        scene_name_full = infos[0]['scene_name']
                        scene_name = scene_name_full.split('/')
                        scene_name = scene_name[-1]
                        scene_name = scene_name[:-4]
                        take_name_flag = False
                    action_count += 1
                    append_progress(action_count, infos[0])
                    exp_ratio = infos[0]['exp_ratio']
                    exp_area = infos[0]['exp_reward'] * 50.0  # convert to m2
                    time_cost_current = time() - t_start
                    # step_list.append(action_count)
                    step_list.append(time_cost_current)
                    cov_ratio_list.append(exp_ratio)
                    cov_area_list.append(exp_area)
                    if bool(np.asarray(done).reshape(-1)[0]):  # means this round should be over
                        last_episode_action_count = action_count
                        last_episode_cov_ratio = float(cov_ratio_list[-1]) if cov_ratio_list else 0.0
                        last_episode_cov_area = float(cov_area_list[-1]) if cov_area_list else 0.0
                        record_event(
                            "episode_done",
                            simulator_step=int(infos[0]["time"]),
                            explored_ratio=last_episode_cov_ratio,
                        )
                        return np.array([None]), np.array([None]), np.array([None]), False
                    # print(infos[0]['sensor_pose'])
                    # print(locs)
                    locs = pu.get_new_pose(locs, infos[0]['sensor_pose'])
                    absolute_locs = locs + origins[0]
                    r, c = absolute_locs[1], absolute_locs[0]
                    loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                    int(c * 100.0 / args.map_resolution)]

                    lmb[0] = get_local_map_boundaries((loc_r, loc_c),
                                                      (local_w, local_h),
                                                      (full_w, full_h))
                    [gx1, gx2, gy1, gy2] = lmb[0]
                    planner_pose_inputs[0, 3:] = lmb[0]
                    planner_pose_inputs[0, :3] = absolute_locs
                    origins[0] = [lmb[0][2] * args.map_resolution / 100.0,
                                  lmb[0][0] * args.map_resolution / 100.0, 0.]
                    locs = absolute_locs - origins[0]
                    # print('origin {}'.format(origins[0]))
                    # print('absolute_locs {}'.format(absolute_locs))
                    # print('locs {}'.format(locs))

                    # print(locs)
                    angle = np.deg2rad(locs[2])
                    dx = np.sin(angle)
                    dy = np.cos(angle)

                    #obs_show = obs[0].cpu().detach().numpy().astype(np.uint8)
                    #obs_show = obs_show.transpose(1, 2, 0)

                    obs_show = infos[0]['door_detection']  # this obs is from detr network after door detection
                    depth_image = infos[0]['depth']

                    #np.save('paper_fig/{}.npy'.format(turn_num), obs[0].cpu().detach().numpy().astype(np.uint8).transpose(1, 2, 0))
                    #np.save('paper_fig/depth{}.npy'.format(turn_num),
                    #        depth_image)
                    render_dashboard(infos[0], absolute_locs)
                    # plt.imsave('pic_save/{}.png'.format(action_count), obs_show)
                # print('origin {}'.format(origins[0]))
                # print('absolute_locs {}'.format(absolute_locs))
                # print('locs {}'.format(locs))
                # gt_map = infos[0]['pano_map']  # ['gt_map']
                # gt_exp = infos[0]['pano_exp']
                #cov_ratio_list.append(infos[0]['exp_ratio'])
                #cov_area_list.append(infos[0]['exp_reward']*50.0)
                gt_map = infos[0]['gt_map']  # ['gt_map']
                gt_exp = infos[0]['gt_exp']
                gt_pano_map = infos[0]['pano_map']
                gt_pano_exp = infos[0]['pano_exp']
                gt_door_map = infos[0]['door_obs_map']
                gt_door_local_map = infos[0]['door_local_map']  # this is for door detection

                # obs_show = hough_detection(obs_show)

                if exp_ratio:
                    cov_ratio = exp_ratio
                if exp_area:
                    cov_area = exp_area
                # print('exp ratio {}'.format(cov_ratio))
                # gt_map = gt_map.transpose()
                # gt_exp = gt_exp.transpose()

                # gt_map_local_grid = np.rint(gt_map[gy1:gy2, gx1:gx2])
                # gt_exp_local_grid = np.rint(gt_exp[gy1:gy2, gx1:gx2])
                gt_map_local_grid = np.rint(gt_map[gx1:gx2, gy1:gy2])
                gt_exp_local_grid = np.rint(gt_exp[gx1:gx2, gy1:gy2])
                # plt.imshow(gt_map_local_grid.transpose())
                # plt.plot(locs[1]*100/5, locs[0]*100/5, 'o', color = 'green')
                # plt.show()
                # exit()
                gt_map = gt_map.transpose()
                gt_exp = gt_exp.transpose()
                gt_pano_map = gt_pano_map.transpose()
                gt_pano_exp = gt_pano_exp.transpose()
                gt_door_map = gt_door_map.transpose()
                gt_door_local_map = gt_door_local_map.transpose()
                """plt.subplot(1,2,1)
                plt.imshow(gt_pano_map)
                plt.subplot(1,2,2)
                plt.imshow(gt_door_local_map)
                plt.show()"""

                c, r = absolute_locs[:-1]
                global_loc_xy_pix = [round(r * 100.0 / args.map_resolution),
                                     round(c * 100.0 / args.map_resolution)]
                """start = [round(r * 100.0 / args.map_resolution - gx1),
                         round(c * 100.0 / args.map_resolution - gy1)]"""
                goal_list = generate_12_parts(locs)
                planner_pose_inputs_copy = planner_pose_inputs.copy()
                unexplorable_list = []
                create_door_detection = True
                # door_detect = Door_detection(list_x, list_y)
                full_list_x = []
                full_list_y = []
                for j, goal in enumerate(goal_list):
                    # goal = [int(302-origins[0][1]*100/5), int(237-origins[0][0]*100/5)]  # [x, y]
                    # goal = [182, 117]
                    # print(goal)

                    planner_inputs = [{} for e in range(num_scenes)]
                    for e, p_input in enumerate(planner_inputs):
                        p_input['goal'] = goal
                        p_input['map_pred'] = gt_map_local_grid
                        p_input['exp_pred'] = gt_exp_local_grid
                        p_input['pose_pred'] = planner_pose_inputs[0]
                        p_input['mid_out'] = True
                    loc_y = (planner_pose_inputs[0][0] - origins[0][0]) * 100 / 5
                    loc_x = (planner_pose_inputs[0][1] - origins[0][1]) * 100 / 5
                    total_dist = 0

                    list_x = []
                    list_y = []
                    for _ in range(150):  # 150
                        output = envs.get_short_term_goal(planner_inputs)
                        # print(output)
                        dist = as_numpy(output[0][0])
                        total_dist += dist
                        stg = as_numpy(output[0][1:])
                        stg_x, stg_y = stg
                        # global_stg = np.array([stg_y, stg_x]) + np.array([origins[0][0] * 100 / 5, origins[0][1] * 100 / 5])
                        list_x.append(stg_x + origins[0][1] * 100 / 5)
                        list_y.append(stg_y + origins[0][0] * 100 / 5)
                        full_list_x.append(stg_x + origins[0][1] * 100 / 5)
                        full_list_y.append(stg_y + origins[0][0] * 100 / 5)
                        # print(total_dist)
                        dist2goal = pu.get_l2_distance(stg_x, goal[0], stg_y, goal[1])
                        if total_dist > explorable_threshold:
                            if dist2goal > 3:  # remove the point that the distance between goal and final pos larger than 3 pix
                                unexplorable_list.append(goal)
                                break
                        if dist2goal < 1:
                            break

                        planner_pose_inputs[0, :2] = stg_y * 5 / 100 + origins[0][0], stg_x * 5 / 100 + origins[0][1]
                        for e, p_input in enumerate(planner_inputs):
                            p_input['goal'] = goal
                            p_input['map_pred'] = gt_map_local_grid
                            p_input['exp_pred'] = gt_exp_local_grid
                            p_input['pose_pred'] = planner_pose_inputs[0]
                            p_input['mid_out'] = True
                    # print(total_dist)
                    if total_dist == 0.0:
                        # print('goal in obstacle')
                        unexplorable_list.append(goal)
                    planner_pose_inputs[0, :2] = planner_pose_inputs_copy[0, :2]

                    if create_door_detection and first_flag:
                        global door_detect
                        door_detect = Door_detection(list_x, list_y)
                        create_door_detection = False
                    else:
                        door_detect.new_list(list_x, list_y)
                    if create_door_detection and not first_flag:
                        door_detect.reset(list_x, list_y)
                        create_door_detection = False
                    # np.save('list_y{}.npy'.format(j), list_y)
                    # np.save('list_x{}.npy'.format(j), list_x)

                # global_stg = np.array([stg_y, stg_x]) + np.array([origins[0][0]*100/5, origins[0][1]*100/5])
                # print(global_stg)
                # print(dist)
                # start = pu.threshold_poses(start, gt_map_local_grid.shape)
                # stg = envs._get_stg(gt_map_local_grid, gt_exp_local_grid, start, goal, [gx1, gx2, gy1, gy2])

                # gt_map = gt_map[::-1, :]  # filp vertically
                # gt_map = gt_map[::-1, ::-1]
                # exit()
                # raw_door_list = door_detect.get_door_point()

                # following two lines is using 12 points method to detect door
                # door_list, raw_list = door_detect.door_filter(gt_map)
                # detected_door_list.extend(door_list)
                # print(door_list)

                hough_door_list, laser_list = convert_2_laser(gt_pano_map, gt_pano_exp, absolute_locs)

                #plt.clf()
                #plt.subplot(1,2,1)
                #plt.ion()
                #plt.subplot(1,2,1)
                #plt.imshow(gt_door_map)
                for grid_idx in hough_door_list:
                    #plt.plot(grid_idx[0], grid_idx[1], 'o', color='red')
                    stage1_door_map[grid_idx[1], grid_idx[0]] = 1
                """plt.subplot(1,2,2)
                plt.imshow(gt_map)
                for grid_idx in hough_door_list:
                    plt.plot(grid_idx[0], grid_idx[1], 'o', color='red')"""
                #plt.subplot(1,2,2)
                #plt.imshow(stage1_door_map)
                #plt.show()
                #plt.pause(1)
                #plt.ioff()
                #plt.close()

                # following method is using RGB to filter the door
                filterd_hough_list = []
                for grid_idx in hough_door_list:

                    if np.sum(gt_door_map[grid_idx[1] - 2:grid_idx[1] + 3, grid_idx[0] - 2:grid_idx[0] + 3]) > 0:
                        filterd_hough_list.append(grid_idx)
                        stage2_door_map[grid_idx[1], grid_idx[0]] = 1

                door_filter_started = perf_counter()
                door_list, raw_list = \
                    door_detect.door_filter(gt_door_local_map, gt_map, gt_exp, global_loc_xy_pix, bot_last_loc,
                                            detected_door_list,
                                            use_12point=False,
                                            external_door_point=filterd_hough_list)
                door_filter_seconds = perf_counter() - door_filter_started
                accumulate_timing("door_filter", door_filter_seconds)

                # uncomment these two is frontier method
                #door_list.clear()
                #raw_list.clear()


                bot_last_loc.clear()
                close_door_list = []
                close_door_list.extend(door_list)
                detected_door_list.extend(door_list)
                raw_detect_list.extend(raw_list)
                #np.save('door_dict_main.npy', detected_door_list)


                """for door in prior_door_list:
                    mid_point = door['mid']
                    start = door['start']
                    end = door['end']
                    print('---')
                    print('current_door {}'.format(door))
                    print('dist {}'.format(pu.get_l2_distance(r * 100 / 5, mid_point[0], c * 100 / 5, mid_point[1])))
                    print('rob loc {}, {}'.format(r * 100 / 5, c * 100 / 5))
                    print('mid {}'.format(mid_point))
                    print('gt_exp {}'.format(gt_exp[int(mid_point[1]), int(mid_point[0])]))
                    print('gt_map_start {}'.format(np.sum(gt_map[start[1]-1:start[1]+2, start[0]-1:start[0]+2])))
                    print('gt_map_end {}'.format(np.sum(gt_map[end[1]-1:end[1]+2, end[0]-1:end[0]+2])))
                    print('---')
                    if pu.get_l2_distance(r * 100 / 5, mid_point[0], c * 100 / 5, mid_point[1]) <= args.vision_range \
                            and gt_exp[int(mid_point[1]), int(mid_point[0])] == 1 \
                            and np.sum(gt_map[start[1] - 2:start[1] + 3, start[0] - 2:start[0] + 3]) > 1 \
                            and np.sum(
                        gt_map[end[1] - 2:end[1] + 3, end[0] - 2:end[0] + 3]) > 1:  # if cantwell this line > 1

                        # print('find door {}'.format(door))
                        close_door_list.append(door)"""

                        # pass
                #for door in close_door_list:  # remove the chosen door
                #    prior_door_list.remove(door)
                # close_door_list = [{'start': [238, 255], 'end': [228, 244], 'mid': [233., 249.5]}]
                # print(planner_pose_inputs[0, :2]*100/5)
                #if established_graph:
                #    close_door_list = prior_door_list_store

                current_loc = planner_pose_inputs[0,
                              :2] * 100 / args.map_resolution  # add this line, means searching from robot's current location
                #f_start_time = time()
                f_list, info_gain_list, show_map, room_exp_list, door_grid = frontier_detector.frontier_detection(np.array(current_loc),
                                                                                       planner_pose_inputs[0,
                                                                                       :2] * 100 / args.map_resolution,
                                                                                       np.rint(gt_map), np.rint(gt_exp),
                                                                                       lmb[0],
                                                                                       detected_door_list,
                                                                                       laser_list
                                                                                       )  # [y, x] close_door_list
                #step_list.append(time()-f_start_time)
                primary_frontier_profile = frontier_detector.last_profile.copy()
                accumulate_timing(
                    "frontier_primary",
                    primary_frontier_profile["total_seconds"],
                )
                accumulate_timing(
                    "frontier_primary_wavefront",
                    primary_frontier_profile["wavefront_seconds"],
                )
                same_node_started = perf_counter()
                detected_door_list = topo.same_node_check(room_exp_list, detected_door_list)
                same_node_seconds = perf_counter() - same_node_started
                accumulate_timing("topology_same_node", same_node_seconds)
                # combine the node for loop case, common this if pure frontier
                topo.frontier_detector.last_profile = {}
                topology_check_started = perf_counter()
                door_list, door_remove_list = topo.check_topomap(door_list, detected_door_list,
                                                                 room_exp_list, current_loc, gt_map, gt_exp, lmb[0], door_grid, scene_idx, scene_name, laser_list)
                topology_check_seconds = perf_counter() - topology_check_started
                topology_frontier_profile = topo.frontier_detector.last_profile.copy()
                accumulate_timing("topology_check", topology_check_seconds)
                if topology_frontier_profile:
                    accumulate_timing(
                        "frontier_topology",
                        topology_frontier_profile["total_seconds"],
                    )
                    accumulate_timing(
                        "frontier_topology_wavefront",
                        topology_frontier_profile["wavefront_seconds"],
                    )

                # these two for pure frontier
                #door_list = []
                #door_remove_list = []
                # check topomap checks the detected door's relation with the current node
                add_room_started = perf_counter()
                in_point, return_flag = topo.add_room(door_list, [absolute_locs[1] * 100 / 5, absolute_locs[0] * 100 / 5],
                                         gt_map, gt_exp, lmb[0])
                add_room_seconds = perf_counter() - add_room_started
                accumulate_timing("topology_add_room", add_room_seconds)
                record_event(
                    "room_scan_profile",
                    timings_seconds={
                        "door_filter": door_filter_seconds,
                        "topology_same_node": same_node_seconds,
                        "topology_check": topology_check_seconds,
                        "topology_add_room": add_room_seconds,
                    },
                    primary_frontier=primary_frontier_profile,
                    topology_frontier=topology_frontier_profile,
                )
                in_point.reverse()  # in point is the entry of the current node, this can make sure that the searching range
                # is within the current room even if the robot is going out of the current room

                for door in detected_door_list:
                    stage3_door_map[door['start'][1], door['start'][0]] = 1
                    stage3_door_map[door['end'][1], door['end'][0]] = 1

                for door in door_remove_list:  # common this for if pure frontier

                    detected_door_list.remove(door)  # remove the door that does not enclose
                    stage3_door_map[door['end'][1], door['end'][0]] = 0
                    stage3_door_map[door['start'][1], door['start'][0]] = 0
                    #door_detect.door_remove(door)  # remove the door in door detect class

                """plt.ion()
                plt.subplot(1, 3, 1)
                plt.imshow(stage1_door_map)
                plt.subplot(1, 3, 2)
                plt.imshow(stage2_door_map)
                plt.subplot(1, 3, 3)
                plt.imshow(stage3_door_map)
                plt.show()
                plt.pause(5)
                plt.ioff()
                plt.close()"""
                print('num in stg3 {}'.format(np.sum(stage3_door_map)))

                print("return_flag {}".format(return_flag))
                if not return_flag:
                    log_num += 1
                    final_goal = None
                    shortest_dist = 10000000

                    if len(f_list) == 1:  # choose the nearest frontier to go
                        final_goal_ = f_list[0]
                        """bot_goal_dist = pu.get_l2_distance(final_goal_[1],
                                                           planner_pose_inputs[0, 1] * 100 / args.map_resolution,
                                                           final_goal_[0],
                                                           planner_pose_inputs[0, 0] * 100 / args.map_resolution)"""
                        final_goal_ = final_goal_ - origins[0, :2] * 100 / args.map_resolution
                        final_goal_ = final_goal_.astype(int).tolist()  # convert into local frame

                        if final_goal_ == last_goal:
                            final_goal = None

                        else:
                            final_goal = final_goal_
                        #if bot_goal_dist <= 30:
                        #    final_goal = None
                    elif len(f_list) == 0:
                        pass
                    else:
                        for f_idx, goal_frontier in enumerate(f_list):
                            goal_frontier = goal_frontier.astype(int).tolist()
                            info_gain = info_gain_list[f_idx]
                            bot_goal_dist = pu.get_l2_distance(goal_frontier[1],
                                                               planner_pose_inputs[0, 1] * 100 / args.map_resolution,
                                                               goal_frontier[0],
                                                               planner_pose_inputs[0, 0] * 100 / args.map_resolution)
                            #print('f_dist {}'.format(bot_goal_dist))
                            #print('info_gain {}'.format(info_gain))
                            #bot_goal_dist = bot_goal_dist/info_gain

                            w1 = 1
                            w2 = 2
                            bot_goal_dist = w1*bot_goal_dist - w2*info_gain
                            #print('cost value {}'.format(bot_goal_dist))
                            if bot_goal_dist < shortest_dist:# and bot_goal_dist > 30:  # 30 is the bot near range in frontier detection

                                final_goal_ = goal_frontier
                                final_goal_ = final_goal_ - origins[0, :2] * 100 / args.map_resolution
                                final_goal_ = final_goal_.astype(int).tolist()

                                if final_goal_ == last_goal:

                                    final_goal = None
                                else:
                                    shortest_dist = bot_goal_dist
                                    final_goal = final_goal_
                else:

                    final_goal = in_point - origins[0, :2] * 100 / args.map_resolution
                    final_goal = final_goal.astype(int).tolist()
                last_goal = (
                    final_goal.copy()
                    if final_goal is not None
                    else None
                )
                #print('cov_ ration {}'.format(cov_ratio))
                #if cov_ratio > 0.99:
                #    final_goal = None  # only use this during pure frontier method, prevent stuck

                if final_goal:
                    # print('the chosen frontier is {}[y,x]'.format(final_goal))
                    final_goal.reverse()  # [x, y]
                    dashboard_goal_xy = (
                        float(final_goal[0] + origins[0][1] * 100 / args.map_resolution),
                        float(final_goal[1] + origins[0][0] * 100 / args.map_resolution),
                    )
                    # print('final_goal{}'.format(final_goal))
                    planner_inputs_frontier = [{} for e in range(num_scenes)]
                    for e, p_input in enumerate(planner_inputs_frontier):
                        p_input['goal'] = final_goal
                        p_input['map_pred'] = gt_map_local_grid
                        p_input['exp_pred'] = gt_exp_local_grid
                        p_input['pose_pred'] = planner_pose_inputs[0]
                        p_input['mid_out'] = True
                    output = envs.get_short_term_goal(planner_inputs_frontier)
                    stg = as_numpy(output[0][1:])
                    # stg = stg.tolist()  # under local frame
                    stg_x, stg_y = stg
                    # stg_x = stg_x + origins[0][1]*100/args.map_resolution
                    # stg_y = stg_y + origins[0][0]*100/args.map_resolution
                    stg = [stg_x * args.map_resolution / 100,
                           stg_y * args.map_resolution / 100]  # [x, y]  may be should change xy
                    # print('short term goal is {}'.format(stg))
                    # del door_detect

                    # print(len(goal_list)-len(unexplorable_list))
                    # print(unexplorable_list)
                    """plt.clf()
                    plt.subplot(1, 2, 1)
                    plt.imshow(gt_map + gt_exp)
                    plt.plot(lmb[0][0], lmb[0][2], 'o', color='green')
                    plt.plot(lmb[0][1], lmb[0][2], 'o', color='green')
                    plt.plot(lmb[0][0], lmb[0][3], 'o', color='green')
                    plt.plot(lmb[0][1], lmb[0][3], 'o', color='green')
                    plt.arrow(absolute_locs[1] * 100 / 5, absolute_locs[0] * 100 / 5, dx * 8, dy * (8 * 1.25), head_width=8,
                              head_length=8 * 1.25,
                              length_includes_head=True, fc='Red', ec='Red',
                              alpha=0.9)  # absolute_locs[1]*100/5, absolute_locs[0]*100/5
                    plt.plot(final_goal[0] + origins[0][1] * 100 / 5, final_goal[1] + origins[0][0] * 100 / 5, 'o',
                             color='lime')
                    plt.plot(full_list_x, full_list_y, 'o', color='blue')
                    plt.plot((stg[0] + origins[0][1]) * 100 / 5, (stg[1] + origins[0][0]) * 100 / 5, 'o', color='aqua')
                    # np.save('list_x.npy', list_x)
                    # np.save('list_y.npy', list_y)

                    for door in door_list:
                        plt.plot([door['start'][0], door['end'][0]], [door['start'][1], door['end'][1]], color='fuchsia')
                        # plt.plot(door['end'][0], door['end'][1], 'o', color='pink')
                    for door in raw_list:
                        plt.plot(door[0], door[1], 'o', color='pink')

                    for goal in goal_list:
                        if goal not in unexplorable_list:
                            plt.plot(goal[0] + origins[0][1] * 100 / 5, goal[1] + origins[0][0] * 100 / 5, 'o', color='red')
                        else:
                            plt.plot(goal[0] + origins[0][1] * 100 / 5, goal[1] + origins[0][0] * 100 / 5, 'o',
                                     color='cyan')
                    plt.subplot(1, 2, 2)
                    plt.imshow(obs_show, cmap='gray')
                    plt.show()"""
                else:
                    stg = None
                    dashboard_goal_xy = None
                render_dashboard(
                    infos[0],
                    absolute_locs,
                    goal_xy=dashboard_goal_xy,
                    frontiers=f_list,
                )
            return locs, stg, final_goal, return_flag

        def go2goal(locs, stg, long_term_goal, first_flag, achieve_criterion=10, consider_door = False):
            # locs[y, x o]/ stg input [x, y] local, m; long term goal local pix [x,y]
            # achieve_criterion = 10
            # consider_door if true, consider door when path planning
            if stg:
                stg.reverse()
                short_term_distance = float(
                    np.linalg.norm(
                        np.asarray(stg, dtype=np.float64)
                        - np.asarray(locs[:2], dtype=np.float64)
                    )
                )
                action_value = (
                    None
                    if short_term_distance <= 1.0e-8
                    else action_generator(locs, stg)
                )
                if action_value is None and short_term_distance > 1.0e-8:
                    raise RuntimeError(
                        "Original action generator returned no action for a nonzero short-term goal"
                    )
            else:
                action_value = 0  # defualt action: turn_left
            achieve_flag = False
            if pu.get_l2_distance(long_term_goal[0], (locs[1]) * 100 / 5, long_term_goal[1], (locs[
                0]) * 100 / 5) > achieve_criterion:  # pu.get_l2_distance(long_term_goal[0], 120, long_term_goal[1], 120)
                #print('dist to goal {}'.format(pu.get_l2_distance(long_term_goal[0], (locs[1])*100/5, long_term_goal[1], (locs[0])*100/5)))
                # means not achieving the goal
                if action_value is None:
                    achieve_flag = True
                    record_event(
                        "voxroom_reachable_frontier_boundary_reached",
                        long_term_goal=long_term_goal,
                    )
                else:
                    locs, stg, long_term_goal, _ = take_action(action_value, locs, first_flag, long_term_goal, consider_door)
            else:
                achieve_flag = True
            return locs, stg, long_term_goal, achieve_flag

        def generate_return_list(visitied_list, exit_point):  # both [x, y], global, pix
            visitied_list.reverse()  # calculate from current position
            shortest_dist = 10000
            for idx, visitied_point in enumerate(visitied_list):
                dist_to_door = pu.get_l2_distance(visitied_point[0], exit_point[0], visitied_point[1], exit_point[1])
                if dist_to_door < shortest_dist:
                    shortest_dist = dist_to_door
                    shortest_idx = idx + 1
            crop_list = visitied_list[:shortest_idx]
            filter_list = []
            for i in range(len(crop_list) - 1):
                if pu.get_l2_distance(crop_list[i][0], crop_list[i + 1][0], crop_list[i][1],
                                      crop_list[i + 1][1]) > 0.18 * 100 / 5:
                    filter_list.append(crop_list[i])
            filter_list.append(exit_point)
            return filter_list

        def room_searching(locs, stg, long_term_goal, first_flag, whether_returning):
            if long_term_goal:
                achieve_flag = False
                room_search_flag = True
                step_limit = 20  # 20 for my method
                step_count = 0

                while room_search_flag:  # this is the room searching part
                    while not achieve_flag:
                        set_runtime_phase("room_search_navigation")
                        if not whether_returning:
                            locs, stg, long_term_goal, achieve_flag = go2goal(locs, stg, long_term_goal, first_flag, consider_door=True)
                        else:
                            locs, stg, long_term_goal, achieve_flag = go2goal(locs, stg, long_term_goal, first_flag,
                                                                              consider_door=False)
                        if not locs.any():
                            return locs, stg, long_term_goal
                        step_count += 1
                        # dist = pu.get_l2_distance(120, long_term_goal[0], 120, long_term_goal[1])
                        if step_count > step_limit:
                            print('failed to achieve')
                            record_event(
                                "room_search_goal_failed",
                                long_term_goal=long_term_goal,
                            )
                            step_count = 0
                            break
                        if achieve_flag:
                            print('achieved')
                            record_event(
                                "room_search_goal_reached",
                                long_term_goal=long_term_goal,
                            )
                            step_count = 0

                    achieve_flag = False
                    set_runtime_phase("room_frontier_scan")
                    locs, stg, long_term_goal, whether_returning = take_action(4, locs, first_flag)
                    if not locs.any():
                        return locs, stg, long_term_goal
                    print('long tt {}'.format(long_term_goal))
                    if not long_term_goal:
                        room_search_flag = False
                        print('room search done')
                        record_event(
                            "room_search_completed",
                            current_node_id=topo.current_node_id,
                        )
            return locs, stg, long_term_goal

        def room_moving(locs, stg, long_term_goal, first_flag):
            def global_xy(local_locs):
                return [
                    float(
                        (local_locs[1] + origins[0][1])
                        * 100
                        / args.map_resolution
                    ),
                    float(
                        (local_locs[0] + origins[0][0])
                        * 100
                        / args.map_resolution
                    ),
                ]

            achieve_flag = False
            set_runtime_phase("topology_exit_selection")
            exit_goal_list = topo.choose_door([(locs[0] + origins[0][0]) * 100 / args.map_resolution,
                                               (locs[1] + origins[0][1]) * 100 / args.map_resolution])
            # return_list = generate_return_list(visited_waypoint, exit_goal)
            print('exit_goal {}'.format(exit_goal_list))
            return_step = 0
            return_threshold = 100#60
            reached_exit_count = 0
            transition_trajectories = []
            for exit_goal in exit_goal_list:
                segment_trajectory = [global_xy(locs)]
                # for return_waypoint in return_list[1:]:
                long_term_goal = [exit_goal[0] - origins[0][1] * 100 / args.map_resolution,
                                  exit_goal[1] - origins[0][0] * 100 / args.map_resolution]  # convert to local frame
                # print(long_term_goal)
                stg = [long_term_goal[0] * 5 / 100, long_term_goal[1] * 5 / 100]
                while not achieve_flag:
                    set_runtime_phase("room_transition_navigation")
                    locs, stg, long_term_goal, achieve_flag = go2goal(locs, stg, long_term_goal, first_flag,
                                                                      achieve_criterion=5)  # default is 5
                    if not locs.any():
                        transition_trajectories.append(segment_trajectory)
                        return (
                            locs,
                            stg,
                            long_term_goal,
                            len(exit_goal_list),
                            reached_exit_count,
                            transition_trajectories,
                        )
                    segment_trajectory.append(global_xy(locs))
                    return_step += 1
                    # dist = pu.get_l2_distance(120, long_term_goal[0], 120, long_term_goal[1])
                    if return_step > return_threshold:
                        print('failed to achieve')
                        return_step = 0
                        break
                    if achieve_flag:
                        print('achieved')
                        reached_exit_count += 1
                        record_event(
                            "topology_exit_waypoint_reached",
                            exit_goal=exit_goal,
                        )
                        return_step = 0
                achieve_flag = False
                transition_trajectories.append(segment_trajectory)
            return (
                locs,
                stg,
                long_term_goal,
                len(exit_goal_list),
                reached_exit_count,
                transition_trajectories,
            )

        def exploration(locs):
            global t_start
            t_start = time()

            def abort(reason):
                runtime_state["completion_reason"] = str(reason)
                set_runtime_phase("aborted", reason)
                record_event("exploration_aborted", reason=reason)
                raise RuntimeError(
                    "Topology exploration aborted before completion: {}".format(
                        reason
                    )
                )

            print('locs in exp {}'.format(locs))
            # auto exploration stage

            first_flag = True  # to check whether it is the first scan
            # take the initialization step, first we scan the surrounding
            set_runtime_phase("initial_scan")
            record_event("exploration_started")
            locs, stg, long_term_goal, whether_returning = take_action(4, locs, first_flag)  # long term goal is under local frame

            if not locs.any():
                abort("initial_scan_exhausted")
            print('first long term goal {}'.format(long_term_goal))
            first_flag = False
            print(topo.stop_exp())
            while not topo.stop_exp():

                # first searching the current room
                locs, stg, long_term_goal = room_searching(locs, stg, long_term_goal, first_flag, whether_returning)
                if not locs.any():
                    abort("room_search_exhausted")
                # here is the room to room moving part
                (
                    locs,
                    stg,
                    long_term_goal,
                    exit_goal_count,
                    reached_exit_count,
                    transition_trajectories,
                ) = room_moving(
                    locs,
                    stg,
                    long_term_goal,
                    first_flag,
                )
                if not locs.any():
                    abort("room_transition_exhausted")
                set_runtime_phase("transition_confirmation_scan")
                locs, stg, long_term_goal, whether_returning = take_action(4, locs, first_flag)
                if not locs.any():
                    abort("transition_confirmation_scan_exhausted")
                transition_confirmed, transition_evidence = (
                    topo.confirm_pending_transition(
                        transition_trajectories,
                        reached_exit_count,
                    )
                )
                if exit_goal_count > 0 and transition_confirmed:
                    print('room transition confirmed')
                    record_event(
                        "room_transition_confirmed",
                        exit_goal_count=exit_goal_count,
                        reached_exit_count=reached_exit_count,
                        current_node_id=topo.current_node_id,
                        confirmation_method="trajectory_geometry",
                        evidence=transition_evidence,
                    )
                elif exit_goal_count > 0:
                    print('room transition not confirmed')
                    record_event(
                        "room_transition_not_confirmed",
                        exit_goal_count=exit_goal_count,
                        reached_exit_count=reached_exit_count,
                        current_node_id=topo.current_node_id,
                        confirmation_method="trajectory_geometry",
                        evidence=transition_evidence,
                    )
                else:
                    print('no room transition: topology has no exit goal')
                    record_event(
                        "room_transition_skipped_no_exit",
                        current_node_id=topo.current_node_id,
                    )
            t_end = time()
            print('time cost {}'.format(t_end-t_start))
            runtime_state["topology_exploration_complete_step"] = action_count
            runtime_state["completion_reason"] = "topology_exploration_completed"
            set_runtime_phase("topology_complete")
            record_event(
                "topology_exploration_completed",
                topology=topo.snapshot(),
                elapsed_seconds=t_end - t_start,
            )
            if args.pad_episode_to_max_steps:
                set_runtime_phase("episode_tail")
                while action_count < args.max_episode_length:
                    locs, stg, long_term_goal, _ = take_action(
                        0,
                        locs,
                        first_flag,
                        [
                            args.map_size_cm / 20,
                            args.map_size_cm / 20,
                        ],
                    )
            set_runtime_phase("completed")
            record_event("episode_control_completed", executed_steps=action_count)
            return runtime_state["completion_reason"]

        completion_reason = exploration(locs)
        if completion_reason != "topology_exploration_completed":
            raise RuntimeError("Exploration returned without a completion reason")

        topology_snapshot = topo.snapshot()
        record_event(
            "run_completed",
            executed_steps=int(last_episode_action_count or action_count),
            completion_reason=completion_reason,
            topology=topology_snapshot,
        )
        if voxroom_sidecar is not None:
            runtime_state["voxroom_result"] = voxroom_sidecar.close()
        if dashboard is not None:
            if (
                runtime_state["last_dashboard_info"] is None
                or runtime_state["last_dashboard_absolute_locs"] is None
            ):
                raise RuntimeError("The live dashboard never rendered a control step")
            render_dashboard(
                runtime_state["last_dashboard_info"],
                runtime_state["last_dashboard_absolute_locs"],
            )
        event_summary = event_recorder.summary()
        topology_transition_count = int(
            event_summary["event_counts"].get("room_transition_confirmed", 0)
        )
        door_crossing_count = int(
            event_summary["event_counts"].get("door_crossing_confirmed", 0)
        )
        surviving_door_crossing_count = surviving_crossing_count(
            confirmed_crossing_evidence,
            topology_snapshot,
        )
        if topology_transition_count > 0 and surviving_door_crossing_count > 0:
            topology_status = "cross_room_verified"
        elif topology_snapshot["edge_count"] > 0:
            topology_status = "topology_built_no_confirmed_crossing"
        else:
            topology_status = "single_room_no_exit"
        topology_artifact = {
            "run_id": args.run_id,
            "process_id": os.getpid(),
            "source_commit": source_commit,
            "actual_episode_id": actual_episode_id,
            "actual_episode_contract_sha256": actual_episode_contract_sha256,
            "actual_scene_id": actual_scene_id,
            "task_config": args.task_config,
            "split": args.split,
            "runtime_install_sha256": runtime_install_sha256,
            "completion_reason": completion_reason,
            "status": topology_status,
            "transition_count": topology_transition_count,
            "door_crossing_count": door_crossing_count,
            "surviving_door_crossing_count": surviving_door_crossing_count,
            "require_topology_transition": bool(args.require_topology_transition),
            "requirement_met": (
                not bool(args.require_topology_transition)
                or (
                    topology_transition_count > 0
                    and surviving_door_crossing_count > 0
                )
            ),
            "topology_exploration_complete_step": runtime_state[
                "topology_exploration_complete_step"
            ],
            "snapshot": topology_snapshot,
            "events": event_summary,
        }
        write_json_atomic(run_dir / "topology_final.json", topology_artifact)
        visualization_manifest = (
            dashboard.finalize(topology_snapshot, event_summary)
            if dashboard is not None
            else {"frame_count": 0}
        )
        executed_steps = int(last_episode_action_count or action_count)
        topology_complete_step = runtime_state["topology_exploration_complete_step"]
        if topology_complete_step is None:
            raise RuntimeError("Topology completion step was not recorded")
        summary = {
            "status": "completed",
            "run_id": args.run_id,
            "process_id": os.getpid(),
            "started_at_unix": started_at,
            "finished_at_unix": time(),
            "scene_name": scene_name,
            "episode_id": actual_episode_id,
            "episode_contract_sha256": actual_episode_contract_sha256,
            "requested_max_episode_steps": int(args.max_episode_length),
            "source_commit": source_commit,
            "executed_steps": executed_steps,
            "completion_reason": completion_reason,
            "pad_episode_to_max_steps": bool(args.pad_episode_to_max_steps),
            "require_topology_transition": bool(args.require_topology_transition),
            "task_config": args.task_config,
            "split": args.split,
            "runtime_install_sha256": runtime_install_sha256,
            "explored_ratio": float(last_episode_cov_ratio or cov_ratio),
            "explored_area": float(last_episode_cov_area or cov_area),
            "visualization": bool(args.visualize),
            "window_title": args.window_title,
            "detector_device": args.detector_device,
            "detr_source_dir": args.detr_source_dir,
            "progress_file": progress_path.name,
            "topology_file": "topology_final.json",
            "topology_event_file": event_recorder.path.name,
            "topology_status": topology_status,
            "topology_room_count": topology_snapshot["room_count"],
            "topology_edge_count": topology_snapshot["edge_count"],
            "topology_transition_count": topology_transition_count,
            "door_crossing_count": door_crossing_count,
            "surviving_door_crossing_count": surviving_door_crossing_count,
            "topology_requirement_met": topology_artifact["requirement_met"],
            "topology_exploration_steps": int(topology_complete_step),
            "episode_tail_steps": int(max(0, executed_steps - topology_complete_step)),
            "visualization_manifest": "visualization_manifest.json",
            "visualization_frame_count": int(
                visualization_manifest.get("frame_count", 0)
            ),
            "visualization_final": visualization_manifest.get("final_image"),
            "runtime_timing": timing_summary(),
            "voxroom_sidecar": bool(voxroom_sidecar is not None),
            "voxroom_result": runtime_state["voxroom_result"],
            "run_context": run_context,
            "x11_client_window_id": x11_client_window_id,
        }
        if run_context is not None:
            if live_figure is None or x11_client_window_id is None:
                raise RuntimeError("Strict run has no terminal dashboard to capture")
            terminal_ready_path = run_dir / "terminal_capture_ready.json"
            terminal_ack_path = run_dir / "terminal_capture_ack.txt"
            final_frame_path = run_dir / visualization_manifest["final_image"]
            if (
                dashboard.last_render_step != executed_steps
                or not final_frame_path.is_file()
            ):
                raise RuntimeError(
                    "Strict terminal capture has no final rendered frame"
                )
            write_json_atomic(
                terminal_ready_path,
                {
                    "run_id": args.run_id,
                    "process_id": os.getpid(),
                    "window_id": x11_client_window_id,
                    "step": executed_steps,
                    "render_step": dashboard.last_render_step,
                    "frame_file": final_frame_path.relative_to(
                        run_dir
                    ).as_posix(),
                    "frame_sha256": sha256(final_frame_path),
                    "completion_reason": completion_reason,
                },
            )
            terminal_deadline = time() + 120.0
            while not terminal_ack_path.is_file():
                if time() >= terminal_deadline:
                    raise TimeoutError(
                        "Timed out waiting for the terminal dashboard capture"
                    )
                live_figure.canvas.flush_events()
                plt.pause(0.05)
            if terminal_ack_path.read_text(encoding="utf-8").strip() != args.run_id:
                raise RuntimeError("Terminal dashboard capture acknowledgement mismatch")
        write_json_atomic(run_dir / "result.json", summary)
        envs.close()
        plt.close("all")




        # plt.close()

        """if action_count != 0:
            locs = np.array([6, 6, 0])
            while locs.any():
                print('action_ {}'.format(action_count))
                locs, _, _ = take_action(4, locs, False)"""
    """while keep_exploring:
        keystroke = input('input action command: ')#cv.waitKey(0)
        if keystroke == 'w':
            action_value = 2
            #locs, origins = take_action(action_value, locs, origins)
            locs = take_action(action_value, locs, first_flag)
            #first_flag = False
        elif keystroke == 'a':
            action_value = 0
            locs = take_action(action_value, locs, first_flag)
            #first_flag = False
        elif keystroke == 'd':
            action_value = 1
            locs = take_action(action_value, locs, first_flag)
            #first_flag = False
        elif keystroke == 'f':
            keep_exploring = False
        elif keystroke == 's':
            action_value = 4
            locs = take_action(action_value, locs, first_flag)
            first_flag = False"""

    # exit()


if __name__ == "__main__":
    main()
