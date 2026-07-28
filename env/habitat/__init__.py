# Parts of the code in this file have been borrowed from:
#    https://github.com/facebookresearch/habitat-api

import numpy as np
import torch
from habitat.config.default import get_config as cfg_env
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1

from .exploration_env import Exploration_Env
from .vector_env import VectorEnv



def make_env_fn(args, config_env, rank):
    dataset = PointNavDatasetV1(config_env.DATASET)
    config_env.defrost()
    config_env.SIMULATOR.SCENE = dataset.episodes[0].scene_id
    print("Loading {}".format(config_env.SIMULATOR.SCENE))
    config_env.freeze()

    env = Exploration_Env(
        args=args,
        rank=rank,
        config_env=config_env,
        dataset=dataset,
    )

    env.seed(rank)
    return env


def construct_envs(args):
    env_configs = []
    args_list = []

    basic_config = cfg_env(config_paths=
                           ["env/habitat/habitat_api/configs/" + args.task_config])  # ["env/habitat/habitat_api/configs/" + args.task_config]

    basic_config.defrost()
    basic_config.DATASET.SPLIT = args.split
    if args.run_context_dataset:
        basic_config.DATASET.DATA_PATH = args.run_context_dataset
    basic_config.freeze()

    scenes = PointNavDatasetV1.get_scenes_to_load(basic_config.DATASET)

    if len(scenes) > 0:
        assert len(scenes) >= args.num_processes, (
            "reduce the number of processes as there "
            "aren't enough number of scenes"
        )
        scene_split_size = int(np.floor(len(scenes) / args.num_processes))

    for i in range(args.num_processes):
        config_env = cfg_env(config_paths=
                             ["env/habitat/habitat_api/configs/" + args.task_config])
        config_env.defrost()
        if args.run_context_dataset:
            config_env.DATASET.DATA_PATH = args.run_context_dataset

        if len(scenes) > 0:
            config_env.DATASET.CONTENT_SCENES = scenes[
                                                i * scene_split_size: (i + 1) * scene_split_size
                                                ]

        if i < args.num_processes_on_first_gpu:
            gpu_id = 0
        else:
            gpu_id = int((i - args.num_processes_on_first_gpu)
                         // args.num_processes_per_gpu) + args.sim_gpu_id
        gpu_id = min(torch.cuda.device_count() - 1, gpu_id)
        config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu_id

        agent_sensors = []
        agent_sensors.append("RGB_SENSOR")
        agent_sensors.append("DEPTH_SENSOR")

        #agent_sensors.append("SEMANTIC_SENSOR")  #boggy add this

        config_env.SIMULATOR.AGENT_0.SENSORS = agent_sensors

        config_env.ENVIRONMENT.MAX_EPISODE_STEPS = args.max_episode_length
        config_env.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False

        config_env.SIMULATOR.RGB_SENSOR.WIDTH = args.env_frame_width
        config_env.SIMULATOR.RGB_SENSOR.HEIGHT = args.env_frame_height
        config_env.SIMULATOR.RGB_SENSOR.HFOV = args.hfov
        config_env.SIMULATOR.RGB_SENSOR.POSITION = [0, args.camera_height, 0]#args.camera_height

        config_env.SIMULATOR.DEPTH_SENSOR.WIDTH = args.env_frame_width
        config_env.SIMULATOR.DEPTH_SENSOR.HEIGHT = args.env_frame_height
        config_env.SIMULATOR.DEPTH_SENSOR.HFOV = args.hfov
        config_env.SIMULATOR.DEPTH_SENSOR.POSITION = [0, args.camera_height, 0]#args.camera_height

        """config_env.SIMULATOR.SEMANTIC_SENSOR.WIDTH = args.env_frame_width
        config_env.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = args.env_frame_height
        config_env.SIMULATOR.SEMANTIC_SENSOR.HFOV = args.hfov
        config_env.SIMULATOR.SEMANTIC_SENSOR.POSITION = [0, args.camera_height, 0]  # args.camera_height"""

        config_env.SIMULATOR.TURN_ANGLE = 30#10
        config_env.DATASET.SPLIT = args.split

        config_env.freeze()

        env_configs.append(config_env)

        args_list.append(args)

    envs = VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(
            tuple(
                zip(args_list, env_configs, range(args.num_processes))
            )
        ),
        auto_reset_done=False,
    )

    return envs
