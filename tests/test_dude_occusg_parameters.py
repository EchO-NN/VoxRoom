import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from voxroom_online.isaac_runtime.baselines.data_contract import resolve_map_info
from voxroom_online.isaac_runtime.baselines.ros_grid_io import snapshot_to_ros_occupancy_grid
from voxroom_online.isaac_runtime.baselines.offline.dude_runner import DudeIncrementalRunner, build_dude_rosrun_shell
from voxroom_online.isaac_runtime.baselines.ros_subprocess import RosSubprocessConfig

spec = importlib.util.spec_from_file_location("dude_tau15", Path(__file__).parents[1]/"scripts/run_dude_occusg_parameters.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def test_only_explicit_experiment_changes_tau():
    assert experiment.TAU == 1.5
    assert DudeIncrementalRunner(fallback_python=True).concavity_threshold_m == 2.5
    assert build_dude_rosrun_shell(RosSubprocessConfig(), concavity_threshold_m=experiment.TAU).splitlines()[-1] == "exec rosrun inc_dude inc_dude 1.5"


def data():
    free = np.zeros((18, 26), bool)
    free[3:10, 13:22] = True
    wall = np.zeros_like(free)
    wall[2, 13:22] = True
    return dict(voxel_vertical_free_xy=free, voxel_wall_xy=wall, occupancy_map=np.zeros_like(free),
                observed_free_mask=free, map_resolution_m=.05, map_origin_x_m=-2., map_origin_y_m=1.,
                map_width_cells=26, map_height_cells=18, step=5)


def test_input_mask_and_geometry_identical_after_minimal_loading(tmp_path):
    arrays = data()
    source = tmp_path/"snapshot.npz"
    np.savez(source, **arrays, final_room_label_map=np.full((18,26), 99),
             voxel_occupancy_state_zyx=np.ones((8,18,26)), voxel_door_raw_seed_mask=np.ones((18,26)))
    original = tmp_path/"old.npz"
    metadata = {"map_info": resolve_map_info(snapshot_arrays=arrays).to_metadata(),
                "parameters": {"concavity_threshold_m": 2.5}, "segmentation_input_mode": "raw_vertical_free"}
    expected = snapshot_to_ros_occupancy_grid(arrays)
    np.savez(original, dude_ros_occupancy_grid=expected, baseline_metadata_json=json.dumps(metadata))
    minimal = experiment.load_input(source)
    assert not {"final_room_label_map", "voxel_occupancy_state_zyx", "voxel_door_raw_seed_mask"} & set(minimal)
    grid, checksum = experiment.assert_identical_input(minimal, original)
    np.testing.assert_array_equal(grid, expected)
    assert len(checksum) == 64
    minimal["voxel_vertical_free_xy"][4,14] = False
    with pytest.raises(AssertionError):
        experiment.assert_identical_input(minimal, original)


def test_same_pixels_but_changed_origin_rejected(tmp_path):
    arrays = data()
    metadata = {"map_info": resolve_map_info(snapshot_arrays=arrays).to_metadata(),
                "parameters": {"concavity_threshold_m": 2.5}, "segmentation_input_mode": "raw_vertical_free"}
    original = tmp_path/"old.npz"
    np.savez(original, dude_ros_occupancy_grid=snapshot_to_ros_occupancy_grid(arrays), baseline_metadata_json=json.dumps(metadata))
    arrays["map_origin_x_m"] = -1.95
    with pytest.raises(AssertionError):
        experiment.assert_identical_input(arrays, original)
