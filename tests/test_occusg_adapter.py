"""Data-boundary tests; not replacements for the upstream native tests."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("shapely", minversion="2.0")
spec = importlib.util.spec_from_file_location("occusg_adapter", Path(__file__).parents[1] / "scripts/run_occusg_saved_voxels.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def test_world_polygon_source_grid_without_y_flip():
    labels = adapter.rasterize([[(2, -1), (4, -1), (4, 0), (2, 0)]], (8, 10), [1, -2], .5)
    expected = np.zeros((8, 10), np.int32)
    expected[2:4, 2:6] = 1
    np.testing.assert_array_equal(labels, expected)


def test_two_adjacent_rooms_do_not_merge():
    labels = adapter.rasterize([[(0, 0), (1, 0), (1, 2), (0, 2)], [(1, 0), (2, 0), (2, 2), (1, 2)]], (4, 4), [0, 0], .5)
    np.testing.assert_array_equal(labels[:, :2], np.ones((4, 2), np.int32))
    np.testing.assert_array_equal(labels[:, 2:], np.full((4, 2), 2, np.int32))


def test_encoder_whitelisted_fields_and_all_heights(tmp_path, monkeypatch):
    state = np.zeros((12, 8, 10), np.uint8)
    state[1, 2, 3], state[11, 7, 9] = 1, 2
    source = tmp_path / "snapshot.npz"
    np.savez(source, voxel_occupancy_state_zyx=state, voxel_occupancy_z_centers_m=np.arange(12)*.05+.025,
             map_resolution_m=.05, map_origin_x_m=-.1, map_origin_y_m=-.2, demo_pose_world=[0., 0., .05, 0.],
             final_room_label_map=np.full((8, 10), 7654), voxel_door_raw_seed_mask=np.ones((8, 10)))
    calls = []
    monkeypatch.setattr(adapter.subprocess, "run", lambda args, **kwargs: calls.append(args))
    info = adapter.encode_snapshot(source, tmp_path)
    assert len(calls) == 1
    assert info["state_counts"] == [958, 1, 1, 0]
    points = np.fromfile(tmp_path / "known_voxels.bin", dtype="<f4").reshape(-1, 4)
    np.testing.assert_allclose(points, [[.075, -.075, .075, 1], [.375, .175, .575, 2]], atol=1e-7)


def test_conflicts_rejected_not_silently_free(tmp_path, monkeypatch):
    source = tmp_path / "snapshot.npz"
    np.savez(source, voxel_occupancy_state_zyx=np.array([[[3]]], np.uint8), voxel_occupancy_z_centers_m=[.025],
             map_resolution_m=.05, map_origin_x_m=0., map_origin_y_m=0., demo_pose_world=[0., 0., 0., 0.])
    with pytest.raises(ValueError, match="conflicting"):
        adapter.encode_snapshot(source, tmp_path)


def vertical_fixture(tmp_path):
    from voxroom_online.isaac_runtime.baselines.data_contract import resolve_map_info
    free = np.zeros((8, 10), bool)
    free[2:6, 3:9] = True
    wall = np.zeros_like(free)
    wall[1, 3:9] = True
    # Free must take precedence over the original overlapping wall mask.
    wall[3, 4] = True
    arrays = dict(voxel_vertical_free_xy=free, voxel_wall_xy=wall, occupancy_map=np.zeros_like(free),
                  map_resolution_m=.05, map_origin_x_m=-.2, map_origin_y_m=.4,
                  map_height_cells=8, map_width_cells=10, demo_pose_world=[0., .5, .8, .1])
    expected = np.full(free.shape, -1, np.int8)
    expected[wall & ~free] = 100
    expected[free] = 0
    source, reference = tmp_path / "source.npz", tmp_path / "reference.npz"
    # Conflicting distractors must never enter input construction.
    np.savez(source, **arrays, final_room_label_map=np.full(free.shape, 99),
             voxel_nav_free_xy=~free, roomseg_eval_reference_explorable_mask=np.zeros_like(free),
             voxel_occupancy_state_zyx=np.full((3, 8, 10), 3), voxel_door_raw_seed_mask=np.ones_like(free))
    np.savez(reference, dude_ros_occupancy_grid=expected,
             baseline_metadata_json=json.dumps({"map_info": resolve_map_info(snapshot_arrays=arrays).to_metadata()}))
    return source, reference, expected


def test_vertical_input_preserves_raw_grid_ignores_predictions_and_gt(tmp_path):
    source, reference, expected = vertical_fixture(tmp_path)
    grid, info = adapter.encode_vertical_snapshot(source, reference)
    np.testing.assert_array_equal(grid, expected)
    assert info["origin"] == [-.2, .4]
    assert info["pose"] == [0., .5, .8, .1]
    assert info["input_identical_to_dude"] is True


def test_vertical_input_refuses_different_grid(tmp_path):
    source, reference, expected = vertical_fixture(tmp_path)
    with np.load(reference) as data:
        metadata = str(data["baseline_metadata_json"])
    expected[0, 0] = 0
    np.savez(reference, dude_ros_occupancy_grid=expected, baseline_metadata_json=metadata)
    with pytest.raises(AssertionError):
        adapter.encode_vertical_snapshot(source, reference)


def test_vertical_input_refuses_different_origin(tmp_path):
    source, reference, expected = vertical_fixture(tmp_path)
    with np.load(reference) as data:
        metadata = json.loads(str(data["baseline_metadata_json"]))
    metadata["map_info"]["min_x"] += .05
    np.savez(reference, dude_ros_occupancy_grid=expected, baseline_metadata_json=json.dumps(metadata))
    with pytest.raises(AssertionError):
        adapter.encode_vertical_snapshot(source, reference)


def test_ros_vertical_roundtrip_preserves_unknown_and_y_direction(tmp_path):
    source, reference, expected = vertical_fixture(tmp_path)
    grid, geometry = adapter.encode_vertical_snapshot(source, reference)
    message = SimpleNamespace(data=grid.ravel().tolist(), info=SimpleNamespace(
        height=8, width=10, resolution=float(np.float32(.05)), origin=SimpleNamespace(
            position=SimpleNamespace(x=-.2, y=.4), orientation=SimpleNamespace(w=1.))))
    adapter.assert_received_vertical_grid(message, expected, geometry)
    message.data = np.flipud(grid).ravel().tolist()
    with pytest.raises(AssertionError):
        adapter.assert_received_vertical_grid(message, expected, geometry)


def test_crop_preserves_known_cells_and_internal_unknown():
    full = np.full((81, 91), -1, dtype=np.int8)
    full[21:34, 41:60] = 0
    full[21, 41:60] = 100
    full[27:29, 47:50] = -1  # interior hole must remain unknown
    geometry = {"shape": list(full.shape), "resolution": .05, "origin": [-2.5, 3.1], "pose": [0., 0., .5, 0.]}
    cropped, info = adapter.crop_unknown_border(full, geometry)
    np.testing.assert_array_equal(cropped, full[21:34, 41:60])
    assert info["crop_bounds_yxyx"] == [21, 41, 34, 60]
    assert info["source_shape"] == [81, 91]
    assert info["source_origin"] == [-2.5, 3.1]
    np.testing.assert_allclose(info["origin"], [-.45, 4.15], rtol=0, atol=1e-12)
    assert (cropped[6:8, 6:9] == -1).all()
    assert (cropped == 0).sum() == (full == 0).sum()
    assert (cropped == 100).sum() == (full == 100).sum()
    assert info["removed_known_cells"] == 0
    assert info["unknown_ratio_after_crop"] < info["unknown_ratio_before_crop"]
    assert geometry["origin"] == [-2.5, 3.1]  # input object not modified


def test_crop_keeps_isolated_edge_obstacles():
    full = np.full((12, 16), -1, dtype=np.int8)
    full[5:8, 6:10] = 0
    full[0, 0] = full[-1, -1] = 100
    cropped, info = adapter.crop_unknown_border(full, {"shape": [12, 16], "origin": [0., 0.], "resolution": .05})
    np.testing.assert_array_equal(cropped, full)
    assert info["removed_unknown_cells"] == 0


def test_crop_world_polygons_still_rasterize_on_full_evaluation_grid():
    full = np.full((20, 24), -1, dtype=np.int8)
    full[6:14, 10:18] = 0
    geometry = {"shape": list(full.shape), "origin": [-.5, 2.], "resolution": .05}
    cropped, info = adapter.crop_unknown_border(full, geometry)
    ox, oy = info["origin"]
    polygon = [(ox, oy), (ox+.4, oy), (ox+.4, oy+.4), (ox, oy+.4)]
    labels = adapter.rasterize([polygon], info["source_shape"], info["source_origin"], .05)
    np.testing.assert_array_equal(labels > 0, full == 0)
    assert labels.shape != cropped.shape


def test_entirely_unknown_input_is_not_fabricated_as_free():
    with pytest.raises(ValueError, match="entirely unknown"):
        adapter.crop_unknown_border(np.full((3, 4), -1, np.int8), {"shape": [3, 4], "origin": [0., 0.], "resolution": .05})
