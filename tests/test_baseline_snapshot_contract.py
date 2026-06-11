from __future__ import annotations

import json

import numpy as np

from voxroom_online.isaac_runtime.baselines.mask_io import save_baseline_snapshot_npz
from voxroom_online.isaac_runtime.debug.roomseg_layer_dump import ROOMSEG_SNAPSHOT_ARRAY_KEYS, save_roomseg_layer_dump


def test_baseline_snapshot_contract(tmp_path):
    source = tmp_path / "roomseg_step_000001.npz"
    np.savez_compressed(
        source,
        occupancy_map=np.zeros((10, 12), np.uint8),
        observed_free_mask=np.ones((10, 12), bool),
        obstacle_mask=np.zeros((10, 12), bool),
        unknown_mask=np.zeros((10, 12), bool),
        navigation_free_room_domain=np.ones((10, 12), bool),
        final_room_label_map=np.zeros((10, 12), np.int32),
    )
    label = np.zeros((10, 12), np.int32)
    label[:, :6] = 10
    label[:, 6:] = 20
    out = tmp_path / "out.npz"
    save_baseline_snapshot_npz(
        source_npz_path=source,
        output_npz_path=out,
        baseline_label_map=label,
        baseline_name="toy",
        metadata={"runner_type": "unit"},
    )
    data = np.load(out, allow_pickle=False)
    assert data["final_room_label_map"].dtype == np.int32
    assert set(np.unique(data["final_room_label_map"]).tolist()) == {1, 2}
    assert "voxroom_final_room_label_map" in data.files
    meta = json.loads(str(data["baseline_metadata_json"]))
    assert meta["runner_type"] == "unit"


def test_roomseg_snapshot_whitelist_includes_selected_frontier_center(tmp_path):
    dump = save_roomseg_layer_dump(
        out_dir=tmp_path,
        step=7,
        room_debug={"final_room_label_map": np.ones((4, 5), dtype=np.int32)},
        occupancy_map=np.zeros((4, 5), dtype=bool),
        observed_free_mask=np.ones((4, 5), dtype=bool),
        obstacle_mask=np.zeros((4, 5), dtype=bool),
        unknown_mask=np.zeros((4, 5), dtype=bool),
        frontier_map=np.zeros((4, 5), dtype=bool),
        selected_frontier_center_rc=(2, 3),
        agent_rc=(1, 1),
        npz_keys=ROOMSEG_SNAPSHOT_ARRAY_KEYS,
        save_npz=True,
        save_png=False,
        save_summary_json=False,
    )
    arrays = np.load(dump["paths"]["npz"], allow_pickle=False)
    assert "selected_frontier_center_rc" in arrays.files
    assert arrays["selected_frontier_center_rc"].tolist() == [2, 3]

