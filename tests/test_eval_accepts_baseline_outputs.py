from __future__ import annotations

import numpy as np

from voxroom_online.isaac_runtime.baselines.mask_io import save_baseline_snapshot_npz
from voxroom_online.isaac_runtime.evaluation.online_roomseg.snapshot_io import load_snapshot_arrays


def test_eval_accepts_baseline_outputs(tmp_path):
    source = tmp_path / "roomseg_step_000003.npz"
    domain = np.ones((6, 7), dtype=bool)
    np.savez_compressed(
        source,
        occupancy_map=np.zeros((6, 7), dtype=bool),
        observed_free_mask=domain,
        obstacle_mask=np.zeros((6, 7), dtype=bool),
        unknown_mask=np.zeros((6, 7), dtype=bool),
        navigation_free_room_domain=domain,
        final_room_label_map=np.zeros((6, 7), dtype=np.int32),
    )
    labels = np.ones((6, 7), dtype=np.int32)
    out = tmp_path / "baseline" / "roomseg_step_000003.npz"
    save_baseline_snapshot_npz(
        source_npz_path=source,
        output_npz_path=out,
        baseline_label_map=labels,
        baseline_name="toy",
        metadata={"runner_type": "unit"},
    )
    snap = load_snapshot_arrays(out)
    assert snap.final_room_label_map.dtype == np.int32
    assert snap.shape == (6, 7)
    assert snap.domain_key == "navigation_free_room_domain"

