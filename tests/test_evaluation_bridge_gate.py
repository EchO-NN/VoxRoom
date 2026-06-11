from __future__ import annotations

import json

import numpy as np
import pytest

from voxroom_online.isaac_runtime.baselines.evaluation_bridge import link_baseline_run_root


def test_evaluation_bridge_rejects_non_main_fallback(tmp_path):
    snap_dir = tmp_path / "run" / "kujiale_a" / "baselines" / "method_a" / "roomseg_snapshots"
    snap_dir.mkdir(parents=True)
    np.savez_compressed(
        snap_dir / "roomseg_step_000001.npz",
        occupancy_map=np.zeros((3, 4), dtype=bool),
        final_room_label_map=np.zeros((3, 4), dtype=np.int32),
        observed_free_mask=np.ones((3, 4), dtype=bool),
        obstacle_mask=np.zeros((3, 4), dtype=bool),
        unknown_mask=np.zeros((3, 4), dtype=bool),
        navigation_free_room_domain=np.ones((3, 4), dtype=bool),
        baseline_metadata_json=np.asarray(
            json.dumps({"runner_type": "python_fallback", "main_experiment_allowed": False})
        ),
    )
    with pytest.raises(ValueError, match="non-main"):
        link_baseline_run_root(
            source_run_root=tmp_path / "run",
            baseline="method_a",
            linked_run_root=tmp_path / "linked",
        )
    linked = link_baseline_run_root(
        source_run_root=tmp_path / "run",
        baseline="method_a",
        linked_run_root=tmp_path / "linked",
        allow_non_main=True,
    )
    assert linked

