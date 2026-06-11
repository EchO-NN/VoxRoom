from __future__ import annotations

import numpy as np

from voxroom_online.isaac_runtime.baselines.mask_io import enforce_room_mask_contract


def test_baseline_relabel_and_domain_clip():
    domain = np.zeros((5, 6), dtype=bool)
    domain[:, :3] = True
    arrays = {
        "occupancy_map": np.zeros((5, 6), dtype=bool),
        "observed_free_mask": domain.copy(),
        "obstacle_mask": np.zeros((5, 6), dtype=bool),
        "unknown_mask": ~domain,
        "navigation_free_room_domain": domain,
    }
    labels = np.zeros((5, 6), dtype=np.int32)
    labels[:2, :3] = 5
    labels[2:, :3] = 9
    labels[:, 3:] = 12
    out = enforce_room_mask_contract(labels, arrays)
    assert set(np.unique(out).tolist()) == {0, 1, 2}
    assert not np.any(out[:, 3:])

