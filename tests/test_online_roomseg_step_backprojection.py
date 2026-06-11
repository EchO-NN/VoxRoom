from __future__ import annotations

from pathlib import Path

import numpy as np

from voxroom_online.isaac_runtime.evaluation.online_roomseg.snapshot_io import load_snapshot_arrays
from voxroom_online.isaac_runtime.evaluation.online_roomseg.step_backprojection import backproject_final_gt_to_snapshot


def _snapshot(path: Path, domain: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        final_room_label_map=np.ones(domain.shape, dtype=np.int32),
        navigation_free_room_domain=np.zeros(domain.shape, dtype=bool),
        observed_free_mask=np.asarray(domain, dtype=bool),
        obstacle_mask=np.zeros(domain.shape, dtype=bool),
        unknown_mask=np.zeros(domain.shape, dtype=bool),
    )


def test_backproject_keeps_only_visible_left_room(tmp_path: Path) -> None:
    final = np.zeros((4, 4), dtype=np.int32)
    final[:, :2] = 1
    final[:, 2:] = 2
    domain = np.zeros((4, 4), dtype=bool)
    domain[:, :2] = True
    path = tmp_path / "roomseg_step_000010.npz"
    _snapshot(path, domain)
    snapshot = load_snapshot_arrays(path)

    result = backproject_final_gt_to_snapshot(final, snapshot, episode_uid="ep", source_final_step=20)

    assert result.metadata["visible_gt_room_count"] == 1
    assert sorted(int(v) for v in np.unique(result.label_map) if int(v) > 0) == [1]
    assert int(np.count_nonzero(result.label_map == 1)) == 8


def test_backproject_keeps_two_rooms_when_each_has_visible_cell(tmp_path: Path) -> None:
    final = np.zeros((4, 4), dtype=np.int32)
    final[:, :2] = 1
    final[:, 2:] = 2
    domain = np.zeros((4, 4), dtype=bool)
    domain[0, 0] = True
    domain[0, 3] = True
    path = tmp_path / "roomseg_step_000011.npz"
    _snapshot(path, domain)
    snapshot = load_snapshot_arrays(path)

    result = backproject_final_gt_to_snapshot(final, snapshot, episode_uid="ep", source_final_step=20)

    assert result.metadata["visible_gt_room_count"] == 2
    assert sorted(int(v) for v in np.unique(result.label_map) if int(v) > 0) == [1, 2]

