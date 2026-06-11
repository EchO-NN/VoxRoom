from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from voxroom_online.isaac_runtime.evaluation.online_roomseg.snapshot_index import build_index


def _write_snapshot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pred = np.ones((4, 4), dtype=np.int32)
    observed = np.ones((4, 4), dtype=bool)
    empty_nav = np.zeros((4, 4), dtype=bool)
    np.savez_compressed(
        path,
        final_room_label_map=pred,
        navigation_free_room_domain=empty_nav,
        observed_free_mask=observed,
        obstacle_mask=np.zeros_like(observed),
        unknown_mask=np.zeros_like(observed),
    )
    path.with_suffix(".navigation_room_masks.png").write_bytes(b"png")


def _make_scene(tmp_path: Path, *, steps_value: int | None) -> Path:
    root = tmp_path / "result" / "run_a"
    scene = root / "scene_001"
    _write_snapshot(scene / "roomseg_snapshots" / "roomseg_step_000010.npz")
    _write_snapshot(scene / "roomseg_snapshots" / "roomseg_step_000020.npz")
    if steps_value is not None:
        with (scene / "results.jsonl").open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"episode_id": "ep_a", "steps": int(steps_value)}) + "\n")
    return root


def test_index_last_snapshot_within_final(tmp_path: Path) -> None:
    index = build_index([_make_scene(tmp_path, steps_value=21)])
    episode = index["episodes"][0]

    assert episode["last_snapshot_step"] == 20
    assert episode["step_reverse_status"] == "within_final"
    assert episode["episode_uid"] == "run_a__scene_001__ep_a"


def test_index_last_snapshot_exact(tmp_path: Path) -> None:
    index = build_index([_make_scene(tmp_path, steps_value=20)])

    assert index["episodes"][0]["last_snapshot_step"] == 20
    assert index["episodes"][0]["step_reverse_status"] == "exact"


def test_index_last_snapshot_snapshot_only_without_results_jsonl(tmp_path: Path) -> None:
    index = build_index([_make_scene(tmp_path, steps_value=None)])

    assert index["episodes"][0]["last_snapshot_step"] == 20
    assert index["episodes"][0]["step_reverse_status"] == "snapshot_only"

