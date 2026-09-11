import csv
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import numpy as np

scripts = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("occusg_input_summary", scripts / "summarize_occusg_input_comparison.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sample(event="milestone_020"):
    return dict(dataset="interioragent", episode_uid="example", scene_id="scene", coverage_event_id=event,
                source_snapshot_path="/data/source.npz", step=20, n_gt=5, metric_domain_pixels=4000)


def test_comparison_requires_same_domain_and_snapshot(tmp_path):
    row = sample()
    lookup = {summary.key(row): row}
    path = tmp_path / "metrics.csv"
    write_rows(path, [row])
    assert len(summary.matched_rows(path, lookup)) == 1
    write_rows(path, [{**row, "metric_domain_pixels": 3999}])
    with pytest.raises(AssertionError):
        summary.matched_rows(path, lookup)
    write_rows(path, [{**row, "source_snapshot_path": "/data/wrong.npz"}])
    with pytest.raises(AssertionError):
        summary.matched_rows(path, lookup)


def test_duplicate_rows_cannot_replace_missing_progress(tmp_path):
    a, b = sample(), sample("final")
    lookup = {summary.key(r): r for r in (a, b)}
    path = tmp_path / "metrics.csv"
    write_rows(path, [a, a])
    with pytest.raises(AssertionError):
        summary.matched_rows(path, lookup)


def test_boundary_audit_checks_original_domain_and_cropped_coordinates(tmp_path):
    from run_occusg_saved_voxels import crop_unknown_border
    full = np.full((19, 29), -1, dtype=np.int8)
    full[4:15, 12:22] = 0
    full[4, 12:22] = 100
    full[8:10, 17] = -1
    cropped, geometry = crop_unknown_border(full, {"shape": list(full.shape), "origin": [-1., 2.], "resolution": .05})
    reference = tmp_path / "reference.npz"
    prediction = tmp_path / "prediction.npz"
    np.savez(reference, dude_ros_occupancy_grid=full,
             baseline_metadata_json=json.dumps({"map_info": {"min_x": -1., "min_y": 2., "resolution_m": .05}}))
    arrays = dict(map_uav=cropped, room_label_map=np.zeros(full.shape, np.int32),
                  map_uav_origin_xy=geometry["origin"], map_uav_resolution=np.float32(.05))
    np.savez(prediction, **arrays)
    prediction.with_name("metadata.json").write_text(json.dumps({"geometry": geometry}))
    audit = summary.audit_input_grid(prediction, reference, cropped=True)
    assert audit["removed_known_cells"] == 0
    assert audit["removed_unknown_cells"] == full.size-cropped.size
    # A smaller evaluation raster must not pass as a cropped-input experiment.
    np.savez(prediction, **{**arrays, "room_label_map": np.zeros(cropped.shape, np.int32)})
    with pytest.raises(AssertionError):
        summary.audit_input_grid(prediction, reference, cropped=True)
    np.savez(prediction, **{**arrays, "map_uav_origin_xy": [-1., 2.]})
    with pytest.raises(AssertionError):
        summary.audit_input_grid(prediction, reference, cropped=True)


def test_rejection_count_does_not_double_count_stale_messages(tmp_path):
    log = tmp_path / "interioragent/replay/example/native_0.log"
    log.parent.mkdir(parents=True)
    log.write_text("Rejected suspicious decomposition update reason=unknown_ratio_too_high other=x\n"
                   "Published stale regions from last valid decomposition reason=unknown_ratio_too_high\n")
    audit = summary.rejection_summary(tmp_path, "example")
    assert audit["rejected_updates"] == audit["unknown_ratio_rejections"] == 1
    assert audit["affected_scenes"] == 1
