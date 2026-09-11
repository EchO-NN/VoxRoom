"""Result exports must preserve point weighting and expose incomplete runs."""
import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/organize_dude_occusg_csv.py"
SPEC = importlib.util.spec_from_file_location("organize_dude_occusg_csv", SCRIPT)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def point(uid, event, value, dataset="grscene"):
    return {"dataset": dataset, "scene_id": uid, "episode_uid": uid,
            "coverage_event_id": event, **{m: value for m in exporter.METRICS}}


def universe(*rows):
    return {exporter.key(r): r for r in rows}


def test_combined_weights_checkpoints_not_datasets():
    rows = [point("a", "milestone_020", .9, "interioragent"),
            point("b", "milestone_020", .3), point("b", "final", .6)]
    result = exporter.aggregate("test", rows, universe(*rows), "combined", "all_available_progress")
    assert result["F1_percent"] == pytest.approx(60)
    assert result["coverage_status"] == "complete"
    assert result["evaluated_checkpoint_count"] == 3


def test_average_f1_is_not_harmonic_of_average_precision_and_recall():
    rows = [point("a", "milestone_020", .8), point("a", "final", .8)]
    rows[0].update(precision=1, recall=.5, f1=2/3)
    rows[1].update(precision=.5, recall=1, f1=2/3)
    result = exporter.aggregate("test", rows, universe(*rows), "combined", "all_available_progress")
    assert result["P_percent"] == result["R_percent"] == 75
    assert result["F1_percent"] == pytest.approx(100*2/3)


def test_pairing_is_distinct_from_missing_predictions():
    rows = [point("a", "milestone_020", .2), point("a", "final", .8)]
    expected = universe(*rows)
    paired = {exporter.key(rows[0])}
    online = exporter.aggregate("online", rows, expected, "combined", "all_available_progress", paired)
    offline = exporter.aggregate("offline", rows[:1], expected, "combined", "all_available_progress", paired)
    assert online["F1_percent"] == offline["F1_percent"] == 20
    assert online["missing_prediction_count"] == 0
    assert online["excluded_by_pairing_count"] == 1
    assert offline["missing_prediction_count"] == 1
    assert offline["excluded_by_pairing_count"] == 0
    assert offline["coverage_status"] == "paired_subset_not_full_test_set"


def test_stability_is_computed_within_scene_then_averaged():
    rows = [point("a", "milestone_020", .5), point("a", "final", 1),
            point("b", "milestone_020", .4), point("b", "final", .4)]
    scenes = exporter.scene_statistics("test", rows, universe(*rows))
    result = exporter.stability_summary("test", scenes, "combined")
    assert result["F1_average_scene_SD_pp"] == pytest.approx(12.5)
    assert result["F1_average_scene_CV_ratio"] == pytest.approx(1/6)
    assert result["F1_average_scene_worst_percent"] == pytest.approx(45)
    assert result["multi_progress_scene_count"] == 2


def test_missing_scene_is_blank_not_zero_and_natural_gaps_are_distinct():
    available = point("a", "milestone_020", 0)
    absent = point("b", "final", .8)
    scenes = exporter.scene_statistics("test", [available], universe(available, absent))
    present, missing = scenes
    assert present["complete_for_saved_progress"] is True
    assert "final" in present["not_saved_in_original_exploration"]
    assert present["F1_SD_pp"] == 0
    assert present["F1_CV_ratio"] is None
    assert missing["F1_mean_percent"] is None
    assert missing["F1_SD_pp"] is None
    assert missing["missing_predictions_for_saved_progress"] == "final"
    result = exporter.stability_summary("test", scenes, "combined")
    assert result["expected_scene_count"] == 2
    assert result["evaluated_scene_count"] == 1
    assert result["single_progress_scene_count"] == 1
    assert result["F1_CV_defined_scene_count"] == 0


def test_explicit_single_point_exclusion_never_drops_whole_scene_or_counts_zero():
    rows = [point("a", "milestone_020", .8), point("a", "final", .2)]
    excluded = {exporter.key(rows[1])}
    result = exporter.aggregate("offline", rows[:1], universe(*rows), "combined", "all_available_progress", excluded_keys=excluded)
    assert result['F1_percent'] == 80
    assert result['coverage_status'] == 'complete_with_user_exclusion'
    assert result['expected_checkpoint_count'] == 2
    assert result['eligible_checkpoint_count'] == result['evaluated_checkpoint_count'] == 1
    assert result['authorized_exclusion_count'] == 1
    assert result['unexpected_missing_prediction_count'] == 0
    scene, = exporter.scene_statistics('offline', rows[:1], universe(*rows), excluded)
    assert scene['complete_for_required_progress'] is True
    assert scene['complete_for_saved_progress'] is False
    assert scene['user_excluded_progress'] == 'final'
    assert scene['missing_predictions_for_saved_progress'] == ''
    assert scene['F1_mean_percent'] == 80


def test_user_exclusion_cannot_silently_discard_an_existing_prediction():
    row = point('a', 'final', .9)
    with pytest.raises(AssertionError):
        exporter.aggregate('offline', [row], universe(row), 'combined', 'all_available_progress', excluded_keys={exporter.key(row)})
