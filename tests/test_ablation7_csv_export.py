"""Do not confuse validation free-cell classification with room metrics."""
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def exporter(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return importlib.import_module('organize_ablation7_csv')


def test_history_has_explicit_metric_scope_and_stage_numbering(exporter):
    item = {'epoch': 13, 'precision': .6, 'recall': .75, 'f1': 2/3,
            'accuracy': .9, 'threshold': .5, 'train_loss': .02, 'val_loss': .1,
            'checkpoint_selection_score': [2/3, .6, .8]}
    row, = exporter.history_rows([item], 'continuation', selected_epoch=13)
    assert row['stage_epoch'] == 13
    assert row['selected_for_this_evaluation'] is True
    assert row['validation_f1'] == pytest.approx(2/3)
    assert row['validation_threshold'] == .5
    assert 'f1' not in row and 'F1_percent' not in row
    assert row['train_loss'] == .02 and row['val_loss'] == .1
    assert json.loads(row['checkpoint_selection_score']) == [2/3, .6, .8]
    assert row['metric_scope'].endswith('not_room_segmentation')


def test_previous_history_never_marked_as_selected_continuation(exporter):
    item = {'epoch': 2, 'precision': 0, 'recall': 0, 'f1': 0}
    row, = exporter.history_rows([item], 'previous')
    assert row['selected_for_this_evaluation'] is False
    assert row['training_stage'] == 'previous'
    assert row['validation_f1'] == 0


def test_inconsistent_history_f1_is_rejected(exporter):
    with pytest.raises(AssertionError):
        exporter.history_rows([{'epoch': 1, 'precision': .5, 'recall': .5, 'f1': .9}], 'test')
