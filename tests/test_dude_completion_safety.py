"""Completion must never replace a previously recorded prediction."""
import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def module(name):
    path = Path(__file__).resolve().parents[1] / 'scripts' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_prediction_publish_is_exclusive_and_preserves_existing_bytes(tmp_path):
    completion = module('complete_missing_dude_offline')
    path = tmp_path / 'predictions' / 'final.npz'
    completion.publish_exclusive(path, {'labels': np.array([[1, 2]], dtype=np.int32)})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        completion.publish_exclusive(path, {'labels': np.array([[9, 9]], dtype=np.int32)})
    assert path.read_bytes() == original
    assert not path.with_name(path.name + '.completion.tmp').exists()
    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['labels'], [[1, 2]])


def test_empty_missing_list_is_header_only_not_a_zero_prediction(tmp_path):
    exporter = module('export_existing_corrected_dude_metrics')
    path = tmp_path / 'missing.csv'
    exporter.write(path, [], fieldnames=['dataset', 'event', 'prediction_available'])
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ['dataset', 'event', 'prediction_available']
        assert list(reader) == []
