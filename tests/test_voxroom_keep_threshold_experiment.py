import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_voxroom_keep_threshold_experiment as experiment


class KeepThresholdTests(unittest.TestCase):
    def test_only_threshold_changes(self):
        replay = experiment.replay
        base = replay.get_nested(replay.load_config('configs/voxroom_online.yaml'), 'mapping.room_segmentation')
        untouched = copy.deepcopy(base)
        method, parameters = experiment.variant(.2)
        self.assertIn('keep0p2_', method)
        config = replay.segmenter_config(base, parameters, '/tmp/checkpoint.pt', 'cpu')
        reference = replay.segmenter_config(base, replay.METHODS[experiment.CONTROL], '/tmp/checkpoint.pt', 'cpu')
        reference['door_seed_learning']['keep_threshold'] = .2
        self.assertEqual(config, reference)
        self.assertEqual(base, untouched)

    def test_invalid_threshold(self):
        for value in (0, 1, -.1, float('nan'), float('inf'), .25):
            with self.assertRaises(ValueError):
                experiment.variant(value)

    def test_worker_runs_only_new_variant_and_restores_module(self):
        replay = experiment.replay
        previous = copy.deepcopy(replay.METHODS)
        def fake(*args):
            self.assertEqual(len(replay.METHODS), 1)
            self.assertEqual(next(iter(replay.METHODS.values()))['keep'], .2)
            return ['result']
        with patch.object(replay, 'run_scene', side_effect=fake):
            self.assertEqual(experiment.run_variant_scene([], {}, 'ckpt', 'out', 'cpu', 'hash', .2), ['result'])
        self.assertEqual(replay.METHODS, previous)
        with patch.object(replay, 'run_scene', side_effect=RuntimeError('test')):
            with self.assertRaises(RuntimeError):
                experiment.run_variant_scene([], {}, 'ckpt', 'out', 'cpu', 'hash', .2)
        self.assertEqual(replay.METHODS, previous)

    def test_comparison_uses_exact_common_checkpoints(self):
        method, _ = experiment.variant(.2)
        def row(m, event):
            return dict(method_id=m, dataset='interioragent', episode_uid='s', coverage_event_id=event)
        new = [row(method, e) for e in ('milestone_020', 'milestone_040')]
        old = [row(m, e) for m in (experiment.CONTROL, experiment.ORIGINAL)
               for e in ('milestone_040', 'milestone_060')]
        paired, keys = experiment.common_rows(new, old, method)
        self.assertEqual(len(keys), 1)
        for records in paired.values():
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]['coverage_event_id'], 'milestone_040')
        paired, keys = experiment.common_rows(new, [], method)
        self.assertFalse(keys)
        self.assertTrue(all(not records for records in paired.values()))

    def test_control_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'control.csv'
            row = dict(method_id=experiment.CONTROL, dataset='interioragent', episode_uid='s',
                       coverage_event_id='milestone_020', source_sha256='wrong')
            experiment.replay.csv_save(path, [row])
            universe = {experiment.replay.key(row): dict(source_sha256='correct')}
            with self.assertRaises(AssertionError):
                experiment.load_control_rows(path, universe)


if __name__ == '__main__':
    unittest.main()
