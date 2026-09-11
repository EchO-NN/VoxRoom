import csv
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_morph_area_evaluation as runner


class MorphAreaTests(unittest.TestCase):
    def test_comparison_rejects_changed_snapshot(self):
        task = dict(dataset='test', scene_id='room', episode_uid='ep', coverage_event_id='final',
                    step=2, source_snapshot_path='/source/a.npz', source_sha256='abc',
                    reference_n_gt=1, reference_metric_domain_pixels=200)
        row = {**task, 'method_id': 'prior', 'source_snapshot_path': '/source/WRONG.npz',
               'n_gt': 1, 'metric_domain_pixels': 200,
               'precision': .8, 'recall': .8, 'f1': .8, 'miou_room': .7}
        with tempfile.TemporaryDirectory(prefix='morph_compare_test_') as temp:
            path = Path(temp) / 'prior.csv'
            with path.open('w', newline='') as f:
                writer = csv.DictWriter(f, list(row))
                writer.writeheader()
                writer.writerow(row)
            with self.assertRaises(AssertionError):
                runner.comparison_records(path, [task])

    def test_input_does_not_use_reference_or_seeds(self):
        free = np.array([[0, 1], [1, 0]], dtype=bool)
        z = {'voxel_vertical_free_xy': free,
             'roomseg_eval_explored_reference_mask': np.zeros((2, 2)),
             'voxel_final_room_label_map': np.ones((2, 2)),
             'door_seed_map': np.ones((2, 2))}
        np.testing.assert_array_equal(runner.input_image(z), free.astype(np.uint8) * 255)
        z['roomseg_eval_explored_reference_mask'][:] = 1
        np.testing.assert_array_equal(runner.input_image(z), free.astype(np.uint8) * 255)

    def test_missing_vertical_cannot_fallback_to_gt(self):
        with self.assertRaises(KeyError):
            runner.input_image({'roomseg_eval_explored_reference_mask': np.ones((2, 2))})

    def test_empty_vertical_fails_explicitly(self):
        with self.assertRaises(ValueError):
            runner.input_image({'voxel_vertical_free_xy': np.zeros((2, 2))})

    def test_native_area_parameter_changes_connected_rooms(self):
        binary = ROOT / 'results/morph_area15_smoke_20260911/build/morph'
        if not binary.exists():
            self.skipTest('Build the native smoke runner first')
        image = np.zeros((90, 160), dtype=np.uint8)
        image[10:74, 5:69] = 255
        image[10:74, 89:153] = 255
        image[38:46, 69:89] = 255
        with tempfile.TemporaryDirectory(prefix='morph_area_test_') as temp:
            output = []
            for upper in (47., 15.):
                raw = runner.native_labels(binary, image, .05, Path(temp) / str(upper), .8, upper)
                self.assertFalse(np.any(raw[image == 0]))
                output.append(len(np.unique(raw[(raw > 0) & (raw < 65280)])))
            self.assertEqual(output, [1, 2])

    def test_small_region_stays_unassigned_not_fake_room(self):
        binary = ROOT / 'results/morph_area15_smoke_20260911/build/morph'
        if not binary.exists():
            self.skipTest('Build the native smoke runner first')
        image = np.zeros((20, 20), dtype=np.uint8)
        image[5:15, 5:15] = 255
        with tempfile.TemporaryDirectory(prefix='morph_small_test_') as temp:
            raw = runner.native_labels(binary, image, .05, Path(temp) / 'small', .8, 15.)
        self.assertFalse(np.any((raw > 0) & (raw < 65280)))
        self.assertTrue(np.all(raw[image > 0] == 65280))


if __name__ == '__main__':
    unittest.main()
