import unittest
from dataclasses import replace

import numpy as np
from scipy import ndimage

from voxroom_online.isaac_runtime.mapping import voxel_door_detector as door


class LShapeTests(unittest.TestCase):
    def setUp(self):
        self.cfg = door.VoxelDoorDetectorConfig(primitive_min_line_correlation=.9,
            primitive_max_orthogonal_variance_cells2=.1, enable_l_shaped_seed_branches=True)
        self.shape = (48, 48)
        self.cells = sorted({(20, c) for c in range(20, 37)} | {(r, 20) for r in range(20, 34)})

    def group(self, cells):
        center, major, minor, residual, thickness, length, bbox = door._fit_seed_cells(cells, self.shape, .05)
        return door.DoorSeedGroup(1, 'single_cluster', [1], [1], cells,
            door._cells_to_mask(cells, self.shape), bbox, tuple(center), tuple(major), tuple(minor),
            residual, thickness, length, False, 'nonlinear_blob')

    def primitives(self, cells, cfg=None):
        return door.extract_seed_line_primitives_from_group(self.group(cells), primitive_id_start=1,
            shape=self.shape, resolution_m=.05, cfg=cfg or self.cfg)

    def candidates(self):
        candidates = []
        for p in self.primitives(self.cells):
            candidates.append(door.VoxelDoorLineCandidate(p.primitive_id, p.primitive_id, p.cells,
                p.center_rc, p.major_dir_rc, p.minor_dir_rc, p.cells.copy(), p.cells.copy(), p.cells.copy(),
                p.cells[0], p.cells[-1], p.length_cells * .05, True, None,
                {'seed_group_id': 1, 'primitive_id': p.primitive_id, 'score': 1.,
                 'partition_accepted': True, 'partition_effective_verified': True, **p.debug}))
        return candidates

    def test_restored_defaults_and_runtime_yaml(self):
        from voxroom_online.isaac_runtime.config import load_config, get_nested
        from voxroom_online.isaac_runtime.mapping.voxel_occupancy_door_wall_roomseg import VoxelOccupancyDoorWallRoomSegConfig
        defaults = door.VoxelDoorDetectorConfig()
        self.assertEqual(defaults.primitive_min_line_correlation, .95)
        self.assertEqual(defaults.primitive_max_orthogonal_variance_cells2, .65)
        self.assertFalse(defaults.enable_l_shaped_seed_branches)
        base = get_nested(load_config('configs/voxroom_online.yaml'), 'mapping.room_segmentation')
        runtime = VoxelOccupancyDoorWallRoomSegConfig.from_mapping(base)
        self.assertEqual(runtime.door.primitive_min_line_correlation, .95)
        self.assertEqual(runtime.door.primitive_max_orthogonal_variance_cells2, .65)
        self.assertFalse(runtime.door.enable_l_shaped_seed_branches)
        self.assertEqual(runtime.door_seed_learning.keep_threshold, .5)
        self.assertTrue(runtime.door_seed_learning.reuse_column_encodings)

    def test_experimental_thresholds(self):
        self.assertEqual(self.cfg.primitive_min_line_correlation, .9)
        self.assertEqual(self.cfg.primitive_max_orthogonal_variance_cells2, .1)
        kwargs = dict(seed_count=20, length_cells=20, thickness_cells=1, residual_cells=.5,
            elongation=20, max_gap=1, longest_contiguous_run_cells=20, cfg=self.cfg)
        self.assertIsNone(door._primitive_reject_reason(**kwargs, line_correlation=.9, orthogonal_variance_cells2=.1))
        self.assertEqual(door._primitive_reject_reason(**kwargs, line_correlation=.899, orthogonal_variance_cells2=.1), 'primitive_line_correlation_too_low')
        self.assertEqual(door._primitive_reject_reason(**kwargs, line_correlation=.95, orthogonal_variance_cells2=.101), 'primitive_orthogonal_variance_too_high')

    def test_experiment_overrides_reach_actual_yaml_config(self):
        from scripts.run_voxroom_line_threshold_experiment import METHODS, segmenter_config
        from voxroom_online.isaac_runtime.config import load_config, get_nested
        from voxroom_online.isaac_runtime.mapping.voxel_occupancy_door_wall_roomseg import VoxelOccupancyDoorWallRoomSegConfig
        base = get_nested(load_config('configs/voxroom_online.yaml'), 'mapping.room_segmentation')
        for params in METHODS.values():
            cfg = VoxelOccupancyDoorWallRoomSegConfig.from_mapping(segmenter_config(base, params, '/tmp/test.pt', 'cpu'))
            self.assertEqual(cfg.door.primitive_min_line_correlation, params['correlation'])
            self.assertEqual(cfg.door.primitive_max_orthogonal_variance_cells2, params['variance'])
            self.assertEqual(cfg.door.enable_l_shaped_seed_branches, params['l_branches'])
            self.assertEqual(cfg.door_seed_learning.keep_threshold, params['keep'])

    def test_both_arms_all_four_rotations(self):
        base = door._cells_to_mask(self.cells, self.shape)
        for rotation in range(4):
            cells = list(map(tuple, np.argwhere(np.rot90(base, rotation))))
            primitives = self.primitives(cells)
            self.assertEqual(len(primitives), 2)
            self.assertEqual({p.debug['l_shape_arm'] for p in primitives}, {'h', 'v'})
            for p in primitives:
                self.assertTrue(p.accepted_for_extension, p.reject_reason)
                self.assertTrue(set(p.cells) <= set(cells))
                self.assertAlmostEqual(p.debug['primitive_line_correlation'], 1.)
                self.assertAlmostEqual(p.debug['primitive_orthogonal_variance_cells2'], 0.)

    def test_thick_l_and_no_invented_seed(self):
        mask = ndimage.binary_dilation(door._cells_to_mask(self.cells, self.shape), structure=np.ones((3, 3)))
        cells = list(map(tuple, np.argwhere(mask)))
        p = self.primitives(cells)
        self.assertEqual({x.debug.get('l_shape_arm') for x in p}, {'h', 'v'})
        self.assertTrue(all(set(x.cells) <= set(cells) for x in p))

    def test_single_line_rectangle_t_and_disconnected_not_misidentified(self):
        examples = [([(20, c) for c in range(10, 35)]),
                    ([(r, c) for r in range(10, 22) for c in range(10, 22)]),
                    sorted({(20, c) for c in range(10, 35)} | {(r, 20) for r in range(20, 35)}),
                    sorted({(20, c) for c in range(25, 40)} | {(r, 20) for r in range(20, 35)})]
        for cells in examples:
            self.assertEqual(door._extract_l_shaped_seed_line_segments(cells, cfg=self.cfg), [])

    def test_branch_selection_does_not_drop_second_arm(self):
        candidates = self.candidates()
        self.assertEqual(len(door._select_seed_group_candidates(candidates, self.cfg)), 2)
        self.assertEqual(len(door._select_seed_group_candidates(candidates,
            replace(self.cfg, enable_l_shaped_seed_branches=False))), 1)
        old = self.primitives(self.cells, replace(self.cfg, enable_l_shaped_seed_branches=False))
        self.assertFalse(any(p.debug.get('l_shape_arm') for p in old))

    def test_corner_allowed_but_unrelated_intersection_rejected(self):
        raw = door._cells_to_mask(self.cells, self.shape)
        pair = self.candidates()
        debug = door._batch_reject_conflicting_doors(pair, raw, self.shape, self.cfg)
        self.assertTrue(all(p.debug['partition_accepted'] for p in pair))
        self.assertEqual(debug['voxel_door_partition_shared_l_corner_ignored_count'], 1)
        pair = self.candidates()
        pair[1].debug['seed_group_id'] = 2
        debug = door._batch_reject_conflicting_doors(pair, raw, self.shape, self.cfg)
        self.assertEqual(sum(p.debug['partition_accepted'] for p in pair), 1)
        self.assertEqual(debug['voxel_door_partition_nav_free_intersection_reject_count'], 1)

    def test_memory_preserves_both_arms_across_frames_and_serialization(self):
        memory = door.VoxelDoorMemory(self.cfg)
        for step in (1, 2):
            result = memory.update(self.candidates(), step=step, shape=self.shape)
            self.assertEqual(len(result.tracks), 2)
            self.assertEqual({t.l_shape_arm for t in result.tracks}, {'h', 'v'})
        restored = door.VoxelDoorMemory(self.cfg)
        restored.load_state_dict(memory.to_state_dict())
        self.assertEqual({t.l_shape_arm for t in restored._tracks}, {'h', 'v'})
        # Only the vertical arm appears: it must not refresh a nearby H track.
        result = restored.update([self.candidates()[1]], step=3, shape=self.shape)
        self.assertEqual(len(result.tracks), 2)
        self.assertEqual(next(t for t in result.tracks if t.l_shape_arm == 'h').last_seen_step, 2)
        self.assertEqual(next(t for t in result.tracks if t.l_shape_arm == 'v').last_seen_step, 3)

    def test_complete_pipeline_attempts_both_arms(self):
        raw = door._cells_to_mask(self.cells, self.shape)
        zeros = np.zeros(self.shape, np.uint8)
        seed = door.VoxelDoorSeedResult(raw, ndimage.label(raw)[0], zeros, zeros, zeros, zeros, zeros, [], {})
        wall = np.zeros(self.shape, bool)
        wall[[0, -1], :] = True
        wall[:, [0, -1]] = True
        result = door.complete_voxel_doors_from_seeds(seed_result=seed, free_map=~wall,
            anchor_wall_map=wall, unknown_map=np.zeros(self.shape, bool), resolution_m=.05,
            real_wall_barrier_map=wall, config=self.cfg)
        arms = [p for p in result.candidates if p.debug.get('l_shape_arm')]
        self.assertEqual({p.debug['l_shape_arm'] for p in arms}, {'h', 'v'})
        self.assertEqual(len(arms), 2)


if __name__ == '__main__':
    unittest.main()
