"""Regression tests for inference-only, content-addressed column reuse."""
import copy
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from voxroom_online.isaac_runtime.door_seed_learning.column_encoding_cache import ColumnEncodingCache
from voxroom_online.isaac_runtime.door_seed_learning.config import DoorSeedLearningConfig
from voxroom_online.isaac_runtime.door_seed_learning.inference import DoorSeedInferenceEngine
from voxroom_online.isaac_runtime.door_seed_learning.model import DoorSeedModelConfig, build_door_seed_model
from voxroom_online.isaac_runtime.door_seed_learning.patch_extraction import (
    class_patches_to_one_hot, extract_class_patches, extract_local_voxel_patches,
    voxel_states_to_model_channels,
)
from voxroom_online.isaac_runtime.mapping.voxel_door_detector import VoxelDoorSeedResult


def original_forward(model, voxel, context):
    """Inline pre-refactor math, deliberately not using either new model method."""
    features = []
    if model.local_xy is not None:
        b, c, z, h, w = voxel.shape
        columns = voxel.permute(0, 3, 4, 1, 2).reshape(b * h * w, c, z)
        columns = model.column_stem(columns).transpose(1, 2)
        columns = model.column_transformer(columns)
        pooled = torch.cat((columns.mean(1), columns.amax(1)), 1)
        local = model.local_xy(pooled.reshape(b, h, w, -1).permute(0, 3, 1, 2))
        features.append(torch.cat((local.mean((-2, -1)), local.amax((-2, -1))), 1))
    if model.context_xy is not None:
        context = model.context_xy(context)
        features.append(torch.cat((context.mean((-2, -1)), context.amax((-2, -1))), 1))
    return model.classifier(features[0] if len(features) == 1 else torch.cat(features, 1))


class ColumnReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(47)
        rng = np.random.default_rng(47)
        self.model = build_door_seed_model(DoorSeedModelConfig(z_count=6)).eval()
        self.state = rng.integers(0, 4, (6, 24, 27), dtype=np.uint8)
        self.context = rng.integers(0, 3, (24, 27), dtype=np.uint8)
        self.rc = np.array([[0, 0], [11, 12], [11, 13], [23, 26]], dtype=np.int32)
        self.z = np.linspace(-.1, 1., 6, dtype=np.float32)
        self.patches = extract_local_voxel_patches(self.state, self.rc, patch_size=19)[0]
        self.ctx = torch.from_numpy(class_patches_to_one_hot(
            extract_class_patches(self.context, self.rc, patch_size=41)))
        self.cache = ColumnEncodingCache(max_columns=4096, batch_size=53)

    def channels(self, patches=None, z=None, scale=4.):
        return torch.from_numpy(voxel_states_to_model_channels(
            self.patches if patches is None else patches, self.z if z is None else z,
            height_scale_m=scale))

    def cached_map(self, patches=None, z=None, scale=4.):
        z = self.z if z is None else z
        self.cache.start_call(self.model, z, scale)
        return self.cache.encode_patches(self.model, self.patches if patches is None else patches, z, scale)

    def engine(self, reuse=True):
        # Isolate array-inference integration from checkpoint file I/O.
        engine = DoorSeedInferenceEngine.__new__(DoorSeedInferenceEngine)
        engine.model = self.model
        engine.device = torch.device('cpu')
        engine.config = DoorSeedLearningConfig(mode='inference', checkpoint_path='unused.pt',
            device='cpu', inference_batch_size=2, reuse_column_encodings=reuse)
        engine.column_cache = self.cache
        engine.last_column_cache_stats = {}
        engine.checkpoint = {'recommended_keep_threshold': .5}
        engine.load_error = None
        engine.inference_count = 0
        engine.fallback_count = 0
        return engine

    def test_original_forward_and_checkpoint_keys_unchanged(self):
        clone = build_door_seed_model(self.model.model_config).eval()
        clone.load_state_dict(self.model.state_dict(), strict=True)
        with torch.inference_mode():
            a = original_forward(self.model, self.channels(), self.ctx)
            b = self.model(self.channels(), self.ctx)
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_overlap_padding_and_spatial_assembly(self):
        with torch.inference_mode():
            local = self.cached_map()
            columns = self.channels().permute(0, 3, 4, 1, 2).reshape(-1, 4, 6)
            expected = self.model.encode_columns(columns).reshape(4, 19, 19, 80).permute(0, 3, 1, 2)
            torch.testing.assert_close(local, expected, rtol=2e-5, atol=2e-5)
            torch.testing.assert_close(self.model.forward_encoded(local, self.ctx),
                self.model(self.channels(), self.ctx), rtol=2e-5, atol=2e-5)
        self.assertLess(self.cache.statistics()['encoded_columns'], 4 * 361)

    def test_warm_cache_and_rotated_repositioning(self):
        with torch.inference_mode():
            local = self.cached_map()
            warm = self.cached_map()
            torch.testing.assert_close(local, warm, rtol=0, atol=0)
            self.assertEqual(self.cache.statistics()['encoded_columns'], 0)
            for k in (1, 2, 3):
                moved = np.flip(np.rot90(self.patches, k, axes=(-2, -1)), -1).copy()
                encoded = self.cached_map(moved)
                self.assertEqual(self.cache.statistics()['encoded_columns'], 0)
                torch.testing.assert_close(self.model.forward_encoded(encoded, self.ctx),
                    self.model(self.channels(moved), self.ctx), rtol=2e-5, atol=2e-5)

    def test_conflict_and_unknown_share_encoding(self):
        a = np.zeros_like(self.patches)
        with torch.inference_mode():
            x = self.cached_map(a)
            self.assertEqual(self.cache.statistics()['encoded_columns'], 1)
            y = self.cached_map(a + 3)
            self.assertEqual(self.cache.statistics()['encoded_columns'], 0)
        torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_changed_voxel_column_is_reencoded(self):
        a = np.zeros_like(self.patches)
        with torch.inference_mode():
            self.cached_map(a)
            a[0, 2, 10, 11] = 2
            local = self.cached_map(a)
            self.assertEqual(self.cache.statistics()['encoded_columns'], 1)
            torch.testing.assert_close(self.model.forward_encoded(local, self.ctx),
                self.model(self.channels(a), self.ctx), rtol=2e-5, atol=2e-5)

    def test_height_scale_and_model_weight_invalidation(self):
        with torch.inference_mode():
            self.cached_map()
            for z, scale in ((self.z + .3, 4.), (self.z + .3, 3.)):
                local = self.cached_map(z=z, scale=scale)
                self.assertGreater(self.cache.statistics()['invalidated_columns'], 0)
                torch.testing.assert_close(self.model.forward_encoded(local, self.ctx),
                    self.model(self.channels(z=z, scale=scale), self.ctx), rtol=2e-5, atol=2e-5)
            self.cached_map()
            self.model.column_stem[0].weight.add_(.01)
            local = self.cached_map()
            self.assertGreater(self.cache.statistics()['invalidated_columns'], 0)
            torch.testing.assert_close(self.model.forward_encoded(local, self.ctx),
                self.model(self.channels(), self.ctx), rtol=2e-5, atol=2e-5)
            state = copy.deepcopy(self.model.state_dict())
            self.model.load_state_dict(state)
            self.cached_map()
            self.assertGreater(self.cache.statistics()['invalidated_columns'], 0)

    def test_bounded_lru_eviction_and_clear(self):
        self.cache = ColumnEncodingCache(max_columns=2, batch_size=29)
        with torch.inference_mode():
            for _ in range(2):
                local = self.cached_map()
                self.assertEqual(self.cache.statistics()['cached_columns'], 2)
                self.assertGreater(self.cache.statistics()['evicted_columns'], 0)
                torch.testing.assert_close(self.model.forward_encoded(local, self.ctx),
                    self.model(self.channels(), self.ctx), rtol=2e-5, atol=2e-5)
        self.cache.clear()
        self.assertEqual(self.cache.statistics()['cached_columns'], 0)

    def test_reject_training_and_grad_enabled_but_training_gradients_unchanged(self):
        with self.assertRaisesRegex(RuntimeError, 'inference-only'):
            self.cache.start_call(self.model, self.z, 4.)
        self.model.train()
        with torch.inference_mode(), self.assertRaisesRegex(RuntimeError, 'inference-only'):
            self.cache.start_call(self.model, self.z, 4.)
        clone = copy.deepcopy(self.model)
        torch.manual_seed(18)
        y1 = original_forward(self.model, self.channels(), self.ctx)
        y1.sum().backward()
        torch.manual_seed(18)
        y2 = clone(self.channels(), self.ctx)
        y2.sum().backward()
        torch.testing.assert_close(y1, y2, rtol=0, atol=0)
        for (name, p), (_, q) in zip(self.model.named_parameters(), clone.named_parameters()):
            self.assertIsNotNone(p.grad, name)
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)

    def test_inference_arrays_and_full_grid_and_empty_inputs(self):
        e = self.engine()
        kwargs = dict(z_centers_m=self.z, class_map_xy=self.context, seed_rc=self.rc)
        a = e.predict_arrays(voxel_state_nzyx=self.patches, **kwargs)
        b = e.predict_arrays(voxel_state_zyx=self.state, **kwargs)
        self.assertEqual(e.last_column_cache_stats['encoded_columns'], 0)
        e.config = replace(e.config, reuse_column_encodings=False)
        c = e.predict_arrays(voxel_state_nzyx=self.patches, **kwargs)
        np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(a, c, rtol=2e-5, atol=2e-5)
        self.assertEqual(e.predict_arrays(voxel_state_zyx=self.state, **{
            **kwargs, 'seed_rc': np.empty((0, 2), np.int32)}).shape, (0,))
        with self.assertRaisesRegex(ValueError, 'either'):
            e.predict_arrays(voxel_state_nzyx=self.patches, voxel_state_zyx=self.state, **kwargs)

    def test_context_recomputed_and_single_branch_ablations(self):
        for voxel_branch, context_branch in ((True, True), (True, False), (False, True)):
            self.model = build_door_seed_model(DoorSeedModelConfig(z_count=6,
                use_voxel_branch=voxel_branch, use_context_branch=context_branch)).eval()
            e = self.engine()
            kwargs = dict(voxel_state_zyx=self.state if voxel_branch else None,
                z_centers_m=self.z, class_map_xy=self.context, seed_rc=self.rc)
            a = e.predict_arrays(**kwargs)
            e.config = replace(e.config, reuse_column_encodings=False)
            b = e.predict_arrays(**kwargs)
            np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-5)
            e.config = replace(e.config, reuse_column_encodings=True)
            kwargs['class_map_xy'] = np.zeros_like(self.context)
            c = e.predict_arrays(**kwargs)
            if voxel_branch:
                self.assertEqual(e.last_column_cache_stats['encoded_columns'], 0)
            if context_branch:
                self.assertGreater(float(np.max(np.abs(c - a))), 1e-6)

    def test_invalid_states_and_cache_configuration(self):
        bad = self.patches.copy()
        bad[0, 0, 0, 0] = 4
        with torch.inference_mode(), self.assertRaisesRegex(ValueError, 'unsupported'):
            self.cached_map(bad)
        with self.assertRaises(ValueError):
            ColumnEncodingCache(max_columns=0)
        with self.assertRaises(ValueError):
            DoorSeedLearningConfig(column_encoding_batch_size=0).validate()

    def test_dtype_change_and_inference_tensor_parameters(self):
        with torch.inference_mode():
            self.cached_map()
            self.model.double()
            local = self.cached_map()
            self.assertGreater(self.cache.statistics()['invalidated_columns'], 0)
            torch.testing.assert_close(self.model.forward_encoded(local, self.ctx.double()),
                self.model(self.channels().double(), self.ctx.double()), rtol=1e-10, atol=1e-10)
            self.model = build_door_seed_model(DoorSeedModelConfig(z_count=6)).eval()
            self.cached_map()
            self.cached_map()
            self.assertGreater(self.cache.statistics()['invalidated_columns'], 0)

    def test_filter_stage_batched_extraction_and_model_masks(self):
        raw = np.zeros(self.context.shape, bool)
        raw[tuple(self.rc.T)] = True
        zeros = np.zeros(self.context.shape, np.uint8)
        raw_result = VoxelDoorSeedResult(raw, zeros, zeros, zeros, zeros, zeros, zeros, [], {})
        stage = SimpleNamespace(raw_seed_result=raw_result, raw_seed_mask_xy=raw,
            vertical_class_map_xy=self.context, nav_class_map_xy=self.context,
            raw_seed_config_hash='test', input_semantics_hash='test')
        grid = SimpleNamespace(state=self.state, z_centers_m=self.z,
            z_min_m=-.21, z_resolution_m=.22, map_info=SimpleNamespace(resolution_m=.05))
        engine = self.engine()
        engine._validate_runtime_metadata = Mock()
        expected = engine.predict_arrays(voxel_state_nzyx=self.patches, z_centers_m=self.z,
            class_map_xy=self.context, seed_rc=self.rc)
        module = 'voxroom_online.isaac_runtime.door_seed_learning.inference'
        for keep_uninformative in (False, True):
            engine.config = replace(engine.config, keep_uninformative_seed=keep_uninformative)
            with patch(module + '.extract_local_voxel_patches', wraps=extract_local_voxel_patches) as extraction:
                result = engine.filter_stage(stage=stage, voxel_grid=grid, seed_connectivity=4)
                counts = [len(call.args[1]) for call in extraction.call_args_list]
            self.assertEqual(counts, [4] if keep_uninformative else [2, 2])
            self.assertIsNone(result.fallback_reason)
            np.testing.assert_array_equal(result.keep_mask_xy[tuple(self.rc.T)], expected >= .5)
            self.assertEqual(result.seed_result.debug['voxel_door_seed_model_column_cache']['encoded_columns'], 0)
        self.assertEqual(engine._validate_runtime_metadata.call_count, 2)
        stage.raw_seed_mask_xy = np.zeros_like(raw)
        result = engine.filter_stage(stage=stage, voxel_grid=grid, seed_connectivity=4)
        self.assertEqual(int(result.keep_mask_xy.sum()), 0)
        self.assertEqual(engine.last_column_cache_stats, {})


if __name__ == '__main__':
    unittest.main()
