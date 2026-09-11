"""Bounded, exact-input memoization of independent voxel-column encodings.

There is no coordinate-only cache: updated columns, shifted grids and rotated
patches are keyed by the actual model input. CNNs/context/head are never cached.
"""
from __future__ import annotations

from collections import OrderedDict
import numpy as np

from .patch_extraction import voxel_states_to_model_channels


class ColumnEncodingCache:
    def __init__(self, max_columns: int = 65536, batch_size: int = 2048):
        if int(max_columns) <= 0 or int(batch_size) <= 0:
            raise ValueError("column cache capacity and encoding batch size must be positive")
        self.max_columns = int(max_columns)
        self.batch_size = int(batch_size)
        self._values = OrderedDict()
        self._signature = None
        self._feature_dtype = None
        self.stats = {}

    def clear(self):
        self._values.clear()
        self._signature = None
        self._feature_dtype = None

    def start_call(self, model, z_centers_m, height_scale_m):
        import torch

        if any(module.training for module in model.modules()) or torch.is_grad_enabled():
            raise RuntimeError("column reuse is inference-only; eval and no gradients are required")
        z = np.ascontiguousarray(z_centers_m, dtype=np.float32)
        if z.shape != (int(model.model_config.z_count),) or not np.all(np.isfinite(z)):
            raise ValueError("column cache requires finite, matching absolute heights")
        if not np.isfinite(height_scale_m) or float(height_scale_m) <= 0:
            raise ValueError("height scale must be positive and finite")
        tensors = list(model.parameters()) + list(model.buffers())
        device = next(model.parameters()).device
        autocast = torch.is_autocast_enabled(device.type)
        def version(tensor):
            try:
                return tensor._version
            except RuntimeError:
                # Parameters created inside inference_mode have no version
                # counter. Still deduplicate within this call, but do not reuse
                # them across calls where a mutation cannot be detected.
                return object()

        signature = (id(model), z.tobytes(), float(height_scale_m),
                     tuple((t.data_ptr(), version(t), str(t.device), str(t.dtype), tuple(t.shape)) for t in tensors),
                     autocast, str(torch.get_autocast_dtype(device.type)) if autocast else None,
                     torch.get_float32_matmul_precision(),
                     torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32,
                     torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
        invalidated = 0
        if signature != self._signature:
            invalidated = len(self._values)
            self.clear()
            self._signature = signature
        self.stats = dict(requested_columns=0, unique_columns_per_batch=0,
                          encoded_columns=0, cache_hit_columns=0,
                          evicted_columns=0, invalidated_columns=invalidated)

    def encode_patches(self, model, states_nzyx, z_centers_m, height_scale_m):
        import torch

        if self._signature is None or any(m.training for m in model.modules()) or torch.is_grad_enabled():
            raise RuntimeError("start an inference-only column cache call before encoding")
        states = np.asarray(states_nzyx, dtype=np.uint8)
        size = int(model.model_config.local_patch_size)
        if states.ndim != 4 or tuple(states.shape[1:]) != (int(model.model_config.z_count), size, size):
            raise ValueError("voxel patches do not match model column geometry")
        if np.any(states > 3):
            raise ValueError("voxel state contains unsupported codes")
        # State 3 and state 0 have exactly the same model input channels.
        states = np.where(states == 3, 0, states).astype(np.uint8)
        n, z, h, w = states.shape
        columns = np.ascontiguousarray(states.transpose(0, 2, 3, 1).reshape(-1, z))
        packed = columns.view(np.dtype((np.void, z))).reshape(-1)
        unique, inverse = np.unique(packed, return_inverse=True)
        keys = [item.tobytes() for item in unique]
        parameter = next(model.parameters())
        table_dtype = np.float64 if parameter.dtype == torch.float64 else np.float32
        table = np.empty((len(keys), 2 * int(model.model_config.column_channels)), dtype=table_dtype)
        missing = []
        for i, key in enumerate(keys):
            value = self._values.get(key)
            if value is None:
                missing.append(i)
            else:
                table[i] = value
                self._values.move_to_end(key)
        self.stats['requested_columns'] += len(columns)
        self.stats['unique_columns_per_batch'] += len(keys)
        self.stats['cache_hit_columns'] += len(keys) - len(missing)
        for start in range(0, len(missing), self.batch_size):
            indices = missing[start:start + self.batch_size]
            state = np.frombuffer(b''.join(keys[i] for i in indices), dtype=np.uint8).reshape(-1, z, 1, 1)
            channels = voxel_states_to_model_channels(state, z_centers_m, height_scale_m=height_scale_m)[:, :, :, 0, 0]
            encoded = model.encode_columns(torch.from_numpy(channels).to(device=parameter.device, dtype=parameter.dtype))
            if not torch.all(torch.isfinite(encoded)):
                raise RuntimeError("column encoder produced NaN or Inf")
            self._feature_dtype = encoded.dtype
            values = (encoded if encoded.dtype == torch.float64 else encoded.float()).cpu().numpy()
            for index, value in zip(indices, values):
                table[index] = value
                self._values[keys[index]] = value.copy()
                if len(self._values) > self.max_columns:
                    self._values.popitem(last=False)
                    self.stats['evicted_columns'] += 1
            self.stats['encoded_columns'] += len(indices)
        if not keys:
            raise ValueError("cannot assemble an empty local-feature batch")
        features = torch.from_numpy(table).to(device=parameter.device, dtype=self._feature_dtype)
        positions = torch.from_numpy(inverse.astype(np.int64, copy=False)).to(parameter.device)
        return features.index_select(0, positions).reshape(n, h, w, -1).permute(0, 3, 1, 2)

    def statistics(self):
        result = dict(self.stats)
        result['cached_columns'] = len(self._values)
        result['cache_capacity_columns'] = self.max_columns
        requested = result.get('requested_columns', 0)
        result['avoided_column_encodings'] = requested - result.get('encoded_columns', 0)
        result['reuse_fraction'] = result['avoided_column_encodings'] / requested if requested else 0.
        return result
