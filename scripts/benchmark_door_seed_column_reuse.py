#!/usr/bin/env python3
"""Compare unchanged checkpoints with/without inference column memoization.

Read-only inputs, separate output directory. Includes patch extraction, transfers,
both CNN branches, classifier and probability download; excludes checkpoint/map
load, raw seed generation and downstream room segmentation. No GT is accessed.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from voxroom_online.isaac_runtime.door_seed_learning.config import DoorSeedLearningConfig
from voxroom_online.isaac_runtime.door_seed_learning.inference import DoorSeedInferenceEngine
from voxroom_online.isaac_runtime.door_seed_learning.schema import source_tree_hash
from voxroom_online.isaac_runtime.door_seed_learning.stage_extractor import encode_vertical_class_map


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--column-batch-size', type=int, default=2048)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--limit', type=int, default=0, help='0: all saved raw seed cells')
    parser.add_argument('--max-probability-error', type=float, default=2e-5)
    parser.add_argument('--strict-fp32', action='store_true', help='Disable TF32 for both paths in this benchmark process only')
    parser.add_argument('--allow-source-code-hash-mismatch', action='store_true',
                        help='Explicitly acknowledge changed inference source, not changed checkpoint tensors')
    args = parser.parse_args()
    if args.repeats < 1 or args.limit < 0:
        parser.error('repeats must be positive; limit must be non-negative')
    if args.output_dir.exists():
        parser.error('output directory already exists; use a new directory to preserve old results')
    torch.set_num_threads(1)
    if args.strict_fp32:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    cfg = DoorSeedLearningConfig(mode='inference', checkpoint_path=str(args.checkpoint.resolve()),
        device=args.device, inference_batch_size=args.batch_size, keep_threshold=.5,
        column_encoding_batch_size=args.column_batch_size,
        allow_source_code_hash_mismatch=args.allow_source_code_hash_mismatch)
    engine = DoorSeedInferenceEngine(cfg)
    with np.load(args.snapshot, allow_pickle=False) as data:
        state = data['voxel_occupancy_state_zyx']
        z = data['voxel_occupancy_z_centers_m']
        context = encode_vertical_class_map(data['voxel_vertical_free_xy'], data['voxel_wall_xy'],
            outside_boundary_mask_xy=data['voxel_outside_xy'] if 'voxel_outside_xy' in data else None)
        rc = np.argwhere(data['voxel_door_raw_seed_mask']).astype(np.int32)
    total_seeds = len(rc)
    if args.limit:
        rc = rc[np.linspace(0, len(rc) - 1, min(args.limit, len(rc)), dtype=int)]
    if not len(rc):
        raise ValueError('snapshot contains no raw seeds')
    kwargs = dict(voxel_state_zyx=state, z_centers_m=z, class_map_xy=context, seed_rc=rc)
    is_cuda = engine.device.type == 'cuda'

    def synchronize():
        if is_cuda:
            torch.cuda.synchronize(engine.device)

    # Warm both execution shapes; cache is cleared before measured cold calls.
    for reuse in (False, True):
        engine.config = replace(cfg, reuse_column_encodings=reuse)
        engine.predict_arrays(**{**kwargs, 'seed_rc': rc[:min(len(rc), args.batch_size)]})
    synchronize()
    rows = []
    reference = None
    predictions = {}
    for mode in ('uncached', 'cold_cache', 'warm_cache'):
        engine.config = replace(cfg, reuse_column_encodings=mode != 'uncached')
        for rep in range(args.repeats):
            if mode != 'warm_cache':
                engine.clear_column_cache()
            synchronize()
            if is_cuda:
                torch.cuda.reset_peak_memory_stats(engine.device)
            started = time.perf_counter()
            pred = engine.predict_arrays(**kwargs)
            synchronize()
            seconds = time.perf_counter() - started
            if reference is None:
                reference = pred.copy()
            delta = np.abs(pred - reference)
            row = dict(mode=mode, repeat=rep + 1, seconds=seconds,
                seeds_per_second=len(rc) / seconds, max_abs_probability_error=float(delta.max()),
                mean_abs_probability_error=float(delta.mean()),
                changed_keep_decisions=int(np.sum((pred >= .5) != (reference >= .5))),
                accepted_seeds=int(np.sum(pred >= .5)),
                cuda_peak_allocated_mib=(torch.cuda.max_memory_allocated(engine.device) / 1024**2 if is_cuda else None),
                column_cache=dict(engine.last_column_cache_stats))
            rows.append(row)
            predictions[mode] = pred
            print(json.dumps(row, ensure_ascii=False), flush=True)
    medians = {mode: statistics.median(r['seconds'] for r in rows if r['mode'] == mode)
               for mode in ('uncached', 'cold_cache', 'warm_cache')}
    passed = not any(r['changed_keep_decisions'] or r['max_abs_probability_error'] > args.max_probability_error for r in rows)
    result = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha256(args.checkpoint),
        snapshot=str(args.snapshot.resolve()), snapshot_sha256=sha256(args.snapshot),
        checkpoint_source_code_hash=engine.checkpoint['source_code_hash'],
        current_source_code_hash=source_tree_hash(ROOT / 'voxroom_online/isaac_runtime/door_seed_learning'),
        allow_source_code_hash_mismatch=args.allow_source_code_hash_mismatch,
        voxel_shape=list(state.shape), total_saved_seeds=total_seeds, evaluated_seeds=len(rc),
        config=cfg.to_dict(), python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
        gpu=torch.cuda.get_device_name(engine.device) if is_cuda else None,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        verification=dict(passed=passed, max_allowed_probability_error=args.max_probability_error,
                          required_changed_keep_decisions=0),
        scope='array inference only; saved raw seeds; original vertical class map; no GT; fixed FP32 checkpoint and threshold 0.5',
        rows=rows, median_seconds=medians,
        speedup={mode: medians['uncached'] / medians[mode] for mode in ('cold_cache', 'warm_cache')})
    args.output_dir.mkdir(parents=True)
    (args.output_dir / 'benchmark.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    np.savez_compressed(args.output_dir / 'prediction_comparison.npz', seed_rc=rc, **predictions)
    # Store evidence even if the equivalence check fails, but fail the command.
    if not passed:
        raise RuntimeError('equivalence check failed; inspect saved benchmark, do not treat as verified')
    print(json.dumps({'verified': True, 'median_seconds': medians, 'speedup': result['speedup']}), flush=True)


if __name__ == '__main__':
    main()
