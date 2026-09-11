#!/usr/bin/env python3
"""Warm-start ablation 7 without overwriting its interrupted run.

The legacy best.pt has no optimizer/RNG state. This is explicitly a new
continuation stage, not an exact optimizer-state resume. Run as a user service
so training and sequential evaluations survive an SSH disconnection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: object) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/media/echo/data/voxroom_ablation_20260828'))
    parser.add_argument('--run', default='all_vertical_free_cells_recovery_20260907')
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in {'.', '..'}:
        raise ValueError('run must be a single directory name')
    root = args.root.resolve()
    output = root / 'training' / args.run
    baseline = root / 'training/all_vertical_free_cells/best.pt'
    dataset = root / 'training/all_free_dataset/dataset_vertical_all_free_unique.jsonl'
    report = json.loads(dataset.with_suffix('.jsonl.build_report.json').read_text())
    bases = {
        'interioragent': Path('/media/echo/data/voxroom_roomseg_evaluation/interioragent_all_available_gt_20260828'),
        'grscene': Path('/media/echo/data/voxroom_roomseg_evaluation/grscene_all_available_gt_20260817'),
    }
    for path in [baseline, dataset, root / 'code', *[base / name for base in bases.values() for name in ('index_voxroom.json', 'annotations', 'final_gt', 'training_scene_exclusion_manifest.json')]]:
        if not path.exists():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(f'Will not overwrite previous continuation: {output}')
    for name in bases:
        if (root / 'evaluation' / f'{name}_{args.run}').exists():
            raise FileExistsError(f'evaluation output already exists: {name}_{args.run}')
    if sha256(dataset) != report['output_index_sha256']:
        raise ValueError('all-free dataset checksum differs from its original build report')

    import torch
    original = torch.load(baseline, map_location='cpu', weights_only=True)
    if original['epoch'] != 2 or original['context_source'] != 'vertical':
        raise ValueError('expected original epoch-2 Vertical-Free checkpoint')
    if original['checkpoint_selection_metric'] != 'f1_at_fixed_threshold':
        raise ValueError('expected F1-selected baseline')
    if original['recommended_keep_threshold'] != 0.5:
        raise ValueError('expected fixed threshold 0.5')

    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(baseline, output / 'baseline_epoch2.pt')
    shutil.copy2(baseline.with_name('training_history.json'), output / 'previous_training_history.json')
    # Freeze this run's implementation; later workspace edits cannot alter it.
    code = output / 'code'
    shutil.copytree(root / 'code', code, ignore=shutil.ignore_patterns('__pycache__', '.git'))
    environment = dict(os.environ, PYTHONPATH=str(code), PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    state = {
        'run': args.run, 'variant': 'all_vertical_free_cells',
        'started_at': datetime.now(timezone.utc).isoformat(), 'status': 'prepared',
        'resume_kind': 'weights_only_warm_start_optimizer_and_rng_reset',
        'initial_checkpoint': str(baseline), 'initial_checkpoint_sha256': sha256(baseline),
        'initial_epoch': 2, 'initial_validation_f1': original['achieved_f1'],
        'dataset': str(dataset), 'dataset_sha256': report['output_index_sha256'],
        'temporal_row_policy': report['temporal_row_policy'], 'label_contract': report['label_contract'],
        'train_samples': report['split_counts']['train'], 'validation_samples': report['split_counts']['val'],
        'selection': 'validation F1 at 0.5, then precision and PR-AUC tie-breakers',
        'remaining_epoch_limit': 48, 'early_stopping_patience': 10,
        'epoch_numbering': 'new history epoch 1 starts from old epoch 2 weights; optimizer not restored',
        'frozen_code': str(code), 'evaluation_execution': 'sequential datasets, all saved checkpoints',
    }

    def update(status: str, **values: object) -> None:
        state.update(status=status, updated_at=datetime.now(timezone.utc).isoformat(), **values)
        write_json(output / 'recovery_status.json', state)
        print(f"[{state['updated_at']}] {status} {values}", flush=True)

    def execute(command: list[str], log: Path, pid_path: Path) -> None:
        with log.open('x') as stream:
            process = subprocess.Popen(command, cwd=code, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            pid_path.write_text(str(process.pid) + '\n')
            if process.wait() != 0:
                raise RuntimeError(f'command failed; see {log}')

    try:
        update('training')
        # Preserve the original batch, worker, cache, LR, augmentation and split.
        command = [sys.executable, '-u', str(code / 'voxroom_online/isaac_runtime/scripts/train_door_seed_classifier.py'),
            '--index', str(dataset), '--out-dir', str(output), '--init-checkpoint', str(output / 'baseline_epoch2.pt'),
            '--context-source', 'vertical', '--device', 'cuda:0', '--precision', 'bfloat16',
            '--batch-size', '128', '--num-workers', '4', '--snapshot-cache-size', '2',
            '--max-epochs', '48', '--patience', '10', '--early-stopping-metric', 'validation_score',
            '--checkpoint-selection-mode', 'fixed_f1', '--learning-rate', '3e-4', '--weight-decay', '1e-4',
            '--threshold-selection-mode', 'fixed', '--fixed-keep-threshold', '0.5', '--max-pos-weight', '10',
            '--seed', '0', '--train-rotation-degrees', '0,90,180,270', '--train-mirror-lr-once']
        write_json(output / 'training_command.json', command)
        execute(command, output / 'train.log', output / 'train.pid')
        summary = json.loads((output / 'training_summary.json').read_text())
        trained = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
        # The interrupted model remains eligible. Never replace a better old
        # validation result with a worse continuation just because it is newer.
        def score(checkpoint: dict) -> tuple:
            return tuple(checkpoint['checkpoint_selection_score'])
        winner = 'best.pt' if score(trained) > score(original) else 'baseline_epoch2.pt'
        checkpoint = output / winner
        write_json(output / 'final_selection.json', {
            'selected_checkpoint': str(checkpoint), 'selected_sha256': sha256(checkpoint),
            'original_validation_score': score(original), 'continuation_validation_score': score(trained),
            'epochs_completed_in_continuation': summary['epochs_completed'],
        })
        update('training_complete', selected_checkpoint=str(checkpoint))
        metrics = {}
        for name, base in bases.items():
            destination = root / 'evaluation' / f'{name}_{args.run}'
            destination.mkdir(parents=True, exist_ok=False)
            update('replaying', dataset_name=name)
            execute([sys.executable, '-u', str(code / 'scripts/replay_voxroom_ablation.py'),
                '--index', str(base / 'index_voxroom.json'), '--annotation-dir', str(base / 'annotations'),
                '--out-dir', str(destination), '--variant', 'all_vertical_free_cells',
                '--config', str(code / 'configs/voxroom_online.yaml'), '--checkpoint', str(checkpoint),
                '--device', 'cuda:0', '--inference-batch-size', '32'], destination / 'replay.log', destination / 'replay.pid')
            prediction = json.loads((destination / 'prediction_index.json').read_text())
            if not prediction.get('complete') or prediction['processed'] != prediction['snapshot_count']:
                raise RuntimeError(f'incomplete replay: {destination}')
            update('evaluating', dataset_name=name)
            execute([sys.executable, '-u', str(code / 'scripts/evaluate_voxroom_ablation.py'),
                '--source-index', str(base / 'index_voxroom.json'), '--prediction-index', str(destination / 'prediction_index.json'),
                '--annotation-dir', str(base / 'annotations'), '--gt-dir', str(base / 'final_gt'),
                '--exclusion-manifest', str(base / 'training_scene_exclusion_manifest.json'),
                '--out-dir', str(destination / 'metrics'), '--dataset', name], destination / 'evaluate.log', destination / 'evaluate.pid')
            metrics[name] = json.loads((destination / 'metrics/summary.json').read_text())
        rows = []
        for scope in ('all_approved', 'excluding_training_and_validation_scenes'):
            selected = [value['scopes'][scope] for value in metrics.values()]
            total = sum(item['checkpoint_evaluation_count'] for item in selected)
            if not total:
                raise ValueError('no checkpoint evaluations in ' + scope)
            for name, value in metrics.items():
                rows.append(dict(dataset=name, scope=scope, **value['scopes'][scope]))
            combined = {'dataset': 'combined_weighted', 'scope': scope, 'checkpoint_evaluation_count': total,
                'process_count': sum(item['process_count'] for item in selected)}
            for metric in ('Precision_percent', 'Recall_percent', 'F1_percent', 'mIoU_room_percent'):
                combined[metric] = sum(item[metric] * item['checkpoint_evaluation_count'] for item in selected) / total
            rows.append(combined)
        write_json(output / 'evaluation_summary.json', rows)
        with (output / 'evaluation_summary.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['dataset', 'scope', 'process_count', 'checkpoint_evaluation_count', 'Precision_percent', 'Recall_percent', 'F1_percent', 'mIoU_room_percent'], extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        update('complete')
    except Exception as error:
        update('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
