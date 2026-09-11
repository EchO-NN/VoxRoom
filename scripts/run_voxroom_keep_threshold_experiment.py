#!/usr/bin/env python3
"""Single-threshold replay, reusing the untouched 0.25 experiment as control."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import csv
import json
import math
import multiprocessing
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_voxroom_line_threshold_experiment as replay

CONTROL = 'VoxRoom_corr0p9_var0p1_keep0p25_Lboth'
ORIGINAL = 'VoxRoom_original_parameters_replay'


def variant(keep):
    if not math.isfinite(keep) or not 0 < keep < 1:
        raise ValueError('keep threshold must be strictly between 0 and 1')
    parameters = dict(replay.METHODS[CONTROL], keep=keep)
    method = 'VoxRoom_corr0p9_var0p1_keep' + format(keep, '.8g').replace('.', 'p') + '_Lboth'
    if method in (CONTROL, ORIGINAL):
        raise ValueError('Choose a new threshold, not an existing control')
    return method, parameters


def run_variant_scene(tasks, base, checkpoint, out, device, protocol_hash, keep):
    # Only the worker's private module is changed, never the running control's
    # source/config/process. Reuse its entire audited inference/metric pipeline.
    method, parameters = variant(keep)
    previous = replay.METHODS
    try:
        replay.METHODS = {method: parameters}
        return replay.run_scene(tasks, base, checkpoint, out, device, protocol_hash)
    finally:
        replay.METHODS = previous


def load_control_rows(path, universe):
    rows = []
    if not path.exists():
        return rows
    with path.open(newline='', encoding='utf-8-sig') as stream:
        for source in csv.DictReader(stream):
            if source['method_id'] not in (CONTROL, ORIGINAL):
                continue
            k = replay.key(source)
            if k not in universe:
                continue
            task = universe[k]
            assert source['source_sha256'] == task['source_sha256']
            assert source['scene_id'] == task['scene_id']
            assert int(source['step']) == task['step']
            row = dict(source)
            for metric in replay.METRICS:
                row[metric] = float(row[metric])
                assert math.isfinite(row[metric]) and 0 <= row[metric] <= 1
            for field in ('step', 'n_gt', 'n_pred', 'metric_domain_pixels'):
                row[field] = int(row[field])
            assert row['n_gt'] == task['reference_n_gt']
            assert row['metric_domain_pixels'] == task['reference_metric_domain_pixels']
            rows.append(row)
    assert len(rows) == len({(r['method_id'], replay.key(r)) for r in rows})
    return rows


def common_rows(rows, controls, method):
    groups = {m: [r for r in rows + controls if r['method_id'] == m]
              for m in (ORIGINAL, CONTROL, method)}
    keys = [{replay.key(r) for r in records} for records in groups.values()]
    common = set.intersection(*keys)
    return {m: [r for r in records if replay.key(r) in common]
            for m, records in groups.items()}, common


def export_groups(out, groups, universe, prefix=''):
    summaries, progress, per_scene, stability = [], [], [], []
    for method, records in groups.items():
        for dataset in ('combined', 'interioragent', 'grscene'):
            summaries.append(replay.aggregate(method, records, universe, dataset, 'all_available_progress'))
            for scope in replay.EVENTS:
                progress.append(replay.aggregate(method, records, universe, dataset, scope))
        scenes = replay.scene_statistics(method, records, universe)
        per_scene.extend(scenes)
        stability.extend(replay.stability_summary(method, scenes, ds)
                         for ds in ('combined', 'interioragent', 'grscene'))
    for filename, records in (('aggregate', summaries), ('progress_metrics', progress),
                              ('per_scene_stability', per_scene), ('stability_summary', stability)):
        replay.csv_save(out / (prefix + filename + '.csv'), records)
    return summaries


def export(out, tasks, rows, failures, method, keep, compare_dir, compare_hash):
    assert replay.digest(compare_dir / 'protocol.json') == compare_hash, 'control protocol changed'
    universe = {replay.key(t): t for t in tasks}
    assert len(rows) == len({replay.key(r) for r in rows})
    rows = sorted(rows, key=lambda r: (r['dataset'], r['scene_id'], replay.EVENTS.index(r['coverage_event_id'])))
    replay.csv_save(out / 'per_checkpoint_metrics.csv', rows)
    complete = len(rows) == len(tasks) and not failures
    controls = load_control_rows(compare_dir / 'per_checkpoint_metrics.csv', universe)
    paired, keys = common_rows(rows, controls, method)
    summary = export_groups(out, {method: rows}, universe)
    comparison = export_groups(out, paired, universe, 'comparison_')
    replay.csv_save(out / 'comparison_per_checkpoint_metrics.csv', [r for records in paired.values() for r in records])
    replay.json_save(out / 'status.json', {'status': 'complete' if complete else 'running_or_incomplete',
        'completed_predictions': len(rows), 'expected_predictions': len(tasks),
        'completed_scenes': len({r['episode_uid'] for r in rows}), 'failed_scenes': len(failures),
        'paired_checkpoint_count': len(keys), 'comparison_complete': len(keys) == len(tasks),
        'updated_at': time.time()})
    replay.json_save(out / 'failures.json', failures)
    report = [f'# VoxRoom：NN 阈值 {keep:g} 重测', '',
        f'状态：{"全部完成" if complete else "运行中/未完成"}；{len(rows)}/{len(tasks)} 个预测已汇总；失败场景 {len(failures)}。', '',
        f'冻结范围：{len({t["episode_uid"] for t in tasks})} 个场景；排除训练和验证场景。',
        f'相对于 0.25 轮只改 NN 保留阈值为 {keep:g}；线性相关性 0.9、正交方差 0.1 格子²、L 型横竖双分支保持不变。',
        '相同 Vertical Free 输入、epoch 14 权重、已保存的 TVARS ∪ VoxRoom raw seed；不重训。',
        '使用原始体素快照重跑分割全链路，不重新运行探索；每个场景按现有进度顺序重建独立门线记忆。',
        '预测保存后才访问 GT；每点独立复核 P/R/F1/mIoU、评测面积和 GT 房间数；不改 GT、预测图或指标。',
        '两个数据集合并按进度点数加权；小于 0.5 m² 的房间忽略。SD/CV 先在同场景实际进度点内计算，再跨场景平均。',
        '这是在已经查看测试集结果后进行的阈值敏感性实验，不能称为未调参测试集泛化结论。', '',
        '## 本轮已完成数据', '', '| F1 (%) | mIoU (%) | 已完成点数 |', '| ---: | ---: | ---: |']
    r = summary[0]
    def score(value):
        return '—' if value is None else f'{value:.3f}'
    report.append(f'| {score(r["F1_percent"])} | {score(r["mIoU_percent"])} | {len(rows)}/{len(tasks)} |')
    report.extend(['', f'## 严格同点对比：当前共同完成 {len(keys)} 个进度点', '',
        '仅复用 0.25 轮已完成的原参数和 0.25 结果，不重复跑这两个对照。各行比较的是完全相同的场景/进度点，未完成部分不补零。', '',
        '| 方法 | P (%) | R (%) | F1 (%) | mIoU (%) |', '| --- | ---: | ---: | ---: | ---: |'])
    names = {ORIGINAL: '原参数：0.95 / 0.65 / NN 0.5，无 L 双支臂',
             CONTROL: '0.9 / 0.1 / L 双支臂 / NN 0.25', method: f'0.9 / 0.1 / L 双支臂 / NN {keep:g}'}
    for r in comparison:
        if r['dataset_scope'] == 'combined':
            report.append('| ' + names[r['method_id']] + ' | ' + ' | '.join(
                score(r[k + '_percent']) for k in ('P', 'R', 'F1', 'mIoU')) + ' |')
    report.extend(['', '逐点/分进度/合并指标和逐场景稳定性分别见 `per_checkpoint_metrics.csv`、`progress_metrics.csv`、`aggregate.csv`、`per_scene_stability.csv`、`stability_summary.csv`。',
        '同点对比文件以 `comparison_` 为前缀；完整配置、源代码/权重 hash 见 `protocol.json`。',
        f'对照结果目录：`{compare_dir}`。本轮结束后若对照尚有未完成点，可用 `--resume` 复用本轮结果重新汇总。'])
    (out / 'report.md').write_text('\n'.join(report) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compare-dir', type=Path, default=ROOT / 'results/voxroom_line_threshold_full_20260912')
    p.add_argument('--keep-threshold', type=float, default=.2)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.workers < 1:
        p.error('--workers must be positive')
    method, parameters = variant(args.keep_threshold)
    out, compare_dir = args.out_dir.resolve(), args.compare_dir.resolve()
    if out == compare_dir or compare_dir in out.parents or out in compare_dir.parents:
        raise ValueError('New output must be separate from the control directory')
    if out.exists() and not args.resume:
        raise FileExistsError('Choose a new directory, or explicitly --resume')
    previous = json.loads((compare_dir / 'protocol.json').read_text())
    compare_hash = replay.digest(compare_dir / 'protocol.json')
    assert previous['methods'][CONTROL] == replay.METHODS[CONTROL]
    for filename, expected in previous['source_hashes'].items():
        assert replay.digest(ROOT / filename) == expected, ('control code/config changed', filename)
    checkpoint = Path(previous['checkpoint'])
    assert replay.digest(checkpoint) == previous['checkpoint_sha256']
    tasks = json.loads((compare_dir / 'frozen_tasks.json').read_text())
    assert len(tasks) == previous['checkpoint_count']
    assert len(tasks) == len({replay.key(t) for t in tasks})
    assert not {t['scene_id'] for t in tasks} & set(previous['excluded_scene_ids'])
    base = copy.deepcopy(previous['base_roomseg_config'])
    cfg = replay.segmenter_config(base, parameters, checkpoint, previous['device'])
    ref = replay.segmenter_config(base, previous['methods'][CONTROL], checkpoint, previous['device'])
    ref['door_seed_learning']['keep_threshold'] = args.keep_threshold
    assert cfg == ref, 'More than NN threshold differs from 0.25 control'
    scenes = defaultdict(list)
    for task in tasks:
        scenes[(task['dataset'], task['episode_uid'])].append(task)
    for scene in scenes.values():
        scene.sort(key=lambda t: replay.EVENTS.index(t['coverage_event_id']))
    protocol = {**previous, 'experiment': 'voxroom_keep_threshold_single_variable',
        'methods': {method: parameters}, 'workers': args.workers,
        'source_hashes': {**previous['source_hashes'], str(Path(__file__).resolve().relative_to(ROOT)): replay.digest(Path(__file__))},
        'control_directory': str(compare_dir), 'control_protocol_sha256': compare_hash,
        'frozen_tasks_sha256': replay.digest(compare_dir / 'frozen_tasks.json'),
        'only_changed_parameter_vs_control': 'door_seed_learning.keep_threshold',
        'actual_effective_roomseg_config': cfg, 'candidate_and_metric_policy': 'identical to control',
        'comparison_policy': 'intersection of completed checkpoint keys; no imputation, no repeated control inference'}
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == protocol, 'resume protocol changed'
    else:
        replay.json_save(path, protocol)
    protocol_hash = replay.digest(path)
    replay.json_save(out / 'frozen_tasks.json', tasks)
    rows, failures = [], []
    export(out, tasks, rows, failures, method, args.keep_threshold, compare_dir, compare_hash)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(run_variant_scene, scene, base, str(checkpoint), str(out), previous['device'],
                               protocol_hash, args.keep_threshold): scene for scene in scenes.values()}
        for future in as_completed(futures):
            try:
                rows.extend(future.result())
            except Exception:
                failures.append({'scene_id': futures[future][0]['scene_id'], 'traceback': traceback.format_exc()})
                print(failures[-1]['traceback'], flush=True)
            export(out, tasks, rows, failures, method, args.keep_threshold, compare_dir, compare_hash)
            print(f'COMPLETE_PREDICTIONS {len(rows)}/{len(tasks)} FAILURES {len(failures)}', flush=True)
    if failures:
        raise RuntimeError(f'{len(failures)} scenes failed; retained outputs, no zero-filled metrics')


if __name__ == '__main__':
    main()
