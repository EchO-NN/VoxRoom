#!/usr/bin/env python3
"""Export completed all-free-cell ablation results and training history to CSV.

No training, prediction, GT editing, or change to the archived measurements.
Held-out paper tables and all-approved diagnostic archives remain separate.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile

from organize_dude_occusg_csv import (
    DATASETS, EVENTS, METRICS, aggregate, key, normalize, scene_statistics,
    stability_summary,
)

ROOT = Path(__file__).resolve().parents[1]
METHOD = 'VoxRoom_ablation7_all_vertical_free_cells_recovery_epoch13'
RUN = 'all_vertical_free_cells_recovery_20260907'
CLASS_METRICS = {'accuracy', 'precision', 'recall', 'f1', 'pr_auc', 'roc_auc',
                 'negative_rejection_rate', 'rejected_seed_accuracy', 'rejected_seed_count',
                 'tp', 'fp', 'tn', 'fn', 'threshold'}


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def history_rows(history, stage, selected_epoch=None):
    """Keep classification validation metrics distinct from room metrics."""
    output = []
    for item in history:
        row = {'training_stage': stage, 'stage_epoch': item['epoch'],
               'selected_for_this_evaluation': item['epoch'] == selected_epoch,
               'metric_scope': 'validation_free_coordinate_classification_not_room_segmentation'}
        for field, value in item.items():
            if field == 'epoch':
                continue
            name = 'validation_' + field if field in CLASS_METRICS else field
            row[name] = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
        p, r = float(item['precision']), float(item['recall'])
        assert math.isclose(item['f1'], 2*p*r/(p+r) if p+r else 0, abs_tol=1e-10)
        output.append(row)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.zip').exists():
        raise FileExistsError('Choose a fresh export destination; previous packages are preserved')
    source, training = args.source.resolve(), args.source.resolve() / 'training'
    sources = set()
    def load(path):
        sources.add(path)
        return json.loads(path.read_text())
    recovery = load(training / 'recovery_status.json')
    selection = load(training / 'final_selection.json')
    training_summary = load(training / 'training_summary.json')
    augmentation = load(training / 'augmentation_manifest.json')
    command = load(training / 'training_command.json')
    history = load(training / 'training_history.json')
    previous_history = load(training / 'previous_training_history.json')
    assert recovery['status'] == 'complete' and recovery['variant'] == 'all_vertical_free_cells'
    assert training_summary['selected_checkpoint_epoch'] == 13 and training_summary['epochs_completed'] == 23
    assert len(history) == 23 and len(previous_history) == 2
    assert selection['selected_checkpoint'] == training_summary['checkpoint'] == recovery['selected_checkpoint']
    assert training_summary['checkpoint_selection_metric'] == 'f1_at_fixed_threshold'
    assert training_summary['fixed_keep_threshold'] == .5
    best = history[12]
    assert best['epoch'] == 13
    assert best['checkpoint_selection_score'] == selection['continuation_validation_score']
    assert list(max(history, key=lambda r: tuple(r['checkpoint_selection_score']))['checkpoint_selection_score']) == selection['continuation_validation_score']
    for metric, value in training_summary['selected_checkpoint_metrics'].items():
        assert math.isclose(float(best[metric]), float(value), abs_tol=1e-10), metric

    excluded_scenes, approved_universe, heldout_universe = set(), {}, {}
    prediction_indexes, summaries, raw_rows = {}, {}, []
    for dataset in DATASETS:
        frozen_base = ROOT / 'results/dude_corrected_20260910' / dataset
        excluded_scenes.update(load(frozen_base / 'input_audit.json')['excluded_scene_ids'])
        frozen_index = load(frozen_base / 'inputs/index_voxroom.json')
        for episode in frozen_index['episodes']:
            for item in episode['snapshots']:
                record = {'dataset': dataset, 'scene_id': episode['scene_id'], 'episode_uid': episode['episode_uid'],
                          'coverage_event_id': item['coverage_event_id'], 'step': int(item['step']),
                          'source_snapshot_path': item['snapshot_path']}
                heldout_universe[key(record)] = record
        index = load(source / dataset / 'prediction_index.json')
        prediction_indexes[dataset] = index
        assert index['complete'] and index['processed'] == index['snapshot_count']
        assert index['candidate_policy'] == 'vertical_free_all' and index['variant'] == 'all_vertical_free_cells'
        assert index['checkpoint_sha256'] == selection['selected_sha256']
        learning = index['learning_config']
        assert learning['raw_seed_source'] == 'vertical_free_all'
        assert learning['context_source'] == 'vertical' and learning['keep_threshold'] == .5
        assert learning['local_voxel_patch_size'] == 19 and learning['context_patch_size'] == 41
        assert not learning['fallback_to_rule_seed_on_error']
        for episode in index['episodes']:
            for item in episode['snapshots']:
                event = item['coverage_event_id']
                assert event in EVENTS
                record = {'dataset': dataset, 'scene_id': episode['scene_id'], 'episode_uid': episode['episode_uid'],
                          'coverage_event_id': event, 'step': int(item['step']),
                          'progress': 'final' if event == 'final' else str(int(event[-3:]))+'%',
                          'actual_coverage_percent': 100*float(item['coverage_ratio']),
                          'source_snapshot_path': item['source_snapshot_path'], 'prediction_path': item['snapshot_path'],
                          'runtime_seconds': float(item['runtime_seconds']),
                          'candidate_free_cells': int(item['raw_seed_cells']), 'network_kept_seed_cells': int(item['kept_seed_cells'])}
                assert key(record) not in approved_universe
                approved_universe[key(record)] = record
        summary = load(source / dataset / 'metrics/summary.json')
        summaries[dataset] = summary
        assert summary['min_room_area_m2'] == .5 and summary['min_room_area_cells'] == 200
        path = source / dataset / 'metrics/per_checkpoint_metrics.csv'
        sources.add(path)
        raw_rows.extend((row, path) for row in read_csv(path))

    all_rows = []
    for raw, path in raw_rows:
        assert raw['variant'] == 'all_vertical_free_cells'
        row = normalize(raw, METHOD, approved_universe, path)
        ref = approved_universe[key(row)]
        assert row['prediction_path'] == ref['prediction_path']
        assert math.isclose(float(raw['coverage_ratio'])*100, row['actual_coverage_percent'], abs_tol=1e-10)
        for metric in METRICS:
            assert math.isclose(float(raw[metric])*100, float(raw[metric+'_percent']), abs_tol=1e-10)
        row.pop('source_input_sha256')  # This run did not log per-grid hashes; do not invent them.
        row.update({k: ref[k] for k in ('runtime_seconds', 'candidate_free_cells', 'network_kept_seed_cells')})
        assert 0 <= row['network_kept_seed_cells'] <= row['candidate_free_cells']
        row['network_rejected_free_cells'] = row['candidate_free_cells']-row['network_kept_seed_cells']
        row['network_keep_percent'] = 100*row['network_kept_seed_cells']/row['candidate_free_cells'] if row['candidate_free_cells'] else None
        row['heldout_test_scene'] = row['scene_id'] not in excluded_scenes
        row['checkpoint_sha256'] = selection['selected_sha256']
        all_rows.append(row)
    all_rows.sort(key=lambda r: (r['dataset'], r['scene_id'], EVENTS.index(r['coverage_event_id'])))
    assert len(all_rows) == len(approved_universe) == 598
    assert len({key(r) for r in all_rows}) == 598
    heldout = [r for r in all_rows if r['heldout_test_scene']]
    assert len(heldout) == 462 and {key(r) for r in heldout} == set(heldout_universe)
    for row in heldout:
        for field in ('scene_id', 'step', 'source_snapshot_path'):
            assert row[field] == heldout_universe[key(row)][field]
    heldout_universe = {key(r): approved_universe[key(r)] for r in heldout}
    anchor_path = ROOT / 'results/dude_occusg_csv_completed_20260911/dude_per_checkpoint_metrics.csv'
    sources.add(anchor_path)
    anchor = {key(r): r for r in read_csv(anchor_path) if r['method_id'] == 'DUDE_online_vertical_tau2p5'}
    assert set(anchor) == set(heldout_universe)
    for row in heldout:
        for field in ('n_gt', 'metric_domain_pixels', 'step'):
            assert row[field] == int(anchor[key(row)][field])
    for dataset in DATASETS:
        for scope, rows in (('all_approved', all_rows), ('excluding_training_and_validation_scenes', heldout)):
            selected = [r for r in rows if r['dataset'] == dataset]
            original = summaries[dataset]['scopes'][scope]
            assert len(selected) == original['checkpoint_evaluation_count']
            for raw, label in (('precision', 'Precision'), ('recall', 'Recall'), ('f1', 'F1'), ('miou_room', 'mIoU_room')):
                assert math.isclose(statistics.fmean(r[raw]*100 for r in selected), original[label+'_percent'], abs_tol=1e-9)

    args.output.mkdir(parents=True)
    files = []
    def write(name, rows, description):
        assert rows
        path = args.output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with path.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        files.append({'file': name, 'row_count': len(rows), 'description': description, 'sha256': digest(path)})
    write('ablation7_per_checkpoint_metrics.csv', heldout, '论文测试集：68 场景、462 进度；P/R/F1/mIoU、房间数、候选/保留数和离线回放耗时')
    overall = [aggregate(METHOD, heldout, heldout_universe, d, e) for d in ('combined', *DATASETS) for e in ('all_available_progress', 'final')]
    progress = [aggregate(METHOD, heldout, heldout_universe, d, e) for d in ('combined', *DATASETS) for e in EVENTS]
    write('ablation7_overall_metrics.csv', overall, '独立测试集：合并加权及两数据集分项的全过程均值和 final')
    write('ablation7_progress_metrics.csv', progress, '独立测试集：20/40/60/70/80/90/final 分进度统计')
    scenes = scene_statistics(METHOD, heldout, heldout_universe)
    write('ablation7_per_scene_stability.csv', scenes, '逐场景对已有进度先算 mean、总体 SD、CV、worst；原探索没存的进度不补齐')
    write('ablation7_stability_summary.csv', [stability_summary(METHOD, scenes, d) for d in ('combined', *DATASETS)], '各场景稳定性统计再平均，不把所有场景混起来求 SD')
    write('training_history.csv', history_rows(history, 'recovery_20260907_weights_only_warm_start', 13), '23 轮续训完整 loss 和验证分类指标；validation_f1 不是房间分割 F1')
    write('training_history_previous_stage.csv', history_rows(previous_history, 'original_interrupted_stage'), '原始 2 轮训练历史，与续训轮次分开编号；续训重置了优化器和随机状态')
    chosen = {'method_id': METHOD, 'selected_training_stage': 'recovery_20260907', 'selected_stage_epoch': 13,
              'stage_epochs_completed': 23, 'stop_reason': training_summary['stop_reason'],
              'selected_checkpoint': selection['selected_checkpoint'], 'checkpoint_sha256': selection['selected_sha256'],
              'selection_metric': 'validation_free_coordinate_classification_f1_at_0p5',
              **{'classification_'+k: v for k, v in training_summary['selected_checkpoint_metrics'].items()}}
    write('selected_checkpoint.csv', [chosen], '选用权重、哈希与验证集分类成绩；不作为独立测试分割成绩')
    write('training_summary.csv', [{'item': k, 'value': json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v} for k, v in training_summary.items()], '完整训练汇总；嵌套值保留为 JSON 单元格')
    write('training_augmentation.csv', [{'item': k, 'value': json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v} for k, v in augmentation.items()], '基础样本数、0/90/180/270 旋转及一次左右镜像、验证集大小')
    write('run_provenance.csv', [{'item': k, 'value': json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v} for k, v in recovery.items()]+[{'item': 'training_command_argv_json', 'value': json.dumps(command, ensure_ascii=False)}], '运行状态、冻结代码、原始 checkpoint、数据集哈希和训练命令')
    excluded_counts = {s: sum(r['scene_id'] == s for r in all_rows) for s in excluded_scenes}
    write('excluded_training_validation_scenes.csv', [{'scene_id': s, 'excluded_recorded_checkpoint_count': excluded_counts[s]} for s in sorted(excluded_scenes)], '从所有论文表中排除的训练/验证场景；0 表示此次没有该场景的已批准评测记录')
    archive = 'archive_all_approved_not_heldout/'
    write(archive+'ablation7_per_checkpoint_metrics_all_approved.csv', all_rows, '598 条原始已批准记录，包含训练/验证场景！只有 heldout_test_scene=True 可进入论文测试表')
    write(archive+'ablation7_overall_metrics_all_approved.csv', [aggregate(METHOD, all_rows, approved_universe, d, e) for d in ('combined', *DATASETS) for e in ('all_available_progress', 'final')], '包含训练/验证场景的存档汇总，不用于论文独立测试结论')
    definitions = [
        ('method', '第七项消融：取消 TVARS 和 voxel raw-seed 预筛，网络逐个推理所有 Vertical-Free 坐标；使用 19×19 三维体素与 41×41 Vertical-Free 二维上下文，固定阈值 0.5'),
        ('labels', recovery['label_contract']),
        ('training_temporal_policy', recovery['temporal_row_policy']),
        ('resume', '从旧 epoch 2 权重继续训练；优化器、RNG 重置，并非精确断点续训。续训完成 23 轮，验证 F1 选中续训 epoch 13；不用测试集选权重'),
        ('test_scope', '主 CSV 只含排除训练和验证场景后的 68 场景/462 点，IA 10/64、GR 58/398；没有跳过 DUDE 崩溃点，那个排除只属于 DUDE 离线'),
        ('scope_completeness', '第七项全部 598 个已批准预测完成；独立测试 462 点全齐。包含训练/验证场景的 598 点结果只放 archive_all_approved_not_heldout'),
        ('combined_weighting', '对有效 scene×checkpoint 指标取均值，即按 IA/GR 实际 checkpoint 数加权；不是两数据集各占 50%'),
        ('F1', '每个进度点先用 P/R 算调和平均，再对进度取均值；不同于先平均 P/R 再算 F1'),
        ('mIoU', '预测与 GT 房间 Hungarian 一对一匹配，IoU 总和除以 GT 房间数，未匹配 GT 计 0'),
        ('P_R', 'P：每个预测房间最大 GT 重叠面积/预测面积再平均；R：每个 GT 最大预测重叠面积/GT 面积再平均'),
        ('room_cutoff', '沿用 0.5 m² 小房间过滤、0.05 m 栅格分辨率，等于 200 格；GT 和评测域与其它对照一致'),
        ('stability', '每个场景已有进度先算 mean、总体 SD(ddof=0)、CV=SD/mean、worst=min，再跨场景等权平均；非 pooled SD。零均值 CV 留空'),
        ('units', '逐点 precision/recall/f1/miou_room 为 0..1；P_percent/R_percent/F1_percent/mIoU_percent 为百分数；SD_pp 为百分点，CV_ratio 与 CV_percent 分开'),
        ('history_metric_scope', 'training_history 的 validation_* 为 free 坐标分类验证指标，不是房间分割指标；train_loss 和 val_loss 单独列出'),
        ('progress', '20/40/60/70/80/90/final。final 是实际探索终点，不代表 100%；原探索未保存的进度不插值补齐'),
        ('candidate_counts', 'candidate_free_cells 是该点实际送入网络的全部 Vertical-Free 坐标数，不是几何预筛后的 raw-seed 数'),
        ('timing', 'runtime_seconds 来自离线分割回放记录；不是机器人探索总时间，不是纯网络 forward 计时'),
        ('export', '只整理已有结果，没有重训、重跑分割、修改 GT 或覆盖旧结果')]
    write('protocol.csv', [{'item': k, 'definition': v} for k, v in definitions], '实验定义、评测协议、单位、范围及避免混淆的说明')
    sources.update({Path(__file__).resolve(), ROOT/'scripts/organize_dude_occusg_csv.py', training/'evaluation_summary.csv'})
    write('source_manifest.csv', [{'source_path': str(p), 'sha256': digest(p)} for p in sorted(sources)], '结果来源、生成脚本及 SHA256')
    write('files.csv', list(files), '文件导航、行数及校验和')
    combined = next(r for r in overall if r['dataset_scope'] == 'combined' and r['progress_scope'] == 'all_available_progress')
    (args.output/'README.md').write_text(f'''# 第七项消融 CSV 数据包

主表为独立测试集：68 场景、462 个有效进度点，已排除训练/验证场景。两个数据集按进度点数加权。

合并全过程：P {combined['P_percent']:.6f}%，R {combined['R_percent']:.6f}%，F1 {combined['F1_percent']:.6f}%，mIoU {combined['mIoU_percent']:.6f}%。

先看 `files.csv`。逐点在 `ablation7_per_checkpoint_metrics.csv`，总分与分进度在 `ablation7_overall_metrics.csv`、`ablation7_progress_metrics.csv`。逐场景及汇总 mean/SD/CV/worst 另列。

`training_history.csv` 为续训 23 轮的 loss/验证分类指标，采用第 13 轮最佳权重；原 2 轮历史单独保存。续训从旧模型权重开始，但重置优化器和随机状态。分类 F1 不能混用为房间分割 F1。

`archive_all_approved_not_heldout/` 仅为完整原始记录存档，包含训练/验证场景，不用于独立测试结论。此次没有沿用 DUDE 离线的单点排除：第七项独立测试 462 点全部完成。

所有 CSV 为 UTF-8 BOM，可直接用 Excel 打开。具体方法、指标、统计和数据标签定义见 `protocol.csv`。
''', encoding='utf-8')
    with zipfile.ZipFile(args.output.with_suffix('.zip'), 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(args.output.rglob('*')):
            if path.is_file():
                z.write(path, path.relative_to(args.output))
    print(json.dumps({'directory': str(args.output), 'archive': str(args.output.with_suffix('.zip')),
                      'heldout_records': len(heldout), 'all_approved_archived_records': len(all_rows),
                      'training_history_epochs': len(history), 'csv_files': len(files), 'combined': combined}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
