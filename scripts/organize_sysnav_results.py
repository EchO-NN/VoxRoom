#!/usr/bin/env python3
"""Package the full-input SysNav replay without modifying measured results."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import statistics as stats
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/sysnav_source_roomseg_fullinput_20260905'
OUT = ROOT / 'results/sysnav_paper_tables_20260907'
EVENTS = ['milestone_020', 'milestone_040', 'milestone_060', 'milestone_070', 'milestone_080', 'milestone_090', 'final']
LABELS = dict(zip(EVENTS, ['20%', '40%', '60%', '70%', '80%', '90%', 'Final']))
METRICS = {'precision': 'P', 'recall': 'R', 'f1': 'F1', 'miou_room': 'room_mIoU'}
EXCLUSIONS = [
    ROOT / 'results/interioragent_paper_baselines_20260828/training_scene_exclusion_manifest.json',
    ROOT / 'results/sysnav_source_roomseg_20260905/gt_cache/grscene/training_scene_exclusion_manifest.json',
]


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def write_csv(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(rows, label):
    return {'scope': label, 'scene_count': len({(r['dataset'], r['scene_id']) for r in rows}),
        'evaluation_count': len(rows),
        **{key + '_percent': stats.fmean(100 * float(r[key]) for r in rows) for key in METRICS},
        'mean_gt_rooms': stats.fmean(int(r['n_gt']) for r in rows),
        'mean_predicted_rooms': stats.fmean(int(r['n_pred']) for r in rows)}


def main():
    summary = json.loads((SOURCE / 'summary.json').read_text())
    assert summary['method'] == 'cmu_sysnav_official_room_segmentation_full_snapshot_input'
    all_rows = read_csv(SOURCE / 'per_checkpoint_metrics.csv')
    prediction_rows = read_csv(SOURCE / 'prediction_manifest.csv')
    assert len(prediction_rows) == summary['prediction_checkpoint_count']
    assert all(r['status'] in {'written', 'resumed'} for r in prediction_rows)
    assert all(Path(r['output']).is_file() for r in prediction_rows)
    excluded = set().union(*(set(json.loads(p.read_text())['excluded_scene_ids']) for p in EXCLUSIONS))
    for row in all_rows:
        assert (row['excluded_as_training_or_validation'] == 'True') == (row['scene_id'] in excluded)
        p, r, f1 = (float(row[k]) for k in ('precision', 'recall', 'f1'))
        assert math.isclose(f1, 2*p*r/(p+r) if p+r else 0, abs_tol=1e-12)
        for key in METRICS:
            value = float(row[key])
            assert math.isfinite(value) and 0 <= value <= 1
            assert math.isclose(100*value, float(row[key+'_percent']), abs_tol=1e-10)
    rows = [r for r in all_rows if r['dataset'] in {'interioragent', 'grscene'} and r['scene_id'] not in excluded]
    assert len({(r['dataset'], r['scene_id'], r['coverage_event_id']) for r in rows}) == len(rows)
    assert all(r['coverage_event_id'] in EVENTS for r in rows)
    rows.sort(key=lambda r: (r['dataset'], r['scene_id'], EVENTS.index(r['coverage_event_id'])))
    groups = defaultdict(list)
    for row in rows:
        groups[row['dataset'], row['scene_id']].append(row)
    assert len(groups) == 68 and len(rows) == 472
    assert Counter(map(len, groups.values())) == {7: 64, 6: 4}

    progress = [dict(checkpoint=event, checkpoint_label=LABELS[event], **aggregate([r for r in rows if r['coverage_event_id'] == event], 'combined_held_out')) for event in EVENTS]
    overall = [aggregate(rows, 'all_checkpoints_weighted'), aggregate([r for r in rows if r['coverage_event_id'] == 'final'], 'final_only')]
    # Independently recompute the existing full-input summary and fail if it disagrees.
    evaluation = summary['evaluation']
    for result, previous in [(overall[0], evaluation['paper_datasets_excluding_training_all_checkpoints']),
                             (overall[1], evaluation['paper_datasets_excluding_training_final']),
                             *[(r, evaluation['by_checkpoint_paper_datasets_excluding_training'][r['checkpoint']]) for r in progress]]:
        assert result['evaluation_count'] == previous['evaluation_count']
        for key in METRICS:
            assert math.isclose(result[key+'_percent'], previous[key+'_percent'], abs_tol=1e-9)

    per_scene, scene_list = [], []
    for (dataset, scene), selected in sorted(groups.items()):
        present = [r['coverage_event_id'] for r in selected]
        item = {'dataset': dataset, 'scene_id': scene, 'episode_uid': selected[0]['episode_uid'],
            'progress_point_count': len(selected), 'progress_points': ';'.join(present)}
        scene_list.append({**item, 'missing_points': ';'.join(e for e in EVENTS if e not in present)})
        for key in METRICS:
            values = [100*float(r[key]) for r in selected]
            mean, sd = stats.fmean(values), stats.pstdev(values)
            assert mean > 0  # Do not silently turn undefined CV into zero.
            item.update({key+'_mean_percent': mean, key+'_sd_pp': sd,
                key+'_cv_percent': 100*sd/mean, key+'_worst_percent': min(values)})
        per_scene.append(item)
    stability = [{'metric': label, 'scene_count': len(groups),
        'mean_percent': stats.fmean(r[key+'_mean_percent'] for r in per_scene),
        'mean_within_scene_sd_pp': stats.fmean(r[key+'_sd_pp'] for r in per_scene),
        'mean_within_scene_cv_percent': stats.fmean(r[key+'_cv_percent'] for r in per_scene),
        'mean_scene_worst_percent': stats.fmean(r[key+'_worst_percent'] for r in per_scene)} for key, label in METRICS.items()]
    by_dataset = [aggregate([r for r in rows if r['dataset'] == dataset], dataset+'_all_checkpoints') for dataset in ('interioragent', 'grscene')]
    excluded_rows = [r for r in all_rows if r['dataset'] in {'interioragent', 'grscene'} and r['scene_id'] in excluded]

    OUT.mkdir(exist_ok=True)
    write_csv('sysnav_progress_metrics.csv', progress)
    write_csv('sysnav_overall_metrics.csv', overall)
    write_csv('sysnav_per_checkpoint_metrics.csv', rows)
    write_csv('sysnav_per_scene_stability.csv', per_scene)
    write_csv('sysnav_stability_summary.csv', stability)
    write_csv('sysnav_test_scenes.csv', scene_list)
    write_csv('sysnav_dataset_audit.csv', by_dataset)
    old_scene_rows = read_csv(ROOT / 'results/non_ablation_comparisons_20260831/comparison_per_scene_stability.csv')
    old_scenes = {(r['dataset'], r['scene_id']) for r in old_scene_rows}
    additional = sorted(set(groups) - old_scenes)
    provenance = {'created_at': datetime.now(timezone.utc).isoformat(), 'source_directory': str(SOURCE),
        'method': summary['method'], 'official_repository': summary['sysnav_repository'],
        'official_commit': summary['sysnav_commit'], 'input_contract': summary['algorithm_input_contract'],
        'source_sha256': {str(p): digest(p) for p in [SOURCE/'summary.json', SOURCE/'per_checkpoint_metrics.csv', SOURCE/'prediction_manifest.csv', *EXCLUSIONS]},
        'excluded_scene_ids': sorted(excluded),
        'excluded_approved_paper_scene_count': len({(r['dataset'], r['scene_id']) for r in excluded_rows}),
        'excluded_approved_paper_checkpoint_count': len(excluded_rows),
        'all_prediction_scenes': summary['prediction_scene_count'], 'all_predictions': len(prediction_rows),
        'all_approved_metric_count': len(all_rows), 'paper_test_scenes': len(groups), 'paper_test_checkpoints': len(rows),
        'scene_counts': dict(Counter(k[0] for k in groups)), 'progress_count_distribution': dict(Counter(map(len, groups.values()))),
        'gt_matching_policy_counts': dict(Counter(r['gt_match'] for r in rows)),
        'additional_scenes_vs_previous_comparison_table': additional,
        'aggregation': 'checkpoint-count weighted; F1 calculated per checkpoint before averaging',
        'stability': 'per scene over its actual checkpoints: population SD (ddof=0), CV=100*SD/mean, minimum; then scene-equal mean',
        'scope_caveat': 'Existing comparison table has 66 scenes / 458 points. This run has 68 / 472. Same scene/event names need not imply identical episode or voxel snapshot.',
        'excluded_obsolete_runs': ['sysnav_source_roomseg_20260905 (incomplete input)', 'sgnav_source_roomseg_20260905 (not SysNav)']}
    (OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False)+'\n')
    lines = ['# SysNav 完整输入版结果整理', '',
        '采用已完成的 `sysnav_source_roomseg_fullinput_20260905`，从逐进度 CSV 重新计算并核对原汇总；未重新运行算法、未修改预测或 GT。', '',
        '## 统计口径', '',
        '- 仅 InteriorAgent 与 GRScene，排除全部训练和验证场景；不包含 Habitat。保留 68 个测试场景（10 + 58）、472 个评测点。',
        '- 两套数据按有效评测点数加权合并，不对两个数据集均值简单各取一半。每个进度均为场景等权。',
        '- 包含 20%、40%、60%、70%、80%、90%、Final；64 个场景有 7 点，4 个场景缺少 90% 点，按实际 6 点统计，不补值，不丢弃整个场景。Final 指探索终点，不等于一定到达 100%。',
        '- P/R 是房间区域指标，不是神经网络 seed 分类指标；每个 checkpoint 先计算 F1=2PR/(P+R)，再平均。room-mIoU 为一对一房间匹配指标；过滤小于 0.5 平方米的房间。',
        '- 稳定性先逐场景计算 Mean、总体 SD（ddof=0）、CV=SD/Mean×100% 和 Worst=min，再跨场景等权平均。表中 Avg.Worst 是各场景最差进度的平均，不是全测试集单个最差值。', '',
        '## 各探索进度（合并独立测试集）', '',
        '| 进度 | 场景/点数 | P (%) | R (%) | F1 (%) | room-mIoU (%) |',
        '|---|---:|---:|---:|---:|---:|']
    for r in progress:
        lines.append(f"| {r['checkpoint_label']} | {r['scene_count']} | {r['precision_percent']:.3f} | {r['recall_percent']:.3f} | {r['f1_percent']:.3f} | {r['miou_room_percent']:.3f} |")
    r = overall[0]
    lines.append(f"| 全过程加权平均 | {r['evaluation_count']} 点 | {r['precision_percent']:.3f} | {r['recall_percent']:.3f} | {r['f1_percent']:.3f} | {r['miou_room_percent']:.3f} |")
    lines.extend(['', '## 同场景过程稳定性', '',
        '| 指标 | 场景内 Mean 的平均 (%) | SD 的平均 (百分点) | CV 的平均 (%) | Avg.Worst (%) |',
        '|---|---:|---:|---:|---:|'])
    for r in stability:
        lines.append(f"| {r['metric']} | {r['mean_percent']:.3f} | {r['mean_within_scene_sd_pp']:.3f} | {r['mean_within_scene_cv_percent']:.3f} | {r['mean_scene_worst_percent']:.3f} |")
    lines.extend(['', '稳定性表的 Mean 是先场景内平均，再平均 68 个场景，因此与全过程 472 点等权平均略有不同；这不是计算错误。', '',
        '## 实验版本及边界', '',
        f"- 源码记录：`{summary['sysnav_repository']}`，commit `{summary['sysnav_commit']}`。",
        '- 输入来自保存的 occupied 三维体素、无 clearance 的 `voxel_nav_free_xy` 和当时机器人位姿，适配为 registered scan、freespace、occupied state、odometry；平面拟合范围覆盖整张快照。',
        '- 输入未使用 VoxRoom 的 room labels、door seeds、分割门线或 GT；GT 仅用于指标计算。这里是保存快照的离线房间分割回放，不是原生机器人传感器时序的完整在线导航复现。',
        '- 全部生成 108 个场景、727 个预测点；其中有可用批准 GT 的 102 个场景、710 点参与原始评测。按本论文的数据集和独立测试范围筛选后为 68 场景、472 点。',
        '- 旧 `sysnav_source_roomseg_20260905` 缺少输入，已经作废；`sgnav_source_roomseg_20260905` 属于另一方法。两者都没有混入这里。', '',
        '## 与已有总对比表的衔接', '',
        '已有 `non_ablation_comparisons_20260831` 使用 66 场景、458 点，而本次 SysNav 完整测试范围为 68 场景、472 点，不能直接当成严格同样本对比。',
        '此外，同场景同百分比的记录可能来自不同 attempt/episode；合表前应核对实际体素快照，不能仅匹配场景名或进度名。本次不覆盖原对比总表。',
        '较旧表多出的场景：' + '、'.join(f'`{d}/{s}`' for d,s in additional) + '。', '',
        '## 文件', '',
        '- `sysnav_progress_metrics.csv`：各进度的合并指标。',
        '- `sysnav_overall_metrics.csv`：全过程加权平均和 Final。',
        '- `sysnav_stability_summary.csv`：场景内统计再平均的 Mean/SD/CV/Avg.Worst。',
        '- `sysnav_per_scene_stability.csv`：68 场景的逐场景稳定性。',
        '- `sysnav_per_checkpoint_metrics.csv`：仅测试集的 472 条原始指标及来源路径。',
        '- `sysnav_test_scenes.csv`：保留场景、实际进度、缺失进度。',
        '- `sysnav_dataset_audit.csv`：两个数据集的单独汇总，仅供核对加权计算。',
        '- `provenance.json`：输入版本、排除清单、源文件校验和及统计方法。', '',
        '复现整理命令：`python3 scripts/organize_sysnav_results.py`。CSV 采用 UTF-8 BOM，便于 Excel 打开。'])
    (OUT / 'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    with zipfile.ZipFile(OUT.with_suffix('.zip'), 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT.iterdir()):
            if path.is_file():
                archive.write(path, OUT.name+'/'+path.name)
    print(json.dumps({'out': str(OUT), 'overall': overall, 'stability': stability, 'extra_scenes': additional}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
