#!/usr/bin/env python3
"""Original C++ Morph area sensitivity; paired snapshots, independent metric audit."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import math
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from organize_dude_occusg_csv import aggregate, key, scene_statistics, stability_summary, EVENTS, METRICS
from verify_aligned_sysnav_results import digest, independent_metrics
from voxroom_online.isaac_runtime.baselines.mask_io import relabel_consecutive
from voxroom_online.isaac_runtime.evaluation.online_roomseg.metrics import prepare_metric_label_maps, compute_snapshot_metrics

UPSTREAM = ROOT / 'external_baselines/ipa_ws/src/ipa_coverage_planning'
SOURCE = UPSTREAM / 'ipa_room_segmentation/common'
COMMIT = '986c18384ed884dadd3bc857cd0c47c13b7d4716'
OLD_CSVS = [ROOT / 'results' / folder / 'per_checkpoint_metrics.csv' for folder in (
    'interioragent_paper_baselines_raw_vertical_20260828',
    'grscene_paper_segmentation_baselines_raw_vertical_20260828')]


def json_save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def csv_save(path, rows):
    if not rows:
        return
    with path.with_suffix('.partial.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix('.partial.csv').replace(path)


def build(out):
    assert subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
    assert not subprocess.check_output(['git', '-C', str(UPSTREAM), 'status', '--porcelain'], text=True).strip()
    binary = out / 'build/morph'
    binary.parent.mkdir(parents=True, exist_ok=True)
    code = [ROOT / 'scripts/run_ipa_morphological_standalone.cpp',
            SOURCE / 'src/morphological_segmentation.cpp', SOURCE / 'src/wavefront_region_growing.cpp']
    hashed = code + [SOURCE / 'include/ipa_room_segmentation' / n for n in (
        'morphological_segmentation.h', 'wavefront_region_growing.h', 'contains.h')]
    hashed += [ROOT / 'scripts/ipa_morph_logging_stub/ros/ros.h']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in hashed}
    command = ['g++', '-O2', '-std=c++17', '-I' + str(ROOT / 'scripts/ipa_morph_logging_stub'),
               '-I' + str(SOURCE / 'include'), *map(str, code), '-o', str(binary)]
    command += shlex.split(subprocess.check_output(['pkg-config', '--cflags', '--libs', 'opencv4'], text=True))
    manifest = binary.parent / 'provenance.json'
    if binary.exists():
        recorded = json.loads(manifest.read_text())
        assert recorded['source_sha256'] == hashes and recorded['binary_sha256'] == digest(binary)
        return binary
    subprocess.run(command, check=True)
    json_save(manifest, {'upstream_commit': COMMIT, 'upstream_clean': True, 'command': command,
        'source_sha256': hashes, 'binary_sha256': digest(binary),
        'algorithm_source_modified': False, 'ros_stub': 'ROS_INFO logging only',
        'opencv_version': subprocess.check_output(['pkg-config', '--modversion', 'opencv4'], text=True).strip()})
    return binary


def input_image(z):
    # Deliberate whitelist: no GT, reference masks, seed maps or predicted rooms.
    free = np.asarray(z['voxel_vertical_free_xy'], dtype=bool)
    if free.ndim != 2 or not free.any():
        raise ValueError('Expected nonempty raw voxel_vertical_free_xy')
    return free.astype(np.uint8) * 255


def native_labels(binary, image, resolution, prefix, lower, upper):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    image_path, result_path = prefix.with_suffix('.png'), prefix.with_suffix('.yml.gz')
    assert cv2.imwrite(str(image_path), image)
    with prefix.with_suffix('.log').open('w') as f:
        subprocess.run([str(binary), str(image_path), str(result_path), repr(resolution), repr(lower), repr(upper)],
                       check=True, stdout=f, stderr=subprocess.STDOUT, timeout=180)
    storage = cv2.FileStorage(str(result_path), cv2.FILE_STORAGE_READ)
    raw = storage.getNode('segmented_map').mat()
    recorded = [storage.getNode(k).real() for k in ('lower_m2', 'upper_m2')]
    storage.release()
    assert recorded == [lower, upper]
    assert raw is not None and raw.shape == image.shape
    assert not np.any((raw > 0) & (image == 0))
    return raw.astype(np.int32)


def metrics(task, raw):
    # Everything involving GT occurs after the standalone program returns.
    with np.load(task['local_source'], allow_pickle=False) as z:
        explored = z['roomseg_eval_explored_reference_mask'].astype(bool)
        resolution = float(z['map_resolution_m'])
    final_gt = np.load(task['gt_path'])
    labels = relabel_consecutive(np.where((raw > 0) & (raw < 65280) & explored, raw, 0))
    threshold = int(math.ceil(.5 / resolution ** 2 - 1e-9))
    prepared = prepare_metric_label_maps(np.where(explored, final_gt, 0), labels, min_room_area_cells=threshold)
    scores = compute_snapshot_metrics(prepared.gt, prepared.pred)
    p, r = scores['precision'], scores['recall']
    scores['f1'] = 2 * p * r / (p + r) if p + r else 0.
    scores['metric_domain_pixels'] = int(np.count_nonzero(prepared.gt))
    independent = independent_metrics(final_gt, explored, labels, threshold)
    error = max(abs(scores[k] - independent[k]) for k in METRICS)
    assert error < 1e-10
    for k in ('n_gt', 'n_pred', 'metric_domain_pixels'):
        assert scores[k] == independent[k]
    assert scores['n_gt'] == task['reference_n_gt']
    assert scores['metric_domain_pixels'] == task['reference_metric_domain_pixels']
    return labels, scores, error


def old_records(tasks):
    universe = {key(t): t for t in tasks}
    records = {}
    for path in OLD_CSVS:
        with path.open(encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                if row['method'] != 'morphological' or key(row) not in universe:
                    continue
                task = universe[key(row)]
                assert key(row) not in records
                assert int(row['step']) == task['step'] and row['source_snapshot_path'] == task['source_snapshot_path']
                assert row['runner_type'] == 'original_ros_action' and row['implementation_commit'] == COMMIT
                for field in ('n_gt', 'metric_domain_pixels'):
                    assert int(row[field]) == task['reference_' + field]
                records[key(row)] = {**row, **{k: float(row[k]) for k in METRICS}}
    assert set(records) == set(universe)
    return records


def comparison_records(path, tasks):
    """Only compare complete runs with identical episode/step/snapshot/GT domains."""
    universe = {key(t): t for t in tasks}
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len({key(r) for r in rows}) == len(universe)
    assert {key(r) for r in rows} == set(universe)
    methods = {r['method_id'] for r in rows}
    assert len(methods) == 1
    for r in rows:
        t = universe[key(r)]
        assert r['scene_id'] == t['scene_id'] and int(r['step']) == t['step']
        assert r['source_snapshot_path'] == t['source_snapshot_path']
        assert r['source_sha256'] == t['source_sha256']
        for field in ('n_gt', 'metric_domain_pixels'):
            assert int(r[field]) == t['reference_' + field]
        for field in METRICS:
            r[field] = float(r[field])
            assert math.isfinite(r[field]) and 0 <= r[field] <= 1
        p, recall = r['precision'], r['recall']
        assert math.isclose(r['f1'], 2*p*recall/(p+recall) if p+recall else 0, abs_tol=1e-10)
    return methods.pop(), rows


def worker(task, out, binary, lower, upper, old):
    started = time.monotonic()
    assert digest(Path(task['local_source'])) == task['source_sha256']
    with np.load(task['local_source'], allow_pickle=False) as z:
        image, resolution = input_image(z), float(z['map_resolution_m'])
    folder = out / 'predictions' / task['episode_uid'] / task['coverage_event_id']
    target = folder / 'prediction.npz'
    # Paired same-build old parameter control: any change in adapter shows here.
    raw_control = native_labels(binary, image, resolution, folder / 'control_47', .8, 47.)
    _, control, _ = metrics(task, raw_control)
    old_error = max(abs(control[k] - old[key(task)][k]) for k in METRICS)
    # Record differences rather than hiding points. Main reports this before attribution.
    raw = native_labels(binary, image, resolution, folder / 'morph_candidate', lower, upper)
    labels, scores, independent_error = metrics(task, raw)
    method = f'Morph_vertical_area_{lower:g}_{upper:g}'
    metadata = {'method_id': method, 'lower_m2': lower, 'upper_m2': upper,
        'input_key': 'voxel_vertical_free_xy', 'input_mode': 'raw_vertical_free',
        'gt_used_for_inference': False, 'runner_type': 'standalone_original_cpp',
        'source_sha256': task['source_sha256'], 'source_snapshot_path': task['source_snapshot_path'],
        'upstream_commit': COMMIT, 'binary_sha256': digest(binary),
        'independent_metric_max_error': independent_error, 'old47_metric_max_error': old_error,
        'runtime_seconds': time.monotonic() - started}
    with target.with_suffix('.partial.npz').open('wb') as f:
        np.savez_compressed(f, final_room_label_map=labels, morphological_raw_segmented_map=raw,
            control47_raw_segmented_map=raw_control, morphological_input_image=image,
            metadata_json=np.asarray(json.dumps(metadata)))
    target.with_suffix('.partial.npz').replace(target)
    row = {k: task[k] for k in ('dataset', 'scene_id', 'episode_uid', 'coverage_event_id', 'progress', 'step', 'actual_coverage_percent')}
    row.update(method_id=method, **{k: scores[k] for k in METRICS})
    row.update({label + '_percent': 100 * scores[k] for k, label in METRICS.items()})
    row.update({k: scores[k] for k in ('n_gt', 'n_pred', 'metric_domain_pixels')})
    row.update(source_snapshot_path=task['source_snapshot_path'], source_sha256=task['source_sha256'],
        prediction_path=str(target), lower_m2=lower, upper_m2=upper,
        independent_metric_max_error=independent_error, old47_metric_max_error=old_error)
    control_row = {**row, 'method_id': 'Morph_vertical_area_0.8_47_same_build',
                   **{k: control[k] for k in METRICS}, 'upper_m2': 47.}
    control_row.update({label + '_percent': 100 * control[k] for k, label in METRICS.items()})
    control_row.update({k: control[k] for k in ('n_gt', 'n_pred', 'metric_domain_pixels')})
    return row, control_row


def export(out, tasks, rows, controls, old, failures, comparisons=()):
    universe = {key(t): t for t in tasks}
    rows.sort(key=lambda r: (r['dataset'], r['scene_id'], EVENTS.index(r['coverage_event_id'])))
    csv_save(out / 'per_checkpoint_metrics.csv', rows)
    csv_save(out / 'control47_per_checkpoint_metrics.csv', controls)
    json_save(out / 'failures.json', failures)
    complete = len(rows) == len(tasks) and not failures
    json_save(out / 'status.json', {'status': 'complete' if complete else 'incomplete',
        'completed': len(rows), 'expected': len(tasks), 'failed': len(failures), 'updated_at': time.strftime('%Y-%m-%d %H:%M:%S')})
    if not rows:
        return
    method = rows[0]['method_id']
    overall = [aggregate(method, rows, universe, d, e) for d in ('combined', 'interioragent', 'grscene') for e in ('all_available_progress', 'final')]
    csv_save(out / 'overall_metrics.csv', overall)
    csv_save(out / 'progress_metrics.csv', [aggregate(method, rows, universe, d, e) for d in ('combined', 'interioragent', 'grscene') for e in EVENTS])
    scenes = scene_statistics(method, rows, universe)
    csv_save(out / 'per_scene_stability.csv', scenes)
    csv_save(out / 'stability_summary.csv', [stability_summary(method, scenes, d) for d in ('combined', 'interioragent', 'grscene')])
    done = {key(r) for r in rows}
    comparison = []
    for name, records in ((method, rows), ('Morph_vertical_area_0.8_47_same_build', controls),
                          ('Morph_vertical_area_0.8_47_historical', list(old.values())),
                          *comparisons,
                          ('VoxRoom_vertical_epoch14', [{**t, **t['main_reference_metrics']} for t in tasks])):
        records = [r for r in records if key(r) in done]
        for event in ('all_available_progress', 'final'):
            comparison.append(aggregate(name, records, universe, 'combined', event))
    csv_save(out / 'comparison.csv', comparison)
    audit = {'independently_verified_points': len(rows),
        'max_independent_metric_error': max(r['independent_metric_max_error'] for r in rows),
        'max_control47_vs_historical_metric_error': max(r['old47_metric_max_error'] for r in rows),
        'control47_points_differing_from_history': sum(r['old47_metric_max_error'] > 1e-8 for r in rows)}
    json_save(out / 'verification.json', audit)
    report = [f'# Morph 面积阈值 {rows[0]["lower_m2"]:g}～{rows[0]["upper_m2"]:g} m² 重测', '',
        f'完成 {len(rows)}/{len(tasks)} 点，失败 {len(failures)} 点。排除训练和验证场景；两数据集合并按点数加权。', '',
        '输入仅为原始 voxel_vertical_free_xy。原版 C++ 腐蚀、轮廓面积判断与 wavefront 回填源码未改；ROS 日志以独立编译桩替代。',
        '面积阈值控制腐蚀后候选区域的接收，不是回填后房间的面积硬上限。公共评测最小房间面积仍为 0.5 m²。',
        '这是用户在查看测试结果后指定的参数敏感性实验，不能宣称为未触碰测试集的默认参数结果。',
        '每点重跑 0.8～47 m² 作为同构建对照；历史差异见 verification.json。', '',
        '| 方法 | P (%) | R (%) | F1 (%) | mIoU (%) |', '|---|---:|---:|---:|---:|']
    for r in comparison:
        if r['progress_scope'] == 'all_available_progress':
            report.append('| ' + r['method_id'] + ' | ' + ' | '.join(f'{r[k+"_percent"]:.3f}' for k in METRICS.values()) + ' |')
    report += ['', '稳定性：每个场景按已有进度先计算总体 SD / CV / 最低值，再对场景取平均；不补齐缺失进度。', '', json.dumps(audit, ensure_ascii=False), '']
    (out / 'report.md').write_text('\n'.join(report))
    if complete:
        package = out.with_suffix('.zip')
        with zipfile.ZipFile(package, 'x', zipfile.ZIP_DEFLATED) as z:
            for p in sorted(out.glob('*.csv')) + [out / n for n in ('report.md', 'protocol.json', 'status.json', 'failures.json', 'verification.json')]:
                z.write(p, p.name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tasks', type=Path, default=ROOT / 'results/sysnav_aligned_20260911/inputs/frozen_tasks.json')
    p.add_argument('--lower-m2', type=float, default=.8)
    p.add_argument('--upper-m2', type=float, default=15.)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--comparison-csv', type=Path, action='append', default=[],
                   help='Complete prior per-checkpoint CSV to include after strict pairing checks')
    a = p.parse_args()
    assert math.isfinite(a.lower_m2) and math.isfinite(a.upper_m2) and 0 <= a.lower_m2 < a.upper_m2
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'protocol.json').exists():
        raise RuntimeError('Use a new output directory; finished or partial experiments are not overwritten.')
    tasks = json.loads(a.tasks.read_text())
    assert len(tasks) == len({key(t) for t in tasks}) == 462
    assert len({t['episode_uid'] for t in tasks}) == 68
    old = old_records(tasks)
    comparisons = [comparison_records(path, tasks) for path in a.comparison_csv]
    binary = build(out)
    json_save(out / 'protocol.json', {'method': 'Morphological', 'lower_m2': a.lower_m2, 'upper_m2': a.upper_m2,
        'input': 'raw_vertical_free', 'input_key': 'voxel_vertical_free_xy', 'gt_used_for_inference': False,
        'source_manifest': str(a.tasks.resolve()), 'source_manifest_sha256': digest(a.tasks),
        'test_points': 462, 'test_scenes': 68, 'exclude_training_and_validation': True,
        'parameter_sensitivity_after_test_inspection': True,
        'build': json.loads((out / 'build/provenance.json').read_text()),
        'metric_min_room_area_m2': .5, 'script_sha256': digest(Path(__file__)),
        'comparison_csv_sha256': {str(path.resolve()): digest(path) for path in a.comparison_csv}})
    frozen = out / 'code'
    frozen.mkdir()
    for name in ('run_morph_area_evaluation.py', 'verify_aligned_sysnav_results.py', 'organize_dude_occusg_csv.py', 'run_ipa_morphological_standalone.cpp'):
        shutil.copy2(ROOT / 'scripts' / name, frozen / name)
    selected = tasks[:a.limit] if a.limit else tasks
    rows, controls, failures = [], [], []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = {pool.submit(worker, t, out, binary, a.lower_m2, a.upper_m2, old): t for t in selected}
        for f in as_completed(futures):
            task = futures[f]
            try:
                row, control = f.result()
                rows.append(row)
                controls.append(control)
                print(f'[{len(rows)}/{len(selected)}] {task["scene_id"]} {task["coverage_event_id"]} old47_error={row["old47_metric_max_error"]:.3g}', flush=True)
            except Exception as e:
                failures.append({'dataset': task['dataset'], 'episode_uid': task['episode_uid'],
                    'event': task['coverage_event_id'], 'error': str(e), 'traceback': traceback.format_exc()})
                print(f'FAIL {task["scene_id"]}: {e}', flush=True)
            if (len(rows) + len(failures)) % 25 == 0:
                export(out, tasks, rows, controls, old, failures, comparisons)
    export(out, tasks, rows, controls, old, failures, comparisons)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
