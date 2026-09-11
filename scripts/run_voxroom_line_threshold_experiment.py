#!/usr/bin/env python3
"""Paired, stateful VoxRoom replay of the frozen held-out snapshot universe."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import csv
from dataclasses import replace
import json
import math
import multiprocessing
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from organize_dude_occusg_csv import aggregate, scene_statistics, stability_summary, key, EVENTS, METRICS
from verify_aligned_sysnav_results import digest, independent_metrics
from replay_voxroom_ablation import _load_arrays, _map_info
from voxroom_online.isaac_runtime.config import load_config, get_nested
from voxroom_online.isaac_runtime.mapping.voxel_occupancy_door_wall_roomseg import VoxelOccupancyDoorWallRoomSegmenter, VoxelOccupancyDoorWallRoomSegConfig
from voxroom_online.isaac_runtime.scripts.replay_voxel_roomseg_snapshots import _voxel_grid_from_snapshot
from voxroom_online.isaac_runtime.evaluation.online_roomseg.metrics import prepare_metric_label_maps, compute_snapshot_metrics

METHODS = {
    'VoxRoom_original_parameters_replay': dict(correlation=.95, variance=.65, keep=.5, l_branches=False),
    'VoxRoom_corr0p9_var0p1_keep0p25_Lboth': dict(correlation=.9, variance=.1, keep=.25, l_branches=True),
}


def json_save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def csv_save(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.partial.csv')
    with temp.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


class FrozenRawUnion:
    """Replay the exact historical union BEFORE neural filtering, never old keeps."""
    mask = None

    def combine(self, stage, **_kwargs):
        if self.mask is None or self.mask.shape != stage.raw_seed_mask_xy.shape:
            raise ValueError('frozen raw candidate shape mismatch')
        return replace(stage, raw_seed_mask_xy=self.mask.copy(), debug={**stage.debug,
            'candidate_policy': 'exact_saved_voxel_door_raw_seed_mask_no_GT_no_old_model_keep'})


def segmenter_config(base, parameters, checkpoint, device):
    cfg = copy.deepcopy(base)
    cfg.setdefault('voxel_door', {}).update(
        primitive_min_line_correlation=parameters['correlation'],
        primitive_max_orthogonal_variance_cells2=parameters['variance'],
        enable_l_shaped_seed_branches=parameters['l_branches'])
    cfg.setdefault('door_seed_learning', {}).update(mode='inference',
        checkpoint_path=str(checkpoint), device=device, keep_threshold=parameters['keep'],
        context_source='vertical', raw_seed_source='voxroom_tvars_vertical_union',
        inference_batch_size=32, reuse_column_encodings=True, fallback_to_rule_seed_on_error=False,
        keep_uninformative_seed=False, allow_source_code_hash_mismatch=True,
        ablation_allow_checkpoint_mismatch=True)
    cfg['min_room_area_m2'] = .5
    effective = VoxelOccupancyDoorWallRoomSegConfig.from_mapping(cfg)
    assert effective.door.primitive_min_line_correlation == parameters['correlation']
    assert effective.door.primitive_max_orthogonal_variance_cells2 == parameters['variance']
    assert effective.door.enable_l_shaped_seed_branches == parameters['l_branches']
    assert effective.door_seed_learning.keep_threshold == parameters['keep']
    return cfg


def metric_scores(task, raw_labels, explored, resolution):
    # GT first becomes accessible AFTER a prediction file has been committed.
    gt = np.load(task['gt_path'], allow_pickle=False)
    threshold = int(math.ceil(.5 / resolution**2 - 1e-9))
    prepared = prepare_metric_label_maps(np.where(explored, gt, 0), raw_labels, min_room_area_cells=threshold)
    scores = compute_snapshot_metrics(prepared.gt, prepared.pred)
    p, r = scores['precision'], scores['recall']
    scores['f1'] = 2*p*r/(p+r) if p+r else 0.
    scores['metric_domain_pixels'] = int(np.count_nonzero(prepared.gt))
    independent = independent_metrics(gt, explored, raw_labels, threshold)
    error = max(abs(scores[k] - independent[k]) for k in METRICS)
    assert error < 1e-10, (key(task), error)
    for field in ('n_gt', 'n_pred', 'metric_domain_pixels'):
        assert scores[field] == independent[field]
    assert scores['n_gt'] == task['reference_n_gt']
    assert scores['metric_domain_pixels'] == task['reference_metric_domain_pixels']
    return scores, error


def run_scene(tasks, base, checkpoint, out_string, device, protocol_hash):
    import torch
    torch.set_num_threads(1)
    out = Path(out_string)
    scene_dir = out / 'scenes' / tasks[0]['episode_uid']
    done = scene_dir / 'result.json'
    if done.exists():
        saved = json.loads(done.read_text())
        assert saved['protocol_hash'] == protocol_hash
        assert saved['complete'] and len(saved['rows']) == len(tasks) * len(METHODS)
        return saved['rows']
    first = _load_arrays(Path(tasks[0]['local_source']))
    info = _map_info(first, first['occupancy_map'].shape)
    segmenters = {}
    for method, parameters in METHODS.items():
        s = VoxelOccupancyDoorWallRoomSegmenter(config=segmenter_config(base, parameters, checkpoint, device), map_info=info)
        s.door_seed_raw_seed_accumulator = FrozenRawUnion()
        segmenters[method] = s
    rows = []
    for number, task in enumerate(tasks):
        source = Path(task['local_source'])
        assert digest(source) == task['source_sha256']
        arrays = first if number == 0 else _load_arrays(source)
        frozen = np.asarray(arrays['voxel_door_raw_seed_mask'], bool)
        assert arrays['occupancy_map'].shape == first['occupancy_map'].shape
        for method, s in segmenters.items():
            started = time.monotonic()
            s.door_seed_raw_seed_accumulator.mask = frozen
            # Fresh grid for each variant; no replayed labels or saved memories.
            grid = _voxel_grid_from_snapshot(arrays, info, base.get('voxel_grid', {}), recompute_state_from_logodds=False)
            pose = np.asarray(arrays.get('demo_pose_world', arrays.get('demo_camera_pose_world', np.zeros(4)))).reshape(-1)
            agent = np.asarray(arrays.get('agent_rc', np.zeros(2)), np.int32).reshape(-1)
            s.update(occupancy_map=arrays['occupancy_map'].astype(bool),
                observed_free_mask=arrays['observed_free_mask'].astype(bool),
                obstacle_mask=arrays['obstacle_mask'].astype(bool), unknown_mask=arrays['unknown_mask'].astype(bool),
                voxel_grid=grid, step=int(task['step']), navigation_free_mask=arrays['voxel_nav_free_xy'].astype(bool),
                navigation_obstacle_mask=arrays['voxel_nav_occupied_xy'].astype(bool),
                door_seed_no_clearance_free_mask=arrays['voxel_nav_free_xy'].astype(bool),
                agent_rc=tuple(map(int, agent[:2])), agent_yaw_deg=float(np.degrees(pose[3])) if len(pose) >= 4 else 0.)
            result = s.last_result
            assert result is not None
            debug = result.debug
            assert not debug.get('voxel_door_seed_model_fallback', False)
            np.testing.assert_array_equal(debug['voxel_door_raw_seed_mask'], frozen)
            labels = np.asarray(result.room_label_map, np.int32)
            folder = out / 'predictions' / method / task['episode_uid'] / task['coverage_event_id']
            folder.mkdir(parents=True, exist_ok=True)
            prediction = folder / 'prediction.npz'
            payload = {'final_room_label_map': labels, 'voxel_final_separator_map': result.separator_map,
                'voxel_door_raw_seed_mask': frozen, 'source_sha256': np.asarray(task['source_sha256']),
                'protocol_hash': np.asarray(protocol_hash)}
            for field in ('voxel_vertical_free_xy', 'voxel_nav_free_xy', 'voxel_nav_occupied_xy',
                'voxel_door_seed_model_probability_xy', 'voxel_door_seed_model_keep_mask',
                'voxel_door_seed_model_reject_mask', 'voxel_door_seed_line_primitive_id_map',
                'voxel_door_extensible_primitive_mask', 'voxel_door_rejected_primitive_mask',
                'voxel_door_partition_cut_mask', 'voxel_stable_door_cut_mask'):
                if field in debug:
                    payload[field] = np.asarray(debug[field])
            with prediction.with_suffix('.partial.npz').open('wb') as f:
                np.savez_compressed(f, **payload)
            prediction.with_suffix('.partial.npz').replace(prediction)
            json_save(folder / 'diagnostics.json', {'method_id': method, 'parameters': METHODS[method],
                'column_cache': debug.get('voxel_door_seed_model_column_cache', {}),
                'candidates': debug.get('voxel_door_candidates', []),
                'primitives': debug.get('voxel_door_line_primitives', []),
                'effective_parameters': {'correlation': s.config.door.primitive_min_line_correlation,
                    'variance': s.config.door.primitive_max_orthogonal_variance_cells2,
                    'l_branches': s.config.door.enable_l_shaped_seed_branches,
                    'keep': s.door_seed_inference_engine.keep_threshold},
                'primitive_reject_reasons': debug.get('voxel_door_primitive_reject_reason_counts', {}),
                'memory_after': s.export_replay_state(), 'protocol_hash': protocol_hash})
            seconds = time.monotonic() - started
            scores, error = metric_scores(task, labels, arrays['roomseg_eval_explored_reference_mask'].astype(bool), float(arrays['map_resolution_m']))
            row = {k: task[k] for k in ('dataset','scene_id','episode_uid','coverage_event_id','progress','step','actual_coverage_percent')}
            row.update(method_id=method, **{k: scores[k] for k in METRICS},
                **{label+'_percent': 100*scores[k] for k, label in METRICS.items()},
                **{k: scores[k] for k in ('n_gt','n_pred','metric_domain_pixels')})
            primitives = debug.get('voxel_door_line_primitives', [])
            row.update(raw_seed_cells=int(frozen.sum()), kept_seed_cells=int(np.count_nonzero(debug['voxel_door_seed_model_keep_mask'])),
                l_shape_primitive_count=sum(str(p.get('extraction_method','')).startswith('l_shape_') for p in primitives),
                selected_door_candidates=int(debug.get('voxel_door_candidate_count',0)),
                partition_accepted_doors=int(debug.get('voxel_door_partition_accepted_count',0)),
                runtime_seconds=seconds, independent_metric_max_error=error,
                historical_main_metric_max_error=max(abs(scores[k]-task['main_reference_metrics'][k]) for k in METRICS),
                source_snapshot_path=task['source_snapshot_path'], source_sha256=task['source_sha256'],
                prediction_path=str(prediction))
            rows.append(row)
            json_save(scene_dir / 'progress.json', {'processed':len(rows),'expected':len(tasks)*len(METHODS),
                'last_event':task['coverage_event_id'],'last_method':method,'updated_at':time.time()})
            print(json.dumps({'scene':task['scene_id'],'event':task['progress'],'method':method,
                'F1':row['F1_percent'],'mIoU':row['mIoU_percent'],'seconds':seconds,
                'kept_seeds':row['kept_seed_cells'],'L_arms':row['l_shape_primitive_count']},ensure_ascii=False),flush=True)
            del grid, result
        if number:
            del arrays
    json_save(done, {'complete': True, 'protocol_hash':protocol_hash, 'rows':rows})
    return rows


def export(out, tasks, rows, failures):
    universe = {key(t):t for t in tasks}
    rows = sorted(rows, key=lambda r:(r['method_id'],r['dataset'],r['scene_id'],EVENTS.index(r['coverage_event_id'])))
    csv_save(out / 'per_checkpoint_metrics.csv', rows)
    complete = len(rows) == 2*len(tasks) and not failures
    json_save(out / 'status.json', {'status':'complete' if complete else 'running_or_incomplete',
        'completed_predictions':len(rows),'expected_predictions':2*len(tasks),'failed_scenes':len(failures),'updated_at':time.time()})
    json_save(out / 'failures.json', failures)
    if not rows:
        return
    groups = {method:[r for r in rows if r['method_id']==method] for method in METHODS}
    groups['VoxRoom_historical_epoch14'] = [{**t, 'method_id':'VoxRoom_historical_epoch14', **t['main_reference_metrics']} for t in tasks]
    summaries, progress, per_scene, stability = [], [], [], []
    for method, records in groups.items():
        for dataset in ('combined','interioragent','grscene'):
            summaries.append(aggregate(method,records,universe,dataset,'all_available_progress'))
            for scope in EVENTS:
                progress.append(aggregate(method,records,universe,dataset,scope))
        scenes = scene_statistics(method,records,universe)
        per_scene.extend(scenes)
        stability.extend(stability_summary(method,scenes,ds) for ds in ('combined','interioragent','grscene'))
    csv_save(out/'aggregate.csv', summaries)
    csv_save(out/'progress_metrics.csv', progress)
    csv_save(out/'per_scene_stability.csv', per_scene)
    csv_save(out/'stability_summary.csv', stability)
    report = ['# VoxRoom 联合参数实验：0.9 / 0.1 / L 双支臂 / 0.25','',
        '状态：'+('全部完成' if complete else '尚未完成；以下含部分结果，不能当作完整总分')+'。', '',
        f'冻结测试范围：{len({t["episode_uid"] for t in tasks})} 场景、{len(tasks)} 个已有进度点，排除训练和验证场景。',
        '两个数据集合并按进度点数加权；小于 0.5 m² 的房间忽略。mIoU 为一对一匹配后按 GT 房间数归一。',
        '同场景按已有进度顺序重放；独立建立两份门线/分隔线记忆，不导入旧分割记忆。仅重跑分割全链路，不重跑 Isaac 探索轨迹。',
        '原始 voxel、导航观测和已保存的 TVARS ∪ VoxRoom raw seed 候选相同。不是重新生成探索期间的射线历史，也不读旧 NN keep/probability 作为模型输出。',
        'GT 只在保存预测之后用于评测；每个点另用独立列联表实现复核指标并核对 GT 房间数、评测面积。', '',
        '阈值含义：线性相关性最低 0.9；正交方差最高 0.1 格子²；NN 保留阈值 0.25。L 支臂分别通过原有几何验证，未强制接受失败门线。',
        '原参数对照为同一当前代码关闭 L 双支臂、恢复 0.95 / 0.65 / 0.5；历史 epoch 14 单列，不能把它与当前代码对照混淆。',
        '这是看过先前测试结果后的联合调参，不是单因素消融，也不能据此宣称独立未调参的泛化提升。', '',
        '| 方法 | 已完成/应有点数 | P (%) | R (%) | F1 (%) | mIoU (%) |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in summaries:
        if r['dataset_scope']=='combined':
            report.append(f"| {r['method_id']} | {r['evaluated_checkpoint_count']}/{r['expected_checkpoint_count']} | "+' | '.join(f"{r[k+'_percent']:.3f}" if r[k+'_percent'] is not None else '—' for k in ('P','R','F1','mIoU'))+' |')
    report += ['', '配置、模型、源代码和数据 hash：`protocol.json`。逐点预测及 primitive/candidate/memory 诊断：`predictions/`。',
        '完整逐点指标：`per_checkpoint_metrics.csv`；分进度：`progress_metrics.csv`；同场景稳定性：`per_scene_stability.csv`、`stability_summary.csv`。',
        'L 型检测只针对有实际 seed 支撑、两端相交、能在既有厚度范围内解释完整组的横竖双支臂，不将任意大块噪声拆成门线。']
    (out/'report.md').write_text('\n'.join(report)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks',type=Path,default=ROOT/'results/sysnav_aligned_20260911/inputs/frozen_tasks.json')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'results/kujiale_0003_vertical_epoch14_fullviz_20260901/checkpoint_vertical_epoch14.pt')
    p.add_argument('--config',type=Path,default=ROOT/'configs/voxroom_online.yaml')
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--max-scenes',type=int,default=0)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    out=args.out_dir.resolve()
    if out.exists() and not args.resume:
        raise FileExistsError('Choose a new output directory, or explicitly --resume this experiment')
    tasks=json.loads(args.tasks.read_text())
    excluded=set(json.loads((args.tasks.parent/'excluded_scenes.json').read_text()))
    import torch
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    excluded.update(checkpoint['training_scene_ids']); excluded.update(checkpoint['validation_scene_ids'])
    assert not {t['scene_id'] for t in tasks}&excluded
    assert len(tasks)==len({key(t) for t in tasks})
    scenes=defaultdict(list)
    for task in tasks:
        scenes[(task['dataset'],task['episode_uid'])].append(task)
    selected=list(scenes.values())[:args.max_scenes or None]
    for scene in selected:
        scene.sort(key=lambda t:EVENTS.index(t['coverage_event_id']))
    tasks=[t for scene in selected for t in scene]
    base=dict(get_nested(load_config(args.config),'mapping.room_segmentation'))
    files=[ROOT/'scripts/run_voxroom_line_threshold_experiment.py', args.config,
        *sorted((ROOT/'voxroom_online/isaac_runtime/mapping').glob('*.py')),
        *sorted((ROOT/'voxroom_online/isaac_runtime/door_seed_learning').glob('*.py'))]
    protocol={'experiment':'voxroom_line_threshold_Lboth_20260912','methods':METHODS,
        'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':digest(args.checkpoint),
        'tasks_sha256':digest(args.tasks),'excluded_scene_ids':sorted(excluded),
        'scene_count':len(selected),'checkpoint_count':len(tasks),'base_roomseg_config':base,
        'source_hashes':{str(f.relative_to(ROOT)):digest(f) for f in files},
        'model_selection':'same existing vertical epoch14; no retraining',
        'metadata_hash_exceptions':'explicit source/config-hash allowance for current code and changed fitting parameters; raw candidates frozen; Z/height/resolution/architecture checks remain active',
        'memory_policy':'stateful at saved checkpoints; independent histories per variant; no old predicted memory import',
        'candidate_policy':'exact saved union before NN filter; TVARS+VoxRoom overlap retained',
        'test_set_tuning':True,'workers':args.workers,'device':args.device}
    out.mkdir(parents=True,exist_ok=True)
    path=out/'protocol.json'
    if path.exists():
        assert json.loads(path.read_text())==protocol, 'resume protocol changed'
    else:
        json_save(path,protocol)
    protocol_hash=digest(path)
    json_save(out/'frozen_tasks.json',tasks)
    rows,failures=[],[]
    export(out,tasks,rows,failures)
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(run_scene,scene,base,str(args.checkpoint.resolve()),str(out),args.device,protocol_hash):scene for scene in selected}
        for future in as_completed(futures):
            scene=futures[future]
            try:
                rows.extend(future.result())
            except Exception:
                failures.append({'scene_id':scene[0]['scene_id'],'traceback':traceback.format_exc()})
                print(failures[-1]['traceback'],flush=True)
            export(out,tasks,rows,failures)
            print(f'COMPLETE_PREDICTIONS {len(rows)}/{2*len(tasks)} FAILURES {len(failures)}',flush=True)
    if failures:
        raise RuntimeError(f'{len(failures)} scenes failed; predictions retained, no zero-filled metrics')


if __name__=='__main__':
    main()
