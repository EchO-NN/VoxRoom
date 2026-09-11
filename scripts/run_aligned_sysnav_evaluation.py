#!/usr/bin/env python3
"""Stage exactly the main-method test snapshots, run SysNav, verify and export."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import io
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sysnav_aligned_input import METHOD, CONTRACT, geometry, sha256, parameters
from run_sysnav_aligned_checkpoint import run as infer
from organize_dude_occusg_csv import key, aggregate, scene_statistics, stability_summary, EVENTS, METRICS
from voxroom_online.isaac_runtime.evaluation.online_roomseg.metrics import prepare_metric_label_maps, compute_snapshot_metrics

HOST = 'echo@fe80::630c:c787:c8e7:5fcb%enp129s0'
REMOTE_DATA = Path('/media/echo/data/voxroom_roomseg_evaluation')
DATASETS = {'interioragent': 'interioragent_all_available_gt_20260828', 'grscene': 'grscene_all_available_gt_20260817'}
SCRIPTS = ['run_aligned_sysnav_evaluation.py', 'sysnav_aligned_input.py', 'run_sysnav_aligned_checkpoint.py',
           'sysnav_snapshot_bridge.cpp', 'build_sysnav_snapshot_bridge.py', 'organize_dude_occusg_csv.py']


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temporary.replace(path)


def read_csv(path):
    return list(csv.DictReader(path.open(encoding='utf-8-sig', newline='')))


def save_csv(path, rows):
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.partial.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    path.with_suffix('.partial.csv').replace(path)


def ssh(command, **kwargs):
    return subprocess.run(['ssh','-6','-o','BatchMode=yes','-o','ConnectTimeout=8',HOST,command],
                          check=True, text=True, capture_output=True, **kwargs).stdout


def prepare(out):
    if (out/'inputs/frozen_tasks.json').exists():
        return json.loads((out/'inputs/frozen_tasks.json').read_text())
    inputs = out/'inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    tasks, files, excluded = [], set(), set()
    for dataset, gt_folder in DATASETS.items():
        base = ROOT/'results/dude_corrected_20260910'/dataset
        exclusion = json.loads((base/'inputs/exclusion_manifest.json').read_text())
        excluded.update(exclusion['excluded_scene_ids'])
        idx_path = base/'inputs/index_voxroom.json'
        index = json.loads(idx_path.read_text())
        shutil.copy2(idx_path, inputs/f'{dataset}_reference_index.json')
        remote_metrics = f'/media/echo/data/voxroom_ablation_20260828/evaluation/{dataset}_vertical_full_retrain_f1_preview_epoch14/metrics/per_checkpoint_metrics.csv'
        metrics_text = ssh('cat '+shlex.quote(remote_metrics), timeout=30)
        (inputs/f'{dataset}_main_epoch14_metrics.csv').write_text(metrics_text)
        main_rows = {key(r):r for r in csv.DictReader(io.StringIO(metrics_text))}
        for ep in index['episodes']:
            uid, scene = ep['episode_uid'], ep['scene_id']
            assert scene not in excluded
            gt = REMOTE_DATA/gt_folder/'final_gt'/uid/'last_step.gt_labels.npy'
            gt_meta = gt.with_name('last_step.gt_metadata.json')
            annotation = REMOTE_DATA/gt_folder/'annotations'/uid/'last_step.annotation.json'
            for p in (gt,gt_meta,annotation): files.add(str(p.relative_to(REMOTE_DATA)))
            final = next(s for s in ep['snapshots'] if s['coverage_event_id']=='final')['snapshot_path']
            for s in ep['snapshots']:
                event = s['coverage_event_id']
                assert event in EVENTS
                remote_source = Path(s['snapshot_path'])
                relative = remote_source.relative_to(REMOTE_DATA)
                files.add(str(relative))
                ref = main_rows[(dataset,uid,event)]
                assert ref['source_snapshot_path'] == str(remote_source)
                assert int(ref['step']) == int(s['step'])
                tasks.append({'dataset':dataset,'scene_id':scene,'episode_uid':uid,
                    'coverage_event_id':event,'progress':'final' if event=='final' else str(int(event[-3:]))+'%',
                    'step':int(s['step']),'actual_coverage_percent':100*float(s['coverage_ratio']),
                    'source_snapshot_path':str(remote_source),'local_source':str(out/'source_data'/relative),
                    'local_final':str(out/'source_data'/Path(final).relative_to(REMOTE_DATA)),
                    'gt_path':str(out/'source_data'/gt.relative_to(REMOTE_DATA)),
                    'gt_meta_path':str(out/'source_data'/gt_meta.relative_to(REMOTE_DATA)),
                    'annotation_path':str(out/'source_data'/annotation.relative_to(REMOTE_DATA)),
                    'reference_n_gt':int(ref['n_gt']),'reference_metric_domain_pixels':int(ref['metric_domain_pixels']),
                    'main_reference_metrics':{k:float(ref[k]) for k in METRICS},
                    'prediction_path':str(out/'predictions'/uid/(event+'.npz'))})
    assert len(tasks)==462 and len({t['episode_uid'] for t in tasks})==68
    assert len({key(t) for t in tasks})==462
    assert not excluded & {t['scene_id'] for t in tasks}
    script = 'import pathlib,hashlib,json\nroot=pathlib.Path('+repr(str(REMOTE_DATA))+')\nfiles='+repr(sorted(files))+'''
result=[]
for name in files:
 p=root/name
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 result.append({'relative_path':name,'bytes':p.stat().st_size,'sha256':h.hexdigest()})
print(json.dumps(result))
'''
    print('Hashing exact remote inputs (462 snapshots plus GT/annotation files)', flush=True)
    manifest=json.loads(ssh('python3 -',input=script,timeout=300))
    save_json(inputs/'remote_file_manifest.json',manifest)
    listing=inputs/'rsync_files.txt'; listing.write_text('\n'.join(sorted(files))+'\n')
    print(f'Staging {sum(r["bytes"] for r in manifest)/1e9:.3f} GB; no raw training data copied',flush=True)
    destination=out/'source_data'; destination.mkdir(exist_ok=True)
    rsync_host='echo@[fe80::630c:c787:c8e7:5fcb%enp129s0]'
    with (inputs/'rsync.log').open('w') as log:
        subprocess.run(['rsync','-a','--checksum','--files-from='+str(listing),'-e','ssh -6 -o BatchMode=yes -o ConnectTimeout=8',
                        rsync_host+':'+str(REMOTE_DATA)+'/',str(destination)+'/'],stdout=log,stderr=subprocess.STDOUT,check=True)
    for r in manifest:
        assert sha256(destination/r['relative_path'])==r['sha256'],r['relative_path']
    hashes={str(REMOTE_DATA/r['relative_path']):r['sha256'] for r in manifest}
    for task in tasks:
        task['source_sha256']=hashes[task['source_snapshot_path']]
        annotation=json.loads(Path(task['annotation_path']).read_text())
        assert annotation['episode_uid']==task['episode_uid'] and annotation['review']['status']=='approved'
        gt_meta=json.loads(Path(task['gt_meta_path']).read_text())
        assert gt_meta['annotation_review_status']=='approved'
        with np.load(task['local_source']) as z: res,shape,bounds=geometry(z)
        with np.load(task['local_final']) as z: final_res,final_shape,final_bounds=geometry(z)
        assert shape==final_shape and np.isclose(res,final_res) and np.allclose(bounds,final_bounds)
        assert sha256(task['local_final'])==annotation['snapshot_sha256'], 'GT is for a different final snapshot'
    save_json(inputs/'excluded_scenes.json',sorted(excluded))
    save_json(inputs/'frozen_tasks.json',tasks)
    return tasks


def evaluate(task, meta):
    with np.load(task['local_source']) as z:
        domain=z['roomseg_eval_explored_reference_mask'].astype(bool)
        resolution=float(z['map_resolution_m'])
    final_gt=np.load(task['gt_path'])
    gt=np.where(domain,final_gt,0)
    with np.load(task['prediction_path']) as z: pred=z['sysnav_room_label_map']
    prepared=prepare_metric_label_maps(gt,pred,min_room_area_cells=int(math.ceil(.5/resolution**2-1e-9)))
    metrics=compute_snapshot_metrics(prepared.gt,prepared.pred)
    assert metrics['n_gt']==task['reference_n_gt'], 'GT room count differs from main evaluation'
    assert int(np.count_nonzero(prepared.gt))==task['reference_metric_domain_pixels'], 'Evaluation domain differs from main'
    p,r=metrics['precision'],metrics['recall']
    row={k:task[k] for k in ('dataset','scene_id','episode_uid','coverage_event_id','progress','step','actual_coverage_percent')}
    row.update(method_id=METHOD,precision=p,recall=r,f1=2*p*r/(p+r) if p+r else 0,miou_room=metrics['miou_room'])
    row.update({v+'_percent':100*row[k] for k,v in METRICS.items()})
    row.update(n_gt=metrics['n_gt'],n_pred=metrics['n_pred'],metric_domain_pixels=int(np.count_nonzero(prepared.gt)),
        unlabeled_domain_pixels=int(np.count_nonzero((prepared.gt>0)&(prepared.pred==0))),
        native_room_count=meta['native_room_count'],runtime_seconds=meta['runtime_seconds'],
        source_snapshot_path=task['source_snapshot_path'],local_source_snapshot_path=task['local_source'],
        source_sha256=task['source_sha256'],prediction_path=task['prediction_path'],
        free_cleared_voxel_count=meta['native_audit']['accumulated_before_free']-meta['native_audit']['accumulated_after_free'],
        state_free_cells=meta['native_audit']['state_free_cells'])
    return row


def worker(task, out, upstream):
    prediction=Path(task['prediction_path'])
    assert sha256(task['local_source'])==task['source_sha256']
    if prediction.exists():
        with np.load(prediction) as z:meta=json.loads(str(z['metadata_json'].item()))
        assert meta['input_contract']==CONTRACT and meta['source_sha256']==task['source_sha256']
        assert meta['bridge_sha256']==sha256(out/'build/sysnav_snapshot_bridge')
        assert meta['parameters']==parameters(upstream)[0]
    else:
        meta=infer(Path(task['local_source']),prediction,out/'build/sysnav_snapshot_bridge',upstream)
    return evaluate(task,meta)


def export(out,tasks,rows,failures):
    universe={key(t):t for t in tasks}
    rows=sorted(rows,key=lambda r:(r['dataset'],r['scene_id'],EVENTS.index(r['coverage_event_id'])))
    save_csv(out/'per_checkpoint_metrics.csv',rows)
    save_json(out/'failures.json',failures)
    save_json(out/'status.json',{'status':'complete' if len(rows)==len(tasks) else 'incomplete',
        'completed':len(rows),'expected':len(tasks),'failed':len(failures),'updated_at':time.strftime('%Y-%m-%d %H:%M:%S')})
    if not rows:return
    over=[aggregate(METHOD,rows,universe,d,event) for d in ('combined',*DATASETS) for event in ('all_available_progress','final')]
    progress=[aggregate(METHOD,rows,universe,d,event) for d in ('combined',*DATASETS) for event in EVENTS]
    scenes=scene_statistics(METHOD,rows,universe)
    stability=[stability_summary(METHOD,scenes,d) for d in ('combined',*DATASETS)]
    save_csv(out/'overall_metrics.csv',over);save_csv(out/'progress_metrics.csv',progress)
    save_csv(out/'per_scene_stability.csv',scenes);save_csv(out/'stability_summary.csv',stability)
    main=[{**t,**t['main_reference_metrics']} for t in tasks]
    main_over=[aggregate('VoxRoom_vertical_epoch14',main,universe,d,event) for d in ('combined',*DATASETS) for event in ('all_available_progress','final')]
    save_csv(out/'main_comparison.csv',over+main_over)
    full=over[0]
    report=['# SysNav：同快照、官方仿真参数的离线重测','',
        f'已完成 {len(rows)}/462 点，68 个独立测试场景；失败 {len(failures)} 点。', '',
        f'当前已完成部分：P {full["P_percent"]:.3f}%，R {full["R_percent"]:.3f}%，F1 {full["F1_percent"]:.3f}%，mIoU {full["mIoU_percent"]:.3f}%。', '',
        '协议：排除训练和验证场景；逐点对齐 VoxRoom epoch 14 的 episode、step、快照 SHA256、GT 和评测域。',
        '每个进度独立运行，未恢复原始传感器时序；不能称为完整在线导航复现。',
        '仅使用真实观测 occupied 体素，不合成人工地板。Nav Free 保守映射为 native free 更新足迹，不能代替真实视点/视线时序。',
        '先调用原版 scan 累积，再 free 清障，再 occupied state 更新，最后原版 roomSegmentation；几何函数体未改。',
        '仿真几何参数来自官方 matterport_sim.yaml，包括上下高度 0.75/0.4 m；默认局部平面拟合半径 15 m，不扩大成全图。',
        '官方 3000×3000×80 网格和 0.1 m 分辨率不变；输出按世界坐标映射回源 0.05 m 网格，再按公共 GT 域评测。',
        'P/R/F1/mIoU 为房间指标。每点先求 F1 再平均；两数据集按点数加权。稳定性先逐场景计算 SD/CV，再平均。',
        '旧 472 点结果与本轮快照不同，只能作为历史记录，不能将分数差全部归因于修复。', '']
    if failures:report+=['失败点没有补零，也没有自动排除；详见 failures.json。','']
    (out/'report.md').write_text('\n'.join(report))
    if len(rows)==len(tasks):
        with zipfile.ZipFile(out.with_suffix('.zip'),'x',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(out.glob('*.csv')):z.write(p,p.name)
            for name in ('report.md','protocol.json','status.json','failures.json'):z.write(out/name,name)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--upstream',type=Path,default=Path('/home/joey/SysNav'))
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    tasks=prepare(out)
    print(f'Exact held-out inputs verified: {len(tasks)} checkpoints / 68 scenes',flush=True)
    if a.prepare_only:return
    if not (out/'protocol.json').exists():
        code=out/'code';code.mkdir(exist_ok=True)
        for name in SCRIPTS:shutil.copy2(ROOT/'scripts'/name,code/name)
        save_json(out/'protocol.json',{'method':METHOD,'input_contract':CONTRACT,'parameters':parameters(a.upstream)[0],
            'code_sha256':{name:sha256(code/name) for name in SCRIPTS},'source_task_manifest_sha256':sha256(out/'inputs/frozen_tasks.json'),
            'build':json.loads((out/'build/build_provenance.json').read_text()),'test_points':462,'test_scenes':68,
            'scope':'offline independent snapshots; original scan/viewpoint trajectory unavailable',
            'gt_used_for_inference':False,'main_reference':'VoxRoom Vertical Free epoch 14', 'parameter_tuning_on_test':False})
    else:
        frozen=json.loads((out/'protocol.json').read_text())
        assert frozen['code_sha256']=={name:sha256(ROOT/'scripts'/name) for name in SCRIPTS},'Frozen code changed; use new result directory'
    selected=tasks[:a.limit] if a.limit else tasks
    rows=[];failures=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(worker,t,out,a.upstream):t for t in selected}
        for f in as_completed(futures):
            t=futures[f]
            try:rows.append(f.result());print(f'[{len(rows)}/{len(selected)}] {t["scene_id"]} {t["coverage_event_id"]}',flush=True)
            except Exception as e:
                failures.append({'dataset':t['dataset'],'episode_uid':t['episode_uid'],'event':t['coverage_event_id'],'error':str(e),'traceback':traceback.format_exc()})
                print('FAILED '+t['scene_id']+' '+t['coverage_event_id']+': '+str(e),flush=True)
            save_csv(out/'per_checkpoint_metrics.csv',rows)
            save_json(out/'status.json',{'status':'running','completed':len(rows),'expected':462,'failed':len(failures),
                'updated_at':time.strftime('%Y-%m-%d %H:%M:%S')})
            save_json(out/'failures.json',failures)
    export(out,tasks,rows,failures)


if __name__=='__main__':main()
