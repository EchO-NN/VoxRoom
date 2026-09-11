#!/usr/bin/env python3
"""Independent contingency-table verification and paper-result packaging."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile
import numpy as np
from scipy.optimize import linear_sum_assignment


def read(path):
    return list(csv.DictReader(path.open(encoding='utf-8-sig',newline='')))


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def remove_small(arr, threshold):
    values,counts=np.unique(arr,return_counts=True)
    return np.where(np.isin(arr,values[(values>0)&(counts<threshold)]),0,arr)


def independent_metrics(final_gt, explored, pred, threshold):
    raw_gt=np.where(explored,final_gt,0)
    prediction=remove_small(np.where(raw_gt>0,pred,0),threshold)
    gt=remove_small(raw_gt,threshold)
    prediction=remove_small(np.where(gt>0,prediction,0),threshold)
    gv,gi=np.unique(gt,return_inverse=True)
    pv,pi=np.unique(prediction,return_inverse=True)
    table=np.bincount(gi.ravel()*len(pv)+pi.ravel(),minlength=len(gv)*len(pv)).reshape(len(gv),len(pv))
    ga=table.sum(axis=1)[gv>0];pa=table.sum(axis=0)[pv>0]
    overlap=table[np.ix_(gv>0,pv>0)]
    if len(ga) and len(pa):
        precision=float(np.mean(overlap.max(axis=0)/pa))
        recall=float(np.mean(overlap.max(axis=1)/ga))
        iou=overlap/(ga[:,None]+pa[None,:]-overlap)
        i,j=linear_sum_assignment(-iou)
        miou=float(iou[i,j].sum()/len(ga))
    else:precision=recall=miou=0.
    return dict(precision=precision,recall=recall,f1=2*precision*recall/(precision+recall) if precision+recall else 0,
                miou_room=miou,n_gt=len(ga),n_pred=len(pa),metric_domain_pixels=int(ga.sum()))


def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();out=a.output.resolve()
    status=json.loads((out/'status.json').read_text())
    assert status['status']=='complete' and status['completed']==462 and status['failed']==0
    assert json.loads((out/'failures.json').read_text())==[]
    rows=read(out/'per_checkpoint_metrics.csv')
    tasks=json.loads((out/'inputs/frozen_tasks.json').read_text())
    key=lambda r:(r['dataset'],r['episode_uid'],r['coverage_event_id'])
    assert len(rows)==len(tasks)==len({key(r) for r in rows})==462
    tasks={key(t):t for t in tasks}
    assert set(tasks)=={key(r) for r in rows}
    audit=[];max_error=0.
    for n,row in enumerate(rows):
        task=tasks[key(row)]
        assert row['source_snapshot_path']==task['source_snapshot_path']
        assert int(row['step'])==task['step']
        assert digest(Path(task['local_source']))==task['source_sha256']==row['source_sha256']
        with np.load(task['local_source']) as z:
            explored=z['roomseg_eval_explored_reference_mask'].astype(bool)
            cellsize=float(z['map_resolution_m'])
        with np.load(task['prediction_path']) as z:
            pred=z['sysnav_room_label_map'];meta=json.loads(str(z['metadata_json'].item()))
        assert meta['synthetic_floor_point_count']==0
        assert meta['native_audit']['sequence']=='scan5_then_free_then_occupied_then_segmentation'
        assert meta['parameters']['kViewPointCollisionMarginZPlus']==.75
        assert meta['parameters']['kViewPointCollisionMarginZMinus']==.4
        assert meta['parameters']['region_growing_radius']==15.
        metrics=independent_metrics(np.load(task['gt_path']),explored,pred,int(math.ceil(.5/cellsize**2-1e-9)))
        error=max(abs(float(row[k])-metrics[k]) for k in ('precision','recall','f1','miou_room'))
        max_error=max(error,max_error);assert error<1e-10,(row,error)
        for k in ('n_gt','n_pred','metric_domain_pixels'):assert int(row[k])==metrics[k]
        assert metrics['n_gt']==task['reference_n_gt'] and metrics['metric_domain_pixels']==task['reference_metric_domain_pixels']
        audit.append({'dataset':row['dataset'],'episode_uid':row['episode_uid'],'coverage_event_id':row['coverage_event_id'],
                      'source_sha256_verified':True,'main_gt_domain_verified':True,'max_metric_error':error,
                      'prediction_sha256':digest(Path(task['prediction_path']))})
        if (n+1)%50==0:print(f'Independently verified {n+1}/462',flush=True)
    labels={'precision':'P','recall':'R','f1':'F1','miou_room':'mIoU'}
    for group in read(out/'per_scene_stability.csv'):
        selected=[r for r in rows if r['episode_uid']==group['episode_uid']]
        for raw,label in labels.items():
            v=np.array([100*float(r[raw]) for r in selected]);mean=float(v.mean());sd=float(np.sqrt(np.mean((v-mean)**2)))
            for field,value in (('mean_percent',mean),('SD_pp',sd),('worst_percent',min(v))):
                assert math.isclose(float(group[label+'_'+field]),value,abs_tol=1e-9)
            if mean:assert math.isclose(float(group[label+'_CV_percent']),100*sd/mean,abs_tol=1e-9)
    summary={'verified_points':462,'verified_scenes':68,'independent_metric_max_absolute_error':max_error,
             'same_main_episode_step_snapshot_gt_domain':True,'within_scene_stability_verified':True,
             'inference_floor_points_added':0,'all_native_free_updates_effective':all(int(r['free_cleared_voxel_count'])>0 for r in rows),
             'verifier_sha256':digest(Path(__file__))}
    (out/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (out/'per_checkpoint_verification.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,list(audit[0]));w.writeheader();w.writerows(audit)
    overview=read(out/'main_comparison.csv')
    combined=[r for r in overview if r['dataset_scope']=='combined' and r['progress_scope']=='all_available_progress']
    report=(out/'report.md').read_text()
    report+='\n## 同快照主方法对比（全部 462 点）\n\n| 方法 | P (%) | R (%) | F1 (%) | mIoU (%) |\n|---|---:|---:|---:|---:|\n'
    for row in combined:
        method='SysNav 对齐重测' if row['method_id'].startswith('sysnav') else 'VoxRoom Vertical-Free epoch 14'
        report+=f'| {method} | '+ ' | '.join(f'{float(row[k+"_percent"]):.3f}' for k in ('P','R','F1','mIoU'))+' |\n'
    report+='\n## SysNav 全过程（两个数据集合并，按评测点数加权）\n\n| 进度 | 点数 | P (%) | R (%) | F1 (%) | mIoU (%) |\n|---|---:|---:|---:|---:|---:|\n'
    for row in read(out/'progress_metrics.csv'):
        if row['dataset_scope']!='combined':continue
        event=row['progress_scope'];name='Final' if event=='final' else str(int(event[-3:]))+'%'
        report+=f'| {name} | {row["evaluated_checkpoint_count"]} | '+' | '.join(f'{float(row[k+"_percent"]):.3f}' for k in ('P','R','F1','mIoU'))+' |\n'
    report+=f'\n独立校验：从 462 个原始预测重新构建交集矩阵和 Hungarian IoU 匹配；最大指标误差 {max_error:.3g}。所有点的 GT 房间数与评测面积均和主方法一致。\n'
    report+='\n接入实现说明：隔离构建副本中只将一个 private 访问标记改为 public，并重命名原 main，便于固定调用顺序；官方算法函数体与原仓库均未改。详细构建命令和校验和见 protocol.json。\n'
    (out/'verified_report.md').write_text(report)
    files=sorted(out.glob('*.csv'))+[out/name for name in ('verified_report.md','protocol.json','status.json','verification.json','failures.json')]
    file_rows=[{'file':p.name,'bytes':p.stat().st_size,'sha256':digest(p)} for p in files]
    with (out/'files.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,['file','bytes','sha256']);w.writeheader();w.writerows(file_rows)
    package=out.with_name(out.name+'_verified_results.zip')
    with zipfile.ZipFile(package,'x',zipfile.ZIP_DEFLATED) as z:
        for p in files+[out/'files.csv']:z.write(p,p.name)
    print(json.dumps(summary));print(str(package))


if __name__=='__main__':main()
