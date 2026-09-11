#!/usr/bin/env python3
"""Export auditable result CSVs without rerunning algorithms or mixing variants."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ("milestone_020", "milestone_040", "milestone_060", "milestone_070", "milestone_080", "milestone_090", "final")
METRICS = {"precision": "P", "recall": "R", "f1": "F1", "miou_room": "mIoU"}
DATASETS = ("interioragent", "grscene")
ONLINE = "DUDE_online_vertical_tau2p5"
OFFLINE = "DUDE_offline_vertical_tau2p5"
CATALOG = {
    ONLINE: ("DUDE 在线增量修复版", "raw_vertical_free", 2.5, "每个场景初始化，按现有进度增量回放", "primary", ""),
    OFFLINE: ("DUDE 离线修复版", "raw_vertical_free", 2.5, "每个进度独立重置原版节点", "primary_incomplete", "GRScene 180 秒超时中断；仅整理已有预测，不补跑、不补零"),
    "OccuSG_official_projection": ("OccuSG 官方投影", "octomap_contiguous_free_0p3m", 1.5, "官方投影、增量 DUDE、跟踪与房间生成", "primary", "已存体素快照回放，不是完整 RGB-D/语义流复现"),
    "OccuSG_vertical_uncropped": ("OccuSG Vertical Free 未裁外围", "raw_vertical_free_full_canvas", 1.5, "官方后端；完整源栅格", "diagnostic_padding_affected", "外围 unknown 导致 42 次拒绝更新，涉及 32 场景；不能冒充裁剪修复版"),
    "OccuSG_vertical_cropped": ("OccuSG Vertical Free 外围裁剪修复", "raw_vertical_free_crop_unknown_border", 1.5, "仅裁全 unknown 外围；保留已知格和世界坐标；官方后端不变", "primary_input_ablation", "0 次拒绝更新；输出和评测仍是原始完整地图"),
}


def read(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(row):
    return row["dataset"], row["episode_uid"], row["coverage_event_id"]


def normalize(row, method, universe, source_csv):
    ref = universe[key(row)]
    assert row["scene_id"] == ref["scene_id"]
    assert int(row["step"]) == ref["step"]
    assert Path(row["source_snapshot_path"]).resolve() == Path(ref["source_snapshot_path"]).resolve()
    result = {"method_id": method, "dataset": row["dataset"], "scene_id": row["scene_id"], "episode_uid": row["episode_uid"],
              "progress": ref["progress"], "coverage_event_id": row["coverage_event_id"], "step": ref["step"],
              "actual_coverage_percent": ref["actual_coverage_percent"]}
    for raw, label in METRICS.items():
        value = float(row[raw])
        assert math.isfinite(value) and 0 <= value <= 1, (method, raw, row)
        result[raw] = value
        result[label+"_percent"] = 100*value
    p, r = result["precision"], result["recall"]
    assert math.isclose(result["f1"], 2*p*r/(p+r) if p+r else 0, abs_tol=1e-10)
    for field in ("n_gt", "n_pred", "metric_domain_pixels"):
        result[field] = int(row[field])
    result.update({"source_snapshot_path": ref["source_snapshot_path"], "prediction_path": row["prediction_path"],
                   "source_metrics_csv": str(source_csv), "source_input_sha256": row.get("input_grid_sha256", "")})
    return result


def in_scope(row, dataset, progress):
    return (dataset == "combined" or row["dataset"] == dataset) and (progress == "all_available_progress" or row["coverage_event_id"] == progress)


def aggregate(method, records, universe, dataset, progress, selected_keys=None, excluded_keys=None):
    expected = [r for r in universe.values() if in_scope(r, dataset, progress)]
    available = [r for r in records if in_scope(r, dataset, progress)]
    selected = available if selected_keys is None else [r for r in available if key(r) in selected_keys]
    excluded = {key(r) for r in expected} & (excluded_keys or set())
    assert not excluded & {key(r) for r in available}
    status = "complete" if len(selected) == len(expected) else "partial_available"
    if excluded and len(selected) == len(expected)-len(excluded):
        status = "complete_with_user_exclusion"
    if selected_keys is not None and len(selected) < len(expected):
        status = "paired_subset_not_full_test_set"
    result = {"method_id": method, "dataset_scope": dataset, "progress_scope": progress,
              "selection_scope": "all_available_predictions" if selected_keys is None else "online_offline_common_available",
              "coverage_status": status, "expected_scene_count": len({r["episode_uid"] for r in expected}),
              "evaluated_scene_count": len({r["episode_uid"] for r in selected}), "expected_checkpoint_count": len(expected),
              "evaluated_checkpoint_count": len(selected), "missing_prediction_count": len(expected)-len(available),
              "authorized_exclusion_count": len(excluded), "eligible_checkpoint_count": len(expected)-len(excluded),
              "unexpected_missing_prediction_count": len(expected)-len(excluded)-len(available),
              "excluded_by_pairing_count": len(available)-len(selected)}
    result.update({label+"_percent": statistics.fmean(r[raw]*100 for r in selected) if selected else None for raw, label in METRICS.items()})
    return result


def scene_statistics(method, records, universe, excluded_keys=None):
    scenes = sorted({(r["dataset"], r["scene_id"], r["episode_uid"]) for r in universe.values()})
    output = []
    for dataset, scene, uid in scenes:
        expected = [r for r in universe.values() if r["episode_uid"] == uid and r["dataset"] == dataset]
        actual = [r for r in records if r["episode_uid"] == uid and r["dataset"] == dataset]
        expected_events = {r["coverage_event_id"] for r in expected}
        events = {r["coverage_event_id"] for r in actual}
        excluded_events = {r['coverage_event_id'] for r in expected if key(r) in (excluded_keys or set())}
        row = {"method_id": method, "dataset": dataset, "scene_id": scene, "episode_uid": uid,
               "expected_checkpoint_count": len(expected), "evaluated_checkpoint_count": len(actual),
               "complete_for_saved_progress": events == expected_events, "temporal_stability_has_2_or_more_points": len(actual) >= 2,
               "complete_for_required_progress": events == expected_events-excluded_events,
               "user_excluded_progress": ";".join(e for e in EVENTS if e in excluded_events),
               "evaluated_progress": ";".join(e for e in EVENTS if e in events),
               "not_saved_in_original_exploration": ";".join(e for e in EVENTS if e not in expected_events),
               "missing_predictions_for_saved_progress": ";".join(e for e in EVENTS if e in expected_events-events-excluded_events)}
        for raw, label in METRICS.items():
            values = [100*r[raw] for r in actual]
            mean = statistics.fmean(values) if values else None
            sd = statistics.pstdev(values) if values else None
            row.update({label+"_mean_percent": mean, label+"_SD_pp": sd,
                        label+"_CV_ratio": sd/mean if mean else None,
                        label+"_CV_percent": 100*sd/mean if mean else None,
                        label+"_worst_percent": min(values) if values else None})
        output.append(row)
    return output


def stability_summary(method, scenes, dataset):
    expected = [r for r in scenes if dataset == "combined" or r["dataset"] == dataset]
    selected = [r for r in expected if r["evaluated_checkpoint_count"] > 0]
    out = {"method_id": method, "dataset_scope": dataset, "expected_scene_count": len(expected),
           "evaluated_scene_count": len(selected),
           "complete_for_saved_progress_scene_count": sum(r["complete_for_saved_progress"] for r in selected),
           "complete_for_required_progress_scene_count": sum(r["complete_for_required_progress"] for r in selected),
           "single_progress_scene_count": sum(r["evaluated_checkpoint_count"] == 1 for r in selected),
           "multi_progress_scene_count": sum(r["evaluated_checkpoint_count"] >= 2 for r in selected)}
    for label in METRICS.values():
        for suffix in ("mean_percent", "SD_pp", "CV_ratio", "CV_percent", "worst_percent"):
            values = [r[label+"_"+suffix] for r in selected if r[label+"_"+suffix] is not None]
            out[label+"_average_scene_"+suffix] = statistics.fmean(values) if values else None
        out[label+"_CV_defined_scene_count"] = sum(r[label+"_CV_ratio"] is not None for r in selected)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dude-export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--completion-audit", type=Path)
    parser.add_argument("--excluded-predictions-json", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "files.csv").exists():
        raise FileExistsError("Choose a new output directory; finalized exports are immutable")
    sources, universe, exclusions = [], {}, set()
    omitted_rows = json.loads(args.excluded_predictions_json.read_text())['excluded_predictions'] if args.excluded_predictions_json else []
    assert all(r['method'] == 'dude_offline' for r in omitted_rows)
    omitted_keys = {key(r) for r in omitted_rows}
    assert len(omitted_keys) == len(omitted_rows)
    if args.excluded_predictions_json:
        sources.append(args.excluded_predictions_json)
    for dataset in DATASETS:
        source = ROOT / "results/dude_corrected_20260910" / dataset / "inputs/index_voxroom.json"
        sources.append(source)
        audit_path = source.parents[1] / "input_audit.json"
        sources.append(audit_path)
        exclusions.update(json.loads(audit_path.read_text())["excluded_scene_ids"])
        for episode in json.loads(source.read_text())["episodes"]:
            for record in episode["snapshots"]:
                event = record["coverage_event_id"]
                assert event in EVENTS
                ratio = record.get("coverage_ratio")
                ref = {"dataset": dataset, "scene_id": episode["scene_id"], "episode_uid": episode["episode_uid"],
                       "coverage_event_id": event, "progress": "final" if event == "final" else str(int(event[-3:]))+"%",
                       "step": int(record["step"]), "actual_coverage_percent": 100*float(ratio) if ratio is not None else None,
                       "source_snapshot_path": record["snapshot_path"]}
                assert key(ref) not in universe
                universe[key(ref)] = ref
    assert len(universe) == 462 and len({r["episode_uid"] for r in universe.values()}) == 68
    assert not exclusions & {r["scene_id"] for r in universe.values()}
    groups = {m: [] for m in CATALOG}
    export = args.dude_export / "per_checkpoint_metrics.csv"
    sources.append(export)
    for row in read(export):
        method = {"dude_incremental": ONLINE, "dude_offline": OFFLINE}[row["method"]]
        groups[method].append(normalize(row, method, universe, export))
    occusg_sources = {
        "OccuSG_official_projection": ("occusg_official_saved_voxels_20260910", "occusg_official_saved_voxel_replay"),
        "OccuSG_vertical_uncropped": ("occusg_vertical_free_20260910", "occusg_official_raw_vertical_free"),
        "OccuSG_vertical_cropped": ("occusg_vertical_free_cropped_20260910", "occusg_official_raw_vertical_free_cropped"),
    }
    for method, (folder, source_method) in occusg_sources.items():
        run_root = ROOT / "results" / folder
        assert json.loads((run_root / "completed.json").read_text())["status"] == "complete"
        sources.append(run_root / "completed.json")
        for dataset in DATASETS:
            path = run_root / dataset / "per_checkpoint_metrics.csv"
            sources.extend([path, run_root / dataset / "inputs/provenance.json"])
            for row in read(path):
                assert row["method"] == source_method
                groups[method].append(normalize(row, method, universe, path))
    for method, rows in groups.items():
        assert len({key(r) for r in rows}) == len(rows)
        assert (266 <= len(rows) <= 462) if method == OFFLINE else (len(rows) == 462)
        rows.sort(key=lambda r: (r["dataset"], r["scene_id"], EVENTS.index(r["coverage_event_id"])))
    offline_count = len(groups[OFFLINE])
    missing_count = len(universe) - offline_count
    assert omitted_keys <= set(universe)-{key(r) for r in groups[OFFLINE]}
    unexpected_missing_count = missing_count-len(omitted_keys)
    offline_gr_count = sum(r['dataset'] == 'grscene' for r in groups[OFFLINE])
    offline_status = (f"离线 {offline_count}/462（IA 64/64，GR {offline_gr_count}/398），尚缺 {missing_count} 个预测"
                      if missing_count else "离线 462/462（IA 64/64，GR 398/398），原缺失的 196 个预测已补齐")
    if omitted_keys and unexpected_missing_count == 0:
        offline_status = f"离线 {offline_count}/462（IA 64/64，GR {offline_gr_count}/398），用户排除 {len(omitted_keys)} 个崩溃点，其余全部完成；均值分母为 {offline_count}"
    anchor = {key(r): r for r in groups[ONLINE]}
    for rows in groups.values():
        for row in rows:
            for field in ("n_gt", "metric_domain_pixels", "step", "source_snapshot_path"):
                assert row[field] == anchor[key(row)][field], (field, row)
    # Independently verify all complete-method overall means against archived tables.
    expected_path = ROOT / "results/dude_tau1p5_vertical_20260910/tables/aggregate.csv"
    sources.append(expected_path)
    prior = [r for r in read(expected_path) if r["method"] == "DUDE_corrected_tau2p5_vertical" and r["scope"] == "all_checkpoints_weighted"]
    assert len(prior) == 1
    for raw, label in METRICS.items():
        assert math.isclose(statistics.fmean(r[raw]*100 for r in groups[ONLINE]), float(prior[0][raw+"_percent"]), abs_tol=1e-9)
    for method, (folder, _) in occusg_sources.items():
        path = ROOT / "results" / folder / "tables/aggregate.csv"
        sources.append(path)
        archived_id = {"OccuSG_official_projection": "OccuSG_official_saved_voxel_replay", "OccuSG_vertical_uncropped": "OccuSG_VerticalFree", "OccuSG_vertical_cropped": "OccuSG_VerticalFree_cropped"}[method]
        prior = [r for r in read(path) if r["method"] == archived_id and r["scope"] in ("all_checkpoints_weighted", "all_available_checkpoints_weighted")]
        assert len(prior) == 1, (method, prior)
        for raw in METRICS:
            assert math.isclose(statistics.fmean(r[raw]*100 for r in groups[method]), float(prior[0][raw+"_percent"]), abs_tol=1e-9)
    files = []
    def write(name, rows, description, fieldnames=None):
        assert rows or fieldnames, name
        path = args.output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames or list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        files.append({"file": name, "row_count": len(rows), "description": description, "encoding": "UTF-8 BOM", "sha256": digest(path)})
    write("dude_per_checkpoint_metrics.csv", groups[ONLINE]+groups[OFFLINE], f"修复 DUDE 在线/离线，均为 2.5 m、Vertical Free；已有 {462+offline_count} 条指标")
    write("occusg_per_checkpoint_metrics.csv", [r for m, rows in groups.items() if m.startswith("OccuSG") for r in rows], "OccuSG 三个输入版本，各 462 条指标；用 method_id 区分")
    overall, progress, scenes, stability, inventory, catalog = [], [], [], [], [], []
    for method, rows in groups.items():
        available = {key(r) for r in rows}
        description, input_map, tau, operation, role, caveat = CATALOG[method]
        if method == OFFLINE:
            role = 'primary_incomplete' if missing_count else 'primary'
            caveat = (offline_status + "；缺失不补零" if missing_count else
                      "仅补跑原缺失的 196 点；保持 2.5 m、同一原版二进制/修复桥接、相同输入，原 728 个预测未改动")
            if omitted_keys and not unexpected_missing_count:
                role = 'primary_with_disclosed_user_exclusion'
        method_omissions = omitted_keys if method == OFFLINE else set()
        run_status = 'complete' if len(rows) == len(universe) else ('complete_with_user_exclusion' if method_omissions and not unexpected_missing_count else 'partial')
        catalog.append({"method_id": method, "description": description, "input_map": input_map, "threshold_m": tau,
                        "operation": operation, "role": role, "run_status": run_status,
                        "expected_checkpoints": len(universe), "available_checkpoints": len(rows), "missing_checkpoints": len(universe)-len(rows),
                        "user_excluded_checkpoints": len(method_omissions), "eligible_checkpoints": len(universe)-len(method_omissions),
                        "available_scenes": len({r["episode_uid"] for r in rows}), "caveat": caveat})
        for ref in universe.values():
            inventory.append({"method_id": method, **ref, "prediction_available": key(ref) in available,
                              "authorized_exclusion": key(ref) in method_omissions,
                              "missing_reason": "user_excluded_native_assertion_dude_cut_cpp_399" if key(ref) in method_omissions else ("" if key(ref) in available else "original_offline_replay_timeout_prediction_absent")})
        method_scenes = scene_statistics(method, rows, universe, method_omissions)
        scenes.extend(method_scenes)
        for dataset in ("combined", *DATASETS):
            overall.extend(aggregate(method, rows, universe, dataset, event, excluded_keys=method_omissions) for event in ("all_available_progress", "final"))
            progress.extend(aggregate(method, rows, universe, dataset, event, excluded_keys=method_omissions) for event in EVENTS)
            stability.append(stability_summary(method, method_scenes, dataset))
    write("method_catalog.csv", catalog, "方法、输入、阈值、完成状态和使用限制")
    write("overall_metrics.csv", overall, "全过程均值和 final；同时提供合并与分数据集统计，部分结果明确标记")
    write("progress_metrics.csv", progress, "20/40/60/70/80/90/final 的 P、R、F1、mIoU 与实际点数")
    write("per_scene_stability.csv", scenes, "先逐场景对已有进度计算 mean、总体 SD、CV、worst；区分未存进度与未跑完")
    write("stability_summary.csv", stability, "逐场景统计后再平均；不是把所有场景进度混起来求 SD")
    write("checkpoint_coverage.csv", inventory, "全部预期 method×checkpoint 的可用状态")
    missing = [r for r in inventory if not r["prediction_available"]]
    assert len(missing) == missing_count
    write("dude_offline_missing_predictions.csv", missing, f"离线 DUDE 缺失的 {missing_count} 个已存进度点；0 行表示已完整", fieldnames=list(inventory[0]))
    common = {key(r) for r in groups[ONLINE]} & {key(r) for r in groups[OFFLINE]}
    paired = [aggregate(m, groups[m], universe, d, e, common, omitted_keys if m == OFFLINE else set()) for m in (ONLINE, OFFLINE) for d in ("combined", *DATASETS) for e in ("all_available_progress", *EVENTS)]
    write("dude_online_offline_common_subset.csv", paired, f"在线和离线共同已有 {len(common)} 个点；" + ("完整测试集" if not missing_count else "不完整子集"))
    for name in ("update_guard_audit.csv", "input_boundary_audit.csv"):
        path = ROOT / "results/occusg_vertical_free_cropped_20260910/tables" / name
        sources.append(path)
        write("occusg_"+name, read(path), "OccuSG 外围裁剪与拒绝更新审计，保留原版本标签")
    write("excluded_training_validation_scenes.csv", [{"scene_id": s, "policy": "excluded_from_all_exported_metrics"} for s in sorted(exclusions)], "所有导出结果一致排除的训练及验证场景")
    # Keep the later online-only threshold experiment separate, not substituted for offline.
    extra_path = ROOT / "results/dude_tau1p5_vertical_20260910/tables/per_checkpoint_metrics.csv"
    sources.append(extra_path)
    extra_method = "DUDE_online_vertical_tau1p5_supplementary"
    extra = [normalize(r, extra_method, universe, extra_path) for r in read(extra_path)]
    assert len(extra) == 462
    write("supplementary/dude_online_tau1p5_per_checkpoint_metrics.csv", extra, "额外的 1.5 m 在线版；不混入 2.5 m 在线/离线主对照")
    write("supplementary/dude_online_tau1p5_overall_metrics.csv", [aggregate(extra_method, extra, universe, d, e) for d in ("combined", *DATASETS) for e in ("all_available_progress", "final")], "1.5 m 在线版的单独补充汇总")
    protocol = [
        ("selection", "仅已批准的独立测试场景，排除训练和验证；InteriorAgent 10/64 点，GRScene 58/398 点"),
        ("DUDE_versions", "用户指定的两种为在线增量和离线；主表两者均 2.5 m、raw Vertical Free；1.5 m 在线另外存 supplementary"),
        ("completion", "DUDE 在线 462/462；" + offline_status + "。所有 OccuSG 版本各 462/462"),
        ("units", "逐点 precision/recall/f1/miou_room 为 0..1；P_percent/R_percent/F1_percent/mIoU_percent 为百分数；SD_pp 是百分点；CV_ratio 和 CV_percent 明确区分"),
        ("progress", "20/40/60/70/80/90/final；缺失原始进度不补齐；final 是探索终点，不强行当 100%；actual_coverage_percent 来自保存元数据"),
        ("P", "预测房间分别寻找最大重叠 GT，交集除以该预测房间面积，再平均"),
        ("R", "GT 房间分别寻找最大重叠预测，交集除以该 GT 面积，再平均"),
        ("F1", "每个进度点先算 2PR/(P+R)，再平均；不是对总平均 P/R 再算 F1，也不是 seed 分类 F1"),
        ("mIoU", "Hungarian 房间一对一匹配 IoU 总和除以 GT 房间数；未匹配 GT 计 0，不是仅已匹配房间平均"),
        ("min_room_area", "统一 0.5 m²、0.05 m 分辨率，即 200 格；GT 仅用于预测后的统一评测"),
        ("combined", "按有效进度点数合并，不将 IA/GR 两均值各取一半；partial_available 不能当完整成绩"),
        ("paired_DUDE", f"common_subset 表中在线与离线使用同一批现有 {len(common)} 点；" + ("已覆盖完整测试集" if not missing_count else "仍不是全 462 点总分")),
        ("stability", "同一场景已有进度先算均值、总体 SD(ddof=0)、CV=SD/mean、worst=min；再对有结果的场景等权平均"),
        ("single_progress", "单进度场景 SD 数学上为 0，但不证明时间稳定；有独立标记和数量。均值为 0 的 CV 留空；无预测场景的全部指标留空"),
        ("worst", "average_scene_worst 是各场景最差进度的平均，不是全测试集单个最差值"),
        ("OccuSG_scope", "保存体素/地图与稀疏位姿的官方源码快照回放；不是完整 RGB-D/语义系统或原论文 matched-only 指标"),
        ("OccuSG_uncropped", "未裁外围版本含 42 次 unknown 比例拒绝更新，仅作历史诊断；不能混用为裁剪修复后的版本"),
        ("export_operation", "本脚本只计分/整理现有预测，不运行分割。补跑实验见单独的 completion 审计；原预测、GT 和旧结果包保持不变"),
    ]
    write("protocol.csv", [{"item": k, "definition": v} for k, v in protocol], "指标定义、单位、加权、缺失值与实验边界")
    if args.completion_audit:
        completion = json.loads(args.completion_audit.read_text())
        assert completion['original_728_sha256_unchanged'] is True
        assert completion['new_offline_predictions'] == 196-len(omitted_keys)
        assert completion['checkpoint_count'] == 924-len(omitted_keys) and unexpected_missing_count == 0
        assert {key(r) for r in completion.get('excluded_predictions', [])} == omitted_keys
        sources.append(args.completion_audit)
        write('dude_completion_audit.csv', [{'original_predictions_preserved': 728,
              'original_sha256_unchanged': True, 'new_offline_predictions': completion['new_offline_predictions'],
              'total_dude_predictions': completion['checkpoint_count'], 'missing_predictions': missing_count,
              'user_excluded_predictions': len(omitted_keys), 'unexpected_missing_predictions': unexpected_missing_count,
              'input': 'raw_vertical_free', 'concavity_threshold_m': 2.5,
              'algorithm_changed': False, 'source_audit': str(args.completion_audit)}],
              '请求范围内补跑完成，用户排除点明确列出；原 728 个预测哈希完全不变')
    if omitted_rows:
        write('user_excluded_predictions.csv', omitted_rows, '用户明确排除的崩溃点，仅针对 DUDE 离线；不补零，不丢弃整个场景')
    sources.extend([args.dude_export / "export_audit.json", args.dude_export / "checkpoint_inventory.csv"])
    write("source_manifest.csv", [{"source_path": str(p), "sha256": digest(p)} for p in sorted(set(sources))], "输入结果文件校验和，用于追溯")
    rows_for_files = list(files)
    write("files.csv", rows_for_files, "文件导航、行数、编码与 SHA256")
    text = f"""# DUDE / OccuSG CSV 结果包

这是已有结果整理，不是重新运行算法。CSV 均为 UTF-8 BOM，可用 Excel 打开。

- DUDE 两种：修复后的在线增量和离线，主表都使用 2.5 m 阈值及原始 Vertical Free。在线 462/462；{offline_status}。
- OccuSG 三种：官方投影、Vertical Free 未裁外围、Vertical Free 裁外围修复。每种 462/462；未裁外围有已知拒绝更新干扰，用标签保留，不当成修复版。
- 后来 1.5 m 的 DUDE 在线实验仅放在 supplementary，不替代在线/离线主表任何行。
- 所有结果排除相同训练、验证场景。合并指标按有效进度点数量加权。缺失预测不补零；部分汇总有明确 coverage_status 和点数，不能当成完整成绩。
- 用户另行排除的离线崩溃点数为 {len(omitted_keys)}，未授权的缺失点为 {unexpected_missing_count}；这类点从离线均值分母中移除。其他方法保留原始全部点，共同样本比较单独列出。

先看 files.csv 和 method_catalog.csv。逐点数据在 dude_per_checkpoint_metrics.csv、occusg_per_checkpoint_metrics.csv；总分在 overall_metrics.csv；进度在 progress_metrics.csv；同场景稳定性在 per_scene_stability.csv、stability_summary.csv。

dude_online_offline_common_subset.csv 提供在线/离线共同已有 {len(common)} 点的公平比较；462 点表示完整。protocol.csv 包含指标定义、单位和统计规则。
"""
    (args.output / "README.md").write_text(text, encoding="utf-8")
    archive_path = args.output.with_suffix(".zip")
    if archive_path.exists():
        raise FileExistsError(archive_path)
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(args.output.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(args.output))
    print(json.dumps({"export_directory": str(args.output), "archive": str(archive_path),
                      "dude_online_rows": len(groups[ONLINE]), "dude_offline_rows": len(groups[OFFLINE]),
                      "occusg_rows": sum(len(v) for k, v in groups.items() if k.startswith("OccuSG")),
                      "missing_offline_predictions": len(missing), "csv_files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
