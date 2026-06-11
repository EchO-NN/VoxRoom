from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .annotation_app import run_annotation_app
from .annotation_schema import approved_or_raise, load_annotation, save_annotation_atomic, validate_annotation
from .common import read_json, write_json_atomic
from .csr_review import CsrReview, csr_path, load_csr, save_csr_atomic
from .mask_generation import GtGenerationConfig, generate_gt_from_annotation
from .metrics import compute_snapshot_metrics, min_area_cells_from_m2, prepare_metric_label_maps
from .reporting import (
    aggregate_episode_first,
    write_metrics_outputs,
    write_review_gallery,
    write_summary_markdown,
)
from .snapshot_index import build_index, load_index, write_index
from .snapshot_io import load_snapshot_arrays
from .step_backprojection import backproject_episode
from .validation import validate_gt_label_map, validate_metric_inputs
from .visualization import save_label_overlay, save_match_visualization


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="voxroom-roomseg-eval", description="Evaluate online room segmentation snapshots.")
    sub = parser.add_subparsers(dest="command", required=True)

    index_p = sub.add_parser("index", help="scan result roots and build a snapshot index")
    index_p.add_argument("--result-root", action="append", required=True)
    index_p.add_argument("--out", required=True)
    index_p.add_argument("--snapshot-policy", choices=("all", "last"), default="all")
    index_p.add_argument("--require-npz", dest="require_npz", action="store_true", default=True)
    index_p.add_argument("--no-require-npz", dest="require_npz", action="store_false")
    index_p.add_argument("--allow-missing-png", action="store_true")
    index_p.add_argument("--no-validate-npz", action="store_true")

    annotate_p = sub.add_parser("annotate-last", help="interactively annotate final snapshots")
    annotate_p.add_argument("--index", required=True)
    annotate_p.add_argument("--annotation-dir", required=True)
    annotate_p.add_argument("--gt-dir", required=True)
    annotate_p.add_argument("--line-width-cells", type=int, default=3)
    annotate_p.add_argument("--preclose-radius-cells", type=int, default=1)
    annotate_p.add_argument("--figure-scale", type=float, default=1.4)
    annotate_p.add_argument("--episode-uid", action="append")
    annotate_p.add_argument("--skip-approved", action="store_true")
    annotate_p.add_argument("--min-room-area-m2", type=float, default=0.3)
    annotate_p.add_argument("--cell-size-m", type=float, default=0.05)
    annotate_p.add_argument("--source-view", choices=("navigation", "segmentation"), default="navigation")

    build_p = sub.add_parser("build-gt", help="build final GT masks from saved annotations")
    build_p.add_argument("--index", required=True)
    build_p.add_argument("--annotation-dir", required=True)
    build_p.add_argument("--gt-dir", required=True)
    build_p.add_argument("--allow-snapshot-changed", action="store_true")
    build_p.add_argument("--include-draft", action="store_true")

    back_p = sub.add_parser("backproject", help="backproject approved final GT to all snapshots")
    back_p.add_argument("--index", required=True)
    back_p.add_argument("--gt-dir", required=True)
    back_p.add_argument("--step-gt-dir", required=True)
    back_p.add_argument("--eval-snapshot-policy", choices=("all", "last"), default="all")
    back_p.add_argument("--min-visible-gt-cells", type=int, default=1)
    back_p.add_argument("--allow-draft", action="store_true")

    review_p = sub.add_parser("review-csr", help="record per-snapshot CSR review labels")
    review_p.add_argument("--index", required=True)
    review_p.add_argument("--step-gt-dir", required=True)
    review_p.add_argument("--csr-dir", required=True)
    review_p.add_argument("--reviewer", default="annotator")
    review_p.add_argument("--episode-uid", action="append")
    review_p.add_argument("--default-csr", choices=("0", "1"))
    review_p.add_argument("--overwrite", action="store_true")
    review_p.add_argument("--no-open-preview", action="store_true")
    review_p.add_argument("--min-room-area-m2", type=float, default=0.3)
    review_p.add_argument("--cell-size-m", type=float, default=0.05)

    compute_p = sub.add_parser("compute", help="compute online room segmentation metrics")
    compute_p.add_argument("--index", required=True)
    compute_p.add_argument("--step-gt-dir", required=True)
    compute_p.add_argument("--csr-dir")
    compute_p.add_argument("--out-dir", required=True)
    compute_p.add_argument("--strict-paper", action="store_true")
    compute_p.add_argument("--require-csr", dest="require_csr", action="store_true", default=True)
    compute_p.add_argument("--no-require-csr", dest="require_csr", action="store_false")
    compute_p.add_argument("--allow-invalid", action="store_true")
    compute_p.add_argument("--min-room-area-m2", type=float, default=0.3)
    compute_p.add_argument("--cell-size-m", type=float, default=0.05)

    report_p = sub.add_parser("report", help="render a report from computed metrics")
    report_p.add_argument("--metrics-dir", required=True)
    report_p.add_argument("--out", required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "index":
            return _cmd_index(args)
        if args.command == "annotate-last":
            return _cmd_annotate_last(args)
        if args.command == "build-gt":
            return _cmd_build_gt(args)
        if args.command == "backproject":
            return _cmd_backproject(args)
        if args.command == "review-csr":
            return _cmd_review_csr(args)
        if args.command == "compute":
            return _cmd_compute(args)
        if args.command == "report":
            return _cmd_report(args)
    except Exception as exc:
        print("voxroom-roomseg-eval %s failed: %s" % (args.command, exc), file=sys.stderr)
        return 1
    parser.error("unknown command: %s" % args.command)
    return 2


def _cmd_index(args: argparse.Namespace) -> int:
    index = build_index(
        [Path(p) for p in args.result_root],
        snapshot_policy=str(args.snapshot_policy),
        require_npz=bool(args.require_npz),
        allow_missing_png=bool(args.allow_missing_png),
        validate_npz=not bool(args.no_validate_npz),
    )
    write_index(index, Path(args.out))
    print("indexed %d episode(s) -> %s" % (len(index.get("episodes", [])), str(args.out)))
    return 0


def _cmd_annotate_last(args: argparse.Namespace) -> int:
    index = load_index(Path(args.index))
    episodes = _select_episodes(index, args.episode_uid)
    min_room_area_cells = min_area_cells_from_m2(float(args.min_room_area_m2), float(args.cell_size_m))
    for episode in episodes:
        snapshot = load_snapshot_arrays(Path(episode["last_snapshot_path"]))
        annotation_path = _annotation_path(Path(args.annotation_dir), episode)
        if bool(args.skip_approved) and annotation_path.exists():
            try:
                if load_annotation(annotation_path).review.status == "approved":
                    print("skip approved annotation: %s" % episode["episode_uid"])
                    continue
            except Exception:
                pass
        run_annotation_app(
            episode=episode,
            snapshot=snapshot,
            annotation_path=annotation_path,
            gt_dir=Path(args.gt_dir),
            line_width_cells=int(args.line_width_cells),
            preclose_radius_cells=int(args.preclose_radius_cells),
            figure_scale=float(args.figure_scale),
            min_room_area_cells=int(min_room_area_cells),
            source_view=str(args.source_view),
        )
    return 0


def _cmd_build_gt(args: argparse.Namespace) -> int:
    index = load_index(Path(args.index))
    outputs: list[dict[str, Any]] = []
    for episode in index.get("episodes", []):
        snapshot = load_snapshot_arrays(Path(episode["last_snapshot_path"]))
        annotation_path = _annotation_path(Path(args.annotation_dir), episode)
        annotation = load_annotation(annotation_path)
        validate_annotation(annotation, snapshot, allow_snapshot_changed=bool(args.allow_snapshot_changed))
        if not bool(args.include_draft):
            approved_or_raise(annotation)
        result = generate_gt_from_annotation(
            eval_domain=snapshot.segmentation_domain,
            split_lines=annotation.split_lines,
            merge_groups=annotation.merge_groups,
            obstacle_mask=snapshot.obstacle_mask,
            segmentation_domain=snapshot.segmentation_domain,
            config=GtGenerationConfig(
                line_width_cells=int(annotation.line_width_cells_default),
                preclose_radius_cells=int(annotation.preclose_radius_cells),
                min_room_area_cells=int(annotation.min_room_area_cells),
            ),
        )
        validate_gt_label_map(result.labels, snapshot.shape, domain=snapshot.segmentation_domain)
        out_paths = _write_final_gt(
            episode=episode,
            annotation=annotation,
            annotation_path=annotation_path,
            snapshot=snapshot,
            result=result,
            gt_dir=Path(args.gt_dir),
        )
        generated = {
            "gt_label_npy": str(out_paths["gt_label_npy"]),
            "gt_label_png": str(out_paths["gt_label_png"]),
            "gt_metadata_json": str(out_paths["gt_metadata_json"]),
            "room_count": int(result.metadata["room_count"]),
            "domain_pixels": int(result.metadata["domain_pixels"]),
            "unlabeled_domain_pixels": int(result.metadata["unlabeled_domain_pixels"]),
        }
        save_annotation_atomic(replace(annotation, generated_gt=generated), annotation_path)
        outputs.append({"episode_uid": episode["episode_uid"], **out_paths})
    print("built final GT for %d episode(s)" % len(outputs))
    return 0


def _cmd_backproject(args: argparse.Namespace) -> int:
    index = load_index(Path(args.index))
    rows: list[dict[str, Any]] = []
    for episode in index.get("episodes", []):
        final = _final_gt_paths(Path(args.gt_dir), str(episode["episode_uid"]))
        if not bool(args.allow_draft):
            status = _final_gt_review_status(final["metadata"])
            if status != "approved":
                raise ValueError("final GT is not approved for %s: %s" % (episode["episode_uid"], status))
        outputs = backproject_episode(
            episode=episode,
            final_gt_path=final["labels"],
            step_gt_dir=Path(args.step_gt_dir),
            min_visible_gt_cells=int(args.min_visible_gt_cells),
            snapshot_policy=str(args.eval_snapshot_policy),
            extra_metadata={
                "annotation_review_status": _final_gt_review_status(final["metadata"]),
                "final_gt_label_npy": str(final["labels"]),
                "final_gt_metadata_json": str(final["metadata"]),
            },
        )
        rows.extend(outputs)
    write_json_atomic(Path(args.step_gt_dir) / "backprojection_manifest.json", {"rows": rows})
    print("backprojected %d snapshot GT file(s)" % len(rows))
    return 0


def _cmd_review_csr(args: argparse.Namespace) -> int:
    index = load_index(Path(args.index))
    episodes = _select_episodes(index, args.episode_uid)
    count = 0
    for episode in episodes:
        for snap in episode.get("snapshots", []):
            stem = Path(snap["snapshot_path"]).stem
            out = csr_path(Path(args.csr_dir), str(episode["episode_uid"]), stem)
            if out.exists() and not bool(args.overwrite):
                continue
            gt_path = Path(args.step_gt_dir) / str(episode["episode_uid"]) / f"{stem}.gt_labels.npy"
            if not gt_path.exists():
                raise FileNotFoundError("missing step GT for CSR review: %s" % gt_path)
            preview = _write_csr_preview(
                episode=episode,
                snap=snap,
                gt_path=gt_path,
                csr_dir=Path(args.csr_dir),
                min_room_area_m2=float(args.min_room_area_m2),
                cell_size_m=float(args.cell_size_m),
            )
            if args.default_csr is None:
                if not bool(args.no_open_preview):
                    _open_preview_nonblocking(preview)
                print("CSR preview: %s" % preview)
                answer = input("%s step=%s CSR [0/1]: " % (episode["episode_uid"], snap["step"])).strip()
            else:
                answer = str(args.default_csr)
            if answer not in {"0", "1"}:
                raise ValueError("CSR must be 0 or 1 for %s step %s" % (episode["episode_uid"], snap["step"]))
            review = CsrReview(
                episode_uid=str(episode["episode_uid"]),
                step=int(snap["step"]),
                snapshot_path=str(snap["snapshot_path"]),
                csr=int(answer),
                reviewer=str(args.reviewer),
            )
            save_csr_atomic(review, out)
            count += 1
    print("wrote %d CSR review file(s)" % count)
    return 0


def _write_csr_preview(
    *,
    episode: Mapping[str, Any],
    snap: Mapping[str, Any],
    gt_path: Path,
    csr_dir: Path,
    min_room_area_m2: float,
    cell_size_m: float,
) -> Path:
    episode_uid = str(episode["episode_uid"])
    stem = Path(str(snap["snapshot_path"])).stem
    snapshot = load_snapshot_arrays(Path(str(snap["snapshot_path"])))
    gt = np.asarray(np.load(gt_path), dtype=np.int32)
    min_room_area_cells = min_area_cells_from_m2(float(min_room_area_m2), float(cell_size_m))
    prepared = prepare_metric_label_maps(gt, snapshot.final_room_label_map, min_room_area_cells=min_room_area_cells)
    metric = compute_snapshot_metrics(prepared.gt, prepared.pred, csr=None)
    metric.update(
        {
            **prepared.stats,
            "min_room_area_m2": float(min_room_area_m2),
            "cell_size_m": float(cell_size_m),
        }
    )
    preview = Path(csr_dir) / episode_uid / f"{stem}.csr_preview.png"
    save_match_visualization(
        preview,
        pred=prepared.pred,
        gt=prepared.gt,
        iou_matrix=np.asarray(metric["iou_matrix"]),
        metric={
            "episode_uid": episode_uid,
            "scene_id": episode.get("scene_id"),
            "step": int(snap["step"]),
            **metric,
        },
    )
    return preview


def _open_preview_nonblocking(path: Path) -> None:
    import shutil
    import subprocess

    opener = shutil.which("xdg-open")
    if opener is None:
        return
    try:
        subprocess.Popen([opener, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return


def _cmd_compute(args: argparse.Namespace) -> int:
    if bool(args.strict_paper) and not bool(args.require_csr):
        raise ValueError("--strict-paper requires CSR; remove --no-require-csr")
    if bool(args.require_csr) and not args.csr_dir:
        raise ValueError("--csr-dir is required unless --no-require-csr is set")
    index = load_index(Path(args.index))
    out_dir = Path(args.out_dir)
    rows: list[dict[str, Any]] = []
    invalid_count = 0
    for episode in index.get("episodes", []):
        episode_uid = str(episode["episode_uid"])
        for snap in episode.get("snapshots", []):
            row = _compute_one_snapshot(
                episode=episode,
                snap=snap,
                step_gt_dir=Path(args.step_gt_dir),
                csr_dir=None if not args.csr_dir else Path(args.csr_dir),
                out_dir=out_dir,
                strict_paper=bool(args.strict_paper),
                require_csr=bool(args.require_csr),
                min_room_area_m2=float(args.min_room_area_m2),
                cell_size_m=float(args.cell_size_m),
            )
            if not bool(row.get("valid", True)):
                invalid_count += 1
            rows.append(row)
            if bool(args.strict_paper) and not bool(row.get("valid", True)):
                raise ValueError("invalid strict snapshot %s step %s: %s" % (episode_uid, snap.get("step"), row.get("invalid_reason")))
    if invalid_count and not bool(args.allow_invalid):
        write_metrics_outputs(out_dir, rows, [], {"schema_version": "voxroom_online_roomseg_metrics_v1", "invalid_count": int(invalid_count)})
        raise ValueError("%d invalid snapshot(s); pass --allow-invalid to keep non-strict outputs" % invalid_count)
    per_episode, summary = aggregate_episode_first(rows, strict_paper=bool(args.strict_paper), require_csr=bool(args.require_csr))
    write_metrics_outputs(out_dir, rows, per_episode, summary)
    print("computed metrics for %d valid snapshot(s), %d invalid -> %s" % (len(rows) - invalid_count, invalid_count, out_dir))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    metrics_dir = Path(args.metrics_dir)
    summary = read_json(metrics_dir / "summary_metrics.json")
    per_episode = read_json(metrics_dir / "per_episode_metrics.json")
    per_snapshot = _read_jsonl(metrics_dir / "per_snapshot_metrics.jsonl")
    write_summary_markdown(Path(args.out), summary, per_episode)
    write_review_gallery(metrics_dir / "review_gallery.html", per_snapshot)
    print("wrote report -> %s" % args.out)
    return 0


def _compute_one_snapshot(
    *,
    episode: Mapping[str, Any],
    snap: Mapping[str, Any],
    step_gt_dir: Path,
    csr_dir: Path | None,
    out_dir: Path,
    strict_paper: bool,
    require_csr: bool,
    min_room_area_m2: float,
    cell_size_m: float,
) -> dict[str, Any]:
    episode_uid = str(episode["episode_uid"])
    stem = Path(str(snap["snapshot_path"])).stem
    base = {
        "episode_uid": episode_uid,
        "scene_id": episode.get("scene_id"),
        "step": int(snap["step"]),
        "snapshot_path": str(snap["snapshot_path"]),
        "source_final_step": int(episode["last_snapshot_step"]),
        "final_reported_step": episode.get("final_reported_step"),
        "step_delta_to_final_snapshot": int(snap["step"]) - int(episode["last_snapshot_step"]),
        "valid": True,
        "invalid_reason": None,
    }
    try:
        snapshot = load_snapshot_arrays(Path(str(snap["snapshot_path"])))
        gt_path = step_gt_dir / episode_uid / f"{stem}.gt_labels.npy"
        meta_path = step_gt_dir / episode_uid / f"{stem}.gt_metadata.json"
        if not gt_path.exists():
            raise FileNotFoundError("missing step GT: %s" % gt_path)
        if not meta_path.exists():
            raise FileNotFoundError("missing step GT metadata: %s" % meta_path)
        gt_meta = read_json(meta_path)
        if str(gt_meta.get("annotation_review_status")) != "approved":
            raise ValueError("step GT not derived from approved final GT")
        gt = np.asarray(np.load(gt_path), dtype=np.int32)
        min_room_area_cells = min_area_cells_from_m2(float(min_room_area_m2), float(cell_size_m))
        prepared = prepare_metric_label_maps(gt, snapshot.final_room_label_map, min_room_area_cells=min_room_area_cells)
        gt_metric = prepared.gt
        pred_metric = prepared.pred
        metric_domain = prepared.metric_domain
        csr = None
        csr_exists = False
        if require_csr:
            assert csr_dir is not None
            csr_file = csr_path(csr_dir, episode_uid, stem)
            csr_exists = csr_file.exists()
            if csr_exists:
                csr_review = load_csr(csr_file)
                csr = int(csr_review.csr)
        invalid = validate_metric_inputs(gt_metric, pred_metric, metric_domain, strict_paper=strict_paper, csr_exists=csr_exists, require_csr=require_csr)
        if invalid is not None:
            raise ValueError(invalid)
        metric = compute_snapshot_metrics(gt_metric, pred_metric, csr=csr)
        metric.update(
            {
                **prepared.stats,
                "min_room_area_m2": float(min_room_area_m2),
                "cell_size_m": float(cell_size_m),
            }
        )
        match_path = out_dir / "match_visualizations" / episode_uid / f"{stem}.pred_gt_match.png"
        vis_metric = {**base, **metric, "match_visualization": str(match_path)}
        save_match_visualization(match_path, pred=pred_metric, gt=gt_metric, iou_matrix=np.asarray(metric["iou_matrix"]), metric=vis_metric)
        metric.pop("iou_matrix", None)
        return {
            **base,
            "metric_domain_pixels": int(np.count_nonzero(metric_domain)),
            **metric,
            "match_visualization": str(match_path),
        }
    except Exception as exc:
        return {
            **base,
            "metric_domain_pixels": 0,
            "min_room_area_m2": float(min_room_area_m2),
            "cell_size_m": float(cell_size_m),
            "min_room_area_cells": 0,
            "gt_label_masks_before_filter": 0,
            "gt_label_masks_after_filter": 0,
            "gt_label_masks_filtered_small": 0,
            "gt_cells_before_filter": 0,
            "gt_cells_after_filter": 0,
            "pred_label_masks_before_filter": 0,
            "pred_label_masks_after_filter": 0,
            "pred_label_masks_filtered_small": 0,
            "pred_cells_before_filter": 0,
            "pred_cells_after_filter": 0,
            "n_gt": 0,
            "n_pred": 0,
            "usr": 0.0,
            "osr": 0.0,
            "miou_room": 0.0,
            "csr": None,
            "matched_pairs": [],
            "unmatched_pred_labels": [],
            "match_visualization": None,
            "valid": False,
            "invalid_reason": str(exc),
        }


def _write_final_gt(*, episode: Mapping[str, Any], annotation, annotation_path: Path, snapshot, result, gt_dir: Path) -> dict[str, str]:
    out_dir = Path(gt_dir) / str(episode["episode_uid"])
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_label = out_dir / "last_step.gt_labels.npy"
    gt_png = out_dir / "last_step.gt_overlay.png"
    gt_meta = out_dir / "last_step.gt_metadata.json"
    np.save(gt_label, np.asarray(result.labels, dtype=np.int32))
    save_label_overlay(
        gt_png,
        labels=result.labels,
        domain=snapshot.segmentation_domain,
        obstacle=snapshot.obstacle_mask,
        unknown=snapshot.unknown_mask,
        split_lines=[line.to_dict() for line in annotation.split_lines],
        title="approved GT" if annotation.review.status == "approved" else "draft GT",
    )
    metadata = {
        **result.metadata,
        "episode_uid": str(episode["episode_uid"]),
        "scene_id": episode.get("scene_id"),
        "last_step": int(episode["last_snapshot_step"]),
        "snapshot_path": str(snapshot.path),
        "annotation_path": str(annotation_path),
        "annotation_review_status": str(annotation.review.status),
        "annotation_snapshot_sha256": str(annotation.snapshot_sha256),
        "segmentation_domain_key": str(snapshot.segmentation_domain_key),
        "output_domain_key": str(snapshot.segmentation_domain_key),
        "navigation_domain_key": str(snapshot.domain_key),
        "gt_label_npy": str(gt_label),
        "gt_label_png": str(gt_png),
        "gt_metadata_json": str(gt_meta),
    }
    write_json_atomic(gt_meta, metadata)
    return {"gt_label_npy": str(gt_label), "gt_label_png": str(gt_png), "gt_metadata_json": str(gt_meta)}


def _select_episodes(index: Mapping[str, Any], episode_uids: list[str] | None) -> list[dict[str, Any]]:
    episodes = list(index.get("episodes", []))
    if not episode_uids:
        return episodes
    wanted = set(str(v) for v in episode_uids)
    selected = [dict(ep) for ep in episodes if str(ep.get("episode_uid")) in wanted]
    missing = wanted - {str(ep.get("episode_uid")) for ep in selected}
    if missing:
        raise ValueError("episode_uid not found in index: %s" % sorted(missing))
    return selected


def _annotation_path(annotation_dir: Path, episode: Mapping[str, Any]) -> Path:
    return Path(annotation_dir) / str(episode["episode_uid"]) / "last_step.annotation.json"
def _final_gt_paths(gt_dir: Path, episode_uid: str) -> dict[str, Path]:
    base = Path(gt_dir) / str(episode_uid)
    paths = {
        "labels": base / "last_step.gt_labels.npy",
        "overlay": base / "last_step.gt_overlay.png",
        "metadata": base / "last_step.gt_metadata.json",
    }
    for key, path in paths.items():
        if key != "overlay" and not path.exists():
            raise FileNotFoundError("missing final GT %s: %s" % (key, path))
    return paths


def _final_gt_review_status(metadata_path: Path) -> str:
    meta = read_json(Path(metadata_path))
    return str(meta.get("annotation_review_status", "unknown"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                import json

                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
