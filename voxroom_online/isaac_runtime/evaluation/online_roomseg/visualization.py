from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def colorize_labels(labels: np.ndarray, *, background: tuple[int, int, int] = (18, 20, 26)) -> np.ndarray:
    arr = np.asarray(labels, dtype=np.int32)
    rgb = np.zeros(arr.shape + (3,), dtype=np.uint8)
    rgb[:] = background
    for label in np.unique(arr):
        if int(label) <= 0:
            continue
        digest = hashlib.blake2b(str(int(label)).encode("ascii"), digest_size=3).digest()
        rgb[arr == int(label)] = tuple(int(55 + (byte % 190)) for byte in digest)
    return rgb


def save_label_overlay(
    out: Path,
    *,
    labels: np.ndarray,
    domain: np.ndarray,
    obstacle: np.ndarray | None = None,
    unknown: np.ndarray | None = None,
    split_lines: Iterable[Mapping[str, object]] | None = None,
    title: str = "",
) -> None:
    labels = np.asarray(labels, dtype=np.int32)
    domain = np.asarray(domain, dtype=bool)
    base = np.zeros(labels.shape + (3,), dtype=np.uint8)
    base[:] = (25, 27, 34)
    base[domain] = (105, 105, 112)
    if unknown is not None and np.asarray(unknown).shape == labels.shape:
        base[np.asarray(unknown, dtype=bool)] = (55, 55, 62)
    if obstacle is not None and np.asarray(obstacle).shape == labels.shape:
        base[np.asarray(obstacle, dtype=bool)] = (0, 0, 0)
    colors = colorize_labels(labels)
    mask = labels > 0
    overlay = base.astype(np.float32)
    overlay[mask] = 0.35 * overlay[mask] + 0.65 * colors[mask].astype(np.float32)
    image = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    if split_lines:
        for line in split_lines:
            p0 = line.get("p0_rc", (0, 0))
            p1 = line.get("p1_rc", (0, 0))
            width = max(1, int(line.get("width_cells", 3)))
            color = (215, 25, 28) if str(line.get("kind", "separator")) == "wall_completion" else (139, 43, 226)
            draw.line([(int(p0[1]), int(p0[0])), (int(p1[1]), int(p1[0]))], fill=color, width=width)
    _draw_label_ids(draw, labels)
    if title:
        draw.rectangle((0, 0, max(120, 8 * len(title)), 16), fill=(0, 0, 0))
        draw.text((4, 2), title, fill=(255, 255, 255))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)


def save_step_gt_overlay(
    out: Path,
    *,
    gt: np.ndarray,
    metric_domain: np.ndarray,
    raw_domain: np.ndarray,
    title: str = "",
) -> None:
    gt = np.asarray(gt, dtype=np.int32)
    metric = np.asarray(metric_domain, dtype=bool)
    raw = np.asarray(raw_domain, dtype=bool)
    base = np.zeros(gt.shape + (3,), dtype=np.uint8)
    base[:] = (22, 24, 30)
    base[raw] = (80, 80, 84)
    base[raw & ~metric] = (210, 0, 210)
    colors = colorize_labels(gt)
    mask = gt > 0
    overlay = base.astype(np.float32)
    overlay[mask] = 0.35 * overlay[mask] + 0.65 * colors[mask].astype(np.float32)
    image = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    _draw_label_ids(draw, gt)
    if title:
        draw.rectangle((0, 0, max(120, 8 * len(title)), 16), fill=(0, 0, 0))
        draw.text((4, 2), title, fill=(255, 255, 255))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    image.save(out)


def save_match_visualization(
    out: Path,
    *,
    pred: np.ndarray,
    gt: np.ndarray,
    iou_matrix: np.ndarray,
    metric: Mapping[str, object],
) -> None:
    pred_img = Image.fromarray(colorize_labels(pred), mode="RGB")
    gt_img = Image.fromarray(colorize_labels(gt), mode="RGB")
    match = np.zeros_like(np.asarray(pred, dtype=np.int32))
    for pair in metric.get("matched_pairs", []) or []:
        gt_label = pair.get("gt_label")
        pred_label = pair.get("pred_label")
        if pred_label is not None and gt_label is not None:
            match[(np.asarray(gt) == int(gt_label)) & (np.asarray(pred) == int(pred_label))] = int(gt_label)
    match_img = Image.fromarray(colorize_labels(match), mode="RGB")
    w, h = pred_img.size
    table_w = max(260, int(w * 0.45))
    canvas = Image.new("RGB", (w * 3 + table_w, h), (16, 18, 24))
    canvas.paste(pred_img, (0, 0))
    canvas.paste(gt_img, (w, 0))
    canvas.paste(match_img, (2 * w, 0))
    draw = ImageDraw.Draw(canvas)
    x0 = 3 * w + 8
    lines = [
        "step=%s" % metric.get("step"),
        "n_gt=%s n_pred=%s" % (metric.get("n_gt"), metric.get("n_pred")),
        "USR=%.4f OSR=%.4f" % (float(metric.get("usr", 0.0)), float(metric.get("osr", 0.0))),
        "mIoU=%.4f CSR=%s" % (float(metric.get("miou_room", 0.0)), metric.get("csr")),
        "IoU matrix:",
    ]
    for i, line in enumerate(lines):
        draw.text((x0, 8 + i * 15), line, fill=(255, 255, 255))
    mat = np.asarray(iou_matrix, dtype=float)
    y = 8 + len(lines) * 15
    for r in range(min(mat.shape[0], 12)):
        row = " ".join("%.2f" % float(v) for v in mat[r, : min(mat.shape[1], 8)])
        draw.text((x0, y + r * 15), row, fill=(230, 230, 230))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


def _draw_label_ids(draw: ImageDraw.ImageDraw, labels: np.ndarray) -> None:
    arr = np.asarray(labels, dtype=np.int32)
    for label in np.unique(arr):
        if int(label) <= 0:
            continue
        rr, cc = np.nonzero(arr == int(label))
        if rr.size == 0:
            continue
        r = int(np.median(rr))
        c = int(np.median(cc))
        text = str(int(label))
        draw.rectangle((c - 2, r - 2, c + 7 * len(text) + 2, r + 12), fill=(0, 0, 0))
        draw.text((c, r), text, fill=(255, 255, 255))
