from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from voxroom_online.isaac_runtime.evaluation.online_roomseg.common import now_iso, read_json, write_json_atomic
from voxroom_online.isaac_runtime.evaluation.online_roomseg.snapshot_index import load_index


def link_method_gt(
    *,
    source_index_path: Path,
    method_index_path: Path,
    source_step_gt_dir: Path,
    out_step_gt_dir: Path,
    source_csr_dir: Path | None = None,
    out_csr_dir: Path | None = None,
    overwrite: bool = False,
    copy: bool = False,
) -> dict[str, Any]:
    source_index = load_index(Path(source_index_path))
    method_index = load_index(Path(method_index_path))
    source_by_key = {_episode_key(ep): ep for ep in source_index.get("episodes", [])}
    rows: list[dict[str, Any]] = []
    csr_rows: list[dict[str, Any]] = []
    for method_ep in method_index.get("episodes", []):
        key = _episode_key(method_ep)
        if key not in source_by_key:
            raise KeyError("method episode has no source match by scene_id/episode_id: %s" % (key,))
        source_ep = source_by_key[key]
        src_uid = str(source_ep["episode_uid"])
        dst_uid = str(method_ep["episode_uid"])
        source_snaps = _snapshots_by_step(source_ep)
        for method_snap in method_ep.get("snapshots", []):
            step = int(method_snap["step"])
            if step not in source_snaps:
                raise KeyError("method snapshot step has no source match: %s %s" % (key, step))
            source_snap = source_snaps[step]
            stem = Path(str(method_snap["snapshot_path"])).stem
            src_stem = Path(str(source_snap["snapshot_path"])).stem
            if stem != src_stem:
                raise ValueError("snapshot stem mismatch for %s step %s: %s != %s" % (key, step, stem, src_stem))
            rows.append(
                _link_one_step_gt(
                    source_step_gt_dir=Path(source_step_gt_dir),
                    out_step_gt_dir=Path(out_step_gt_dir),
                    src_uid=src_uid,
                    dst_uid=dst_uid,
                    source_snap=source_snap,
                    method_snap=method_snap,
                    stem=stem,
                    src_stem=src_stem,
                    overwrite=overwrite,
                    copy=copy,
                )
            )
            if source_csr_dir is not None and out_csr_dir is not None:
                linked = _link_one_csr(
                    source_csr_dir=Path(source_csr_dir),
                    out_csr_dir=Path(out_csr_dir),
                    src_uid=src_uid,
                    dst_uid=dst_uid,
                    source_snap=source_snap,
                    method_snap=method_snap,
                    stem=stem,
                    src_stem=src_stem,
                    overwrite=overwrite,
                    copy=copy,
                )
                if linked is not None:
                    csr_rows.append(linked)
    manifest = {
        "schema_version": "voxroom_comparison_link_method_gt_v1",
        "created_at": now_iso(),
        "source_index": str(source_index_path),
        "method_index": str(method_index_path),
        "source_step_gt_dir": str(source_step_gt_dir),
        "out_step_gt_dir": str(out_step_gt_dir),
        "source_csr_dir": None if source_csr_dir is None else str(source_csr_dir),
        "out_csr_dir": None if out_csr_dir is None else str(out_csr_dir),
        "copy": bool(copy),
        "rows": rows,
        "csr_rows": csr_rows,
    }
    write_json_atomic(Path(out_step_gt_dir) / "link_method_gt_manifest.json", manifest)
    return manifest


def _episode_key(episode: Mapping[str, Any]) -> tuple[str, str]:
    return (str(episode.get("scene_id")), str(episode.get("episode_id")))


def _snapshots_by_step(episode: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    out: dict[int, Mapping[str, Any]] = {}
    for snap in episode.get("snapshots", []):
        step = int(snap["step"])
        if step in out:
            raise ValueError("duplicate snapshot step in source episode %s: %s" % (episode.get("episode_uid"), step))
        out[step] = snap
    return out


def _link_one_step_gt(
    *,
    source_step_gt_dir: Path,
    out_step_gt_dir: Path,
    src_uid: str,
    dst_uid: str,
    source_snap: Mapping[str, Any],
    method_snap: Mapping[str, Any],
    stem: str,
    src_stem: str,
    overwrite: bool,
    copy: bool,
) -> dict[str, Any]:
    src_dir = source_step_gt_dir / src_uid
    dst_dir = out_step_gt_dir / dst_uid
    dst_dir.mkdir(parents=True, exist_ok=True)
    labels_src = src_dir / f"{src_stem}.gt_labels.npy"
    labels_dst = dst_dir / f"{stem}.gt_labels.npy"
    meta_src = src_dir / f"{src_stem}.gt_metadata.json"
    meta_dst = dst_dir / f"{stem}.gt_metadata.json"
    if not labels_src.exists():
        raise FileNotFoundError("missing source step GT labels: %s" % labels_src)
    if not meta_src.exists():
        raise FileNotFoundError("missing source step GT metadata: %s" % meta_src)
    _copy_or_symlink(labels_src, labels_dst, overwrite=overwrite, copy=copy)
    metadata = dict(read_json(meta_src))
    metadata.update(
        {
            "source_episode_uid": src_uid,
            "episode_uid": dst_uid,
            "source_snapshot_path": str(source_snap["snapshot_path"]),
            "snapshot_path": str(method_snap["snapshot_path"]),
            "source_gt_metadata_json": str(meta_src),
            "gt_label_npy": str(labels_dst),
            "gt_metadata_json": str(meta_dst),
        }
    )
    overlay_src = src_dir / f"{src_stem}.gt_overlay.png"
    overlay_dst = dst_dir / f"{stem}.gt_overlay.png"
    if overlay_src.exists():
        _copy_or_symlink(overlay_src, overlay_dst, overwrite=overwrite, copy=copy)
        metadata["gt_overlay_png"] = str(overlay_dst)
    write_json_atomic(meta_dst, metadata)
    return {
        "source_episode_uid": src_uid,
        "episode_uid": dst_uid,
        "step": int(method_snap["step"]),
        "source_stem": src_stem,
        "stem": stem,
        "gt_label_npy": str(labels_dst),
        "gt_metadata_json": str(meta_dst),
    }


def _link_one_csr(
    *,
    source_csr_dir: Path,
    out_csr_dir: Path,
    src_uid: str,
    dst_uid: str,
    source_snap: Mapping[str, Any],
    method_snap: Mapping[str, Any],
    stem: str,
    src_stem: str,
    overwrite: bool,
    copy: bool,
) -> dict[str, Any] | None:
    csr_src = source_csr_dir / src_uid / f"{src_stem}.csr.json"
    if not csr_src.exists():
        return None
    dst_dir = out_csr_dir / dst_uid
    dst_dir.mkdir(parents=True, exist_ok=True)
    csr_dst = dst_dir / f"{stem}.csr.json"
    payload = dict(read_json(csr_src))
    payload.update(
        {
            "source_episode_uid": src_uid,
            "episode_uid": dst_uid,
            "source_snapshot_path": str(source_snap["snapshot_path"]),
            "snapshot_path": str(method_snap["snapshot_path"]),
        }
    )
    write_json_atomic(csr_dst, payload)
    for suffix in (".csr_preview.png",):
        preview_src = source_csr_dir / src_uid / f"{src_stem}{suffix}"
        preview_dst = dst_dir / f"{stem}{suffix}"
        if preview_src.exists():
            _copy_or_symlink(preview_src, preview_dst, overwrite=overwrite, copy=copy)
    return {"source_episode_uid": src_uid, "episode_uid": dst_uid, "step": int(method_snap["step"]), "csr_json": str(csr_dst)}


def _copy_or_symlink(src: Path, dst: Path, *, overwrite: bool, copy: bool) -> None:
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(os.path.relpath(src, dst.parent), dst)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Map VoxRoom step GT/CSR files onto a baseline method episode uid.")
    parser.add_argument("--source-index", required=True)
    parser.add_argument("--method-index", required=True)
    parser.add_argument("--source-step-gt-dir", required=True)
    parser.add_argument("--out-step-gt-dir", required=True)
    parser.add_argument("--source-csr-dir")
    parser.add_argument("--out-csr-dir")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy", action="store_true")
    args = parser.parse_args(argv)
    manifest = link_method_gt(
        source_index_path=Path(args.source_index),
        method_index_path=Path(args.method_index),
        source_step_gt_dir=Path(args.source_step_gt_dir),
        out_step_gt_dir=Path(args.out_step_gt_dir),
        source_csr_dir=None if args.source_csr_dir is None else Path(args.source_csr_dir),
        out_csr_dir=None if args.out_csr_dir is None else Path(args.out_csr_dir),
        overwrite=bool(args.overwrite),
        copy=bool(args.copy),
    )
    print("linked %d GT file(s) and %d CSR file(s)" % (len(manifest["rows"]), len(manifest["csr_rows"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

