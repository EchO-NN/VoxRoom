from __future__ import annotations

import datetime as _dt
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


SNAPSHOT_RE = re.compile(r"roomseg_step_(?P<step>\d+)")


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def make_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Mapping):
        return {str(k): make_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_jsonable(v) for v in value]
    return value


def write_json_atomic(path: Path, payload: Mapping[str, Any] | list[Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(make_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True)
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(data)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def slug(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
    return out.strip("_") or "episode"


def relabel_positive_labels_sequentially(labels: np.ndarray) -> np.ndarray:
    arr = np.asarray(labels, dtype=np.int32)
    out = np.zeros_like(arr, dtype=np.int32)
    next_label = 1
    for label in sorted(int(v) for v in np.unique(arr) if int(v) > 0):
        out[arr == label] = int(next_label)
        next_label += 1
    return out


def relabel_subset_sequential(gt: np.ndarray, visible_labels: Iterable[int]) -> np.ndarray:
    arr = np.asarray(gt, dtype=np.int32)
    out = np.zeros_like(arr, dtype=np.int32)
    for next_label, label in enumerate([int(v) for v in visible_labels if int(v) > 0], start=1):
        out[arr == int(label)] = int(next_label)
    return out


def positive_labels(label_map: np.ndarray) -> list[int]:
    return sorted(int(v) for v in np.unique(np.asarray(label_map, dtype=np.int32)) if int(v) > 0)


def parse_snapshot_step(path: Path, summary_json: Path | None = None, metadata_step: int | None = None) -> tuple[int | None, str]:
    match = SNAPSHOT_RE.search(Path(path).stem)
    if match:
        return int(match.group("step")), "filename"
    if summary_json is not None and Path(summary_json).exists():
        try:
            data = read_json(Path(summary_json))
            if "step" in data:
                return int(data["step"]), "summary_json"
        except Exception:
            pass
    if metadata_step is not None:
        return int(metadata_step), "npz_metadata"
    return None, "unparsed"

