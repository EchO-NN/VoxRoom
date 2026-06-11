from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Iterable


AUTOTUNE_VERSION = "voxel_v35_autotune_1"


def parse_thread_candidates(value: object, *, requested_max: int) -> tuple[int, ...]:
    if isinstance(value, str):
        raw = [int(v.strip()) for v in value.split(",") if v.strip()]
    elif isinstance(value, Iterable):
        raw = [int(v) for v in value]
    else:
        raw = [2, 4, 8, 14, int(requested_max)]
    cpu_count = max(1, int(os.cpu_count() or 1))
    out = sorted({max(1, min(int(v), int(requested_max), cpu_count)) for v in raw if int(v) > 0})
    return tuple(out or (max(1, min(int(requested_max), cpu_count)),))


def autotune_cache_key(grid, *, candidates: tuple[int, ...], threading_layer: str) -> str:
    cfg = grid.config
    payload = {
        "version": AUTOTUNE_VERSION,
        "shape_zyx": [int(grid.state.shape[0]), int(grid.state.shape[1]), int(grid.state.shape[2])],
        "z_resolution_m": float(grid.z_resolution_m),
        "resolution_m": float(grid.map_info.resolution_m),
        "candidates": [int(v) for v in candidates],
        "threading_layer": str(threading_layer),
        "sensor_ray_enabled": bool(getattr(cfg, "sensor_range_mark_ray_samples_enabled", False)),
        "endpoint_column_enabled": bool(getattr(cfg, "sensor_range_mark_endpoint_column_enabled", True)),
        "event_block_size": int(getattr(cfg, "cpu_numba_event_block_size", 4096)),
        "event_chunk_multiplier": int(getattr(cfg, "cpu_numba_event_chunk_count_multiplier", 4)),
        "cpu_count": int(os.cpu_count() or 1),
        "cpu": str(platform.processor() or _cpu_model_name()),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def load_cached_thread_count(path: str | os.PathLike[str], key: str) -> int | None:
    cache_path = Path(path).expanduser()
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    entry = data.get(key) if isinstance(data, dict) else None
    if not isinstance(entry, dict):
        return None
    try:
        return int(entry.get("best_threads"))
    except Exception:
        return None


def store_cached_thread_count(path: str | os.PathLike[str], key: str, *, best_threads: int, results: list[dict[str, object]]) -> None:
    cache_path = Path(path).expanduser()
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}
        data[str(key)] = {
            "best_threads": int(best_threads),
            "results": results,
        }
        cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError:
        return


def _cpu_model_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "unknown"
