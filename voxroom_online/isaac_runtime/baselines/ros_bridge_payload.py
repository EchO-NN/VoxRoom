from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = 2


def write_request(
    path: Path,
    *,
    method: str,
    snapshot_path: Path | str,
    input_npz: Path,
    output_npz: Path,
    scene_id: str | None,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    request = {
        "schema_version": SCHEMA_VERSION,
        "method": str(method),
        "snapshot_path": str(snapshot_path),
        "input_npz": str(input_npz),
        "output_npz": str(output_npz),
        "scene_id": scene_id,
        "params": _json_ready(dict(params or {})),
    }
    Path(path).write_text(json.dumps(request, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return request


def load_request(path: Path | str) -> dict[str, Any]:
    request = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(request.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported ROS bridge request schema_version=%s" % request.get("schema_version"))
    return dict(request)


def save_input_arrays(path: Path | str, arrays: Mapping[str, Any]) -> None:
    serializable = {str(k): np.asarray(v) for k, v in arrays.items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **serializable)


def load_input_arrays(request: Mapping[str, Any]) -> dict[str, np.ndarray]:
    with np.load(Path(str(request["input_npz"])), allow_pickle=False) as data:
        return {str(name): np.asarray(data[name]).copy() for name in data.files}


def write_result(
    request: Mapping[str, Any],
    *,
    label_map: np.ndarray,
    metadata: Mapping[str, Any],
    debug_arrays: Mapping[str, Any] | None = None,
) -> None:
    output_npz = Path(str(request["output_npz"]))
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "label_map": np.asarray(label_map, dtype=np.int32),
        "metadata_json": np.asarray(json.dumps(_json_ready(dict(metadata)), ensure_ascii=False, sort_keys=True)),
    }
    for key, value in dict(debug_arrays or {}).items():
        arrays["debug__" + str(key)] = np.asarray(value)
    np.savez_compressed(output_npz, **arrays)


def load_result(path: Path | str) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    with np.load(Path(path), allow_pickle=False) as data:
        label_map = np.asarray(data["label_map"], dtype=np.int32)
        metadata = json.loads(str(data["metadata_json"]))
        debug = {
            str(name).removeprefix("debug__"): np.asarray(data[name]).copy()
            for name in data.files
            if str(name).startswith("debug__")
        }
    return label_map, dict(metadata), debug


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value
