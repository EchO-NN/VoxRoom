from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


def make_jsonable(value):
    try:
        import numpy as np
    except Exception:
        np = None
    if np is not None:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return make_jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): make_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_jsonable(item) for item in value]
    return value


class JsonlEpisodeLogger:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, row: Mapping) -> None:
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(make_jsonable(dict(row)), ensure_ascii=False) + "\n")

