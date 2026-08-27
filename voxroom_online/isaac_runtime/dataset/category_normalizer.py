from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

from voxroom_online.isaac_runtime.config import load_yaml, repo_root

DEFAULT_INSTANCE_PATTERNS = [
    r"[_\-. ]?\d+$",
    r"[_\-. ]?copy[_\-. ]?\d*$",
    r"[_\-. ]?instance[_\-. ]?\d*$",
    r"[_\-. ]?inst[_\-. ]?\d*$",
    r"[_\-. ]?[a-f0-9]{6,}$",
]


@lru_cache(maxsize=1)
def load_default_aliases() -> Dict[str, List[str]]:
    fallback = {
        "sofa": ["couch", "settee", "loveseat"],
        "tv": ["television", "monitor_tv", "tv_monitor"],
        "dining_table": ["dinner_table", "table_dining"],
        "chair": ["dining_chair", "armchair", "seat"],
        "toilet": ["closestool"],
        "sink": ["basin"],
        "cabinet": ["bathroom_cabinet", "kitchen_cabinet"],
    }
    path = repo_root() / "voxroom_online" / "isaac_runtime" / "configs" / "room_category_aliases.yaml"
    try:
        data = load_yaml(path)
        return dict(data.get("canonical", {}))
    except Exception:
        return fallback


def _basic_normalize(raw: object) -> str:
    if raw is None:
        return "unknown"
    text = unicodedata.normalize("NFKC", str(raw)).strip().lower()
    text = text.replace("/", "_").replace("-", "_").replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]+", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def strip_instance_suffix(text: str, patterns: Optional[Iterable[str]] = None) -> str:
    text = _basic_normalize(text)
    prev = None
    patterns = list(patterns or DEFAULT_INSTANCE_PATTERNS)
    while prev != text:
        prev = text
        for pattern in patterns:
            text = re.sub(pattern, "", text).strip("_")
    return text or "unknown"


def normalize_category(raw: object, aliases: Optional[Mapping[str, List[str]]] = None) -> str:
    text = strip_instance_suffix(_basic_normalize(raw))
    aliases = aliases if aliases is not None else load_default_aliases()
    for canonical, names in aliases.items():
        canonical_norm = strip_instance_suffix(canonical)
        all_names = [canonical] + list(names or [])
        if text in {strip_instance_suffix(name) for name in all_names}:
            return canonical_norm
    return text or "unknown"


def normalize_many(values: Iterable[object], aliases: Optional[Mapping[str, List[str]]] = None) -> List[str]:
    return [normalize_category(value, aliases=aliases) for value in values]
