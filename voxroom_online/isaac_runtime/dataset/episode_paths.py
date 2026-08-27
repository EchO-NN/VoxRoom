from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


def resolve_migrated_episode_path(
    raw_path: str | os.PathLike[str],
    *,
    repository_root: Path,
    home_dir: Path | None = None,
) -> Path:
    """Resolve an absolute episode asset path after moving a checkout.

    A working recorded path is preserved.  Deterministic repository and user
    home rebases are attempted only for a missing absolute path, and a rebase
    is accepted only when its target exists.
    """

    recorded = Path(raw_path).expanduser()
    if recorded.exists() or not recorded.is_absolute():
        return recorded

    raw = str(recorded)
    candidates: list[Path] = []
    repository_marker = f"{os.sep}VoxRoom{os.sep}"
    if repository_marker in raw:
        repository_relative = raw.split(repository_marker, 1)[1]
        candidates.append(Path(repository_root) / repository_relative)

    parts = recorded.parts
    if len(parts) >= 4 and parts[1] == "home":
        candidates.append((home_dir or Path.home()).joinpath(*parts[3:]))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return recorded


def resolve_migrated_episode_paths(
    episode: Mapping[str, object],
    *,
    repository_root: Path,
    home_dir: Path | None = None,
) -> dict[str, object]:
    updated = dict(episode)
    for key in ("usd_path", "rooms_json_path", "preprocessed_scene_dir"):
        value = updated.get(key)
        if isinstance(value, str) and value:
            updated[key] = str(
                resolve_migrated_episode_path(
                    value,
                    repository_root=repository_root,
                    home_dir=home_dir,
                )
            )
    return updated
