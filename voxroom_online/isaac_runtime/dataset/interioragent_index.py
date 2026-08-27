from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class InteriorAgentScene:
    scene_id: str
    scene_dir: str
    usd_path: str
    rooms_json_path: str

    def to_dict(self) -> dict:
        return asdict(self)


def find_top_level_usd(scene_dir: Path) -> Optional[Path]:
    scene_id = scene_dir.name
    preferred = list(scene_dir.glob(f"{scene_id}.usd*"))
    if preferred:
        return sorted(preferred)[0]
    candidates = [p for p in scene_dir.glob("*.usd*") if p.is_file()]
    return sorted(candidates)[0] if candidates else None


def discover_scenes(dataset_root: str, scene_glob: str = "kujiale_*", scene_ids: Optional[Iterable[str]] = None) -> List[InteriorAgentScene]:
    root = Path(dataset_root).expanduser().resolve()
    requested = set(scene_ids or [])
    scenes: List[InteriorAgentScene] = []
    for scene_dir in sorted(root.glob(scene_glob)):
        if not scene_dir.is_dir():
            continue
        if requested and scene_dir.name not in requested:
            continue
        usd_path = find_top_level_usd(scene_dir)
        rooms_path = scene_dir / "rooms.json"
        if usd_path is None or not rooms_path.exists():
            continue
        scenes.append(
            InteriorAgentScene(
                scene_id=scene_dir.name,
                scene_dir=str(scene_dir),
                usd_path=str(usd_path),
                rooms_json_path=str(rooms_path),
            )
        )
    return scenes

