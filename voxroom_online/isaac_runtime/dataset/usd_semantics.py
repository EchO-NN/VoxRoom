from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from voxroom_online.isaac_runtime.dataset.category_normalizer import normalize_category
from voxroom_online.isaac_runtime.mapping.room_map_from_rooms_json import RoomGT, assign_object_room

STRUCTURAL_CATEGORIES = {
    "wall",
    "floor",
    "ceiling",
    "roof",
    "door",
    "window",
    "baseboard",
    "light_switch",
    "outlet",
}


@dataclass
class ObjectGT:
    instance_id: str
    prim_path: str
    raw_label: str
    category: str
    labels: List[str]
    center_world: Tuple[float, float, float]
    bbox_min_world: Tuple[float, float, float]
    bbox_max_world: Tuple[float, float, float]
    extent_world: Tuple[float, float, float]
    room_id: Optional[str] = None
    room_type: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _import_pxr():
    try:
        from pxr import Usd, UsdGeom  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "pxr.Usd is unavailable. Run preprocessing through Isaac Kit Python "
            "or use voxroom_online.isaac_runtime.scripts.preprocess_interioragent, which re-execs "
            "with Isaac's USD library paths."
        ) from exc
    try:
        from pxr import UsdSemantics  # type: ignore
    except Exception:
        UsdSemantics = None
    return Usd, UsdGeom, UsdSemantics


def collect_semantic_labels(prim) -> List[str]:
    labels: List[str] = []
    _, _, UsdSemantics = _import_pxr()
    if UsdSemantics is not None:
        try:
            taxonomies = UsdSemantics.LabelsAPI.GetDirectTaxonomies(prim)
            for taxonomy in taxonomies:
                api = UsdSemantics.LabelsAPI(prim, taxonomy)
                attr = api.GetLabelsAttr()
                if attr and attr.HasValue():
                    labels.extend([str(x) for x in attr.Get()])
        except Exception:
            pass
    for attr in prim.GetAttributes():
        name = attr.GetName()
        if name.startswith("semantics:labels") or "semantic" in name.lower():
            try:
                value = attr.Get()
            except Exception:
                value = None
            if isinstance(value, (list, tuple)):
                labels.extend([str(x) for x in value])
            elif value is not None:
                labels.append(str(value))
    return sorted(set(x for x in labels if x))


def fallback_label_from_prim_name(prim) -> str:
    return normalize_category(prim.GetName())


def prim_has_reference(prim) -> bool:
    try:
        return bool(prim.HasAuthoredReferences() or prim.GetMetadata("references"))
    except Exception:
        return False


def compute_world_bbox(prim) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]]:
    Usd, UsdGeom, _ = _import_pxr()
    try:
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=True,
        )
        bound = bbox_cache.ComputeWorldBound(prim)
        box = bound.ComputeAlignedBox()
        mn = tuple(float(box.GetMin()[i]) for i in range(3))
        mx = tuple(float(box.GetMax()[i]) for i in range(3))
    except Exception:
        return None
    if not all(math.isfinite(v) for v in mn + mx):
        return None
    extent = tuple(max(0.0, mx[i] - mn[i]) for i in range(3))
    if max(extent) <= 0.0:
        return None
    center = tuple((mn[i] + mx[i]) * 0.5 for i in range(3))
    return mn, mx, center, extent


def should_consider_prim(prim) -> bool:
    if not prim.IsActive():
        return False
    type_name = prim.GetTypeName()
    if type_name not in ("Xform", "Scope", "Mesh"):
        return False
    path = str(prim.GetPath())
    if "/Materials" in path:
        return False
    labels = collect_semantic_labels(prim)
    return bool(labels or prim_has_reference(prim))


def extract_objects_from_usd(
    usd_path: str,
    rooms: Optional[Iterable[RoomGT]] = None,
    min_bbox_diag_m: float = 0.05,
    include_structural: bool = True,
) -> List[ObjectGT]:
    Usd, _, _ = _import_pxr()
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError("Failed to open USD stage: %s" % usd_path)
    room_list = list(rooms or [])
    objects: List[ObjectGT] = []
    skipped_descendants: List[str] = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if any(path.startswith(prefix + "/") for prefix in skipped_descendants):
            continue
        if not should_consider_prim(prim):
            continue
        labels = collect_semantic_labels(prim)
        raw_label = labels[0] if labels else fallback_label_from_prim_name(prim)
        category = normalize_category(raw_label)
        if (not include_structural) and category in STRUCTURAL_CATEGORIES:
            continue
        bbox = compute_world_bbox(prim)
        if bbox is None:
            continue
        mn, mx, center, extent = bbox
        diag = math.sqrt(sum(v * v for v in extent))
        if diag < min_bbox_diag_m:
            continue
        room_id, room_type = assign_object_room(center, room_list) if room_list else (None, None)
        obj = ObjectGT(
            instance_id=path,
            prim_path=path,
            raw_label=str(raw_label),
            category=category,
            labels=labels or [str(raw_label)],
            center_world=center,
            bbox_min_world=mn,
            bbox_max_world=mx,
            extent_world=extent,
            room_id=room_id,
            room_type=room_type,
        )
        objects.append(obj)
        if prim.GetTypeName() == "Xform" and (labels or prim_has_reference(prim)):
            skipped_descendants.append(path)
    return objects


def write_objects_json(objects: Sequence[ObjectGT], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump([obj.to_dict() for obj in objects], handle, ensure_ascii=False, indent=2)

