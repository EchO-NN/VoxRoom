from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from voxroom_online.isaac_runtime.dataset.category_normalizer import normalize_category
from voxroom_online.isaac_runtime.graph.edge_builder import LLMClient
from voxroom_online.isaac_runtime.mapping.room_segmentation import RoomMask, ObjectRoomAssignment


DEFAULT_ROOM_CATEGORIES = [
    "kitchen",
    "bedroom",
    "bathroom",
    "living_room",
    "dining_room",
    "office",
    "hallway",
    "corridor",
    "laundry_room",
    "storage_room",
    "entryway",
    "balcony",
    "unknown",
]


DIAGNOSTIC_OBJECTS = {
    "bathroom": {"toilet", "sink", "bathtub", "shower", "bathroom sink", "mirror"},
    "bedroom": {"bed", "nightstand", "wardrobe", "dresser", "closet"},
    "kitchen": {"stove", "oven", "refrigerator", "microwave", "sink", "counter"},
    "living_room": {"sofa", "couch", "tv", "coffee table", "lamp", "armchair"},
    "dining_room": {"dining table", "table", "chair", "chandelier"},
    "office": {"desk", "office chair", "computer", "monitor", "bookshelf"},
    "laundry_room": {"washing machine", "dryer", "laundry basket"},
}

CONTRADICTORY_PAIRS = [
    ({"bed", "nightstand"}, {"stove", "oven", "refrigerator"}),
    ({"toilet", "bathtub", "shower"}, {"stove", "oven"}),
]


@dataclass
class ObjectEvidence:
    category: str
    count: int = 1
    mean_confidence: float = 1.0
    hits: int = 1
    object_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "count": int(self.count),
            "mean_confidence": float(self.mean_confidence),
            "hits": int(self.hits),
            "object_ids": list(self.object_ids),
        }


@dataclass
class RoomVisualEvidence:
    has_mask_thumbnail: bool = False
    rgb_view_count: int = 0
    strong_visual_evidence: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "has_mask_thumbnail": bool(self.has_mask_thumbnail),
            "rgb_view_count": int(self.rgb_view_count),
            "strong_visual_evidence": bool(self.strong_visual_evidence),
            "notes": list(self.notes),
        }


@dataclass
class RoomSemanticLabel:
    room_id: str
    category: str
    confidence: float
    supporting_objects: List[str]
    conflicting_evidence: List[str]
    rationale: str
    backend: str
    unknown_reason: Optional[str] = None
    vlm_self_confidence: float = 0.0
    label_reliability: float = 0.0
    reliability_factors: Dict[str, object] = field(default_factory=dict)

    @property
    def is_reliable(self) -> bool:
        return bool(self.category != "unknown" and float(self.label_reliability) >= 0.65)

    def to_dict(self) -> dict:
        return {
            "room_id": self.room_id,
            "category": self.category,
            "confidence": float(self.confidence),
            "vlm_self_confidence": float(self.vlm_self_confidence),
            "label_reliability": float(self.label_reliability),
            "is_reliable": bool(self.is_reliable),
            "reliability_factors": dict(self.reliability_factors),
            "supporting_objects": list(self.supporting_objects),
            "conflicting_evidence": list(self.conflicting_evidence),
            "rationale": self.rationale,
            "backend": self.backend,
            "unknown_reason": self.unknown_reason,
        }


class VLMRoomLabeler:
    def __init__(
        self,
        client: Optional[LLMClient] = None,
        allowed_categories: Optional[Sequence[str]] = None,
        min_confidence: float = 0.60,
        ambiguity_margin: float = 0.15,
        min_reliable_objects: int = 2,
        unknown_category: str = "unknown",
        require_backend: bool = False,
        max_room_objects_in_prompt: int = 25,
    ):
        self.client = client
        self.allowed_categories = _normalize_allowed_categories(allowed_categories or DEFAULT_ROOM_CATEGORIES, unknown_category)
        self.min_confidence = float(min_confidence)
        self.ambiguity_margin = float(ambiguity_margin)
        self.min_reliable_objects = max(0, int(min_reliable_objects))
        self.unknown_category = normalize_room_category(unknown_category)
        self.require_backend = bool(require_backend)
        self.max_room_objects_in_prompt = max(1, int(max_room_objects_in_prompt))
        self.request_count = 0
        self.failure_count = 0
        self.last_error: Optional[str] = None
        self.backend = "vlm" if client is not None else "deterministic_debug"

    def label_room(
        self,
        room_mask: RoomMask,
        object_evidence: Sequence[ObjectEvidence],
        visual_evidence: Optional[RoomVisualEvidence] = None,
    ) -> RoomSemanticLabel:
        evidence = list(object_evidence)[: self.max_room_objects_in_prompt]
        if self.client is None:
            if self.require_backend:
                self.last_error = "room VLM backend unavailable"
                raise RuntimeError("strict benchmark requires an enabled room VLM backend")
            return self._postprocess(
                self._deterministic_debug_label(room_mask, evidence, visual_evidence),
                room_mask,
                evidence,
                visual_evidence,
                backend="deterministic_debug",
            )
        prompt = build_room_label_prompt(room_mask, evidence, visual_evidence, self.allowed_categories)
        self.request_count += 1
        try:
            parsed = _parse_json_object(self.client.complete_json(prompt))
        except Exception as exc:
            self.failure_count += 1
            self.last_error = str(exc)
            if self.require_backend:
                raise RuntimeError("strict room VLM labeling failed: %s" % exc) from exc
            return RoomSemanticLabel(
                room_id=room_mask.room_id,
                category=self.unknown_category,
                confidence=0.0,
                supporting_objects=[],
                conflicting_evidence=["room_vlm_invalid_json"],
                rationale="Room VLM failed; non-metric debug fallback.",
                backend="unavailable",
                unknown_reason="room_vlm_invalid_json",
                vlm_self_confidence=0.0,
                label_reliability=0.0,
                reliability_factors={"backend_available": False},
            )
        return self._postprocess(parsed, room_mask, evidence, visual_evidence, backend="vlm")

    def _deterministic_debug_label(
        self,
        room_mask: RoomMask,
        object_evidence: Sequence[ObjectEvidence],
        visual_evidence: Optional[RoomVisualEvidence],
    ) -> dict:
        _ = visual_evidence
        cats = _expanded_object_categories(object_evidence)
        best_category = self.unknown_category
        best_hits = 0
        support: List[str] = []
        for room_cat, diagnostic in DIAGNOSTIC_OBJECTS.items():
            if room_cat not in self.allowed_categories:
                continue
            hits = len(cats & {normalize_category(item).replace("_", " ") for item in diagnostic})
            if hits > best_hits:
                best_hits = hits
                best_category = room_cat
                support = sorted(cats & {normalize_category(item).replace("_", " ") for item in diagnostic})
        confidence = min(0.95, 0.45 + 0.18 * best_hits + 0.05 * len(object_evidence))
        if best_hits <= 0:
            best_category = self.unknown_category
            confidence = 0.35
        return {
            "room_id": room_mask.room_id,
            "category": best_category,
            "confidence": confidence,
            "ranked_categories": [
                {"category": best_category, "confidence": confidence},
                {"category": self.unknown_category, "confidence": max(0.0, 1.0 - confidence)},
            ],
            "supporting_objects": support,
            "conflicting_evidence": [],
            "rationale": "Deterministic debug room label from diagnostic objects.",
            "unknown_reason": None if best_category != self.unknown_category else "insufficient_or_ambiguous_evidence",
        }

    def _postprocess(
        self,
        parsed: Mapping[str, object],
        room_mask: RoomMask,
        object_evidence: Sequence[ObjectEvidence],
        visual_evidence: Optional[RoomVisualEvidence],
        backend: str,
    ) -> RoomSemanticLabel:
        category = normalize_room_category(parsed.get("category", self.unknown_category))
        vlm_self_confidence = _safe_float(parsed.get("confidence", parsed.get("vlm_self_confidence", 0.0)), 0.0)
        supporting = _string_list(parsed.get("supporting_objects", []))
        conflicting = _string_list(parsed.get("conflicting_evidence", []))
        rationale = str(parsed.get("rationale", parsed.get("reason", "")))
        unknown_reason = parsed.get("unknown_reason")
        unknown_reason = None if unknown_reason in {None, "", "null"} else str(unknown_reason)

        reliability, reliability_factors = self._label_reliability(
            category,
            vlm_self_confidence,
            object_evidence,
            visual_evidence,
            room_mask,
            parsed,
        )
        quality_warning = self._label_quality_warning_reason(
            category,
            vlm_self_confidence,
            reliability_factors,
            parsed,
        )
        if quality_warning == "invalid_category":
            category = self.unknown_category
            reliability = 0.0
            if quality_warning not in conflicting:
                conflicting.append(quality_warning)
            unknown_reason = quality_warning
        elif quality_warning is not None:
            if quality_warning not in conflicting:
                conflicting.append(quality_warning)
            reliability_factors["label_quality_warning"] = quality_warning
            if not rationale:
                rationale = "Room type kept from model output; reliability records weak or ambiguous evidence."
        if category == self.unknown_category and not unknown_reason:
            unknown_reason = "insufficient_or_ambiguous_evidence"
        if category not in self.allowed_categories:
            category = self.unknown_category
            reliability = 0.0
            unknown_reason = "invalid_category"
            if "invalid_category" not in conflicting:
                conflicting.append("invalid_category")
            if not rationale:
                rationale = "Observed room evidence is insufficient or ambiguous."
        return RoomSemanticLabel(
            room_id=str(parsed.get("room_id", room_mask.room_id)),
            category=category,
            confidence=float(np.clip(reliability, 0.0, 1.0)),
            supporting_objects=supporting,
            conflicting_evidence=conflicting,
            rationale=rationale,
            backend=backend,
            unknown_reason=unknown_reason,
            vlm_self_confidence=float(np.clip(vlm_self_confidence, 0.0, 1.0)),
            label_reliability=float(np.clip(reliability, 0.0, 1.0)),
            reliability_factors=reliability_factors,
        )

    def _label_reliability(
        self,
        category: str,
        vlm_self_confidence: float,
        object_evidence: Sequence[ObjectEvidence],
        visual_evidence: Optional[RoomVisualEvidence],
        room_mask: RoomMask,
        parsed: Mapping[str, object],
    ) -> tuple[float, Dict[str, object]]:
        cats = _expanded_object_categories(object_evidence)
        reliable_objects = sum(int(ev.count) for ev in object_evidence if float(ev.mean_confidence) >= 0.45 or int(ev.hits) >= 2)
        strong_visual = bool(visual_evidence and visual_evidence.strong_visual_evidence)
        diagnostic_hits = _diagnostic_hits_for_category(category, object_evidence)
        contradictory = _contradictory_evidence(cats)
        ambiguity_margin_passed = _ambiguity_margin_passed(parsed, self.ambiguity_margin)
        mask_confidence = float(np.clip(float(getattr(room_mask, "mask_confidence", room_mask.confidence)), 0.0, 1.0))
        boundary_unknown = float(np.clip(float(room_mask.boundary_unknown_fraction), 0.0, 1.0))
        evidence_score = min(1.0, 0.25 * float(reliable_objects) + 0.22 * float(diagnostic_hits))
        if strong_visual:
            evidence_score = max(evidence_score, 0.70)
        geometry_score = float(mask_confidence) * (1.0 - 0.35 * boundary_unknown)
        if room_mask.is_partial and diagnostic_hits < 2 and not strong_visual:
            geometry_score *= 0.55
        reliability = 0.15 * float(np.clip(vlm_self_confidence, 0.0, 1.0)) + 0.60 * evidence_score + 0.25 * geometry_score
        if not ambiguity_margin_passed or contradictory:
            reliability *= 0.35
        factors: Dict[str, object] = {
            "vlm_self_confidence": float(np.clip(vlm_self_confidence, 0.0, 1.0)),
            "diagnostic_object_hits": int(diagnostic_hits),
            "reliable_object_count": int(reliable_objects),
            "has_strong_visual_evidence": bool(strong_visual),
            "is_partial_room": bool(room_mask.is_partial),
            "boundary_unknown_fraction": float(boundary_unknown),
            "ambiguity_margin_passed": bool(ambiguity_margin_passed),
            "mask_confidence": float(mask_confidence),
            "contradictory_evidence": bool(contradictory),
        }
        return float(np.clip(reliability, 0.0, 1.0)), factors

    def _label_quality_warning_reason(
        self,
        category: str,
        vlm_self_confidence: float,
        reliability_factors: Mapping[str, object],
        parsed: Mapping[str, object],
    ) -> Optional[str]:
        if category not in self.allowed_categories:
            return "invalid_category"
        if category == self.unknown_category:
            return None
        if vlm_self_confidence < self.min_confidence:
            return "low_confidence"
        reliable_objects = int(reliability_factors.get("reliable_object_count", 0) or 0)
        diagnostic_hits = int(reliability_factors.get("diagnostic_object_hits", 0) or 0)
        strong_visual = bool(reliability_factors.get("has_strong_visual_evidence", False))
        if reliable_objects < self.min_reliable_objects and diagnostic_hits < 2 and not strong_visual:
            return "insufficient_or_ambiguous_evidence"
        if not bool(reliability_factors.get("ambiguity_margin_passed", True)):
            return "ambiguous_ranked_categories"
        if diagnostic_hits <= 0 and not strong_visual:
            return "no_diagnostic_evidence"
        if bool(reliability_factors.get("is_partial_room", False)) and diagnostic_hits < 2 and not strong_visual:
            return "partial_weak_evidence"
        if bool(reliability_factors.get("contradictory_evidence", False)):
            return "contradictory_evidence"
        return None


def build_room_label_prompt(
    room_mask: RoomMask,
    object_evidence: Sequence[ObjectEvidence],
    visual_evidence: Optional[RoomVisualEvidence],
    allowed_categories: Sequence[str],
) -> str:
    payload = {
        "room_id": room_mask.room_id,
        "allowed_categories": list(allowed_categories),
        "must_use_unknown_when_insufficient": False,
        "geometry": {
            "area_m2": float(room_mask.area_m2),
            "boundary_unknown_fraction": float(room_mask.boundary_unknown_fraction),
            "is_partial": bool(room_mask.is_partial),
            "doorway_count": len(room_mask.doorway_edges),
            "source": room_mask.source,
        },
        "objects_inside": [item.to_dict() for item in object_evidence],
        "visual_evidence": None if visual_evidence is None else visual_evidence.to_dict(),
    }
    return (
        "You are assigning a semantic type to an online-discovered room region for a robot navigation scene graph.\n"
        "Use only the provided observed objects, room geometry summary, and optional images.\n"
        "Choose exactly one category from allowed_categories.\n"
        "Prefer the best supported room category when there is any useful evidence; use unknown only when no category can be justified.\n"
        "When evidence is weak, partial, ambiguous, contradictory, or not diagnostic, keep the best category if possible but report low confidence and conflicting_evidence.\n"
        "Do not guess a room type just because one category is common.\n"
        "Do not use any dataset priors or ground-truth room labels.\n"
        "Return strict JSON only with schema:\n"
        "{\"room_id\":\"room_0001\",\"category\":\"unknown\",\"confidence\":0.0,"
        "\"ranked_categories\":[{\"category\":\"unknown\",\"confidence\":0.0}],"
        "\"supporting_objects\":[],\"conflicting_evidence\":[],\"rationale\":\"...\",\"unknown_reason\":\"...\"}\n"
        "Treat confidence as your own self-reported ordinal confidence, not a calibrated probability.\n"
        "Evidence:\n%s" % json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )


def summarize_room_object_evidence(
    object_nodes: Iterable[object],
    assignments: Mapping[str, ObjectRoomAssignment],
    room_id: str,
) -> List[ObjectEvidence]:
    buckets: Dict[str, dict] = {}
    for obj in object_nodes:
        object_id = str(getattr(obj, "id", getattr(obj, "node_id", "")))
        assignment = assignments.get(object_id)
        if assignment is None or assignment.room_id != room_id:
            continue
        category = normalize_category(str(getattr(obj, "category", "unknown"))).replace("_", " ")
        bucket = buckets.setdefault(category, {"count": 0, "conf_sum": 0.0, "hits": 0, "ids": []})
        bucket["count"] += 1
        bucket["conf_sum"] += float(getattr(obj, "confidence", 0.0))
        bucket["hits"] += int(getattr(obj, "observed_count", 1))
        bucket["ids"].append(object_id)
    out = []
    for category, bucket in sorted(buckets.items()):
        count = max(1, int(bucket["count"]))
        out.append(
            ObjectEvidence(
                category=category,
                count=count,
                mean_confidence=float(bucket["conf_sum"]) / float(count),
                hits=int(bucket["hits"]),
                object_ids=list(bucket["ids"]),
            )
        )
    return out


def room_semantics_debug(labels: Mapping[str, RoomSemanticLabel], allowed_categories: Sequence[str], backend: str) -> dict:
    return {
        "backend": backend,
        "allowed_categories": list(allowed_categories),
        "labels": [label.to_dict() for _room_id, label in sorted(labels.items())],
    }


def normalize_room_category(value: object) -> str:
    text = normalize_category(str(value)).replace(" ", "_")
    aliases = {
        "living room": "living_room",
        "livingroom": "living_room",
        "dining room": "dining_room",
        "office_room": "office",
        "unknown_room": "unknown",
    }
    return aliases.get(text, text)


def _normalize_allowed_categories(values: Sequence[str], unknown_category: str) -> List[str]:
    out = []
    for value in values:
        cat = normalize_room_category(value)
        if cat not in out:
            out.append(cat)
    unknown = normalize_room_category(unknown_category)
    if unknown not in out:
        out.append(unknown)
    return out


def _parse_json_object(value: object) -> dict:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        parsed = json.loads(text)
        if isinstance(parsed, Mapping):
            return dict(parsed)
    raise ValueError("room VLM response must be a JSON object")


def _string_list(value: object) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _safe_float(value: object, default: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(parsed):
        return float(default)
    return float(parsed)


def _expanded_object_categories(object_evidence: Sequence[ObjectEvidence]) -> set[str]:
    out = set()
    for item in object_evidence:
        cat = normalize_category(item.category).replace("_", " ")
        out.add(cat)
        out.add(cat.replace(" ", "_"))
    return out


def _contradictory_evidence(cats: set[str]) -> bool:
    flat = {cat.replace("_", " ") for cat in cats}
    for left, right in CONTRADICTORY_PAIRS:
        left_norm = {normalize_category(item).replace("_", " ") for item in left}
        right_norm = {normalize_category(item).replace("_", " ") for item in right}
        if flat & left_norm and flat & right_norm:
            return True
    return False


def _diagnostic_hits_for_category(category: str, object_evidence: Sequence[ObjectEvidence]) -> int:
    diagnostic = DIAGNOSTIC_OBJECTS.get(normalize_room_category(category), set())
    diagnostic_norm = {normalize_category(item).replace("_", " ") for item in diagnostic}
    cats = {normalize_category(item.category).replace("_", " ") for item in object_evidence}
    return len(cats & diagnostic_norm)


def _ambiguity_margin_passed(parsed: Mapping[str, object], margin: float) -> bool:
    ranked = parsed.get("ranked_categories", parsed.get("alternatives", []))
    if not isinstance(ranked, list) or len(ranked) < 2:
        return True
    scores = []
    for item in ranked[:2]:
        if isinstance(item, Mapping):
            scores.append(_safe_float(item.get("confidence", item.get("score", 0.0)), 0.0))
        else:
            try:
                scores.append(float(item))
            except Exception:
                pass
    if len(scores) < 2:
        return True
    return abs(scores[0] - scores[1]) >= float(margin)
