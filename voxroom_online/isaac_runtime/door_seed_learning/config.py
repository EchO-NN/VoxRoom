from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


VALID_MODES = {"disabled", "collect", "inference"}
VALID_CONTEXT_SOURCES = {"vertical", "nav"}
VALID_RAW_SEED_SOURCES = {"voxroom", "voxroom_tvars_vertical_union"}


@dataclass(frozen=True)
class DoorSeedLearningConfig:
    mode: str = "disabled"
    collection_root: str = "outputs/door_seed_collection"
    force_final_snapshot: bool = True
    continue_exploration_after_object_found: bool = True
    collect_frontier_selection_mode: str = "random"
    raw_seed_source: str = "voxroom"
    collection_every_steps: int = 0
    tvars_seed_width_cells: int = 3
    persistent_final_raw_seed_union: bool = False
    save_full_voxel_milestones: bool = False
    full_voxel_coverage_milestones: str = "30,40,50,60,70,80,90"

    local_voxel_patch_size: int = 19
    context_patch_size: int = 41
    context_source: str = "vertical"
    checkpoint_path: str = ""
    device: str = "cuda:0"
    inference_batch_size: int = 256
    keep_threshold: float | None = None
    fallback_to_rule_seed_on_error: bool = False
    keep_uninformative_seed: bool = False
    uninformative_observed_ratio_min: float = 0.02

    height_scale_m: float = 4.0
    dedup_exact: bool = True
    dedup_change_ratio: float = 0.03
    dedup_unknown_drop: float = 0.05
    max_samples_per_scene_column: int = 8
    snapshot_cache_size: int = 6
    target_recall: float = 0.98
    max_pos_weight: float = 10.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "DoorSeedLearningConfig":
        if isinstance(data, cls):
            return data.validate()
        raw = dict(data or {})
        fields = cls.__dataclass_fields__
        cfg = cls(**{key: raw[key] for key in raw if key in fields})
        return cfg.validate()

    def validate(self) -> "DoorSeedLearningConfig":
        mode = str(self.mode).strip().lower()
        context_source = str(self.context_source).strip().lower()
        raw_seed_source = str(self.raw_seed_source).strip().lower()
        if mode not in VALID_MODES:
            raise ValueError("door_seed_learning.mode must be disabled, collect, or inference")
        if context_source not in VALID_CONTEXT_SOURCES:
            raise ValueError("door_seed_learning.context_source must be vertical or nav")
        if raw_seed_source not in VALID_RAW_SEED_SOURCES:
            raise ValueError(
                "door_seed_learning.raw_seed_source must be voxroom or voxroom_tvars_vertical_union"
            )
        for name, value in (
            ("local_voxel_patch_size", self.local_voxel_patch_size),
            ("context_patch_size", self.context_patch_size),
        ):
            if int(value) <= 0 or int(value) % 2 != 1:
                raise ValueError("door_seed_learning.%s must be a positive odd integer" % name)
        if int(self.local_voxel_patch_size) != 19:
            raise ValueError("door_seed_learning.local_voxel_patch_size must be 19 for schema v3")
        if int(self.context_patch_size) != 41:
            raise ValueError("door_seed_learning.context_patch_size must be 41 for schema v3")
        if mode == "collect" and not str(self.collection_root).strip():
            raise ValueError("door_seed_learning.collection_root is required in collect mode")
        if mode == "collect" and not bool(self.force_final_snapshot):
            raise ValueError("door_seed_learning.force_final_snapshot must be true in collect mode")
        if int(self.collection_every_steps) < 0:
            raise ValueError("door_seed_learning.collection_every_steps must be non-negative")
        if bool(self.save_full_voxel_milestones) and int(self.collection_every_steps) <= 0:
            raise ValueError(
                "door_seed_learning.save_full_voxel_milestones requires a positive collection_every_steps"
            )
        _parse_coverage_milestones(self.full_voxel_coverage_milestones)
        if int(self.tvars_seed_width_cells) <= 0 or int(self.tvars_seed_width_cells) % 2 != 1:
            raise ValueError("door_seed_learning.tvars_seed_width_cells must be a positive odd integer")
        if mode == "inference" and not str(self.checkpoint_path).strip():
            raise ValueError("door_seed_learning.checkpoint_path is required in inference mode")
        if self.keep_threshold is not None and not 0.0 <= float(self.keep_threshold) <= 1.0:
            raise ValueError("door_seed_learning.keep_threshold must be in [0, 1]")
        if int(self.inference_batch_size) <= 0:
            raise ValueError("door_seed_learning.inference_batch_size must be positive")
        if float(self.height_scale_m) <= 0.0:
            raise ValueError("door_seed_learning.height_scale_m must be positive")
        for name, value in (
            ("dedup_change_ratio", self.dedup_change_ratio),
            ("dedup_unknown_drop", self.dedup_unknown_drop),
            ("target_recall", self.target_recall),
            ("uninformative_observed_ratio_min", self.uninformative_observed_ratio_min),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError("door_seed_learning.%s must be in [0, 1]" % name)
        if int(self.max_samples_per_scene_column) < 2:
            raise ValueError("door_seed_learning.max_samples_per_scene_column must be at least 2")
        if int(self.snapshot_cache_size) <= 0:
            raise ValueError("door_seed_learning.snapshot_cache_size must be positive")
        if float(self.max_pos_weight) <= 0.0:
            raise ValueError("door_seed_learning.max_pos_weight must be positive")
        if str(self.collect_frontier_selection_mode).strip().lower() not in {"nearest", "random"}:
            raise ValueError("door_seed_learning.collect_frontier_selection_mode must be nearest or random")
        if (
            mode == self.mode
            and context_source == self.context_source
            and raw_seed_source == self.raw_seed_source
        ):
            return self
        values = asdict(self)
        values["mode"] = mode
        values["context_source"] = context_source
        values["raw_seed_source"] = raw_seed_source
        return type(self)(**values)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def parsed_full_voxel_coverage_milestones(self) -> tuple[float, ...]:
        return _parse_coverage_milestones(self.full_voxel_coverage_milestones)


def _parse_coverage_milestones(value: object) -> tuple[float, ...]:
    tokens = [token.strip() for token in str(value).split(",") if token.strip()]
    if not tokens:
        raise ValueError("door_seed_learning.full_voxel_coverage_milestones must not be empty")
    values = [float(token) for token in tokens]
    normalized = [item / 100.0 if item > 1.0 else item for item in values]
    if any(item <= 0.0 or item >= 1.0 for item in normalized):
        raise ValueError(
            "door_seed_learning.full_voxel_coverage_milestones must lie strictly between 0 and 100 percent"
        )
    if normalized != sorted(set(normalized)):
        raise ValueError(
            "door_seed_learning.full_voxel_coverage_milestones must be strictly increasing and unique"
        )
    return tuple(float(item) for item in normalized)
