from __future__ import annotations

import pytest

from voxroom_online.isaac_runtime.door_seed_learning.config import (
    DoorSeedLearningConfig,
)
from voxroom_online.isaac_runtime.door_seed_learning.model import (
    DoorSeedModelConfig,
    build_door_seed_model,
)


def test_ablation_raw_seed_sources_and_rules_only_mode_validate() -> None:
    for source in (
        "voxroom",
        "tvars_vertical",
        "voxroom_tvars_vertical_union",
    ):
        config = DoorSeedLearningConfig.from_mapping(
            {"mode": "rules_only", "raw_seed_source": source}
        )
        assert config.mode == "rules_only"
        assert config.raw_seed_source == source
    dense = DoorSeedLearningConfig.from_mapping(
        {
            "mode": "inference",
            "raw_seed_source": "vertical_free_all",
            "checkpoint_path": "/tmp/ablation.pt",
            "ablation_allow_checkpoint_mismatch": True,
        }
    )
    assert dense.ablation_allow_checkpoint_mismatch is True


def test_rules_only_rejects_dense_all_free_without_classifier() -> None:
    with pytest.raises(ValueError, match="vertical_free_all"):
        DoorSeedLearningConfig.from_mapping(
            {"mode": "rules_only", "raw_seed_source": "vertical_free_all"}
        )


@pytest.mark.parametrize(
    ("use_voxel", "use_context", "parameter_upper_bound"),
    ((True, True, 1_100_000), (True, False, 600_000), (False, True, 550_000)),
)
def test_model_branch_ablations_have_distinct_valid_forward_paths(
    use_voxel: bool,
    use_context: bool,
    parameter_upper_bound: int,
) -> None:
    torch = pytest.importorskip("torch")
    config = DoorSeedModelConfig(
        z_count=4,
        use_voxel_branch=use_voxel,
        use_context_branch=use_context,
    )
    model = build_door_seed_model(config)
    voxel = torch.zeros(2, 4, 4, 19, 19)
    context = torch.zeros(2, 3, 41, 41)
    output = model(voxel, context)
    assert tuple(output.shape) == (2, 1)
    assert sum(parameter.numel() for parameter in model.parameters()) < parameter_upper_bound


def test_model_rejects_disabling_both_branches() -> None:
    with pytest.raises(ValueError, match="at least one"):
        DoorSeedModelConfig(
            z_count=4,
            use_voxel_branch=False,
            use_context_branch=False,
        ).validate()
