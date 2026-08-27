from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Mapping


MODEL_ARCHITECTURE_VERSION = "voxroom_door_seed_column_transformer_v2_patch19_context41_wide"
PREPROCESSOR_VERSION = "unknown_free_occupied_absolute_height_v1"


@dataclass(frozen=True)
class DoorSeedModelConfig:
    z_count: int
    local_patch_size: int = 19
    context_patch_size: int = 41
    input_channels: int = 4
    context_channels: int = 3
    column_channels: int = 40
    transformer_layers: int = 2
    transformer_heads: int = 4
    transformer_feedforward: int = 120
    transformer_dropout: float = 0.1
    group_norm_groups: int = 8
    classifier_dropout: float = 0.2
    use_context_branch: bool = True
    architecture_version: str = MODEL_ARCHITECTURE_VERSION

    def validate(self) -> "DoorSeedModelConfig":
        if int(self.z_count) <= 0:
            raise ValueError("model z_count must be positive")
        if int(self.local_patch_size) != 19 or int(self.context_patch_size) != 41:
            raise ValueError("model schema v2 requires local 19x19 and context 41x41")
        if int(self.input_channels) != 4 or int(self.context_channels) != 3:
            raise ValueError("model schema v2 requires four voxel and three context channels")
        if int(self.column_channels) % int(self.transformer_heads) != 0:
            raise ValueError("column_channels must be divisible by transformer_heads")
        if int(self.group_norm_groups) <= 0:
            raise ValueError("group_norm_groups must be positive")
        for channels in (self.column_channels, 40, 72, 112):
            if int(channels) % int(self.group_norm_groups) != 0:
                raise ValueError("all convolution channels must be divisible by group_norm_groups")
        if str(self.architecture_version) != MODEL_ARCHITECTURE_VERSION:
            raise ValueError("unsupported model architecture version")
        return self

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "DoorSeedModelConfig":
        fields = cls.__dataclass_fields__
        return cls(**{key: data[key] for key in data if key in fields}).validate()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class DoorSeedClassifier:
    """Lazy factory that keeps rules-only imports independent of torch."""

    def __new__(cls, config: DoorSeedModelConfig | Mapping[str, object]):
        return build_door_seed_model(config)


def build_door_seed_model(config: DoorSeedModelConfig | Mapping[str, object]):
    cfg = config if isinstance(config, DoorSeedModelConfig) else DoorSeedModelConfig.from_mapping(config)
    cfg.validate()
    model_class = _model_class()
    return model_class(cfg)


@lru_cache(maxsize=1)
def _model_class():
    import torch
    from torch import nn

    class Residual1d(nn.Module):
        def __init__(self, channels: int, groups: int):
            super().__init__()
            self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
            self.norm = nn.GroupNorm(groups, channels)
            self.activation = nn.GELU()

        def forward(self, value):
            return value + self.activation(self.norm(self.conv(value)))

    class Residual2d(nn.Module):
        def __init__(self, channels: int, groups: int):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.norm1 = nn.GroupNorm(groups, channels)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.norm2 = nn.GroupNorm(groups, channels)
            self.activation = nn.GELU()

        def forward(self, value):
            residual = value
            value = self.activation(self.norm1(self.conv1(value)))
            value = self.norm2(self.conv2(value))
            return self.activation(value + residual)

    class Model(nn.Module):
        def __init__(self, model_config: DoorSeedModelConfig):
            super().__init__()
            self.model_config = model_config
            groups = int(model_config.group_norm_groups)
            channels = int(model_config.column_channels)
            self.column_stem = nn.Sequential(
                nn.Conv1d(4, channels, kernel_size=5, padding=2),
                nn.GroupNorm(groups, channels),
                nn.GELU(),
                Residual1d(channels, groups),
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=channels,
                nhead=int(model_config.transformer_heads),
                dim_feedforward=int(model_config.transformer_feedforward),
                dropout=float(model_config.transformer_dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.column_transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(model_config.transformer_layers))
            self.local_xy = nn.Sequential(
                nn.Conv2d(2 * channels, 72, kernel_size=3, padding=1),
                nn.GroupNorm(groups, 72),
                nn.GELU(),
                Residual2d(72, groups),
                nn.Conv2d(72, 112, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(groups, 112),
                nn.GELU(),
                Residual2d(112, groups),
            )
            if bool(model_config.use_context_branch):
                self.context_xy = nn.Sequential(
                    nn.Conv2d(3, 40, kernel_size=3, padding=1),
                    nn.GroupNorm(groups, 40),
                    nn.GELU(),
                    Residual2d(40, groups),
                    nn.Conv2d(40, 72, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(groups, 72),
                    nn.GELU(),
                    Residual2d(72, groups),
                    nn.Conv2d(72, 112, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(groups, 112),
                    nn.GELU(),
                    Residual2d(112, groups),
                )
            else:
                self.context_xy = None
            fusion_size = 448 if bool(model_config.use_context_branch) else 224
            self.classifier = nn.Sequential(
                nn.Linear(fusion_size, 160),
                nn.GELU(),
                nn.Dropout(float(model_config.classifier_dropout)),
                nn.Linear(160, 40),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(40, 1),
            )

        def forward(self, voxel, context=None):
            if voxel.ndim != 5:
                raise ValueError("voxel input must have shape [B,4,Z,H,W]")
            batch, channels, z_count, height, width = voxel.shape
            expected_local = int(self.model_config.local_patch_size)
            if channels != 4 or z_count != int(self.model_config.z_count) or (height, width) != (expected_local, expected_local):
                raise ValueError("voxel input shape does not match model config")
            columns = voxel.permute(0, 3, 4, 1, 2).reshape(batch * height * width, channels, z_count)
            columns = self.column_stem(columns).transpose(1, 2)
            columns = self.column_transformer(columns)
            pooled = torch.cat((columns.mean(dim=1), columns.amax(dim=1)), dim=1)
            local_map = pooled.reshape(batch, height, width, -1).permute(0, 3, 1, 2)
            local_feature = self._global_mean_max(self.local_xy(local_map))
            if self.context_xy is None:
                fused = local_feature
            else:
                expected_context = int(self.model_config.context_patch_size)
                if context is None or context.ndim != 4 or tuple(context.shape[1:]) != (3, expected_context, expected_context):
                    raise ValueError(
                        "context input must have shape [B,3,%d,%d]"
                        % (expected_context, expected_context)
                    )
                context_feature = self._global_mean_max(self.context_xy(context))
                fused = torch.cat((local_feature, context_feature), dim=1)
            return self.classifier(fused)

        @staticmethod
        def _global_mean_max(value):
            return torch.cat((value.mean(dim=(-2, -1)), value.amax(dim=(-2, -1))), dim=1)

    return Model
