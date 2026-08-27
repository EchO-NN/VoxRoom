from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from voxroom_online.isaac_runtime.door_seed_learning.dataset import (
    DoorSeedDataset,
    GroupedCoordinateSampler,
    normalize_right_angle_rotations,
)
from voxroom_online.isaac_runtime.door_seed_learning.metrics import (
    binary_classification_metrics,
    choose_threshold_for_recall,
    ranking_metrics,
    recall_rejection_table,
)
from voxroom_online.isaac_runtime.door_seed_learning.model import (
    MODEL_ARCHITECTURE_VERSION,
    PREPROCESSOR_VERSION,
    DoorSeedModelConfig,
    build_door_seed_model,
)
from voxroom_online.isaac_runtime.door_seed_learning.schema import (
    DATASET_SCHEMA_VERSION,
    config_hash,
    load_snapshot,
    scalar_value,
    sha256_bytes,
    source_tree_hash,
    write_json_atomic,
)


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    batch_size: int = 128
    max_epochs: int = 50
    early_stopping_patience: int = 8
    early_stopping_metric: str = "validation_score"
    early_stopping_min_delta: float = 0.0
    checkpoint_selection_mode: str = "operating_metric"
    threshold_selection_mode: str = "target_recall"
    fixed_keep_threshold: float = 0.5
    target_recall: float = 0.98
    max_pos_weight: float = 10.0
    num_workers: int = 0
    snapshot_cache_size: int = 0
    seed: int = 0
    device: str = "cuda:0"
    grouped_coordinate_sampling: bool = True
    train_rotation_degrees: tuple[int, ...] = (0, 90, 180, 270)
    train_mirror_lr_once: bool = False
    precision: str = "float32"

    def __post_init__(self) -> None:
        if int(self.batch_size) <= 0 or int(self.max_epochs) <= 0 or int(self.early_stopping_patience) <= 0:
            raise ValueError("training batch_size, max_epochs, and early_stopping_patience must be positive")
        if str(self.early_stopping_metric) not in {"validation_score", "train_loss"}:
            raise ValueError("early_stopping_metric must be validation_score or train_loss")
        if float(self.early_stopping_min_delta) < 0.0:
            raise ValueError("early_stopping_min_delta must be non-negative")
        if int(self.snapshot_cache_size) < 0:
            raise ValueError("training snapshot_cache_size must be non-negative")
        if str(self.checkpoint_selection_mode) not in {"operating_metric", "validation_loss"}:
            raise ValueError("checkpoint_selection_mode must be operating_metric or validation_loss")
        if str(self.threshold_selection_mode) not in {"target_recall", "fixed"}:
            raise ValueError("threshold_selection_mode must be target_recall or fixed")
        if not 0.0 <= float(self.fixed_keep_threshold) <= 1.0:
            raise ValueError("fixed_keep_threshold must be in [0,1]")
        if not 0.0 < float(self.target_recall) <= 1.0:
            raise ValueError("training target_recall must be in (0,1]")
        if str(self.precision) not in {"float32", "bfloat16"}:
            raise ValueError("training precision must be float32 or bfloat16")
        object.__setattr__(
            self,
            "train_rotation_degrees",
            normalize_right_angle_rotations(self.train_rotation_degrees),
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "TrainingConfig":
        raw = dict(data or {})
        fields = cls.__dataclass_fields__
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class EarlyStoppingTracker:
    metric: str
    patience: int
    min_delta: float = 0.0
    best_train_loss: float = float("inf")
    best_validation_score: tuple[float, float] = (-1.0, -1.0)
    epochs_without_improvement: int = 0

    def __post_init__(self) -> None:
        if self.metric not in {"validation_score", "train_loss"}:
            raise ValueError("unsupported early stopping metric: %s" % self.metric)
        if int(self.patience) <= 0:
            raise ValueError("early stopping patience must be positive")
        if float(self.min_delta) < 0.0:
            raise ValueError("early stopping min_delta must be non-negative")

    def update(self, *, train_loss: float, validation_score: tuple[float, float]) -> bool:
        if self.metric == "train_loss":
            improved = float(train_loss) < self.best_train_loss - float(self.min_delta)
            if improved:
                self.best_train_loss = float(train_loss)
        elif self.metric == "validation_score":
            self.best_train_loss = min(self.best_train_loss, float(train_loss))
            improved = tuple(validation_score) > self.best_validation_score
            if improved:
                self.best_validation_score = tuple(float(value) for value in validation_score)
        self.epochs_without_improvement = 0 if improved else self.epochs_without_improvement + 1
        return self.epochs_without_improvement >= int(self.patience)


def checkpoint_selection_score(
    *,
    negative_rejection_rate: float,
    rejected_seed_accuracy: float,
    pr_auc: float,
) -> tuple[float, float, float]:
    values = (
        float(negative_rejection_rate),
        float(rejected_seed_accuracy),
        float(pr_auc),
    )
    if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise ValueError("checkpoint selection metrics must be finite values in [0,1]")
    return values


def fixed_threshold_checkpoint_selection_score(
    *,
    accuracy: float,
    f1: float,
    pr_auc: float,
) -> tuple[float, float, float]:
    values = (float(accuracy), float(f1), float(pr_auc))
    if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise ValueError("checkpoint selection metrics must be finite values in [0,1]")
    return values


def validation_loss_checkpoint_selection_score(
    *,
    validation_loss: float,
    accuracy: float,
    f1: float,
) -> tuple[float, float, float]:
    loss = float(validation_loss)
    if not np.isfinite(loss) or loss < 0.0:
        raise ValueError("validation loss must be a finite non-negative value")
    metrics = (float(accuracy), float(f1))
    if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in metrics):
        raise ValueError("checkpoint selection metrics must be finite values in [0,1]")
    return (-loss, *metrics)


def select_operating_metrics(
    labels,
    probabilities,
    *,
    threshold_selection_mode: str,
    fixed_keep_threshold: float,
    target_recall: float,
) -> dict[str, float | int]:
    mode = str(threshold_selection_mode)
    if mode == "fixed":
        return binary_classification_metrics(labels, probabilities, float(fixed_keep_threshold))
    if mode == "target_recall":
        return choose_threshold_for_recall(labels, probabilities, target_recall=float(target_recall))
    raise ValueError("unsupported threshold_selection_mode: %s" % mode)


def train_classifier(
    *,
    index_path: str | Path,
    output_dir: str | Path,
    context_source: str,
    height_scale_m: float,
    model_config: DoorSeedModelConfig | Mapping[str, object] | None = None,
    training_config: TrainingConfig | Mapping[str, object] | None = None,
    source_root: str | Path | None = None,
) -> dict[str, object]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    train_cfg = training_config if isinstance(training_config, TrainingConfig) else TrainingConfig.from_mapping(training_config)
    torch.manual_seed(int(train_cfg.seed))
    np.random.seed(int(train_cfg.seed))
    index = Path(index_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_dataset = DoorSeedDataset(
        index,
        split="train",
        context_source=context_source,
        height_scale_m=float(height_scale_m),
        augment=False,
        rotation_degrees=train_cfg.train_rotation_degrees,
        mirror_lr_once=bool(train_cfg.train_mirror_lr_once),
        seed=int(train_cfg.seed),
    )
    val_dataset = DoorSeedDataset(
        index,
        split="val",
        context_source=context_source,
        height_scale_m=float(height_scale_m),
        augment=False,
        rotation_degrees=(0,),
        seed=int(train_cfg.seed),
    )
    train_snapshot_count = len({str(row["snapshot_path"]) for row in train_dataset.rows})
    val_snapshot_count = len({str(row["snapshot_path"]) for row in val_dataset.rows})
    requested_cache_size = int(train_cfg.snapshot_cache_size)
    train_dataset.cache_size = requested_cache_size or max(1, train_snapshot_count)
    val_dataset.cache_size = requested_cache_size or max(1, val_snapshot_count)
    if not len(train_dataset) or not len(val_dataset):
        raise ValueError("training requires non-empty scene-level train and val splits")
    geometry = _dataset_geometry(train_dataset.rows + val_dataset.rows)
    model_cfg = (
        model_config
        if isinstance(model_config, DoorSeedModelConfig)
        else DoorSeedModelConfig.from_mapping(model_config)
        if model_config is not None
        else DoorSeedModelConfig(z_count=int(geometry["z_count"]))
    )
    if int(model_cfg.z_count) != int(geometry["z_count"]):
        raise ValueError("model z_count does not match dataset")
    model = build_door_seed_model(model_cfg)
    device = torch.device(str(train_cfg.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but torch.cuda.is_available() is false")
    if str(train_cfg.precision) == "bfloat16" and device.type != "cuda":
        raise RuntimeError("bfloat16 training requires a CUDA device")
    torch.set_float32_matmul_precision("high")
    autocast_dtype = torch.bfloat16 if str(train_cfg.precision) == "bfloat16" else None
    model.to(device)
    train_sampler = (
        GroupedCoordinateSampler(
            train_dataset.rows,
            rotation_count=train_dataset.transform_count,
            seed=int(train_cfg.seed),
            shuffle=True,
        )
        if bool(train_cfg.grouped_coordinate_sampling)
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg.batch_size),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=int(train_cfg.num_workers),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(train_cfg.batch_size),
        shuffle=False,
        num_workers=int(train_cfg.num_workers),
    )
    train_samples_per_epoch = len(train_sampler) if train_sampler is not None else len(train_dataset)
    augmentation_manifest = {
        "mode": "deterministic_index_expansion",
        "base_train_samples": len(train_dataset.rows),
        "rotation_degrees": list(train_dataset.rotation_degrees),
        "rotation_count": int(train_dataset.rotation_count),
        "mirror_lr_once": bool(train_dataset.mirror_lr_once),
        "transform_count": int(train_dataset.transform_count),
        "expanded_train_samples": len(train_dataset),
        "grouped_coordinate_sampling": bool(train_cfg.grouped_coordinate_sampling),
        "samples_per_epoch": int(train_samples_per_epoch),
        "validation_rotation_degrees": list(val_dataset.rotation_degrees),
        "validation_samples": len(val_dataset),
    }
    write_json_atomic(output / "augmentation_manifest.json", augmentation_manifest)
    print(
        "[door-seed-train] augmentation=%s rotations=%s mirror_lr_once=%s base_train_samples=%d "
        "expanded_train_samples=%d samples_per_epoch=%d validation_samples=%d precision=%s"
        % (
            str(augmentation_manifest["mode"]),
            ",".join(str(value) for value in train_dataset.rotation_degrees),
            str(bool(train_dataset.mirror_lr_once)).lower(),
            len(train_dataset.rows),
            len(train_dataset),
            int(train_samples_per_epoch),
            len(val_dataset),
            str(train_cfg.precision),
        ),
        flush=True,
    )
    group_labels = {}
    for row in train_dataset.rows:
        group_labels[str(row["group_id"])] = int(row["label"])
    positive_count = int(sum(value == 1 for value in group_labels.values()))
    negative_count = int(sum(value == 0 for value in group_labels.values()))
    if positive_count <= 0 or negative_count <= 0:
        raise ValueError("training split must contain positive and negative coordinate groups")
    pos_weight_value = min(float(train_cfg.max_pos_weight), float(negative_count) / float(positive_count))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.learning_rate), weight_decay=float(train_cfg.weight_decay))
    history: list[dict[str, object]] = []
    best_checkpoint_score = (float("-inf"), float("-inf"), float("-inf"))
    best_checkpoint_metrics: dict[str, float | int] = {}
    best_checkpoint_epoch = 0
    best_checkpoint = output / "best.pt"
    stopper = EarlyStoppingTracker(
        metric=str(train_cfg.early_stopping_metric),
        patience=int(train_cfg.early_stopping_patience),
        min_delta=float(train_cfg.early_stopping_min_delta),
    )
    stopped_early = False
    for epoch in range(int(train_cfg.max_epochs)):
        epoch_started = time.perf_counter()
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            voxel = batch["voxel"].to(device=device, dtype=torch.float32, non_blocking=True)
            context = batch["context"].to(device=device, dtype=torch.float32, non_blocking=True)
            labels = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=autocast_dtype is not None,
            ):
                logits = model(voxel, context)
                if logits.shape != labels.shape:
                    raise RuntimeError("model logit shape does not match labels")
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_loss, labels, probabilities = evaluate_loader(
            model,
            val_loader,
            criterion,
            device,
            autocast_dtype=autocast_dtype,
        )
        rank = ranking_metrics(labels, probabilities)
        selected = select_operating_metrics(
            labels,
            probabilities,
            threshold_selection_mode=str(train_cfg.threshold_selection_mode),
            fixed_keep_threshold=float(train_cfg.fixed_keep_threshold),
            target_recall=float(train_cfg.target_recall),
        )
        train_loss = float(np.mean(losses)) if losses else 0.0
        epoch_seconds = float(time.perf_counter() - epoch_started)
        if str(train_cfg.checkpoint_selection_mode) == "validation_loss":
            checkpoint_selection_metric = "validation_loss"
            validation_score = (-float(val_loss), float(selected["accuracy"]))
            selection_score = validation_loss_checkpoint_selection_score(
                validation_loss=float(val_loss),
                accuracy=float(selected["accuracy"]),
                f1=float(selected["f1"]),
            )
        elif str(train_cfg.threshold_selection_mode) == "fixed":
            checkpoint_selection_metric = "accuracy_at_fixed_threshold"
            validation_score = (float(selected["accuracy"]), float(selected["f1"]))
            selection_score = fixed_threshold_checkpoint_selection_score(
                accuracy=float(selected["accuracy"]),
                f1=float(selected["f1"]),
                pr_auc=float(rank["pr_auc"]),
            )
        else:
            checkpoint_selection_metric = "negative_rejection_rate"
            validation_score = (float(selected["negative_rejection_rate"]), float(rank["pr_auc"]))
            selection_score = checkpoint_selection_score(
                negative_rejection_rate=float(selected["negative_rejection_rate"]),
                rejected_seed_accuracy=float(selected["rejected_seed_accuracy"]),
                pr_auc=float(rank["pr_auc"]),
            )
        is_best_checkpoint = selection_score > best_checkpoint_score
        should_stop = stopper.update(train_loss=train_loss, validation_score=validation_score)
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_seconds": epoch_seconds,
            **rank,
            **selected,
            "positive_coordinate_groups": positive_count,
            "negative_coordinate_groups": negative_count,
            "pos_weight": pos_weight_value,
            "early_stopping_metric": str(train_cfg.early_stopping_metric),
            "early_stopping_patience": int(stopper.patience),
            "early_stopping_best_train_loss": float(stopper.best_train_loss),
            "early_stopping_epochs_without_improvement": int(stopper.epochs_without_improvement),
            "checkpoint_selection_mode": str(train_cfg.checkpoint_selection_mode),
            "threshold_selection_mode": str(train_cfg.threshold_selection_mode),
            "checkpoint_selection_metric": checkpoint_selection_metric,
            "checkpoint_selection_score": list(selection_score),
            "is_best_checkpoint": bool(is_best_checkpoint),
        }
        if str(train_cfg.threshold_selection_mode) == "target_recall":
            row["recall_rejection"] = recall_rejection_table(labels, probabilities)
        history.append(row)
        write_json_atomic(output / "training_history.json", history)
        print(
            "[door-seed-train] epoch=%d train_loss=%.8f val_loss=%.8f threshold=%.9f "
            "accuracy=%.6f f1=%.6f reject_coverage=%.6f reject_acc=%.6f pr_auc=%.6f "
            "no_improve=%d/%d seconds=%.3f"
            % (
                epoch + 1,
                train_loss,
                val_loss,
                float(selected["threshold"]),
                float(selected["accuracy"]),
                float(selected["f1"]),
                float(selected["negative_rejection_rate"]),
                float(selected["rejected_seed_accuracy"]),
                float(rank["pr_auc"]),
                stopper.epochs_without_improvement,
                stopper.patience,
                epoch_seconds,
            ),
            flush=True,
        )
        if is_best_checkpoint:
            best_checkpoint_score = selection_score
            best_checkpoint_metrics = {
                **selected,
                **rank,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            }
            best_checkpoint_epoch = epoch + 1
            checkpoint = _checkpoint_payload(
                model=model,
                model_config=model_cfg,
                context_source=context_source,
                height_scale_m=float(height_scale_m),
                geometry=geometry,
                selected_threshold=selected,
                train_rows=train_dataset.rows,
                val_rows=val_dataset.rows,
                source_root=source_root,
                training_config=train_cfg,
                epoch=epoch + 1,
                history=history,
                validation_loss=float(val_loss),
                checkpoint_selection_metric=checkpoint_selection_metric,
                checkpoint_selection_score=selection_score,
                augmentation_manifest=augmentation_manifest,
            )
            _torch_save_atomic(checkpoint, best_checkpoint)
        if should_stop:
            stopped_early = True
            break
    result = {
        "checkpoint": str(best_checkpoint),
        "epochs_completed": len(history),
        "threshold_selection_mode": str(train_cfg.threshold_selection_mode),
        "checkpoint_selection_mode": str(train_cfg.checkpoint_selection_mode),
        "fixed_keep_threshold": (
            float(train_cfg.fixed_keep_threshold)
            if str(train_cfg.threshold_selection_mode) == "fixed"
            else None
        ),
        "checkpoint_selection_metric": (
            "validation_loss"
            if str(train_cfg.checkpoint_selection_mode) == "validation_loss"
            else "accuracy_at_fixed_threshold"
            if str(train_cfg.threshold_selection_mode) == "fixed"
            else "negative_rejection_rate"
        ),
        "selected_checkpoint_epoch": int(best_checkpoint_epoch),
        "selected_checkpoint_metrics": best_checkpoint_metrics,
        "best_checkpoint_selection_score": list(best_checkpoint_score),
        "train_base_samples": len(train_dataset.rows),
        "train_samples": len(train_dataset),
        "train_samples_per_epoch": int(train_samples_per_epoch),
        "train_rotation_degrees": list(train_dataset.rotation_degrees),
        "train_mirror_lr_once": bool(train_dataset.mirror_lr_once),
        "precision": str(train_cfg.precision),
        "val_samples": len(val_dataset),
        "train_snapshot_count": train_snapshot_count,
        "val_snapshot_count": val_snapshot_count,
        "train_snapshot_cache_size": int(train_dataset.cache_size),
        "val_snapshot_cache_size": int(val_dataset.cache_size),
        "early_stopping_metric": str(train_cfg.early_stopping_metric),
        "early_stopping_patience": int(train_cfg.early_stopping_patience),
        "early_stopping_min_delta": float(train_cfg.early_stopping_min_delta),
        "best_train_loss": float(stopper.best_train_loss),
        "stop_reason": (
            "%s_no_improvement_%d_epochs"
            % (str(train_cfg.early_stopping_metric), int(train_cfg.early_stopping_patience))
            if stopped_early
            else "max_epochs"
        ),
    }
    write_json_atomic(output / "training_summary.json", result)
    return result


def evaluate_loader(
    model,
    loader,
    criterion,
    device,
    *,
    autocast_dtype=None,
) -> tuple[float, list[int], list[float]]:
    import torch

    model.eval()
    losses: list[float] = []
    labels: list[int] = []
    probabilities: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            voxel = batch["voxel"].to(device=device, dtype=torch.float32, non_blocking=True)
            context = batch["context"].to(device=device, dtype=torch.float32, non_blocking=True)
            target = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=autocast_dtype is not None,
            ):
                logits = model(voxel, context)
                loss = criterion(logits, target)
            if not torch.all(torch.isfinite(logits)):
                raise RuntimeError("model produced non-finite validation logits")
            losses.append(float(loss.detach().cpu()))
            labels.extend(int(value) for value in target.detach().cpu().numpy().reshape(-1))
            probabilities.extend(
                float(value)
                for value in torch.sigmoid(logits.float()).detach().cpu().numpy().reshape(-1)
            )
    return float(np.mean(losses)) if losses else 0.0, labels, probabilities


def _dataset_geometry(rows: list[Mapping[str, object]]) -> dict[str, object]:
    geometries: set[tuple[object, ...]] = set()
    raw_seed_hashes: set[str] = set()
    input_semantics_hashes: set[str] = set()
    z_hashes: set[str] = set()
    for path in sorted({str(row["snapshot_path"]) for row in rows}):
        arrays = load_snapshot(path)
        z = np.asarray(arrays["z_centers_m"], dtype=np.float32)
        z_hash = sha256_bytes(np.ascontiguousarray(z).tobytes())
        geometry = (
            int(len(z)),
            float(scalar_value(arrays, "z_min_m")),
            float(scalar_value(arrays, "z_resolution_m")),
            float(scalar_value(arrays, "resolution_m")),
            z_hash,
        )
        geometries.add(geometry)
        z_hashes.add(z_hash)
        raw_seed_hashes.add(str(scalar_value(arrays, "raw_seed_config_hash", "")))
        input_semantics_hashes.add(str(scalar_value(arrays, "input_semantics_hash", "")))
    if len(geometries) != 1:
        raise ValueError("dataset contains incompatible voxel or XY geometries")
    if len(raw_seed_hashes) != 1:
        raise ValueError("dataset contains multiple raw seed configurations")
    if len(input_semantics_hashes) != 1 or not next(iter(input_semantics_hashes), ""):
        raise ValueError("dataset contains missing or incompatible input semantics")
    z_count, z_min, z_resolution, xy_resolution, z_hash = next(iter(geometries))
    return {
        "z_count": z_count,
        "z_min_m": z_min,
        "z_resolution_m": z_resolution,
        "xy_resolution_m": xy_resolution,
        "z_centers_sha256": z_hash,
        "raw_seed_config_hash": next(iter(raw_seed_hashes)),
        "input_semantics_hash": next(iter(input_semantics_hashes)),
    }


def _checkpoint_payload(
    *,
    model,
    model_config: DoorSeedModelConfig,
    context_source: str,
    height_scale_m: float,
    geometry: Mapping[str, object],
    selected_threshold: Mapping[str, object],
    train_rows: list[Mapping[str, object]],
    val_rows: list[Mapping[str, object]],
    source_root: str | Path | None,
    training_config: TrainingConfig,
    epoch: int,
    history: list[Mapping[str, object]],
    validation_loss: float,
    checkpoint_selection_metric: str,
    checkpoint_selection_score: tuple[float, float, float],
    augmentation_manifest: Mapping[str, object],
) -> dict[str, object]:
    expected_source_root = Path(__file__).resolve().parent
    actual_source_root = expected_source_root if source_root is None else Path(source_root).expanduser().resolve()
    if actual_source_root != expected_source_root:
        raise ValueError("source_root must be the door_seed_learning package for strict online validation")
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    return {
        "model_state_dict": state,
        "model_config": model_config.to_dict(),
        "model_architecture_version": MODEL_ARCHITECTURE_VERSION,
        "preprocessor_version": PREPROCESSOR_VERSION,
        "context_source": str(context_source),
        "local_patch_size": int(model_config.local_patch_size),
        "context_patch_size": int(model_config.context_patch_size),
        "z_count": int(geometry["z_count"]),
        "z_min_m": float(geometry["z_min_m"]),
        "z_resolution_m": float(geometry["z_resolution_m"]),
        "xy_resolution_m": float(geometry["xy_resolution_m"]),
        "z_centers_sha256": str(geometry["z_centers_sha256"]),
        "raw_seed_config_hash": str(geometry["raw_seed_config_hash"]),
        "input_semantics_hash": str(geometry["input_semantics_hash"]),
        "height_scale_m": float(height_scale_m),
        "voxel_state_mapping": {"unknown": [0, 3], "free": [1], "occupied": [2]},
        "recommended_keep_threshold": float(selected_threshold["threshold"]),
        "achieved_recall": float(selected_threshold["recall"]),
        "achieved_accuracy": float(selected_threshold["accuracy"]),
        "achieved_f1": float(selected_threshold["f1"]),
        "negative_rejection_rate": float(selected_threshold["negative_rejection_rate"]),
        "rejected_seed_accuracy": float(selected_threshold["rejected_seed_accuracy"]),
        "threshold_selection_mode": str(training_config.threshold_selection_mode),
        "checkpoint_selection_metric": str(checkpoint_selection_metric),
        "checkpoint_selection_score": list(checkpoint_selection_score),
        "checkpoint_validation_loss": float(validation_loss),
        "training_scene_ids": sorted({str(row["scene_id"]) for row in train_rows}),
        "validation_scene_ids": sorted({str(row["scene_id"]) for row in val_rows}),
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "source_code_hash": source_tree_hash(actual_source_root),
        "training_config": asdict(training_config),
        "training_config_hash": config_hash(training_config),
        "augmentation_manifest": dict(augmentation_manifest),
        "epoch": int(epoch),
        "history": list(history),
        "created_at_unix": time.time(),
    }


def _torch_save_atomic(payload: object, path: Path) -> None:
    import torch

    temp = path.with_name(path.name + ".tmp")
    torch.save(payload, temp)
    with temp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temp, path)
