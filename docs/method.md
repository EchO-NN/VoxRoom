# VoxRoom: implementation notes

This document describes the maintained default implementation accompanying the ICRA 2027 submission. The manuscript calls the planar structural representation **Structural Free Map (SFM)**; code and debug arrays retain **Vertical Free Map** terminology.

## Mapping and structural representation

Posed sensor observations update a gravity-aligned 3D occupancy map. Observed free, occupied, and unobserved states remain distinct. The configured backend is `nvblox_fast_dda`; alternative integration paths in the repository are not automatically interchangeable experimental settings.

Column classification aggregates observed free evidence over height rather than requiring one uninterrupted traversable vertical interval. Strict wall and occluded-wall evidence provide structural barriers when free support is insufficient. Unknown evidence can participate in structural tests without rewriting the original unknown voxels as occupied.

The SFM is a **structural representation**, not a safety certificate. The no-clearance navigation-free map is a separate projection; its furniture obstacles remain obstacles for motion planning. See `mapping/voxel_occupancy_door_wall_roomseg.py` and the voxel column classifiers in the same package.

## Hybrid raw entry seeds

`door_seed_learning/hybrid_raw_seed.py` combines the existing voxel-derived candidates with TVARS-style planar ray candidates. Ray-length discontinuities suggest nearby opening endpoints; geometry constrains their pairing. Candidate lines are rasterized into the structural map.

The combination is **OR**, not AND. A seed tagged by both sources still belongs to both for source-specific analyses. Ray candidates do not import TVARS visual accept/reject decisions, VoxRoom classifier predictions, or GT room labels. The classifier subsequently verifies the union.

## Verifier architecture

Implemented in `door_seed_learning/model.py`:

| Stage | Layers / output |
| --- | --- |
| Voxel column | `[4,Z]`; Conv1d `4→40`, kernel 5; GroupNorm; GELU; residual Conv1d block |
| Height relations | Two pre-norm Transformer layers, width 40, four heads, feed-forward width 120 |
| Height aggregation | Mean and max pooling → 80 features per column |
| Local XY layout | Assemble `[80,19,19]`; Conv2d `80→72`, residual block; stride-2 Conv2d `72→112`, residual block |
| 3D branch output | Global spatial mean + max → 224 features |
| SFM context | `[3,41,41]`; Conv2d `3→40`, residual block; stride-2 `40→72`, residual block; stride-2 `72→112`, residual block |
| 2D branch output | Global spatial mean + max → 224 features |
| Fused classifier | `448→160→40→1`, GELU/dropout between linear layers; sigmoid at inference |

The first three voxel channels encode unknown, free, and occupied; the fourth encodes height relative to the common ground reference, normalized by a fixed scale. The context channels encode unknown, structural free, and occupied. Checkpoint metadata specifies the height discretization, preprocessing, and branch configuration.

### Inference-time column reuse

`door_seed_learning/column_encoding_cache.py` and `inference.py` avoid repeatedly encoding columns shared by overlapping 19×19 patches. The cache stores per-column descriptors, not seed probabilities. Each candidate gathers those descriptors in its own XY order before the unchanged local CNN and context branch run.

Cache lookup depends on the encoded column content and relevant model/input conditions. Changed states or height/model signatures invalidate reuse. The default CPU cache is bounded at 65,536 columns; encoding batches are limited to 2,048 columns. No entire-map feature tensor is required on the GPU.

The mathematical operation is unchanged; GPU batch shapes and floating-point kernels can yield small numerical differences. Regression tests check tolerances and classification decisions, rather than claiming bitwise equality. Training uses the normal differentiable model path.

## Separators and room labels

Verified seeds are grouped into geometric support sets. Fitted linear primitives are checked for correlation, orthogonal variance, support, continuity, thickness, and surrounding geometry. Accepted door candidates contribute separators; persistent door memory stabilizes established partitions as observations change. The release defaults are:

```yaml
door_seed_learning:
  keep_threshold: 0.5
  context_source: vertical
  raw_seed_source: voxroom_tvars_vertical_union
  reuse_column_encodings: true
voxel_door:
  primitive_min_line_correlation: 0.95
  primitive_max_orthogonal_variance_cells2: 0.65
  enable_l_shaped_seed_branches: false
```

These keys are nested under `mapping.room_segmentation`. The variance is measured in **grid cells squared**, not square meters. L-shaped dual-arm fitting remains an explicit experimental option, not the default algorithm.

Connected components of structural-free space separated by the accepted cuts produce room labels. Labels are then projected/aligned onto navigation space. Separators are virtual instance boundaries: they must not overwrite the physical navigation occupancy map.

## Rendering versus evaluation

Room colors are categorical presentation choices, not semantic classes. A matched palette across methods does not merge their predicted labels. Unknown regions remain unassigned; navigation obstacles and structural-free regions should not be conflated.

The metric pipeline consumes integer room-label arrays on the frozen evaluation domain. Recoloring, image cropping, and other publication-figure presentation operations are not part of metric computation. Use saved raw predictions and explicit reference masks when reproducing scores; do not evaluate screenshots.

## References to code

- [Column/door geometry](../voxroom_online/isaac_runtime/mapping/voxel_door_detector.py)
- [Room-segmentation pipeline](../voxroom_online/isaac_runtime/mapping/voxel_occupancy_door_wall_roomseg.py)
- [Hybrid candidates](../voxroom_online/isaac_runtime/door_seed_learning/hybrid_raw_seed.py)
- [Network](../voxroom_online/isaac_runtime/door_seed_learning/model.py)
- [Inference and compatibility checks](../voxroom_online/isaac_runtime/door_seed_learning/inference.py)
- [Overlap metrics](../voxroom_online/isaac_runtime/evaluation/online_roomseg/metrics.py)
