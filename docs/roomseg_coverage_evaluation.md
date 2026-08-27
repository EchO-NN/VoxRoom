# Room Segmentation Coverage Evaluation

This protocol is shared by Isaac Sim and Habitat. It has no runtime fallback.
Any missing full-scene reference, resolution mismatch, coordinate mismatch,
coverage regression, missing voxel state, or unpaired method event aborts the
run.

## Protocol

1. Load the simulator's complete static explorable mask before exploration and
   freeze it as the denominator.
2. Count explored cells only inside that fixed mask.
3. At 20%, 40%, 60%, 80%, and 100%, run VoxRoom and partition the same current
   map with TVARS accepted door lines.
4. Always emit an additional `final` event when the episode ends, even when the
   final coverage is below 100% or exactly equals a milestone.
5. Save both 2D segmentations, their coordinate metadata, and one shared,
   complete 3D voxel state for every event. The TVARS artifact references that
   voxel snapshot instead of duplicating it.
6. Annotate the complete fixed explorable reference on the VoxRoom final event
   once. Backproject that approved GT through each event's cumulative explored
   reference mask, then evaluate both methods with the same GT files.

Precision averages each predicted room's largest overlap fraction with any GT
room. Recall averages each GT room's largest overlap fraction with any predicted
room. Both are macro averages over rooms rather than pixel accuracy.

## Isaac Sim

```bash
MAX_CONTROL_STEPS=5000 \
  scripts/run_roomseg_coverage_eval_isaac.sh \
  data/interioragent_episodes/radius005_all_scenes/kujiale_0003.jsonl
```

The default output is
`outputs/roomseg_coverage_eval_isaac/<scene>/roomseg_coverage_eval`.

## Habitat

Run from the staged Active Room checkout after installing it on the Habitat
machine:

```bash
cd ~/Active_room_segmentation
ROOMSEG_COVERAGE_EVAL=1 \
ROOMSEG_COVERAGE_MILESTONES=20,40,60,80,100 \
  scripts/run_gibson_voxroom_coverage_eval.sh
```

Habitat supplies its fixed `explorable_map`, cumulative `explored_map`, native
occupied map, and the original method's accepted door segments to the strict
VoxRoom sidecar. Habitat arrays are converted to the native VoxRoom grid with
the recorded vertical row flip only; original `(x, y)` door endpoints are
explicitly converted to `(row=y, col=x)`. No resizing or resampling is allowed.

## GT And Metrics

Use one evaluation directory for both methods:

```bash
scripts/roomseg_coverage_eval.sh index RESULT_ROOT EVAL_DIR
scripts/roomseg_coverage_eval.sh annotate RESULT_ROOT EVAL_DIR
scripts/roomseg_coverage_eval.sh build-gt RESULT_ROOT EVAL_DIR
scripts/roomseg_coverage_eval.sh compute RESULT_ROOT EVAL_DIR
```

The paired result is `EVAL_DIR/COVERAGE_COMPARISON.md`, with machine-readable
JSON and CSV beside it. Method-specific reports remain under
`EVAL_DIR/metrics_voxroom` and `EVAL_DIR/metrics_tvars_original`. Per-event rows
retain the measured coverage ratio and event identity.
