#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VOXROOM_PYTHON="${VOXROOM_PYTHON:-$REPO_ROOT/../.conda/envs/voxroom-online/bin/python}"
if [[ ! -x "$VOXROOM_PYTHON" ]]; then
  VOXROOM_PYTHON="${HOME}/.conda/envs/voxroom-online/bin/python"
fi
ROS_ENV="${ROS_ENV:-$REPO_ROOT/external_baselines/ros_noetic_env}"
DATA_ROOT="${DATA_ROOT:-/media/echo/data/voxroom_roomseg_evaluation/grscene_all_available_gt_20260817}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/echo/data/voxroom_roomseg_evaluation/grscene_paper_segmentation_baselines_20260827}"
METRICS_OUT="${METRICS_OUT:-$OUTPUT_ROOT/metrics_all_methods}"
PAPER="${PAPER:-$DATA_ROOT/inputs/Topology-Based_Visual_Active_Room_Segmentation.pdf}"
SEGMENTATION_INPUT_MAP="${SEGMENTATION_INPUT_MAP:-raw_vertical_free}"

export ROS_BASELINE_SETUP="${ROS_BASELINE_SETUP:-$ROS_ENV/setup.bash}"
export ROS_BASELINE_PYTHON="${ROS_BASELINE_PYTHON:-$ROS_ENV/bin/python}"
export PATH="$(dirname "$ROS_BASELINE_PYTHON"):$PATH"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export VOXROOM_REPO_ROOT="$REPO_ROOT"
export DUDE_WS="${DUDE_WS:-$REPO_ROOT/external_baselines/dude_ws}"
export ROSE2_WS="${ROSE2_WS:-$REPO_ROOT/external_baselines/rose2_ws}"
export IPA_WS="${IPA_WS:-$REPO_ROOT/external_baselines/ipa_ws}"
export INCREMENTAL_DUDE_ROS_ROOT="${INCREMENTAL_DUDE_ROS_ROOT:-$DUDE_WS/src/Incremental_DuDe_ROS}"

for required in \
  "$DATA_ROOT/index_voxroom.json" \
  "$DATA_ROOT/index_tvars_original.json" \
  "$DATA_ROOT/annotations" \
  "$DATA_ROOT/final_gt" \
  "$PAPER" \
  "$ROS_BASELINE_SETUP" \
  "$ROS_BASELINE_PYTHON"; do
  if [[ ! -e "$required" ]]; then
    echo "[paper-baselines] missing required input: $required" >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_ROOT/commands"
RUN_LOG="$OUTPUT_ROOT/commands/replay.log"
"$VOXROOM_PYTHON" scripts/replay_grscene_paper_segmentation_baselines.py \
  --voxroom-index "$DATA_ROOT/index_voxroom.json" \
  --tvars-index "$DATA_ROOT/index_tvars_original.json" \
  --approved-summary "$REPO_ROOT/results/grscene_metrics_20260818/summary.json" \
  --output-root "$OUTPUT_ROOT" \
  --segmentation-input-map "$SEGMENTATION_INPUT_MAP" \
  --methods \
    dude_incremental \
    gomez_incremental \
    dude_offline \
    rose2 \
    morphological \
    distance_transform \
    voronoi \
  --strict-main \
  --resume \
  --map-resolution-m 0.05 \
  --ros-baseline-setup "$ROS_BASELINE_SETUP" \
  --ros-baseline-python "$ROS_BASELINE_PYTHON" \
  --dude-ws "$DUDE_WS" \
  --dude-repo-root "$INCREMENTAL_DUDE_ROS_ROOT" \
  --rose2-ros-workspace "$ROSE2_WS" \
  --rose2-launch-file "$REPO_ROOT/configs/ros/rose2_headless.launch" \
  --ipa-ros-workspace "$IPA_WS" \
  2>&1 | tee "$RUN_LOG"

if [[ -e "$METRICS_OUT" ]]; then
  echo "[paper-baselines] refusing to overwrite existing metrics directory: $METRICS_OUT" >&2
  exit 1
fi

"$VOXROOM_PYTHON" scripts/evaluate_grscene_all_paper_segmentation_baselines.py \
  --dataset-name grscene \
  --voxroom-index "$DATA_ROOT/index_voxroom.json" \
  --tvars-index "$DATA_ROOT/index_tvars_original.json" \
  --replay-root "$OUTPUT_ROOT" \
  --annotation-dir "$DATA_ROOT/annotations" \
  --gt-dir "$DATA_ROOT/final_gt" \
  --paper "$PAPER" \
  --out-dir "$METRICS_OUT" \
  --training-exclusion-manifest \
    "$REPO_ROOT/results/grscene_metrics_excluding_training_20260818/training_scene_exclusion_manifest.json" \
  --min-room-area-m2 0.5 \
  --cell-size-m 0.05 \
  2>&1 | tee "$OUTPUT_ROOT/commands/evaluate.log"

echo "[paper-baselines] predictions: $OUTPUT_ROOT/predictions"
echo "[paper-baselines] metrics: $METRICS_OUT"
echo "[paper-baselines] segmentation input: $SEGMENTATION_INPUT_MAP"
