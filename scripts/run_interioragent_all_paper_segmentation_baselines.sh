#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VOXROOM_PYTHON="${VOXROOM_PYTHON:-/home/echo/miniforge3/envs/sgnav-isaac/bin/python}"
EXTERNAL_ROOT="${EXTERNAL_ROOT:-/home/echo/VoxRoom-Online-exp/external_baselines}"
ROS_ENV="${ROS_ENV:-$EXTERNAL_ROOT/ros_noetic_env}"
DATA_ROOT="${DATA_ROOT:-/media/echo/data/voxroom_roomseg_evaluation/interioragent_all_available_gt_20260828}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/echo/data/voxroom_roomseg_evaluation/interioragent_paper_segmentation_baselines_20260828}"
METRICS_OUT="${METRICS_OUT:-$OUTPUT_ROOT/metrics_all_methods}"
PAPER="${PAPER:-$DATA_ROOT/inputs/Topology-Based_Visual_Active_Room_Segmentation.pdf}"
TRAINING_EXCLUSION_MANIFEST="${TRAINING_EXCLUSION_MANIFEST:-$DATA_ROOT/training_scene_exclusion_manifest.json}"
SEGMENTATION_INPUT_MAP="${SEGMENTATION_INPUT_MAP:-raw_vertical_free}"

export ROS_BASELINE_SETUP="${ROS_BASELINE_SETUP:-$ROS_ENV/setup.bash}"
export ROS_BASELINE_PYTHON="${ROS_BASELINE_PYTHON:-$ROS_ENV/bin/python}"
export PATH="$(dirname "$ROS_BASELINE_PYTHON"):$PATH"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export VOXROOM_REPO_ROOT="$REPO_ROOT"
export DUDE_WS="${DUDE_WS:-$EXTERNAL_ROOT/dude_ws}"
export ROSE2_WS="${ROSE2_WS:-$EXTERNAL_ROOT/rose2_ws}"
export IPA_WS="${IPA_WS:-$EXTERNAL_ROOT/ipa_ws}"
export INCREMENTAL_DUDE_ROS_ROOT="${INCREMENTAL_DUDE_ROS_ROOT:-$DUDE_WS/src/Incremental_DuDe_ROS}"

for required in \
  "$DATA_ROOT/index_voxroom.json" \
  "$DATA_ROOT/index_tvars_original.json" \
  "$DATA_ROOT/approved_summary.json" \
  "$DATA_ROOT/annotations" \
  "$DATA_ROOT/final_gt" \
  "$PAPER" \
  "$TRAINING_EXCLUSION_MANIFEST" \
  "$ROS_BASELINE_SETUP" \
  "$ROS_BASELINE_PYTHON"; do
  if [[ ! -e "$required" ]]; then
    echo "[interioragent-paper-baselines] missing required input: $required" >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_ROOT/commands"
"$VOXROOM_PYTHON" scripts/replay_grscene_paper_segmentation_baselines.py \
  --voxroom-index "$DATA_ROOT/index_voxroom.json" \
  --tvars-index "$DATA_ROOT/index_tvars_original.json" \
  --approved-summary "$DATA_ROOT/approved_summary.json" \
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
  2>&1 | tee "$OUTPUT_ROOT/commands/replay.log"

if [[ -e "$METRICS_OUT" ]]; then
  echo "[interioragent-paper-baselines] refusing to overwrite existing metrics: $METRICS_OUT" >&2
  exit 1
fi

"$VOXROOM_PYTHON" scripts/evaluate_grscene_all_paper_segmentation_baselines.py \
  --dataset-name interioragent \
  --voxroom-index "$DATA_ROOT/index_voxroom.json" \
  --tvars-index "$DATA_ROOT/index_tvars_original.json" \
  --replay-root "$OUTPUT_ROOT" \
  --annotation-dir "$DATA_ROOT/annotations" \
  --gt-dir "$DATA_ROOT/final_gt" \
  --paper "$PAPER" \
  --out-dir "$METRICS_OUT" \
  --training-exclusion-manifest "$TRAINING_EXCLUSION_MANIFEST" \
  --min-room-area-m2 0.5 \
  --cell-size-m 0.05 \
  2>&1 | tee "$OUTPUT_ROOT/commands/evaluate.log"

echo "[interioragent-paper-baselines] predictions: $OUTPUT_ROOT/predictions"
echo "[interioragent-paper-baselines] metrics: $METRICS_OUT"
echo "[interioragent-paper-baselines] segmentation input: $SEGMENTATION_INPUT_MAP"
