#!/usr/bin/env bash
set -Eeuo pipefail

export MALLOC_ARENA_MAX="${VOXROOM_MALLOC_ARENA_MAX:-1}"
export GLIBC_TUNABLES="${VOXROOM_GLIBC_TUNABLES:-glibc.malloc.arena_max=1}"
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-131072}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/results/kujiale_0003_vertical_epoch14_fullviz_20260901}"
EPISODE_FILE="${EPISODE_FILE:-/home/joey/VoxRoom/data/interioragent_episodes/radius015_physical010_targetlocal_20260809/kujiale_0003.jsonl}"
CHECKPOINT="${CHECKPOINT:-$RUN_ROOT/checkpoint_vertical_epoch14.pt}"
EXPECTED_CHECKPOINT_SHA256="2e82fba43ab0d484bb7f73b5d752d9a2f7c270d65228eaebcdb77cd2d2c866d9"
CONFIG="${CONFIG:-configs/voxroom_online.yaml}"
FRAMES_DIR="$RUN_ROOT/voxroom_viz_frames"
COVERAGE_DIR="$RUN_ROOT/roomseg_coverage_eval"
RESULTS_FILE="$RUN_ROOT/results.jsonl"
LOG_FILE="$RUN_ROOT/run.log"
PACKAGER_PYTHON="${PACKAGER_PYTHON:-${VOXROOM_PYTHON:-python3}}"

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$LOG_FILE") 2>&1

if [[ -e "$RUN_ROOT/.run_started" || -e "$COVERAGE_DIR" || -s "$RESULTS_FILE" ]]; then
  echo "[kujiale-0003-fullviz] refusing to overwrite an existing run: $RUN_ROOT" >&2
  exit 1
fi
for required in "$EPISODE_FILE" "$CHECKPOINT" "$CONFIG"; do
  if [[ ! -f "$required" ]]; then
    echo "[kujiale-0003-fullviz] missing required file: $required" >&2
    exit 1
  fi
done
actual_checkpoint_sha256="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
if [[ "$actual_checkpoint_sha256" != "$EXPECTED_CHECKPOINT_SHA256" ]]; then
  echo "[kujiale-0003-fullviz] checkpoint hash mismatch: $actual_checkpoint_sha256" >&2
  exit 1
fi

touch "$RUN_ROOT/.run_started"
echo "[kujiale-0003-fullviz] scene=kujiale_0003"
echo "[kujiale-0003-fullviz] checkpoint=$CHECKPOINT sha256=$actual_checkpoint_sha256"
echo "[kujiale-0003-fullviz] output=$RUN_ROOT"
echo "[kujiale-0003-fullviz] navigation=random_frontier seed=0 source=voxel_vertical_free"
echo "[kujiale-0003-fullviz] 70pct_snapshot=$RUN_ROOT/paper_visualization_70pct"

"$PACKAGER_PYTHON" \
  scripts/package_kujiale_0003_60pct_snapshot.py \
  --run-root "$RUN_ROOT" --checkpoint "$CHECKPOINT" \
  --milestone-percent 70 \
  --wait --timeout-s 86400 --poll-s 3 \
  >> "$RUN_ROOT/70pct_packager.log" 2>&1 &
packager_pid=$!

cleanup_packager() {
  if kill -0 "$packager_pid" 2>/dev/null; then
    kill "$packager_pid" 2>/dev/null || true
    wait "$packager_pid" 2>/dev/null || true
  fi
}
trap cleanup_packager EXIT

set +e
DISPLAY="${DISPLAY:-:1}" \
XAUTHORITY="${XAUTHORITY:-/run/user/1000/gdm/Xauthority}" \
VOXROOM_PYTHON="${VOXROOM_PYTHON:-python3}" \
VOXROOM_NUMBA_THREADS="${VOXROOM_NUMBA_THREADS:-28}" \
  scripts/run_voxroom_isaac_env.sh voxroom_online/isaac_runtime/scripts/run_one_episode.py \
  --config "$CONFIG" \
  --episode-file "$EPISODE_FILE" \
  --episode-index 0 \
  --sim-backend isaac \
  --planner astar \
  --detector none \
  --segmenter none \
  --no-llm-enabled \
  --no-vllm-frontier-scoring \
  --no-vllm-image-scoring \
  --no-allow-gt-goal-fallback \
  --no-frontier-allow-near-fallback \
  --frontier-selection-mode random \
  --frontier-random-seed 0 \
  --frontier-source voxel_vertical_free \
  --room-map-mode voxel_occupancy_door_wall_v33 \
  --roomseg-backend voxel_occupancy_door_wall_v33 \
  --no-strict-benchmark \
  --explore-until-no-frontiers \
  --max-control-steps 7000 \
  --max-vx-mps 2.70 \
  --max-vy-mps 0.0 \
  --control-dt 0.2 \
  --max-wz-radps 2.6179938779914944 \
  --forward-heading-tolerance-deg 25.0 \
  --lookahead-m 1.00 \
  --translation-command-scale 6.0 \
  --replan-every-steps 10 \
  --smoothing-max-skip-cells 24 \
  --simulator-collision-guard \
  --simulator-collision-reference-radius-m 0.05 \
  --no-static-nearfield-map \
  --robot-radius-m 0.15 \
  --runtime-planning-clearance-m 0.01 \
  --astar-clearance-cost-enabled \
  --astar-clearance-desired-m 0.20 \
  --astar-clearance-hard-min-m 0.05 \
  --astar-clearance-weight 6.0 \
  --astar-goal-min-clearance-m 0.08 \
  --lookahead-min-clearance-m 0.05 \
  --guard-min-clearance-m 0.05 \
  --headless \
  --read-depth \
  --camera-annotator-device cpu \
  --isaac-width 256 \
  --isaac-height 256 \
  --camera-hfov-deg 90 \
  --camera-mast-height-m 1.25 \
  --panorama-steps 12 \
  --door-seed-learning-mode inference \
  --door-seed-checkpoint "$CHECKPOINT" \
  --door-seed-allow-source-code-hash-mismatch \
  --door-seed-raw-source voxroom_tvars_vertical_union \
  --door-seed-collection-every-steps 5 \
  --door-seed-tvars-seed-width-cells 3 \
  --door-seed-local-voxel-patch-size 19 \
  --door-seed-context-patch-size 41 \
  --door-seed-persistent-final-raw-seed-union \
  --voxroom-viz \
  --voxroom-viz-every-steps 1 \
  --voxroom-viz-save-dir "$FRAMES_DIR" \
  --voxroom-viz-save-every-steps 1 \
  --voxroom-viz-width 2200 \
  --voxroom-viz-height 1400 \
  --no-show-tvars-original-mask \
  --live-roomseg-baseline tvars_original_isaac \
  --live-baseline-output-dir "$RUN_ROOT/baselines/tvars_original_isaac" \
  --live-baseline-door-detector disabled \
  --live-baseline-policy-control never \
  --live-baseline-panorama-views 12 \
  --roomseg-coverage-eval \
  --roomseg-coverage-milestones 20,40,60,70,80,90 \
  --roomseg-full-voxel-milestones 70 \
  --roomseg-coverage-output-dir "$COVERAGE_DIR" \
  --output "$RESULTS_FILE"
run_status=$?
set -e

if [[ "$run_status" -ne 0 || ! -s "$RESULTS_FILE" ]]; then
  touch "$RUN_ROOT/.run_failed"
  echo "[kujiale-0003-fullviz] Isaac run failed with status=$run_status results_present=$([[ -s "$RESULTS_FILE" ]] && echo true || echo false)" >&2
  if [[ "$run_status" -eq 0 ]]; then
    exit 70
  fi
  exit "$run_status"
fi

wait "$packager_pid"
trap - EXIT

if command -v ffmpeg >/dev/null 2>&1; then
  ffmpeg -hide_banner -loglevel warning -y \
    -framerate 10 -pattern_type glob -i "$FRAMES_DIR/voxroom_step_*.jpg" \
    -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
    -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p \
    -movflags +faststart "$RUN_ROOT/kujiale_0003_full_visualization.mp4"
else
  echo "[kujiale-0003-fullviz] ffmpeg is unavailable; frames remain in $FRAMES_DIR" >&2
fi

touch "$RUN_ROOT/.run_complete"
echo "[kujiale-0003-fullviz] complete: $RUN_ROOT"
