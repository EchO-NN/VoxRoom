#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-configs/voxroom_online.yaml}"
DATASET_ROOT="${DATASET_ROOT:-data/InteriorAgent}"
PREPROCESSED_DIR="${PREPROCESSED_DIR:-data/interioragent_preprocessed_radius005}"
EPISODE_DIR="${EPISODE_DIR:-data/interioragent_episodes/radius005_all_scenes}"
RUN_ROOT="${RUN_ROOT:-result/radius005_robot005_random_frontier_all_scenes_collect_topology}"
SCENE_GLOB="${SCENE_GLOB:-kujiale_*}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"
RUN_NUMBA_NUM_THREADS="${RUN_NUMBA_NUM_THREADS:-7}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
FORCE_PREPROCESS="${FORCE_PREPROCESS:-0}"
FORCE_EPISODES="${FORCE_EPISODES:-0}"
GENERATE_ONLY="${GENERATE_ONLY:-0}"

mkdir -p "$EPISODE_DIR" "$RUN_ROOT"

if [[ "$FORCE_PREPROCESS" == "1" || ! -f "$PREPROCESSED_DIR/index.json" ]]; then
  echo "[voxroom-all-topology] preprocessing scenes -> $PREPROCESSED_DIR"
  scripts/run_voxroom_isaac_env.sh voxroom_online/isaac_runtime/scripts/preprocess_interioragent.py \
    --dataset-root "$DATASET_ROOT" \
    --scene-glob "$SCENE_GLOB" \
    --out "$PREPROCESSED_DIR" \
    --resolution 0.05 \
    --robot-radius-m "${ROBOT_RADIUS_M:-0.05}" \
    --inflation-radius-m 0.0
else
  echo "[voxroom-all-topology] reusing preprocessed scenes in $PREPROCESSED_DIR"
fi

if [[ "$FORCE_EPISODES" == "1" || ! -s "$EPISODE_DIR/episode_files.txt" ]]; then
  echo "[voxroom-all-topology] generating one exploration episode per scene"
  scripts/run_voxroom_isaac_env.sh voxroom_online/isaac_runtime/scripts/generate_all_scene_episodes_radius005.py \
    --preprocessed-dir "$PREPROCESSED_DIR" \
    --out-dir "$EPISODE_DIR" \
    --combined-out "$EPISODE_DIR/all_scenes_radius005.jsonl" \
    --report-out "$EPISODE_DIR/all_scenes_radius005_report.json" \
    --file-list-out "$EPISODE_DIR/episode_files.txt" \
    --debug-map-dir "$EPISODE_DIR/debug_maps" \
    --min-planning-clearance-m "${RUNTIME_PLANNING_CLEARANCE_M:-0.01}" \
    --robot-spawn-height-m 0.05
else
  echo "[voxroom-all-topology] reusing $(wc -l < "$EPISODE_DIR/episode_files.txt") episode files in $EPISODE_DIR"
fi

if [[ "$GENERATE_ONLY" == "1" ]]; then
  echo "[voxroom-all-topology] GENERATE_ONLY=1, not launching Isaac"
  exit 0
fi

run_scene() {
  local episode_file="$1"
  local run_index="$2"
  local scene_id
  scene_id="$(basename "$episode_file" .jsonl)"
  local scene_run_dir="$RUN_ROOT/$scene_id"
  mkdir -p "$scene_run_dir"
  echo "[voxroom-all-topology] ($run_index) running $scene_id threads=$RUN_NUMBA_NUM_THREADS"
  {
    echo "config=$CONFIG"
    echo "episode_file=$episode_file"
    echo "threads=$RUN_NUMBA_NUM_THREADS"
    echo "frontier_selection_mode=random"
    echo "frontier_source=voxel_vertical_free"
    echo "panorama_steps=${PANORAMA_STEPS:-12}"
  } > "$scene_run_dir/command.txt"
  if VOXROOM_NUMBA_THREADS="$RUN_NUMBA_NUM_THREADS" \
    VOXROOM_VOXEL_CPU_NUMBA_THREADS="$RUN_NUMBA_NUM_THREADS" \
    NUMBA_NUM_THREADS="$RUN_NUMBA_NUM_THREADS" \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    RUN_DIR="$scene_run_dir" SNAPSHOT_DIR="$scene_run_dir/roomseg_snapshots" CONFIG="$CONFIG" \
    scripts/run_collect_topology_baseline_random_frontier.sh "$episode_file" > "$scene_run_dir/run.log" 2>&1; then
    echo "[voxroom-all-topology] $scene_id finished"
    return 0
  fi
  local status=$?
  echo "[voxroom-all-topology] $scene_id failed with status $status" | tee -a "$scene_run_dir/run.log"
  return "$status"
}

if [[ ! -s "$EPISODE_DIR/episode_files.txt" ]]; then
  echo "[voxroom-all-topology] missing episode file list: $EPISODE_DIR/episode_files.txt" >&2
  exit 1
fi

echo "[voxroom-all-topology] launching runs under $RUN_ROOT jobs=$PARALLEL_JOBS threads_per_job=$RUN_NUMBA_NUM_THREADS"
run_index=0
active_jobs=0
failure_count=0
declare -A scheduled_scene_ids=()
while IFS= read -r episode_file; do
  [[ -z "$episode_file" ]] && continue
  scene_id="$(basename "$episode_file" .jsonl)"
  if [[ -n "${scheduled_scene_ids[$scene_id]+x}" ]]; then
    echo "[voxroom-all-topology] skipping duplicate scene: $scene_id"
    continue
  fi
  scheduled_scene_ids[$scene_id]=1
  run_index=$((run_index + 1))
  run_scene "$episode_file" "$run_index" &
  active_jobs=$((active_jobs + 1))
  if [[ "$active_jobs" -ge "$PARALLEL_JOBS" ]]; then
    if ! wait -n; then
      failure_count=$((failure_count + 1))
    fi
    active_jobs=$((active_jobs - 1))
    if [[ "$CONTINUE_ON_ERROR" != "1" && "$failure_count" -gt 0 ]]; then
      wait
      exit 1
    fi
  fi
done < "$EPISODE_DIR/episode_files.txt"

while [[ "$active_jobs" -gt 0 ]]; do
  if ! wait -n; then
    failure_count=$((failure_count + 1))
  fi
  active_jobs=$((active_jobs - 1))
done

if [[ "$failure_count" -gt 0 ]]; then
  echo "[voxroom-all-topology] done with $failure_count failed scene(s). Runs are under $RUN_ROOT"
  [[ "$CONTINUE_ON_ERROR" == "1" ]] && exit 0
  exit 1
fi

echo "[voxroom-all-topology] done. Runs are under $RUN_ROOT"

