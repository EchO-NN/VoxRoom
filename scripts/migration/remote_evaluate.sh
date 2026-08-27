#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ISAAC_ROOT="${ISAAC_ROOT:-$HOME/isaac-sim-standalone-5.1.0-linux-x86_64}"
INTERIOR_ROOT="${INTERIOR_ROOT:-$HOME/InteriorAgent}"
GRSCENE_ROOT="${GRSCENE_ROOT:-$HOME/GRScenes-100}"
ENV_NAME="${VOXROOM_ENV_NAME:-voxroom-online}"
MAX_CONTROL_STEPS="${MAX_CONTROL_STEPS:-6000}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/outputs/remote_migration_eval_$RUN_TAG}"
CHECKPOINT="$REPO_ROOT/outputs/door_seed_training_runs/interioragent_raw_occ6_20260717_153144/model_vertical/best.pt"
EXPECTED_CHECKPOINT_SHA256="de0397beb366c83839f11dbd1cf42395a5214004b0de62965d4fcab953ed1f20"
LIVE_VISUALIZATION="${LIVE_VISUALIZATION:-1}"
LIVE_VISUALIZATION_TIMEOUT_S="${LIVE_VISUALIZATION_TIMEOUT_S:-120}"
VOXROOM_WINDOW_NAME="${VOXROOM_WINDOW_NAME:-VoxRoom Isaac Debug}"

configure_live_visualization() {
  if [[ "$LIVE_VISUALIZATION" == "0" ]]; then
    echo "[remote-eval] live visualization disabled explicitly"
    return
  fi
  if [[ "$LIVE_VISUALIZATION" != "1" ]]; then
    echo "[remote-eval] LIVE_VISUALIZATION must be 0 or 1" >&2
    exit 1
  fi

  local runtime_dir="/run/user/$(id -u)"
  local -a display_sockets=()
  local display_socket
  shopt -s nullglob
  display_sockets=(/tmp/.X11-unix/X*)
  shopt -u nullglob
  if [[ -n "${VOXROOM_REMOTE_DISPLAY:-}" ]]; then
    export DISPLAY="$VOXROOM_REMOTE_DISPLAY"
  elif [[ "${#display_sockets[@]}" -eq 1 ]]; then
    display_socket="${display_sockets[0]}"
    export DISPLAY=":${display_socket##*/X}"
  else
    echo "[remote-eval] expected one local X11 display; set VOXROOM_REMOTE_DISPLAY" >&2
    exit 1
  fi

  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-$runtime_dir}"
  export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=$runtime_dir/bus}"
  export XAUTHORITY="${XAUTHORITY:-$runtime_dir/gdm/Xauthority}"
  if [[ ! -r "$XAUTHORITY" ]]; then
    echo "[remote-eval] XAuthority is not readable: $XAUTHORITY" >&2
    exit 1
  fi
  if ! command -v xwininfo >/dev/null 2>&1; then
    echo "[remote-eval] xwininfo is required to verify the live popup" >&2
    exit 1
  fi
  if ! xwininfo -root >/dev/null 2>&1; then
    echo "[remote-eval] cannot access remote desktop DISPLAY=$DISPLAY" >&2
    exit 1
  fi
  echo "[remote-eval] live visualization desktop ready: DISPLAY=$DISPLAY"
}

if [[ ! -f "$ISAAC_ROOT/setup_conda_env.sh" ]]; then
  echo "[remote-eval] invalid ISAAC_ROOT: $ISAAC_ROOT" >&2
  exit 1
fi
if [[ ! -d "$INTERIOR_ROOT/kujiale_0003" ]]; then
  echo "[remote-eval] invalid InteriorAgent root: $INTERIOR_ROOT" >&2
  exit 1
fi
if [[ ! -d "$GRSCENE_ROOT/home_scenes/scenes" ]]; then
  echo "[remote-eval] invalid GRScene root: $GRSCENE_ROOT" >&2
  exit 1
fi
if [[ "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" != "$EXPECTED_CHECKPOINT_SHA256" ]]; then
  echo "[remote-eval] checkpoint hash mismatch: $CHECKPOINT" >&2
  exit 1
fi

configure_live_visualization
mkdir -p "$RUN_ROOT"
{
  echo "run_tag=$RUN_TAG"
  echo "hostname=$(hostname)"
  echo "isaac_root=$ISAAC_ROOT"
  echo "interior_root=$INTERIOR_ROOT"
  echo "grscene_root=$GRSCENE_ROOT"
  echo "max_control_steps=$MAX_CONTROL_STEPS"
  echo "live_visualization=$LIVE_VISUALIZATION"
  echo "display=${DISPLAY:-}"
  echo "xauthority=${XAUTHORITY:-}"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
} > "$RUN_ROOT/environment.txt"

echo "[remote-eval] rebasing migrated metadata"
VOXROOM_ENV_NAME="$ENV_NAME" ISAAC_ROOT="$ISAAC_ROOT" \
  "$REPO_ROOT/scripts/run_voxroom_isaac_env.sh" \
  "$REPO_ROOT/scripts/migration/rebase_remote_paths.py" \
  --mapping "/home/echo/VoxRoom=$REPO_ROOT" \
  --mapping "/home/echo/InteriorAgent=$INTERIOR_ROOT" \
  --mapping "/home/echo/GRScenes-100=$GRSCENE_ROOT" \
  --rewrite-usd \
  | tee "$RUN_ROOT/rebase.log"

echo "[remote-eval] running focused migration tests"
VOXROOM_ENV_NAME="$ENV_NAME" ISAAC_ROOT="$ISAAC_ROOT" \
  "$REPO_ROOT/scripts/run_voxroom_isaac_env.sh" -m pytest -q \
  tests/test_voxel_nvblox_fast_dda_backend.py \
  tests/test_voxel_dda_occupancy_integration.py \
  tests/test_voxel_door_seed_semantics.py \
  tests/door_seed_learning \
  | tee "$RUN_ROOT/pytest.log"

INTERIOR_EPISODE="$REPO_ROOT/data/interioragent_episodes/radius005_all_scenes/kujiale_0003.jsonl"
GRSCENE_EPISODE="$REPO_ROOT/data/grscene_home_episodes/radius005_all_render_safe/kujiale_grscene_MV7J6NIKTKJZ2AABAAAAADI8_usd.jsonl"
if [[ ! -s "$INTERIOR_EPISODE" || ! -s "$GRSCENE_EPISODE" ]]; then
  echo "[remote-eval] representative episode files are missing" >&2
  exit 1
fi

run_episode() {
  local dataset_name="$1"
  local episode_file="$2"
  local output_dir="$RUN_ROOT/$dataset_name"
  local episode_pid
  local episode_status
  local live_visualization_confirmed=0
  local waited_s=0
  mkdir -p "$output_dir"
  echo "[remote-eval] running $dataset_name"
  (
    VOXROOM_ENV_NAME="$ENV_NAME" \
      VOXROOM_NUMBA_THREADS=18 \
      ISAAC_ROOT="$ISAAC_ROOT" \
      RUN_DIR="$output_dir" \
      SNAPSHOT_DIR="$output_dir/roomseg_snapshots" \
      VOXROOM_VIZ_SAVE_DIR="$output_dir/voxroom_viz_frames" \
      VOXROOM_VIZ_SAVE_EVERY_STEPS=100 \
      ROOMSEG_SNAPSHOT_MAX_SAVES=160 \
      MAX_CONTROL_STEPS="$MAX_CONTROL_STEPS" \
      "$REPO_ROOT/scripts/run_one_scene_random_frontier.sh" "$episode_file"
  ) > "$output_dir/run.log" 2>&1 &
  episode_pid=$!

  if [[ "$LIVE_VISUALIZATION" == "1" ]]; then
    while [[ "$waited_s" -lt "$LIVE_VISUALIZATION_TIMEOUT_S" ]]; do
      if xwininfo -root -tree 2>/dev/null | grep -Fq "$VOXROOM_WINDOW_NAME"; then
        live_visualization_confirmed=1
        break
      fi
      if ! kill -0 "$episode_pid" 2>/dev/null; then
        break
      fi
      sleep 1
      waited_s=$((waited_s + 1))
    done
  fi

  set +e
  wait "$episode_pid"
  episode_status=$?
  set -e
  if [[ "$episode_status" -ne 0 ]]; then
    echo "[remote-eval] $dataset_name exited with status $episode_status" >&2
    return "$episode_status"
  fi
  if [[ "$LIVE_VISUALIZATION" == "1" && "$live_visualization_confirmed" -ne 1 ]]; then
    echo "[remote-eval] live popup was not observed for $dataset_name" >&2
    exit 1
  fi
  {
    echo "window_name=$VOXROOM_WINDOW_NAME"
    echo "display=${DISPLAY:-}"
    echo "confirmed=$live_visualization_confirmed"
    echo "confirmation_wait_s=$waited_s"
  } > "$output_dir/live_visualization.txt"
  if [[ ! -s "$output_dir/results.jsonl" ]]; then
    echo "[remote-eval] missing result artifact for $dataset_name" >&2
    exit 1
  fi
}

run_episode "interioragent_kujiale_0003" "$INTERIOR_EPISODE"
run_episode "grscene_MV7J6NIKTKJZ2AABAAAAADI8" "$GRSCENE_EPISODE"

validation_args=("$RUN_ROOT" "$MAX_CONTROL_STEPS" "$CHECKPOINT")
if [[ "$LIVE_VISUALIZATION" == "1" ]]; then
  validation_args+=(--live-visualization-required)
fi
VOXROOM_ENV_NAME="$ENV_NAME" ISAAC_ROOT="$ISAAC_ROOT" \
  "$REPO_ROOT/scripts/run_voxroom_isaac_env.sh" \
  "$REPO_ROOT/scripts/migration/validate_remote_evaluation.py" \
  "${validation_args[@]}"

echo "[remote-eval] complete: $RUN_ROOT"
