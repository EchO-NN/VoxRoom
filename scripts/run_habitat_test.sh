#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_PREFIX="${TARGET_PREFIX:-$HOME/.conda/envs/active-room-seg}"
PYTHON="${PYTHON:-$TARGET_PREFIX/bin/python}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-120}"
RUN_ID="${RUN_ID:-$(< /proc/sys/kernel/random/uuid)}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_${RUN_ID:0:8}}"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/outputs/habitat_test_$RUN_TAG}"
DETR_DIR="$ROOT_DIR/third_party/detr"
WINDOW_TITLE="Active Room Segmentation [$RUN_ID]"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-900}"
STALL_TIMEOUT_SECONDS="${STALL_TIMEOUT_SECONDS:-300}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-$((MAX_EPISODE_STEPS * 30 + 900))}"

if [[ ! -x "$PYTHON" ]]; then
    echo "Python environment is missing: $PYTHON" >&2
    exit 1
fi
if [[ -e "$RUN_DIR" ]]; then
    echo "Run directory already exists: $RUN_DIR" >&2
    exit 1
fi
if [[ "${ACTIVE_ROOM_DETECTOR_DEVICE:-cuda}" != "cuda" ]]; then
    echo "The verified run requires ACTIVE_ROOM_DETECTOR_DEVICE=cuda" >&2
    exit 1
fi

uid="$(id -u)"
mapfile -t physical_sessions < <(
    loginctl list-sessions --no-legend |
        awk -v uid="$uid" '$2 == uid && $4 == "seat0" {print $1}'
)
if [[ "${#physical_sessions[@]}" -ne 1 ]]; then
    echo "Expected one physical seat0 session, found ${#physical_sessions[@]}" >&2
    exit 1
fi
session_id="${physical_sessions[0]}"
if [[ "$(loginctl show-session "$session_id" -p Remote --value)" != "no" ]] \
    || [[ "$(loginctl show-session "$session_id" -p Active --value)" != "yes" ]] \
    || [[ "$(loginctl show-session "$session_id" -p Type --value)" != "x11" ]]; then
    echo "The seat0 session is not an active local X11 session" >&2
    exit 1
fi

export XDG_RUNTIME_DIR="/run/user/$uid"
export DBUS_SESSION_BUS_ADDRESS="unix:path=$XDG_RUNTIME_DIR/bus"
desktop_environment="$(
    DBUS_SESSION_BUS_ADDRESS="$DBUS_SESSION_BUS_ADDRESS" \
        systemctl --user show-environment
)"
export DISPLAY="$(
    awk -F= '$1 == "DISPLAY" {sub(/^[^=]*=/, ""); print; exit}' \
        <<<"$desktop_environment"
)"
export XAUTHORITY="$(
    awk -F= '$1 == "XAUTHORITY" {sub(/^[^=]*=/, ""); print; exit}' \
        <<<"$desktop_environment"
)"
if [[ -z "$DISPLAY" || -z "$XAUTHORITY" || ! -S "/tmp/.X11-unix/X${DISPLAY#:}" ]]; then
    echo "The physical X11 display environment is incomplete" >&2
    exit 1
fi

export MPLBACKEND="TkAgg"
export ACTIVE_ROOM_DETR_DIR="$DETR_DIR"
export ACTIVE_ROOM_DETECTOR_DEVICE="cuda"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

DISPLAY="$DISPLAY" XAUTHORITY="$XAUTHORITY" xwininfo -root >/dev/null
"$PYTHON" -c \
    "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); torch.zeros(1, device='cuda')"

mkdir -p "$RUN_DIR"
command=(
    "$PYTHON" -u "$ROOT_DIR/explorable_with_door_detection.py"
    --task_config tasks/pointnav_habitat_test.yaml
    --split train
    --eval 1
    --num_episodes 1
    --max_episode_length "$MAX_EPISODE_STEPS"
    --auto_gpu_config 0
    --num_processes 1
    --num_processes_on_first_gpu 1
    --num_processes_per_gpu 1
    --train_global 0
    --train_local 0
    --train_slam 0
    --visualize 1
    --print_images 0
    --detector_device cuda
    --detr_source_dir "$DETR_DIR"
    --run_id "$RUN_ID"
    --window_title "$WINDOW_TITLE"
    --run_dir "$RUN_DIR"
    --dump_location "$RUN_DIR"
    --exp_name native
)
printf '%q ' "${command[@]}" >"$RUN_DIR/command.txt"
printf '\n' >>"$RUN_DIR/command.txt"

cleanup() {
    if [[ -n "${run_pid:-}" ]] && kill -0 "$run_pid" 2>/dev/null; then
        kill -TERM -- "-$run_pid" 2>/dev/null || true
        sleep 2
        kill -KILL -- "-$run_pid" 2>/dev/null || true
        wait "$run_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT

setsid bash -c 'cd "$1"; shift; exec "$@"' \
    _ "$ROOT_DIR" "${command[@]}" >"$RUN_DIR/runtime.log" 2>&1 &
run_pid=$!
printf '%s\n' "$run_pid" >"$RUN_DIR/process_id.txt"

window_id=""
startup_deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
while ((SECONDS < startup_deadline)); do
    if ! kill -0 "$run_pid" 2>/dev/null; then
        break
    fi
    if [[ -s "$RUN_DIR/progress.jsonl" ]]; then
        while read -r candidate_id; do
            if DISPLAY="$DISPLAY" XAUTHORITY="$XAUTHORITY" \
                xwininfo -id "$candidate_id" 2>/dev/null |
                    grep -Fq "Map State: IsViewable"; then
                window_id="$candidate_id"
                break
            fi
        done < <(
            DISPLAY="$DISPLAY" XAUTHORITY="$XAUTHORITY" \
                xwininfo -root -tree 2>/dev/null |
                awk -v title="\"$WINDOW_TITLE\"" 'index($0, title) {print $1}'
        )
    fi
    [[ -n "$window_id" ]] && break
    sleep 1
done
if [[ -z "$window_id" ]]; then
    echo "A viewable live window carrying run ID $RUN_ID was not observed" >&2
    exit 1
fi

first_refresh_target=$(($(wc -l <"$RUN_DIR/progress.jsonl") + 1))
first_refresh_deadline=$((SECONDS + STALL_TIMEOUT_SECONDS))
while ((SECONDS < first_refresh_deadline)); do
    if ! kill -0 "$run_pid" 2>/dev/null; then
        break
    fi
    first_step="$(wc -l <"$RUN_DIR/progress.jsonl")"
    ((first_step >= first_refresh_target)) && break
    sleep 1
done
first_step="$(wc -l <"$RUN_DIR/progress.jsonl")"
if ((first_step < first_refresh_target)); then
    echo "The live window did not complete its first visual refresh" >&2
    exit 1
fi

DISPLAY="$DISPLAY" XAUTHORITY="$XAUTHORITY" \
    "$PYTHON" "$ROOT_DIR/scripts/capture_x11.py" \
    --window-id "$window_id" \
    --window-output "$RUN_DIR/window_first.png" \
    --desktop-output "$RUN_DIR/live_desktop.png"

later_target=$((first_step + 3))
later_deadline=$((SECONDS + STALL_TIMEOUT_SECONDS))
while ((SECONDS < later_deadline)); do
    if ! kill -0 "$run_pid" 2>/dev/null; then
        break
    fi
    current_step="$(wc -l <"$RUN_DIR/progress.jsonl")"
    ((current_step >= later_target)) && break
    sleep 1
done
current_step="$(wc -l <"$RUN_DIR/progress.jsonl")"
if ((current_step < later_target)); then
    echo "The visualization did not advance by three control steps" >&2
    exit 1
fi

DISPLAY="$DISPLAY" XAUTHORITY="$XAUTHORITY" \
    "$PYTHON" "$ROOT_DIR/scripts/capture_x11.py" \
    --window-id "$window_id" \
    --window-output "$RUN_DIR/window_later.png"
printf 'run_id=%s\nprocess_id=%s\ndisplay=%s\nsession_id=%s\nwindow_id=%s\nwindow_title=%s\nfirst_step=%s\nlater_step=%s\nconfirmed=1\n' \
    "$RUN_ID" "$run_pid" "$DISPLAY" "$session_id" "$window_id" "$WINDOW_TITLE" \
    "$first_step" "$current_step" >"$RUN_DIR/live_visualization.txt"

last_progress="$current_step"
last_progress_at="$SECONDS"
run_deadline=$((SECONDS + RUN_TIMEOUT_SECONDS))
while kill -0 "$run_pid" 2>/dev/null; do
    sleep 2
    current_progress="$(wc -l <"$RUN_DIR/progress.jsonl")"
    if ((current_progress > last_progress)); then
        last_progress="$current_progress"
        last_progress_at="$SECONDS"
    fi
    if ((SECONDS - last_progress_at > STALL_TIMEOUT_SECONDS)); then
        echo "Control progress stalled at step $last_progress" >&2
        exit 1
    fi
    if ((SECONDS > run_deadline)); then
        echo "Run timed out at step $last_progress" >&2
        exit 1
    fi
done

set +e
wait "$run_pid"
run_status=$?
set -e
run_pid=""
if [[ "$run_status" -ne 0 ]]; then
    echo "Active Room Segmentation exited with status $run_status" >&2
    exit "$run_status"
fi

"$PYTHON" "$ROOT_DIR/scripts/validate_run.py" \
    --run-dir "$RUN_DIR" \
    --expected-steps "$MAX_EPISODE_STEPS" \
    --run-id "$RUN_ID"

trap - EXIT
echo "$RUN_DIR"
