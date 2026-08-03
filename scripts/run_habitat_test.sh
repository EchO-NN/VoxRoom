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
TASK_CONFIG="${TASK_CONFIG:-tasks/pointnav_habitat_test.yaml}"
SPLIT="${SPLIT:-train}"
REQUIRE_TOPOLOGY_TRANSITION="${REQUIRE_TOPOLOGY_TRANSITION:-0}"
PAD_EPISODE_TO_MAX_STEPS="${PAD_EPISODE_TO_MAX_STEPS:-1}"
ALLOW_EARLY_COMPLETION="${ALLOW_EARLY_COMPLETION:-0}"
RUN_CONTEXT_REQUIRED="${RUN_CONTEXT_REQUIRED:-0}"
RUN_CONTEXT_MANIFEST="${RUN_CONTEXT_MANIFEST:-}"
RUN_CONTEXT_DATASET="${RUN_CONTEXT_DATASET:-}"
WINDOW_CHECKPOINT_EVERY_STEPS="${WINDOW_CHECKPOINT_EVERY_STEPS:-100}"
VISUALIZATION_FRAME_EVERY_STEPS="${VISUALIZATION_FRAME_EVERY_STEPS:-5}"
VISUALIZATION_REFRESH_SECONDS="${VISUALIZATION_REFRESH_SECONDS:-0.01}"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-900}"
STALL_TIMEOUT_SECONDS="${STALL_TIMEOUT_SECONDS:-300}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-$((MAX_EPISODE_STEPS * 30 + 900))}"
VOXROOM_SIDECAR="${VOXROOM_SIDECAR:-0}"
VOXROOM_ROOT="${VOXROOM_ROOT:-}"
VOXROOM_CONFIG="${VOXROOM_CONFIG:-}"
VOXROOM_MAP_SIZE_M="${VOXROOM_MAP_SIZE_M:-48.0}"
VOXROOM_ROOMSEG_EVERY_STEPS="${VOXROOM_ROOMSEG_EVERY_STEPS:-50}"
VOXROOM_VISUALIZATION_EVERY_STEPS="${VOXROOM_VISUALIZATION_EVERY_STEPS:-5}"
VOXROOM_RESPONSE_TIMEOUT_SECONDS="${VOXROOM_RESPONSE_TIMEOUT_SECONDS:-300}"
ROOMSEG_COVERAGE_EVAL="${ROOMSEG_COVERAGE_EVAL:-0}"
ROOMSEG_COVERAGE_MILESTONES="${ROOMSEG_COVERAGE_MILESTONES:-20,40,60,80,100}"

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
if [[ "$REQUIRE_TOPOLOGY_TRANSITION" != "0" \
    && "$REQUIRE_TOPOLOGY_TRANSITION" != "1" ]]; then
    echo "REQUIRE_TOPOLOGY_TRANSITION must be 0 or 1" >&2
    exit 1
fi
if [[ "$PAD_EPISODE_TO_MAX_STEPS" != "0" \
    && "$PAD_EPISODE_TO_MAX_STEPS" != "1" ]]; then
    echo "PAD_EPISODE_TO_MAX_STEPS must be 0 or 1" >&2
    exit 1
fi
if [[ "$ALLOW_EARLY_COMPLETION" != "0" \
    && "$ALLOW_EARLY_COMPLETION" != "1" ]]; then
    echo "ALLOW_EARLY_COMPLETION must be 0 or 1" >&2
    exit 1
fi
if [[ "$RUN_CONTEXT_REQUIRED" != "0" \
    && "$RUN_CONTEXT_REQUIRED" != "1" ]]; then
    echo "RUN_CONTEXT_REQUIRED must be 0 or 1" >&2
    exit 1
fi
if [[ "$VOXROOM_SIDECAR" != "0" && "$VOXROOM_SIDECAR" != "1" ]]; then
    echo "VOXROOM_SIDECAR must be 0 or 1" >&2
    exit 1
fi
if [[ "$ROOMSEG_COVERAGE_EVAL" != "0" \
    && "$ROOMSEG_COVERAGE_EVAL" != "1" ]]; then
    echo "ROOMSEG_COVERAGE_EVAL must be 0 or 1" >&2
    exit 1
fi
if [[ "$ROOMSEG_COVERAGE_EVAL" == "1" && "$VOXROOM_SIDECAR" != "1" ]]; then
    echo "ROOMSEG_COVERAGE_EVAL requires VOXROOM_SIDECAR=1" >&2
    exit 1
fi
if [[ "$VOXROOM_SIDECAR" == "1" ]]; then
    if [[ -z "$VOXROOM_ROOT" ]]; then
        echo "VOXROOM_ROOT is required when VOXROOM_SIDECAR=1" >&2
        exit 1
    fi
    VOXROOM_CONFIG="${VOXROOM_CONFIG:-$VOXROOM_ROOT/configs/voxroom_online.yaml}"
    if [[ ! -x "$VOXROOM_ROOT/scripts/run_voxroom_isaac_env.sh" ]]; then
        echo "VoxRoom launcher is missing: $VOXROOM_ROOT/scripts/run_voxroom_isaac_env.sh" >&2
        exit 1
    fi
    if [[ ! -f "$VOXROOM_CONFIG" ]]; then
        echo "VoxRoom config is missing: $VOXROOM_CONFIG" >&2
        exit 1
    fi
fi
if [[ ! "$WINDOW_CHECKPOINT_EVERY_STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "WINDOW_CHECKPOINT_EVERY_STEPS must be a positive integer" >&2
    exit 1
fi
if [[ "$PAD_EPISODE_TO_MAX_STEPS" == "$ALLOW_EARLY_COMPLETION" ]]; then
    echo "Padding and early-completion modes are inconsistent" >&2
    exit 1
fi
required_topology_transition=1
if [[ "$ROOMSEG_COVERAGE_EVAL" == "1" ]]; then
    required_topology_transition=0
fi
if [[ "$RUN_CONTEXT_REQUIRED" == "1" \
    && ("$TASK_CONFIG" != "tasks/pointnav_gibson_visual.yaml" \
        || "$SPLIT" != "val" \
        || "$REQUIRE_TOPOLOGY_TRANSITION" != "$required_topology_transition" \
        || "$PAD_EPISODE_TO_MAX_STEPS" != "0" \
        || "$ALLOW_EARLY_COMPLETION" != "1" \
        || "$VISUALIZATION_FRAME_EVERY_STEPS" != "5" \
        || "$WINDOW_CHECKPOINT_EVERY_STEPS" != "100") ]]; then
    echo "Run context requires the fixed strict Gibson contract" >&2
    exit 1
fi
if [[ "$RUN_CONTEXT_REQUIRED" == "1" \
    && (! -f "$RUN_CONTEXT_MANIFEST" || ! -f "$RUN_CONTEXT_DATASET") ]]; then
    echo "Required run context files are missing" >&2
    exit 1
fi
strict_stage_names=()
strict_stage_steps=()
strict_live_canvas_width=0
strict_live_canvas_height=0
if [[ "$RUN_CONTEXT_REQUIRED" == "1" ]]; then
    read -r strict_first_step strict_later_step strict_mid_step < <(
        PYTHONPATH="$ROOT_DIR" "$PYTHON" -c \
            'from run_context_contract import STRICT_VISUAL_CAPTURE_STEPS as s; print(*(step for _, step in s))'
    )
    read -r strict_live_canvas_width strict_live_canvas_height < <(
        PYTHONPATH="$ROOT_DIR" "$PYTHON" -c \
            'from run_context_contract import STRICT_LIVE_CANVAS_SIZE as s; print(*s)'
    )
    strict_stage_names=(first later mid)
    strict_stage_steps=(
        "$strict_first_step"
        "$strict_later_step"
        "$strict_mid_step"
    )
fi

uid="$(id -u)"
session_listing="$(
    timeout --signal=TERM --kill-after=5s 10s \
        loginctl list-sessions --no-legend
)"
mapfile -t physical_sessions < <(
    awk -v uid="$uid" '$2 == uid && $4 == "seat0" {print $1}' \
        <<<"$session_listing"
)
if [[ "${#physical_sessions[@]}" -ne 1 ]]; then
    echo "Expected one physical seat0 session, found ${#physical_sessions[@]}" >&2
    exit 1
fi
session_id="${physical_sessions[0]}"
session_remote="$(
    timeout 10s loginctl show-session "$session_id" -p Remote --value
)"
session_active="$(
    timeout 10s loginctl show-session "$session_id" -p Active --value
)"
session_type="$(
    timeout 10s loginctl show-session "$session_id" -p Type --value
)"
session_seat="$(
    timeout 10s loginctl show-session "$session_id" -p Seat --value
)"
session_vtnr="$(
    timeout 10s loginctl show-session "$session_id" -p VTNr --value
)"
if [[ "$session_remote" != "no" \
    || "$session_active" != "yes" \
    || "$session_type" != "x11" \
    || "$session_seat" != "seat0" \
    || ! "$session_vtnr" =~ ^[1-9][0-9]*$ ]]; then
    echo "The seat0 session is not an active local X11 session" >&2
    exit 1
fi

export XDG_RUNTIME_DIR="/run/user/$uid"
export DBUS_SESSION_BUS_ADDRESS="unix:path=$XDG_RUNTIME_DIR/bus"
desktop_environment="$(
    timeout --signal=TERM --kill-after=5s 10s \
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
if [[ ! "$DISPLAY" =~ ^:[0-9]+$ || -z "$XAUTHORITY" ]]; then
    echo "The physical X11 display environment is incomplete" >&2
    exit 1
fi
x11_socket="/tmp/.X11-unix/X${DISPLAY#:}"
if [[ ! -S "$x11_socket" || ! -r "$XAUTHORITY" ]]; then
    echo "The physical X11 socket or authority file is unavailable" >&2
    exit 1
fi
xserver_pid="$(
    timeout --signal=TERM --kill-after=5s 10s \
        fuser "$x11_socket" 2>/dev/null |
        xargs
)"
if [[ ! "$xserver_pid" =~ ^[1-9][0-9]*$ \
    || "$(<"/proc/$xserver_pid/comm")" != "Xorg" ]]; then
    echo "The display socket is not owned by one Xorg process" >&2
    exit 1
fi
session_control_group="$(
    timeout 10s systemctl show "session-$session_id.scope" \
        -p ControlGroup --value
)"
session_processes="/sys/fs/cgroup${session_control_group}/cgroup.procs"
if [[ ! -r "$session_processes" ]] \
    || ! grep -Fxq "$xserver_pid" "$session_processes"; then
    echo "The Xorg process is outside the selected seat0 session" >&2
    exit 1
fi
mapfile -d '' -t xserver_args <"/proc/$xserver_pid/cmdline"
xserver_auth=""
xserver_vt_confirmed=0
for ((argument_index = 0; argument_index < ${#xserver_args[@]}; argument_index++)); do
    if [[ "${xserver_args[$argument_index]}" == "-auth" ]]; then
        xserver_auth="${xserver_args[$((argument_index + 1))]:-}"
    fi
    if [[ "${xserver_args[$argument_index]}" == "vt$session_vtnr" ]]; then
        xserver_vt_confirmed=1
    fi
done
if [[ "$xserver_auth" != "$XAUTHORITY" || "$xserver_vt_confirmed" != "1" ]]; then
    echo "The X11 environment does not belong to the selected physical VT" >&2
    exit 1
fi

export MPLBACKEND="TkAgg"
export ACTIVE_ROOM_DETR_DIR="$DETR_DIR"
export ACTIVE_ROOM_DETECTOR_DEVICE="cuda"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

DISPLAY="$DISPLAY" XAUTHORITY="$XAUTHORITY" \
    timeout --signal=TERM --kill-after=5s 10s xwininfo -root >/dev/null
timeout --signal=TERM --kill-after=10s 60s "$PYTHON" -c \
    "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); torch.zeros(1, device='cuda')"
timeout --signal=TERM --kill-after=10s 180s \
    "$PYTHON" "$ROOT_DIR/scripts/validate_install.py" >/dev/null

mkdir -p "$RUN_DIR"
install -m 0444 "$ROOT_DIR/reproduction_install.json" \
    "$RUN_DIR/runtime_install.json"
run_context_args=()
if [[ "$RUN_CONTEXT_REQUIRED" == "1" ]]; then
    install -m 0444 "$RUN_CONTEXT_MANIFEST" "$RUN_DIR/input_manifest.json"
    install -m 0444 "$RUN_CONTEXT_DATASET" "$RUN_DIR/input_dataset.json.gz"
    run_context_args=(
        --run_context_manifest "$RUN_DIR/input_manifest.json"
        --run_context_dataset "$RUN_DIR/input_dataset.json.gz"
    )
fi
voxroom_args=(--voxroom_sidecar 0)
if [[ "$VOXROOM_SIDECAR" == "1" ]]; then
    voxroom_args=(
        --voxroom_sidecar 1
        --voxroom_root "$VOXROOM_ROOT"
        --voxroom_config "$VOXROOM_CONFIG"
        --voxroom_map_size_m "$VOXROOM_MAP_SIZE_M"
        --voxroom_roomseg_every_steps "$VOXROOM_ROOMSEG_EVERY_STEPS"
        --voxroom_visualization_every_steps "$VOXROOM_VISUALIZATION_EVERY_STEPS"
        --voxroom_response_timeout_seconds "$VOXROOM_RESPONSE_TIMEOUT_SECONDS"
        --roomseg_coverage_eval "$ROOMSEG_COVERAGE_EVAL"
        --roomseg_coverage_milestones "$ROOMSEG_COVERAGE_MILESTONES"
    )
fi
command=(
    "$PYTHON" -u "$ROOT_DIR/explorable_with_door_detection.py"
    --task_config "$TASK_CONFIG"
    --split "$SPLIT"
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
    --visualization_frame_every_steps "$VISUALIZATION_FRAME_EVERY_STEPS"
    --visualization_refresh_seconds "$VISUALIZATION_REFRESH_SECONDS"
    --window_checkpoint_every_steps "$WINDOW_CHECKPOINT_EVERY_STEPS"
    --require_topology_transition "$REQUIRE_TOPOLOGY_TRANSITION"
    --pad_episode_to_max_steps "$PAD_EPISODE_TO_MAX_STEPS"
    --detector_device cuda
    --detr_source_dir "$DETR_DIR"
    --run_id "$RUN_ID"
    --window_title "$WINDOW_TITLE"
    --run_dir "$RUN_DIR"
    --dump_location "$RUN_DIR"
    --exp_name native
    "${run_context_args[@]}"
    "${voxroom_args[@]}"
)
printf '%q ' "${command[@]}" >"$RUN_DIR/command.txt"
printf '\n' >>"$RUN_DIR/command.txt"

cleanup() {
    if [[ -n "${run_pid:-}" ]] && kill -0 -- "-$run_pid" 2>/dev/null; then
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

verify_physical_session() {
    local current_socket_pid
    [[ "$(
        timeout 10s loginctl show-session "$session_id" -p Active --value
    )" == "yes" ]] || return 1
    [[ "$(
        timeout 10s loginctl show-session "$session_id" -p Remote --value
    )" == "no" ]] || return 1
    [[ "$(
        timeout 10s loginctl show-session "$session_id" -p Type --value
    )" == "x11" ]] || return 1
    [[ "$(
        timeout 10s loginctl show-session "$session_id" -p Seat --value
    )" == "seat0" ]] || return 1
    current_socket_pid="$(
        timeout --signal=TERM --kill-after=5s 10s \
            fuser "$x11_socket" 2>/dev/null |
            xargs
    )"
    [[ "$current_socket_pid" == "$xserver_pid" ]] || return 1
    grep -Fxq "$xserver_pid" "$session_processes" || return 1
    physical_session_checks=$((physical_session_checks + 1))
    last_physical_session_check_unix="$(date +%s.%N)"
}

verify_live_window() {
    local candidate_id="$1"
    local candidate_height
    local candidate_pid
    local candidate_width
    local window_info
    verify_physical_session || return 1
    candidate_pid="$(
        DISPLAY="$DISPLAY" XAUTHORITY="$XAUTHORITY" \
            timeout --signal=TERM --kill-after=2s 5s \
            xprop -id "$candidate_id" _NET_WM_PID 2>/dev/null |
            awk -F' = ' 'NF == 2 {print $2}'
    )"
    [[ "$candidate_pid" == "$run_pid" ]] || return 1
    DISPLAY="$DISPLAY" XAUTHORITY="$XAUTHORITY" \
        timeout --signal=TERM --kill-after=2s 5s \
        xprop -id "$candidate_id" WM_NAME 2>/dev/null |
        grep -Fq "\"$WINDOW_TITLE\"" || return 1
    window_info="$(
        DISPLAY="$DISPLAY" XAUTHORITY="$XAUTHORITY" \
        timeout --signal=TERM --kill-after=2s 5s \
            xwininfo -id "$candidate_id" 2>/dev/null
    )" || return 1
    grep -Fq "Map State: IsViewable" <<<"$window_info" || return 1
    candidate_width="$(
        awk '$1 == "Width:" {print $2; exit}' <<<"$window_info"
    )"
    candidate_height="$(
        awk '$1 == "Height:" {print $2; exit}' <<<"$window_info"
    )"
    [[ "$candidate_width" =~ ^[1-9][0-9]*$ \
        && "$candidate_height" =~ ^[1-9][0-9]*$ ]] || return 1
    if [[ "$RUN_CONTEXT_REQUIRED" == "1" ]]; then
        if ((candidate_width < strict_live_canvas_width \
            || candidate_height < strict_live_canvas_height)); then
            return 1
        fi
        if ((window_width == 0)); then
            window_width="$candidate_width"
            window_height="$candidate_height"
        elif ((candidate_width != window_width \
            || candidate_height != window_height)); then
            return 1
        fi
    elif ((window_width == 0)); then
        window_width="$candidate_width"
        window_height="$candidate_height"
    fi
    window_viewable_checks=$((window_viewable_checks + 1))
    last_window_viewable_unix="$(date +%s.%N)"
    window_geometry_checks=$((window_geometry_checks + 1))
    last_window_geometry_unix="$last_window_viewable_unix"
}

window_id=""
window_width=0
window_height=0
window_geometry_checks=0
last_window_geometry_unix=0
physical_session_checks=0
last_physical_session_check_unix=0
window_viewable_checks=0
last_window_viewable_unix=0
checkpoint_steps=""
checkpoint_count=0
next_checkpoint="$WINDOW_CHECKPOINT_EVERY_STEPS"
stage_capture_index=0
first_step=0
later_step=0
mid_step=0
terminal_captured=0
terminal_step=0

capture_ready_checkpoint() {
    local checkpoint_label
    local checkpoint_ready
    local current_progress
    local ready_run_id
    local ready_process_id
    local ready_frame_file
    local ready_frame_sha256
    local ready_render_step
    local ready_window_id
    local ready_step
    [[ -s "$RUN_DIR/progress.jsonl" ]] || return 0
    checkpoint_label="$(printf '%06d' "$next_checkpoint")"
    checkpoint_ready="$RUN_DIR/window_checkpoints/ready_$checkpoint_label.json"
    [[ -s "$checkpoint_ready" ]] || return 0
    current_progress="$(wc -l <"$RUN_DIR/progress.jsonl")"
    read -r ready_run_id ready_process_id ready_window_id ready_step \
        ready_render_step ready_frame_file ready_frame_sha256 < <(
        timeout --signal=TERM --kill-after=2s 10s \
            "$PYTHON" -c \
            'import json,sys; d=json.load(open(sys.argv[1])); print(d["run_id"], d["process_id"], d["window_id"], d["step"], d["render_step"], d["frame_file"], d["frame_sha256"])' \
            "$checkpoint_ready"
    )
    if [[ "$ready_run_id" != "$RUN_ID" \
        || "$ready_process_id" != "$run_pid" \
        || "$ready_window_id" != "$window_id" \
        || "$ready_step" != "$next_checkpoint" \
        || "$ready_render_step" != "$next_checkpoint" \
        || "$ready_frame_file" \
            != "visualization_frames/frame_$checkpoint_label.png" \
        || ! "$ready_frame_sha256" =~ ^[0-9a-f]{64}$ \
        || "$current_progress" != "$next_checkpoint" ]]; then
        echo "Periodic capture handshake identity mismatch" >&2
        return 1
    fi
    verify_live_window "$window_id"
    DISPLAY="$DISPLAY" XAUTHORITY="$XAUTHORITY" \
        timeout --signal=TERM --kill-after=5s 30s \
        "$PYTHON" "$ROOT_DIR/scripts/capture_x11.py" \
        --window-id "$window_id" \
        --window-output \
        "$RUN_DIR/window_checkpoints/step_$checkpoint_label.png" \
        --receipt-output \
        "$RUN_DIR/window_checkpoints/receipt_$checkpoint_label.json" \
        --run-id "$RUN_ID" \
        --process-id "$run_pid" \
        --capture-label "checkpoint_$checkpoint_label" \
        --step "$next_checkpoint" \
        --render-step "$ready_render_step" \
        --frame-path "$RUN_DIR/$ready_frame_file" \
        --frame-file "$ready_frame_file" \
        --frame-sha256 "$ready_frame_sha256" \
        --window-file "window_checkpoints/step_$checkpoint_label.png"
    printf '%s\n' "$RUN_ID" \
        >"$RUN_DIR/window_checkpoints/.ack_$checkpoint_label.tmp"
    mv "$RUN_DIR/window_checkpoints/.ack_$checkpoint_label.tmp" \
        "$RUN_DIR/window_checkpoints/ack_$checkpoint_label.txt"
    if [[ -n "$checkpoint_steps" ]]; then
        checkpoint_steps+=","
    fi
    checkpoint_steps+="$next_checkpoint"
    checkpoint_count=$((checkpoint_count + 1))
    next_checkpoint=$((next_checkpoint + WINDOW_CHECKPOINT_EVERY_STEPS))
}

capture_ready_stage() {
    local ack_path
    local capture_args
    local current_progress
    local expected_stage
    local expected_step
    local ready_path
    local ready_process_id
    local ready_frame_file
    local ready_frame_sha256
    local ready_render_step
    local ready_run_id
    local ready_stage
    local ready_step
    local ready_window_id
    [[ "$RUN_CONTEXT_REQUIRED" == "1" ]] || return 0
    ((${#strict_stage_names[@]} == ${#strict_stage_steps[@]})) || return 1
    ((stage_capture_index < ${#strict_stage_names[@]})) || return 0
    expected_stage="${strict_stage_names[$stage_capture_index]}"
    expected_step="${strict_stage_steps[$stage_capture_index]}"
    ready_path="$RUN_DIR/window_stages/ready_$expected_stage.json"
    ack_path="$RUN_DIR/window_stages/ack_$expected_stage.txt"
    current_progress="$(wc -l <"$RUN_DIR/progress.jsonl")"
    if [[ ! -s "$ready_path" ]]; then
        if ((current_progress > expected_step)); then
            echo "Application advanced past a required rendered-stage capture" >&2
            return 1
        fi
        return 0
    fi
    read -r ready_run_id ready_process_id ready_window_id ready_stage \
        ready_step ready_render_step ready_frame_file \
        ready_frame_sha256 < <(
        timeout --signal=TERM --kill-after=2s 10s \
            "$PYTHON" -c \
            'import json,sys; d=json.load(open(sys.argv[1])); print(d["run_id"], d["process_id"], d["window_id"], d["stage"], d["step"], d["render_step"], d["frame_file"], d["frame_sha256"])' \
            "$ready_path"
    )
    if [[ "$ready_run_id" != "$RUN_ID" \
        || "$ready_process_id" != "$run_pid" \
        || "$ready_window_id" != "$window_id" \
        || "$ready_stage" != "$expected_stage" \
        || "$ready_step" != "$expected_step" \
        || "$ready_render_step" != "$expected_step" \
        || "$ready_frame_file" \
            != "visualization_frames/frame_$(printf '%06d' "$expected_step").png" \
        || ! "$ready_frame_sha256" =~ ^[0-9a-f]{64}$ \
        || "$current_progress" != "$expected_step" ]]; then
        echo "Rendered-stage capture handshake identity mismatch" >&2
        return 1
    fi
    verify_live_window "$window_id"
    capture_args=(
        "$PYTHON" "$ROOT_DIR/scripts/capture_x11.py"
        --window-id "$window_id"
        --window-output "$RUN_DIR/window_$expected_stage.png"
        --receipt-output \
        "$RUN_DIR/window_stages/receipt_$expected_stage.json"
        --run-id "$RUN_ID"
        --process-id "$run_pid"
        --capture-label "$expected_stage"
        --step "$expected_step"
        --render-step "$ready_render_step"
        --frame-path "$RUN_DIR/$ready_frame_file"
        --frame-file "$ready_frame_file"
        --frame-sha256 "$ready_frame_sha256"
        --window-file "window_$expected_stage.png"
    )
    if [[ "$expected_stage" == "first" ]]; then
        capture_args+=(--desktop-output "$RUN_DIR/live_desktop.png")
    fi
    DISPLAY="$DISPLAY" XAUTHORITY="$XAUTHORITY" \
        timeout --signal=TERM --kill-after=5s 30s "${capture_args[@]}"
    printf '%s\n' "$RUN_ID" >"$RUN_DIR/window_stages/.ack_$expected_stage.tmp"
    mv "$RUN_DIR/window_stages/.ack_$expected_stage.tmp" "$ack_path"
    printf -v "${expected_stage}_step" '%s' "$expected_step"
    stage_capture_index=$((stage_capture_index + 1))
}

capture_terminal_if_ready() {
    local current_progress
    local ready_completion_reason
    local ready_frame_file
    local ready_frame_sha256
    local ready_run_id
    local ready_process_id
    local ready_render_step
    local ready_window_id
    local ready_step
    [[ "$terminal_captured" == "0" ]] || return 0
    [[ -s "$RUN_DIR/terminal_capture_ready.json" ]] || return 0
    if [[ -z "$window_id" || ! -s "$RUN_DIR/progress.jsonl" ]]; then
        echo "Terminal capture request appeared before live-window initialization" >&2
        return 1
    fi
    current_progress="$(wc -l <"$RUN_DIR/progress.jsonl")"
    read -r ready_run_id ready_process_id ready_window_id ready_step \
        ready_render_step ready_frame_file ready_frame_sha256 \
        ready_completion_reason < <(
        timeout --signal=TERM --kill-after=2s 10s \
            "$PYTHON" -c \
            'import json,sys; d=json.load(open(sys.argv[1])); print(d["run_id"], d["process_id"], d["window_id"], d["step"], d["render_step"], d["frame_file"], d["frame_sha256"], d["completion_reason"])' \
            "$RUN_DIR/terminal_capture_ready.json"
    )
    if [[ "$ready_run_id" != "$RUN_ID" \
        || "$ready_process_id" != "$run_pid" \
        || "$ready_window_id" != "$window_id" \
        || "$ready_step" != "$current_progress" \
        || "$ready_render_step" != "$current_progress" \
        || "$ready_frame_file" != "visualization_final.png" \
        || ! "$ready_frame_sha256" =~ ^[0-9a-f]{64}$ ]]; then
        echo "Terminal capture handshake identity mismatch" >&2
        return 1
    fi
    if [[ "$ROOMSEG_COVERAGE_EVAL" == "1" ]]; then
        if [[ "$ready_completion_reason" != "coverage_episode_step_limit_reached" \
            && "$ready_completion_reason" != "coverage_exploration_completed" ]]; then
            echo "Terminal capture has the wrong coverage completion reason" >&2
            return 1
        fi
    elif [[ "$ready_completion_reason" != "topology_exploration_completed" ]]; then
        echo "Terminal capture has the wrong topology completion reason" >&2
        return 1
    fi
    verify_live_window "$window_id"
    DISPLAY="$DISPLAY" XAUTHORITY="$XAUTHORITY" \
        timeout --signal=TERM --kill-after=5s 30s \
        "$PYTHON" "$ROOT_DIR/scripts/capture_x11.py" \
        --window-id "$window_id" \
        --window-output "$RUN_DIR/window_terminal.png" \
        --desktop-output "$RUN_DIR/live_desktop_terminal.png" \
        --receipt-output "$RUN_DIR/terminal_capture_receipt.json" \
        --run-id "$RUN_ID" \
        --process-id "$run_pid" \
        --capture-label terminal \
        --step "$current_progress" \
        --render-step "$ready_render_step" \
        --frame-path "$RUN_DIR/$ready_frame_file" \
        --frame-file "$ready_frame_file" \
        --frame-sha256 "$ready_frame_sha256" \
        --window-file "window_terminal.png"
    printf '%s\n' "$RUN_ID" >"$RUN_DIR/.terminal_capture_ack.tmp"
    mv "$RUN_DIR/.terminal_capture_ack.tmp" \
        "$RUN_DIR/terminal_capture_ack.txt"
    terminal_step="$current_progress"
    terminal_captured=1
}

startup_deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
while ((SECONDS < startup_deadline)); do
    if ! kill -0 "$run_pid" 2>/dev/null; then
        break
    fi
    if [[ -z "$window_id" && -s "$RUN_DIR/run_metadata.json" ]]; then
        window_id="$(
            timeout --signal=TERM --kill-after=2s 10s \
                "$PYTHON" -c \
                'import json,sys; print(json.load(open(sys.argv[1]))["x11_client_window_id"])' \
                "$RUN_DIR/run_metadata.json"
        )"
    fi
    if [[ -n "$window_id" ]] \
        && verify_live_window "$window_id" \
        && [[ -s "$RUN_DIR/progress.jsonl" ]]; then
        break
    fi
    sleep 1
done
if [[ -z "$window_id" ]] || ! verify_live_window "$window_id"; then
    echo "The process-owned live X11 client window was not observed" >&2
    exit 1
fi
if [[ ! -s "$RUN_DIR/progress.jsonl" ]]; then
    echo "The live dashboard did not reach its first control step" >&2
    exit 1
fi

if [[ "$RUN_CONTEXT_REQUIRED" == "1" ]]; then
    current_step="$(wc -l <"$RUN_DIR/progress.jsonl")"
    last_progress="$current_step"
    last_progress_at="$SECONDS"
    run_deadline=$((SECONDS + RUN_TIMEOUT_SECONDS))
    while kill -0 "$run_pid" 2>/dev/null; do
        if ! verify_live_window "$window_id"; then
            if ! kill -0 "$run_pid" 2>/dev/null; then
                break
            fi
            echo "The process-owned live visualization window became unavailable" >&2
            exit 1
        fi
        capture_ready_stage
        capture_ready_checkpoint
        current_progress="$(wc -l <"$RUN_DIR/progress.jsonl")"
        if ((current_progress > last_progress)); then
            last_progress="$current_progress"
            last_progress_at="$SECONDS"
        fi
        capture_terminal_if_ready
        if ((SECONDS - last_progress_at > STALL_TIMEOUT_SECONDS)); then
            echo "Control progress stalled at step $last_progress" >&2
            exit 1
        fi
        if ((SECONDS > run_deadline)); then
            echo "Run timed out at step $last_progress" >&2
            exit 1
        fi
        sleep 1
    done
else
    first_refresh_target=$(($(wc -l <"$RUN_DIR/progress.jsonl") + 1))
    first_refresh_deadline=$((SECONDS + STALL_TIMEOUT_SECONDS))
    while ((SECONDS < first_refresh_deadline)); do
        if ! kill -0 "$run_pid" 2>/dev/null; then
            break
        fi
        if ! verify_live_window "$window_id"; then
            if ! kill -0 "$run_pid" 2>/dev/null; then
                break
            fi
            echo "The process-owned live visualization window became unavailable" >&2
            exit 1
        fi
        capture_ready_checkpoint
        first_step="$(wc -l <"$RUN_DIR/progress.jsonl")"
        if ((first_step >= first_refresh_target)); then
            break
        fi
        capture_terminal_if_ready
        if [[ "$terminal_captured" == "1" ]]; then
            break
        fi
        sleep 1
    done
    first_step="$(wc -l <"$RUN_DIR/progress.jsonl")"
    if ((first_step < first_refresh_target)); then
        if [[ "$terminal_captured" == "1" ]]; then
            echo "The episode completed before the first visual refresh" >&2
            exit 1
        fi
        echo "The live window did not complete its first visual refresh" >&2
        exit 1
    fi

    verify_live_window "$window_id"
    DISPLAY="$DISPLAY" XAUTHORITY="$XAUTHORITY" \
        timeout --signal=TERM --kill-after=5s 30s \
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
        if ! verify_live_window "$window_id"; then
            if ! kill -0 "$run_pid" 2>/dev/null; then
                break
            fi
            echo "The process-owned live visualization window became unavailable" >&2
            exit 1
        fi
        capture_ready_checkpoint
        current_step="$(wc -l <"$RUN_DIR/progress.jsonl")"
        if ((current_step >= later_target)); then
            break
        fi
        capture_terminal_if_ready
        if [[ "$terminal_captured" == "1" ]]; then
            break
        fi
        sleep 1
    done
    current_step="$(wc -l <"$RUN_DIR/progress.jsonl")"
    if ((current_step < later_target)); then
        if [[ "$terminal_captured" == "1" ]]; then
            echo "The episode completed before three visual control steps elapsed" >&2
            exit 1
        fi
        echo "The visualization did not advance by three control steps" >&2
        exit 1
    fi

    verify_live_window "$window_id"
    DISPLAY="$DISPLAY" XAUTHORITY="$XAUTHORITY" \
        timeout --signal=TERM --kill-after=5s 30s \
        "$PYTHON" "$ROOT_DIR/scripts/capture_x11.py" \
        --window-id "$window_id" \
        --window-output "$RUN_DIR/window_later.png"
    later_step="$current_step"

    last_progress="$current_step"
    last_progress_at="$SECONDS"
    mid_target=$((current_step + 20))
    run_deadline=$((SECONDS + RUN_TIMEOUT_SECONDS))
    while kill -0 "$run_pid" 2>/dev/null; do
        sleep 2
        if ! kill -0 "$run_pid" 2>/dev/null; then
            break
        fi
        if ! verify_live_window "$window_id"; then
            if ! kill -0 "$run_pid" 2>/dev/null; then
                break
            fi
            echo "The process-owned live visualization window became unavailable" >&2
            exit 1
        fi
        capture_ready_checkpoint
        current_progress="$(wc -l <"$RUN_DIR/progress.jsonl")"
        if ((current_progress > last_progress)); then
            last_progress="$current_progress"
            last_progress_at="$SECONDS"
        fi
        if ((mid_step == 0 && current_progress >= mid_target)); then
            DISPLAY="$DISPLAY" XAUTHORITY="$XAUTHORITY" \
                timeout --signal=TERM --kill-after=5s 30s \
                "$PYTHON" "$ROOT_DIR/scripts/capture_x11.py" \
                --window-id "$window_id" \
                --window-output "$RUN_DIR/window_mid.png"
            mid_step="$current_progress"
        fi
        capture_terminal_if_ready
        if ((SECONDS - last_progress_at > STALL_TIMEOUT_SECONDS)); then
            echo "Control progress stalled at step $last_progress" >&2
            exit 1
        fi
        if ((SECONDS > run_deadline)); then
            echo "Run timed out at step $last_progress" >&2
            exit 1
        fi
    done
fi
printf 'run_id=%s\nprocess_id=%s\ndisplay=%s\nsession_id=%s\nwindow_id=%s\nwindow_title=%s\nfirst_step=%s\nlater_step=%s\nmid_step=%s\nconfirmed=1\n' \
    "$RUN_ID" "$run_pid" "$DISPLAY" "$session_id" "$window_id" "$WINDOW_TITLE" \
    "$first_step" "$later_step" "$mid_step" >"$RUN_DIR/live_visualization.txt"
printf 'session_vtnr=%s\nxserver_pid=%s\nxauthority=%s\nx11_socket=%s\nphysical_session_confirmed=1\n' \
    "$session_vtnr" "$xserver_pid" "$XAUTHORITY" "$x11_socket" \
    >>"$RUN_DIR/live_visualization.txt"
printf 'window_viewable_checks=%s\nlast_window_viewable_unix=%s\n' \
    "$window_viewable_checks" "$last_window_viewable_unix" \
    >>"$RUN_DIR/live_visualization.txt"
printf 'window_width=%s\nwindow_height=%s\nwindow_locked=%s\nwindow_geometry_checks=%s\nlast_window_geometry_unix=%s\n' \
    "$window_width" "$window_height" "$RUN_CONTEXT_REQUIRED" \
    "$window_geometry_checks" "$last_window_geometry_unix" \
    >>"$RUN_DIR/live_visualization.txt"
printf 'physical_session_checks=%s\nlast_physical_session_check_unix=%s\n' \
    "$physical_session_checks" "$last_physical_session_check_unix" \
    >>"$RUN_DIR/live_visualization.txt"
printf 'checkpoint_every_steps=%s\ncheckpoint_count=%s\ncheckpoint_steps=%s\nterminal_captured=%s\nterminal_step=%s\n' \
    "$WINDOW_CHECKPOINT_EVERY_STEPS" "$checkpoint_count" "$checkpoint_steps" \
    "$terminal_captured" "$terminal_step" >>"$RUN_DIR/live_visualization.txt"

set +e
wait "$run_pid"
run_status=$?
set -e
process_group_exit_deadline=$((SECONDS + 10))
while kill -0 -- "-$run_pid" 2>/dev/null \
    && ((SECONDS < process_group_exit_deadline)); do
    sleep 1
done
if kill -0 -- "-$run_pid" 2>/dev/null; then
    echo "The runtime process group retained descendant processes" >&2
    exit 1
fi
run_pid=""
if [[ "$run_status" -ne 0 ]]; then
    echo "Active Room Segmentation exited with status $run_status" >&2
    exit "$run_status"
fi
if [[ "$RUN_CONTEXT_REQUIRED" == "1" && "$terminal_captured" != "1" ]]; then
    echo "Strict run exited without a terminal physical-window capture" >&2
    exit 1
fi
if [[ "$RUN_CONTEXT_REQUIRED" == "1" \
    && "$stage_capture_index" -ne "${#strict_stage_names[@]}" ]]; then
    echo "Strict run exited without all rendered-stage captures" >&2
    exit 1
fi

timeout --signal=TERM --kill-after=10s 300s \
    "$PYTHON" "$ROOT_DIR/scripts/render_replay.py" \
    --run-dir "$RUN_DIR"

validation_args=(
    "$PYTHON" "$ROOT_DIR/scripts/validate_run.py"
    --run-dir "$RUN_DIR" \
    --expected-steps "$MAX_EPISODE_STEPS" \
    --run-id "$RUN_ID"
)
if [[ "$REQUIRE_TOPOLOGY_TRANSITION" == "1" ]]; then
    validation_args+=(--require-topology-transition)
fi
if [[ "$ALLOW_EARLY_COMPLETION" == "1" ]]; then
    validation_args+=(--allow-early-completion)
fi
if [[ "$RUN_CONTEXT_REQUIRED" == "1" ]]; then
    validation_args+=(--require-run-context)
fi
if [[ "$ROOMSEG_COVERAGE_EVAL" == "1" ]]; then
    validation_args+=(--roomseg-coverage-eval)
fi
timeout --signal=TERM --kill-after=10s 300s "${validation_args[@]}"

trap - EXIT
echo "$RUN_DIR"
