#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EPISODE_FILE="${1:-${EPISODE_FILE:-}}"
if [[ -z "$EPISODE_FILE" || ! -f "$EPISODE_FILE" ]]; then
  echo "usage: $0 PATH/TO/episode.jsonl" >&2
  exit 1
fi

CONFIG="${CONFIG:-configs/voxroom_online.yaml}"
SCENE_ID="$(basename "$EPISODE_FILE" .jsonl)"
RUN_DIR="${RUN_DIR:-outputs/roomseg_coverage_eval_isaac/$SCENE_ID}"
MAX_CONTROL_STEPS="${MAX_CONTROL_STEPS:-5000}"
COVERAGE_MILESTONES="${COVERAGE_MILESTONES:-20,40,60,70,80,90}"
HEADLESS_FLAG="${HEADLESS_FLAG---headless}"
LIVE_BASELINE_POLICY_CONTROL="${LIVE_BASELINE_POLICY_CONTROL:-original_topology}"
LIVE_BASELINE_PANORAMA_VIEWS="${LIVE_BASELINE_PANORAMA_VIEWS:-12}"
NAVIGATION_PLANNER_BACKEND="${NAVIGATION_PLANNER_BACKEND:-astar}"
TRANSLATION_STEP_SCALE="${TRANSLATION_STEP_SCALE:-18.0}"
TRANSLATION_COMMAND_SCALE="${TRANSLATION_COMMAND_SCALE:-6.0}"
BASE_MAX_VX_MPS="${BASE_MAX_VX_MPS:-0.15}"
BASE_MAX_VY_MPS="${BASE_MAX_VY_MPS:-0.0}"
MAX_VX_MPS="${MAX_VX_MPS:-2.70}"
MAX_VY_MPS="${MAX_VY_MPS:-0.0}"
CONTROL_DT="${CONTROL_DT:-0.2}"
TURN_STEP_DEG="${TURN_STEP_DEG:-30.0}"
MAX_WZ_RADPS="${MAX_WZ_RADPS:-2.6179938779914944}"
FORWARD_HEADING_TOLERANCE_DEG="${FORWARD_HEADING_TOLERANCE_DEG:-75.0}"
SIMULATOR_COLLISION_REFERENCE_RADIUS_M="${SIMULATOR_COLLISION_REFERENCE_RADIUS_M:-0.05}"
ASTAR_CLEARANCE_DESIRED_M="${ASTAR_CLEARANCE_DESIRED_M:-0.20}"
ASTAR_CLEARANCE_HARD_MIN_M="${ASTAR_CLEARANCE_HARD_MIN_M:-0.05}"
LOOKAHEAD_MIN_CLEARANCE_M="${LOOKAHEAD_MIN_CLEARANCE_M:-$ASTAR_CLEARANCE_HARD_MIN_M}"
GUARD_MIN_CLEARANCE_M="${GUARD_MIN_CLEARANCE_M:-$ASTAR_CLEARANCE_HARD_MIN_M}"
ASTAR_CLEARANCE_WEIGHT="${ASTAR_CLEARANCE_WEIGHT:-6.0}"
REPLAN_EVERY_STEPS="${REPLAN_EVERY_STEPS:-10}"

if [[ "$LIVE_BASELINE_POLICY_CONTROL" != "original_topology" ]]; then
  echo "[roomseg-coverage-isaac] TVARS navigation requires original_topology control" >&2
  exit 1
fi
if [[ "$LIVE_BASELINE_PANORAMA_VIEWS" -ne 12 ]]; then
  echo "[roomseg-coverage-isaac] strict TVARS requires exactly 12 panorama views" >&2
  exit 1
fi
if [[ "$NAVIGATION_PLANNER_BACKEND" != "astar" ]]; then
  echo "[roomseg-coverage-isaac] this TVARS evaluation requires clearance A*" >&2
  exit 1
fi
python3 - "$TRANSLATION_STEP_SCALE" "$TRANSLATION_COMMAND_SCALE" "$BASE_MAX_VX_MPS" "$BASE_MAX_VY_MPS" "$MAX_VX_MPS" "$MAX_VY_MPS" "$CONTROL_DT" "$TURN_STEP_DEG" "$MAX_WZ_RADPS" "$FORWARD_HEADING_TOLERANCE_DEG" "$ASTAR_CLEARANCE_HARD_MIN_M" "$LOOKAHEAD_MIN_CLEARANCE_M" "$GUARD_MIN_CLEARANCE_M" <<'PY'
import math
import sys

(
    scale,
    command_scale,
    base_vx,
    base_vy,
    max_vx,
    max_vy,
    control_dt,
    turn_step_deg,
    max_wz,
    forward_tolerance_deg,
    astar_hard_min,
    lookahead_min,
    guard_min,
) = map(float, sys.argv[1:])
if not math.isclose(scale, 18.0, rel_tol=0.0, abs_tol=1.0e-9):
    raise SystemExit("strict batch requires TRANSLATION_STEP_SCALE=18.0")
if not math.isclose(command_scale, 6.0, rel_tol=0.0, abs_tol=1.0e-9):
    raise SystemExit("strict batch requires TRANSLATION_COMMAND_SCALE=6.0")
if not math.isclose(max_vx, base_vx * scale, rel_tol=0.0, abs_tol=1.0e-9):
    raise SystemExit("MAX_VX_MPS does not match the requested translation step scale")
if not math.isclose(max_vy, base_vy * scale, rel_tol=0.0, abs_tol=1.0e-9):
    raise SystemExit("MAX_VY_MPS does not match the requested translation step scale")
if not math.isclose(turn_step_deg, 30.0, rel_tol=0.0, abs_tol=1.0e-9):
    raise SystemExit("strict TVARS control requires TURN_STEP_DEG=30.0")
if not math.isclose(forward_tolerance_deg, 75.0, rel_tol=0.0, abs_tol=1.0e-9):
    raise SystemExit("forward-only arc control requires FORWARD_HEADING_TOLERANCE_DEG=75.0")
if not math.isclose(max_wz * control_dt, math.radians(turn_step_deg), rel_tol=0.0, abs_tol=1.0e-9):
    raise SystemExit("MAX_WZ_RADPS and CONTROL_DT do not produce one TVARS turn step")
if not math.isclose(lookahead_min, astar_hard_min, rel_tol=0.0, abs_tol=1.0e-9):
    raise SystemExit("LOOKAHEAD_MIN_CLEARANCE_M must equal ASTAR_CLEARANCE_HARD_MIN_M")
if not math.isclose(guard_min, astar_hard_min, rel_tol=0.0, abs_tol=1.0e-9):
    raise SystemExit("GUARD_MIN_CLEARANCE_M must equal ASTAR_CLEARANCE_HARD_MIN_M")
PY

export ACTIVE_ROOM_SEG_ROOT="${ACTIVE_ROOM_SEG_ROOT:-$HOME/Active_room_segmentation}"
export ACTIVE_ROOM_SEG_EXPECTED_COMMIT="${ACTIVE_ROOM_SEG_EXPECTED_COMMIT:-c6dbe92c55ea34f9710ddcc5b10d59144662fe68}"
export TOPOLOGY_DOOR_DETR_PYTHON="${TOPOLOGY_DOOR_DETR_PYTHON:-$HOME/.conda/envs/active-room-seg/bin/python}"
export TOPOLOGY_DOOR_DETR_CHECKPOINT="${TOPOLOGY_DOOR_DETR_CHECKPOINT:-$ACTIVE_ROOM_SEG_ROOT/detr_door_detection/train_params/detr_resnet50_4/final_doors_dataset/model.pth}"

if [[ ! -d "$ACTIVE_ROOM_SEG_ROOT/.git" ]]; then
  echo "[roomseg-coverage-isaac] Active Room checkout missing: $ACTIVE_ROOM_SEG_ROOT" >&2
  exit 1
fi
actual_commit="$(git -C "$ACTIVE_ROOM_SEG_ROOT" rev-parse HEAD)"
if [[ "$actual_commit" != "$ACTIVE_ROOM_SEG_EXPECTED_COMMIT" ]]; then
  echo "[roomseg-coverage-isaac] source commit mismatch: expected=$ACTIVE_ROOM_SEG_EXPECTED_COMMIT actual=$actual_commit" >&2
  exit 1
fi
if [[ -n "$(git -C "$ACTIVE_ROOM_SEG_ROOT" status --porcelain)" ]]; then
  echo "[roomseg-coverage-isaac] Active Room checkout must be clean" >&2
  exit 1
fi
if [[ ! -x "$TOPOLOGY_DOOR_DETR_PYTHON" ]]; then
  echo "[roomseg-coverage-isaac] DETR Python is not executable: $TOPOLOGY_DOOR_DETR_PYTHON" >&2
  exit 1
fi
if [[ ! -f "$TOPOLOGY_DOOR_DETR_CHECKPOINT" ]]; then
  echo "[roomseg-coverage-isaac] DETR checkpoint missing: $TOPOLOGY_DOOR_DETR_CHECKPOINT" >&2
  exit 1
fi
if [[ -e "$RUN_DIR/roomseg_coverage_eval" ]]; then
  echo "[roomseg-coverage-isaac] output already contains a coverage run: $RUN_DIR" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"
echo "[roomseg-coverage-isaac] scene=$SCENE_ID output=$RUN_DIR milestones=$COVERAGE_MILESTONES"

VOXROOM_NUMBA_THREADS="${VOXROOM_NUMBA_THREADS:-28}" \
  scripts/run_voxroom_isaac_env.sh voxroom_online/isaac_runtime/scripts/run_one_episode.py \
  --config "$CONFIG" \
  --episode-file "$EPISODE_FILE" \
  --episode-index 0 \
  --sim-backend isaac \
  --planner "$NAVIGATION_PLANNER_BACKEND" \
  --detector none \
  --segmenter none \
  --no-llm-enabled \
  --no-vllm-frontier-scoring \
  --no-vllm-image-scoring \
  --no-allow-gt-goal-fallback \
  --no-frontier-allow-near-fallback \
  --frontier-selection-mode tvars_original \
  --frontier-source voxel_vertical_free \
  --room-map-mode "${ROOM_MAP_MODE:-voxel_occupancy_door_wall_v33}" \
  --roomseg-backend "${ROOMSEG_BACKEND:-voxel_occupancy_door_wall_v33}" \
  --no-strict-benchmark \
  --explore-until-no-frontiers \
  --max-control-steps "$MAX_CONTROL_STEPS" \
  --max-vx-mps "$MAX_VX_MPS" \
  --max-vy-mps "$MAX_VY_MPS" \
  --control-dt "$CONTROL_DT" \
  --max-wz-radps "$MAX_WZ_RADPS" \
  --forward-heading-tolerance-deg "$FORWARD_HEADING_TOLERANCE_DEG" \
  --translation-command-scale "$TRANSLATION_COMMAND_SCALE" \
  --replan-every-steps "$REPLAN_EVERY_STEPS" \
  --simulator-collision-guard \
  --simulator-collision-reference-radius-m "$SIMULATOR_COLLISION_REFERENCE_RADIUS_M" \
  --no-static-nearfield-map \
  --robot-radius-m "${ROBOT_RADIUS_M:-0.15}" \
  --runtime-planning-clearance-m "${RUNTIME_PLANNING_CLEARANCE_M:-0.01}" \
  --astar-clearance-cost-enabled \
  --astar-clearance-desired-m "$ASTAR_CLEARANCE_DESIRED_M" \
  --astar-clearance-hard-min-m "$ASTAR_CLEARANCE_HARD_MIN_M" \
  --astar-clearance-weight "$ASTAR_CLEARANCE_WEIGHT" \
  --astar-goal-min-clearance-m "${ASTAR_GOAL_MIN_CLEARANCE_M:-0.08}" \
  --lookahead-min-clearance-m "$LOOKAHEAD_MIN_CLEARANCE_M" \
  --guard-min-clearance-m "$GUARD_MIN_CLEARANCE_M" \
  $HEADLESS_FLAG \
  --read-depth \
  --camera-annotator-device cpu \
  --isaac-width 256 \
  --isaac-height 256 \
  --camera-hfov-deg 90 \
  --camera-mast-height-m 1.25 \
  --panorama-steps "$LIVE_BASELINE_PANORAMA_VIEWS" \
  --live-baseline-panorama-views "$LIVE_BASELINE_PANORAMA_VIEWS" \
  --voxroom-viz \
  --voxroom-viz-every-steps 1 \
  --live-roomseg-baseline tvars_original_isaac \
  --live-baseline-output-dir "$RUN_DIR/baselines/tvars_original_isaac" \
  --live-baseline-door-detector original_detr \
  --live-baseline-policy-control "$LIVE_BASELINE_POLICY_CONTROL" \
  --roomseg-coverage-eval \
  --roomseg-coverage-milestones "$COVERAGE_MILESTONES" \
  --roomseg-coverage-output-dir "$RUN_DIR/roomseg_coverage_eval" \
  --output "$RUN_DIR/results.jsonl"
