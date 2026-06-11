#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source scripts/comparison/env.sh

RUN_ROOT="${RUN_ROOT:-result/comparison_random_frontier_t0}"
EVAL_ROOT="${EVAL_ROOT:-result/comparison_random_frontier_t0_eval}"
EPISODE_DIR="${EPISODE_DIR:-data/interioragent_episodes/radius005_all_scenes}"
SCENE_GLOB="${SCENE_GLOB:-kujiale_*.jsonl}"
SNAPSHOT_POLICY="${SNAPSHOT_POLICY:-all}"
REQUIRE_CSR="${REQUIRE_CSR:-1}"
MAX_CONTROL_STEPS="${MAX_CONTROL_STEPS:-5000}"
PANORAMA_VIEWS="${PANORAMA_VIEWS:-12}"
MAX_SCENES=""
MAX_SNAPSHOTS_PER_SCENE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --eval-root) EVAL_ROOT="$2"; shift 2 ;;
    --episode-dir) EPISODE_DIR="$2"; shift 2 ;;
    --scene-glob) SCENE_GLOB="$2"; shift 2 ;;
    --snapshot-policy) SNAPSHOT_POLICY="$2"; shift 2 ;;
    --require-csr) REQUIRE_CSR=1; shift ;;
    --no-require-csr) REQUIRE_CSR=0; shift ;;
    --max-control-steps) MAX_CONTROL_STEPS="$2"; shift 2 ;;
    --panorama-views) PANORAMA_VIEWS="$2"; shift 2 ;;
    --max-scenes) MAX_SCENES="$2"; shift 2 ;;
    --max-snapshots-per-scene) MAX_SNAPSHOTS_PER_SCENE="$2"; shift 2 ;;
    *) echo "[comparison-closed-loop] unknown argument: $1" >&2; exit 2 ;;
  esac
done

STRICT_VERIFY=1 bash scripts/comparison/00_verify_env.sh --run-root "$RUN_ROOT"

COLLECT_ARGS=(
  --episode-dir "$EPISODE_DIR"
  --scene-glob "$SCENE_GLOB"
  --run-root "$RUN_ROOT"
  --max-control-steps "$MAX_CONTROL_STEPS"
  --panorama-views "$PANORAMA_VIEWS"
  --strict-main-collection
)
if [[ -n "$MAX_SCENES" ]]; then
  COLLECT_ARGS+=(--max-scenes "$MAX_SCENES")
fi
bash scripts/comparison/01_collect_all_scenes_online_random_frontier.sh "${COLLECT_ARGS[@]}"

python -m voxroom_online.isaac_runtime.comparison.validate_collected_run \
  --run-root "$RUN_ROOT" \
  --require-topology \
  --require-active-stream \
  --require-panorama \
  --fail-on-placeholder \
  --fail-on-fallback

OFFLINE_ARGS=(--run-root "$RUN_ROOT" --scene-glob "${SCENE_GLOB%.jsonl}")
if [[ -n "$MAX_SNAPSHOTS_PER_SCENE" ]]; then
  OFFLINE_ARGS+=(--max-snapshots-per-scene "$MAX_SNAPSHOTS_PER_SCENE")
fi
bash scripts/comparison/02_run_offline_native_baselines.sh "${OFFLINE_ARGS[@]}" --strict-native

python -m voxroom_online.isaac_runtime.comparison.validate_all_baselines \
  --run-root "$RUN_ROOT" \
  --methods topology_visual_active dude_incremental rose2 morphological distance_transform voronoi \
  --fail-on-non-main \
  --require-same-snapshot-set

GT_ARGS=(--run-root "$RUN_ROOT" --eval-root "$EVAL_ROOT" --snapshot-policy "$SNAPSHOT_POLICY")
LINK_ARGS=(--run-root "$RUN_ROOT" --eval-root "$EVAL_ROOT")
COMPUTE_ARGS=(--eval-root "$EVAL_ROOT" --strict-paper)
if [[ "$REQUIRE_CSR" == "1" ]]; then
  GT_ARGS+=(--require-csr)
  LINK_ARGS+=(--require-csr)
  COMPUTE_ARGS+=(--require-csr)
else
  GT_ARGS+=(--no-require-csr)
  LINK_ARGS+=(--no-require-csr)
  COMPUTE_ARGS+=(--no-require-csr)
fi
bash scripts/comparison/03_prepare_eval_gt.sh "${GT_ARGS[@]}"
bash scripts/comparison/04_link_method_roots_and_gt.sh "${LINK_ARGS[@]}"
bash scripts/comparison/05_compute_all_methods.sh "${COMPUTE_ARGS[@]}"

python -m voxroom_online.isaac_runtime.comparison.make_final_report \
  --eval-root "$EVAL_ROOT" \
  --methods voxroom topology_visual_active dude_incremental rose2 morphological distance_transform voronoi \
  --out-dir "$EVAL_ROOT/reports"

python -m voxroom_online.isaac_runtime.comparison.validate_paper_ready \
  --run-root "$RUN_ROOT" \
  --eval-root "$EVAL_ROOT" \
  --methods voxroom topology_visual_active dude_incremental rose2 morphological distance_transform voronoi \
  --require-csr \
  --require-native \
  --require-original-commits \
  --require-no-placeholder \
  | tee "$EVAL_ROOT/reports/paper_ready_validation.json"
