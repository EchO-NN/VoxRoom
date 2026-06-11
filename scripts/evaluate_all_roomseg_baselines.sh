#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_ROOT="${RUN_ROOT:-result/radius005_robot005_random_frontier_all_scenes_collect_topology}"
EVAL_ROOT="${EVAL_ROOT:-result/eval_radius005_random_frontier_collect_topology}"
SNAPSHOT_POLICY="${SNAPSHOT_POLICY:-all}"
REQUIRE_CSR="${REQUIRE_CSR:-1}"
ALLOW_NON_MAIN_BASELINES="${ALLOW_NON_MAIN_BASELINES:-0}"
PREPARE_GT="${PREPARE_GT:-1}"
METHODS=(voxroom topology_visual_active dude_incremental rose2 morphological distance_transform voronoi)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --eval-root) EVAL_ROOT="$2"; shift 2 ;;
    --snapshot-policy) SNAPSHOT_POLICY="$2"; shift 2 ;;
    --no-require-csr) REQUIRE_CSR=0; shift ;;
    --skip-prepare-gt) PREPARE_GT=0; shift ;;
    *) echo "[voxroom-eval-all] unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$EVAL_ROOT"

if [[ "$ALLOW_NON_MAIN_BASELINES" == "1" ]]; then
  echo "[voxroom-eval-all] ALLOW_NON_MAIN_BASELINES=1 is only supported by the low-level bridge, not this paper-flow wrapper" >&2
  exit 1
fi

if [[ "$PREPARE_GT" == "1" ]]; then
  GT_ARGS=(--run-root "$RUN_ROOT" --eval-root "$EVAL_ROOT" --snapshot-policy "$SNAPSHOT_POLICY")
  if [[ "$REQUIRE_CSR" == "1" ]]; then
    GT_ARGS+=(--require-csr)
  else
    GT_ARGS+=(--no-require-csr)
  fi
  bash scripts/comparison/03_prepare_eval_gt.sh "${GT_ARGS[@]}"
else
  mkdir -p "$EVAL_ROOT/indexes"
  if [[ ! -f "$EVAL_ROOT/indexes/index_voxroom_all.json" ]]; then
    voxroom-roomseg-eval index \
      --result-root "$RUN_ROOT" \
      --out "$EVAL_ROOT/indexes/index_voxroom_all.json" \
      --snapshot-policy all
  fi
fi

LINK_ARGS=(--run-root "$RUN_ROOT" --eval-root "$EVAL_ROOT")
COMPUTE_ARGS=(--eval-root "$EVAL_ROOT" --strict-paper)
if [[ "$REQUIRE_CSR" == "1" ]]; then
  LINK_ARGS+=(--require-csr)
  COMPUTE_ARGS+=(--require-csr)
else
  LINK_ARGS+=(--no-require-csr)
  COMPUTE_ARGS+=(--no-require-csr)
fi

bash scripts/comparison/04_link_method_roots_and_gt.sh "${LINK_ARGS[@]}"
bash scripts/comparison/05_compute_all_methods.sh "${COMPUTE_ARGS[@]}"
python -m voxroom_online.isaac_runtime.comparison.make_final_report \
  --eval-root "$EVAL_ROOT" \
  --methods "${METHODS[@]}" \
  --out-dir "$EVAL_ROOT/reports"
