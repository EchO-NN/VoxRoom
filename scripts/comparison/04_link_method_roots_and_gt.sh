#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RUN_ROOT="${RUN_ROOT:-result/comparison_random_frontier_v1}"
EVAL_ROOT="${EVAL_ROOT:-result/comparison_random_frontier_v1_eval}"
METHODS=(topology_visual_active dude_incremental rose2 morphological distance_transform voronoi)
REQUIRE_CSR="${REQUIRE_CSR:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --eval-root) EVAL_ROOT="$2"; shift 2 ;;
    --require-csr) REQUIRE_CSR=1; shift ;;
    --no-require-csr) REQUIRE_CSR=0; shift ;;
    --methods)
      shift
      METHODS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do METHODS+=("$1"); shift; done
      ;;
    *) echo "[comparison-link-methods] unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$EVAL_ROOT/method_roots" "$EVAL_ROOT/indexes"
if [[ ! -e "$EVAL_ROOT/method_roots/voxroom" ]]; then
  ln -s "$(realpath --relative-to="$EVAL_ROOT/method_roots" "$RUN_ROOT")" "$EVAL_ROOT/method_roots/voxroom"
fi

for method in "${METHODS[@]}"; do
  python -m voxroom_online.isaac_runtime.baselines.evaluation_bridge \
    --source-run-root "$RUN_ROOT" \
    --baseline "$method" \
    --linked-run-root "$EVAL_ROOT/method_roots/$method" \
    --overwrite
  voxroom-roomseg-eval index \
    --result-root "$EVAL_ROOT/method_roots/$method" \
    --out "$EVAL_ROOT/indexes/index_${method}.json" \
    --snapshot-policy all
  LINK_ARGS=(
    --source-index "$EVAL_ROOT/indexes/index_voxroom_all.json"
    --method-index "$EVAL_ROOT/indexes/index_${method}.json"
    --source-step-gt-dir "$EVAL_ROOT/gt_steps_by_method/voxroom"
    --out-step-gt-dir "$EVAL_ROOT/gt_steps_by_method/$method"
    --overwrite
  )
  if [[ "$REQUIRE_CSR" == "1" ]]; then
    LINK_ARGS+=(
      --source-csr-dir "$EVAL_ROOT/csr_by_method/voxroom"
      --out-csr-dir "$EVAL_ROOT/csr_by_method/$method"
    )
  fi
  python -m voxroom_online.isaac_runtime.comparison.link_method_gt "${LINK_ARGS[@]}"
done
