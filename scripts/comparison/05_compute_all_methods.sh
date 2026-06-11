#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

EVAL_ROOT="${EVAL_ROOT:-result/comparison_random_frontier_v1_eval}"
STRICT_PAPER="${STRICT_PAPER:-1}"
REQUIRE_CSR="${REQUIRE_CSR:-1}"
METHODS=(voxroom topology_visual_active dude_incremental rose2 morphological distance_transform voronoi)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --eval-root) EVAL_ROOT="$2"; shift 2 ;;
    --strict-paper) STRICT_PAPER=1; shift ;;
    --no-strict-paper) STRICT_PAPER=0; shift ;;
    --require-csr) REQUIRE_CSR=1; shift ;;
    --no-require-csr) REQUIRE_CSR=0; shift ;;
    --methods)
      shift
      METHODS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do METHODS+=("$1"); shift; done
      ;;
    *) echo "[comparison-compute] unknown argument: $1" >&2; exit 2 ;;
  esac
done

for method in "${METHODS[@]}"; do
  INDEX="$EVAL_ROOT/indexes/index_${method}.json"
  if [[ "$method" == "voxroom" && ! -f "$INDEX" ]]; then
    INDEX="$EVAL_ROOT/indexes/index_voxroom_all.json"
  fi
  ARGS=(
    compute
    --index "$INDEX"
    --step-gt-dir "$EVAL_ROOT/gt_steps_by_method/$method"
    --out-dir "$EVAL_ROOT/metrics/$method"
  )
  if [[ "$STRICT_PAPER" == "1" ]]; then
    ARGS+=(--strict-paper)
  fi
  if [[ "$REQUIRE_CSR" == "1" ]]; then
    ARGS+=(--csr-dir "$EVAL_ROOT/csr_by_method/$method")
  else
    ARGS+=(--no-require-csr)
  fi
  voxroom-roomseg-eval "${ARGS[@]}"
  voxroom-roomseg-eval report \
    --metrics-dir "$EVAL_ROOT/metrics/$method" \
    --out "$EVAL_ROOT/reports/report_${method}.md"
done
