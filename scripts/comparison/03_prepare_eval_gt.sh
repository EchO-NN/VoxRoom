#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RUN_ROOT="${RUN_ROOT:-result/comparison_random_frontier_v1}"
EVAL_ROOT="${EVAL_ROOT:-result/comparison_random_frontier_v1_eval}"
SNAPSHOT_POLICY="${SNAPSHOT_POLICY:-all}"
REQUIRE_CSR="${REQUIRE_CSR:-1}"
REVIEWER="${REVIEWER:-annotator}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --eval-root) EVAL_ROOT="$2"; shift 2 ;;
    --snapshot-policy) SNAPSHOT_POLICY="$2"; shift 2 ;;
    --require-csr) REQUIRE_CSR=1; shift ;;
    --no-require-csr) REQUIRE_CSR=0; shift ;;
    --reviewer) REVIEWER="$2"; shift 2 ;;
    *) echo "[comparison-prepare-gt] unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$EVAL_ROOT/indexes"
voxroom-roomseg-eval index \
  --result-root "$RUN_ROOT" \
  --out "$EVAL_ROOT/indexes/index_voxroom_all.json" \
  --snapshot-policy all
voxroom-roomseg-eval index \
  --result-root "$RUN_ROOT" \
  --out "$EVAL_ROOT/indexes/index_voxroom_last.json" \
  --snapshot-policy last

voxroom-roomseg-eval annotate-last \
  --index "$EVAL_ROOT/indexes/index_voxroom_last.json" \
  --annotation-dir "$EVAL_ROOT/annotations" \
  --gt-dir "$EVAL_ROOT/gt_final"

voxroom-roomseg-eval build-gt \
  --index "$EVAL_ROOT/indexes/index_voxroom_last.json" \
  --annotation-dir "$EVAL_ROOT/annotations" \
  --gt-dir "$EVAL_ROOT/gt_final"

voxroom-roomseg-eval backproject \
  --index "$EVAL_ROOT/indexes/index_voxroom_all.json" \
  --gt-dir "$EVAL_ROOT/gt_final" \
  --step-gt-dir "$EVAL_ROOT/gt_steps_by_method/voxroom" \
  --eval-snapshot-policy "$SNAPSHOT_POLICY"

if [[ "$REQUIRE_CSR" == "1" ]]; then
  voxroom-roomseg-eval review-csr \
    --index "$EVAL_ROOT/indexes/index_voxroom_all.json" \
    --step-gt-dir "$EVAL_ROOT/gt_steps_by_method/voxroom" \
    --csr-dir "$EVAL_ROOT/csr_by_method/voxroom" \
    --reviewer "$REVIEWER"
fi
