#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/activate_voxroom_env.sh"

if [[ -z "${ISAAC_ROOT:-}" ]]; then
  ISAAC_ROOT="${ISAAC_SIM_ROOT:-}"
fi
if [[ -z "$ISAAC_ROOT" ]]; then
  for candidate in \
    "$HOME/isaac-sim-standalone-5.1.0-linux-x86_64" \
    "/home/echo/isaac-sim-standalone-5.1.0-linux-x86_64"; do
    if [[ -f "$candidate/setup_conda_env.sh" ]]; then
      ISAAC_ROOT="$candidate"
      break
    fi
  done
fi
if [[ -z "$ISAAC_ROOT" || ! -f "$ISAAC_ROOT/setup_conda_env.sh" ]]; then
  echo "[voxroom-env] Isaac Sim setup_conda_env.sh not found; set ISAAC_ROOT or ISAAC_SIM_ROOT" >&2
  exit 1
fi
export ISAAC_SIM_ROOT="$ISAAC_ROOT"

# Isaac Sim exposes isaacsim/omni packages and native libraries through this
# script. It expects an active Python 3.11 conda environment.
set +u
# shellcheck disable=SC1090
source "$ISAAC_ROOT/setup_conda_env.sh"
set -u

THREADS="${VOXROOM_NUMBA_THREADS:-${NUMBA_NUM_THREADS:-28}}"
export NUMBA_NUM_THREADS="$THREADS"
export VOXROOM_NUMBA_THREADS="$THREADS"
export VOXROOM_VOXEL_CPU_NUMBA_THREADS="${VOXROOM_VOXEL_CPU_NUMBA_THREADS:-$THREADS}"

export OMP_NUM_THREADS="${VOXROOM_OMP_THREADS:-1}"
export MKL_NUM_THREADS="${VOXROOM_MKL_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${VOXROOM_OPENBLAS_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${VOXROOM_NUMEXPR_THREADS:-1}"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec python "$@"
