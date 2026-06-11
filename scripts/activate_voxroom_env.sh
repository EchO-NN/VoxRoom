#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
elif [[ -d "$HOME/miniforge3" ]]; then
  CONDA_BASE="$HOME/miniforge3"
elif [[ -d "$HOME/miniconda3" ]]; then
  CONDA_BASE="$HOME/miniconda3"
else
  echo "[voxroom-env] could not find conda/mamba installation" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if [[ -n "${VOXROOM_ENV_NAME:-}" ]]; then
  ENV_NAME="$VOXROOM_ENV_NAME"
elif conda env list | awk '{print $1}' | grep -qx "voxroom-online"; then
  ENV_NAME="voxroom-online"
elif conda env list | awk '{print $1}' | grep -qx "sgnav-isaac"; then
  ENV_NAME="sgnav-isaac"
else
  ENV_NAME="voxroom-online"
fi

conda activate "$ENV_NAME"
