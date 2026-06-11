#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${VOXROOM_ENV_NAME:-voxroom-online}"
ENV_FILE="$REPO_ROOT/envs/voxroom-online-py311.yml"

if command -v mamba >/dev/null 2>&1; then
  CONDA_FRONTEND=mamba
elif command -v conda >/dev/null 2>&1; then
  CONDA_FRONTEND=conda
else
  echo "[voxroom-setup] conda or mamba is required" >&2
  exit 1
fi

if "$CONDA_FRONTEND" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[voxroom-setup] updating env $ENV_NAME"
  "$CONDA_FRONTEND" env update -n "$ENV_NAME" -f "$ENV_FILE"
else
  echo "[voxroom-setup] creating env $ENV_NAME"
  "$CONDA_FRONTEND" env create -n "$ENV_NAME" -f "$ENV_FILE"
fi

echo "[voxroom-setup] installing project in editable mode"
"$CONDA_FRONTEND" run -n "$ENV_NAME" python -m pip install -e "$REPO_ROOT"

cat <<'EOF'
[voxroom-setup] done

Next:
  cd VoxRoom-Online
  source scripts/activate_voxroom_env.sh
EOF
