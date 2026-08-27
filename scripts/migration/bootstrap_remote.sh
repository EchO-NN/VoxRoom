#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ISAAC_ROOT="${ISAAC_ROOT:-$HOME/isaac-sim-standalone-5.1.0-linux-x86_64}"
ENV_NAME="${VOXROOM_ENV_NAME:-voxroom-online}"
NVBLOX_BUNDLE="${NVBLOX_BUNDLE:-$REPO_ROOT/.migration/nvblox_torch_c457c3fc_py311_cu128.tar}"
EXPECTED_NVBLOX_SHA256="${EXPECTED_NVBLOX_SHA256:-742f7d92f550db7bd6e462353f31590d53127e6799cd42b74145f6552f4c99ef}"

if [[ ! -f "$ISAAC_ROOT/setup_conda_env.sh" ]]; then
  echo "[bootstrap] invalid ISAAC_ROOT: $ISAAC_ROOT" >&2
  exit 1
fi
if [[ ! -f "$NVBLOX_BUNDLE" ]]; then
  echo "[bootstrap] missing nvblox bundle: $NVBLOX_BUNDLE" >&2
  exit 1
fi

if [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  CONDA_BASE="$HOME/anaconda3"
elif [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
  CONDA_BASE="$HOME/miniforge3"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  CONDA_BASE="$HOME/miniconda3"
else
  echo "[bootstrap] conda installation not found" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda env update -n "$ENV_NAME" -f "$REPO_ROOT/envs/voxroom-online-py311.yml"
else
  conda env create -n "$ENV_NAME" -f "$REPO_ROOT/envs/voxroom-online-py311.yml"
fi

conda run -n "$ENV_NAME" python -m pip install --no-deps -e "$REPO_ROOT"

actual_bundle_sha256="$(sha256sum "$NVBLOX_BUNDLE" | awk '{print $1}')"
if [[ "$actual_bundle_sha256" != "$EXPECTED_NVBLOX_SHA256" ]]; then
  echo "[bootstrap] nvblox bundle hash mismatch: $actual_bundle_sha256" >&2
  exit 1
fi

site_packages="$(
  conda run -n "$ENV_NAME" python -c \
    'import sysconfig; print(sysconfig.get_paths()["purelib"])'
)"
rm -rf "$site_packages/nvblox_torch" "$site_packages"/nvblox_torch-*.dist-info
tar -xf "$NVBLOX_BUNDLE" -C "$site_packages"

VOXROOM_ENV_NAME="$ENV_NAME" ISAAC_ROOT="$ISAAC_ROOT" \
  "$REPO_ROOT/scripts/run_voxroom_isaac_env.sh" -c '
import torch
import nvblox_torch
from nvblox_torch.mapper import Mapper
from nvblox_torch.projective_integrator_types import ProjectiveIntegratorType
from nvblox_torch.sensor import Sensor

expected_sha = "c457c3fc01003bec6eba3ec1c61e6bf84bc3f51f"
if nvblox_torch.__git_sha__ != expected_sha:
    raise RuntimeError(
        f"nvblox commit mismatch: {nvblox_torch.__git_sha__} != {expected_sha}"
    )
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

mapper = Mapper(0.05, ProjectiveIntegratorType.OCCUPANCY)
sensor = Sensor.from_camera(50.0, 50.0, 7.5, 5.5, 16, 12)
depth = torch.ones((12, 16), device="cuda", dtype=torch.float32)
t_w_c = torch.eye(4, dtype=torch.float32)
mapper.add_depth_frame(depth, t_w_c, sensor)
torch.cuda.synchronize()
print(
    "NVBLOX_CUDA_SMOKE=PASS",
    f"gpu={torch.cuda.get_device_name(0)}",
    f"nvblox_git={nvblox_torch.__git_sha__}",
)
'

echo "[bootstrap] environment ready: $ENV_NAME"
