#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE="${REMOTE:-joey}"
REMOTE_REPO="${REMOTE_REPO:-/home/joey/VoxRoom}"
REMOTE_INTERIOR_ROOT="${REMOTE_INTERIOR_ROOT:-/home/joey/InteriorAgent}"
REMOTE_GRSCENE_ROOT="${REMOTE_GRSCENE_ROOT:-/home/joey/GRScenes-100}"
REMOTE_ISAAC_ROOT="${REMOTE_ISAAC_ROOT:-/home/joey/isaac-sim-standalone-5.1.0-linux-x86_64}"
LOCAL_INTERIOR_ROOT="${LOCAL_INTERIOR_ROOT:-/home/echo/InteriorAgent}"
LOCAL_GRSCENE_ROOT="${LOCAL_GRSCENE_ROOT:-/home/echo/GRScenes-100}"
STAGE_ROOT="${STAGE_ROOT:-/media/echo/data/voxroom_migration_20260727}"
CHECKPOINT_REL="outputs/door_seed_training_runs/interioragent_raw_occ6_20260717_153144/model_vertical"
NVBLOX_BUNDLE="$STAGE_ROOT/nvblox_torch_c457c3fc_py311_cu128.tar"
MANIFEST="$STAGE_ROOT/transfer_manifest.json"
SSH=(ssh -T -o BatchMode=yes -o Compression=no -c aes128-gcm@openssh.com)
RSYNC_SSH="ssh -T -o BatchMode=yes -o Compression=no -c aes128-gcm@openssh.com"

for path in \
  "$LOCAL_INTERIOR_ROOT" \
  "$LOCAL_GRSCENE_ROOT" \
  "$NVBLOX_BUNDLE" \
  "$MANIFEST" \
  "$REPO_ROOT/$CHECKPOINT_REL/best.pt"; do
  if [[ ! -e "$path" ]]; then
    echo "[migrate] required source is missing: $path" >&2
    exit 1
  fi
done

"${SSH[@]}" "$REMOTE" "test -f '$REMOTE_ISAAC_ROOT/setup_conda_env.sh'"
"${SSH[@]}" "$REMOTE" \
  "mkdir -p '$REMOTE_REPO' '$REMOTE_REPO/.migration' '$REMOTE_GRSCENE_ROOT'"

echo "[migrate] syncing current VoxRoom working tree"
rsync -a --human-readable --info=progress2 --partial \
  -e "$RSYNC_SSH" \
  --exclude='/.git/' \
  --exclude='/.pytest_cache/' \
  --exclude='/.vscode/' \
  --exclude='/backups/' \
  --exclude='/outputs/' \
  --exclude='/result/' \
  --exclude='/data/door_seed_learning/' \
  --exclude='**/__pycache__/' \
  --exclude='*.pyc' \
  "$REPO_ROOT/" "$REMOTE:$REMOTE_REPO/"

echo "[migrate] syncing selected deployment checkpoint"
"${SSH[@]}" "$REMOTE" "mkdir -p '$REMOTE_REPO/$CHECKPOINT_REL'"
rsync -a --human-readable --info=progress2 --partial \
  -e "$RSYNC_SSH" \
  "$REPO_ROOT/$CHECKPOINT_REL/best.pt" \
  "$REPO_ROOT/$CHECKPOINT_REL/training_history.json" \
  "$REPO_ROOT/$CHECKPOINT_REL/training_summary.json" \
  "$REMOTE:$REMOTE_REPO/$CHECKPOINT_REL/"

echo "[migrate] syncing manifest and nvblox runtime bundle"
rsync -a --human-readable --info=progress2 --partial \
  -e "$RSYNC_SSH" \
  "$MANIFEST" "$NVBLOX_BUNDLE" \
  "$REMOTE:$REMOTE_REPO/.migration/"

echo "[migrate] verifying existing InteriorAgent copy incrementally"
rsync -a --human-readable --info=progress2 --partial \
  -e "$RSYNC_SSH" \
  --exclude='/.git/' \
  "$LOCAL_INTERIOR_ROOT/" "$REMOTE:$REMOTE_INTERIOR_ROOT/"

echo "[migrate] syncing GRScene assets"
rsync -a --human-readable --info=progress2 --partial \
  -e "$RSYNC_SSH" \
  "$LOCAL_GRSCENE_ROOT/" "$REMOTE:$REMOTE_GRSCENE_ROOT/"

echo "[migrate] verifying checkpoint hash"
remote_checkpoint_sha="$(
  "${SSH[@]}" "$REMOTE" "sha256sum '$REMOTE_REPO/$CHECKPOINT_REL/best.pt'" \
    | awk '{print $1}'
)"
if [[ "$remote_checkpoint_sha" != "de0397beb366c83839f11dbd1cf42395a5214004b0de62965d4fcab953ed1f20" ]]; then
  echo "[migrate] remote checkpoint hash mismatch: $remote_checkpoint_sha" >&2
  exit 1
fi

echo "[migrate] transfer complete"
