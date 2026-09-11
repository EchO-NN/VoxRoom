#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_ROOT=/home/joey/Active_room_segmentation
REMOTE_HOST=echo@10.42.0.1
REMOTE_DIR=/home/echo/下载
STAMP=20260831
LOG=$SOURCE_ROOT/results/transfer_existing_data_to_echo_${STAMP}.log

log_status() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG"
}

transfer_directory() {
  local directory=$1
  local archive=Active_room_segmentation_${directory}_${STAMP}.tar.zst
  local final=$REMOTE_DIR/$archive
  local partial=$final.part

  log_status "packing and transferring $directory to $final"
  (
    cd "$SOURCE_ROOT"
    ZSTD_CLEVEL=1 tar --zstd -cf - "$directory"
  ) | ssh -o BatchMode=yes "$REMOTE_HOST" "cat > '$partial'"

  ssh -o BatchMode=yes "$REMOTE_HOST" \
    "mv '$partial' '$final' && cd '$REMOTE_DIR' && sha256sum '$archive' > '$archive.sha256'"
  local size
  size=$(ssh -o BatchMode=yes "$REMOTE_HOST" "du -h '$final' | cut -f1")
  log_status "complete $directory archive_size=$size"
}

: >"$LOG"
transfer_directory results
log_status "all requested archives transferred"
