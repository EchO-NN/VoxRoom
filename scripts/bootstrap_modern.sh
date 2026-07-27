#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
CONDA_ROOT="${CONDA_ROOT:-$HOME/anaconda3}"
SOURCE_ENV="${SOURCE_ENV:-SG_Nav}"
TARGET_PREFIX="${TARGET_PREFIX:-$HOME/.conda/envs/active-room-seg}"
PYTHON="$TARGET_PREFIX/bin/python"
DETR_DIR="$ROOT_DIR/third_party/detr"
DETR_COMMIT="29901c51d7fe8712168b8d0d64351170bc0f83e0"
WEIGHTS_ZIP="$ROOT_DIR/.downloads/train_params.zip"
WEIGHTS_PATH="$ROOT_DIR/detr_door_detection/train_params/detr_resnet50_4/final_doors_dataset/model.pth"
PROXY_URL="${PROXY_URL:-http://10.42.0.1:7890}"

export HTTP_PROXY="$PROXY_URL"
export HTTPS_PROXY="$PROXY_URL"
export http_proxy="$PROXY_URL"
export https_proxy="$PROXY_URL"
unset ALL_PROXY all_proxy NO_PROXY no_proxy

if [[ ! -x "$CONDA_ROOT/bin/conda" ]]; then
    echo "Conda is missing: $CONDA_ROOT/bin/conda" >&2
    exit 1
fi

if [[ ! -f "$WEIGHTS_PATH" && ! -f "$WEIGHTS_ZIP" ]]; then
    echo "Door detector archive is missing: $WEIGHTS_ZIP" >&2
    exit 1
fi

if [[ ! -x "$PYTHON" ]]; then
    SOURCE_PREFIX="$(
        "$CONDA_ROOT/bin/conda" env list --json |
            "$CONDA_ROOT/bin/python" -c \
                'import json, pathlib, sys; name=sys.argv[1]; matches=[p for p in json.load(sys.stdin)["envs"] if pathlib.Path(p).name == name]; assert len(matches) == 1, matches; print(matches[0])' \
                "$SOURCE_ENV"
    )"
    "$CONDA_ROOT/bin/conda" create -y \
        --prefix "$TARGET_PREFIX" \
        --clone "$SOURCE_PREFIX"
fi

"$PYTHON" -m pip install --proxy "$PROXY_URL" \
    python-igraph==0.11.9 \
    seaborn==0.13.2 \
    treelib==1.7.1

if [[ ! -d "$DETR_DIR/.git" ]]; then
    mkdir -p "$(dirname "$DETR_DIR")"
    git -c "http.proxy=$PROXY_URL" clone \
        https://github.com/facebookresearch/detr.git "$DETR_DIR"
elif [[ -n "$(git -C "$DETR_DIR" status --porcelain)" ]]; then
    echo "Pinned DETR checkout is dirty: $DETR_DIR" >&2
    exit 1
fi
git -C "$DETR_DIR" -c "http.proxy=$PROXY_URL" fetch origin "$DETR_COMMIT"
git -C "$DETR_DIR" checkout --detach "$DETR_COMMIT"
test "$(git -C "$DETR_DIR" rev-parse HEAD)" = "$DETR_COMMIT"
test -z "$(git -C "$DETR_DIR" status --porcelain)"

if [[ ! -f "$WEIGHTS_PATH" ]]; then
    unzip -q -o "$WEIGHTS_ZIP" -d "$ROOT_DIR/detr_door_detection"
fi

if [[ ! -e "$ROOT_DIR/data/scene_datasets/habitat-test-scenes" ]] \
    || [[ ! -e "$ROOT_DIR/data/datasets/pointnav/habitat-test-scenes" ]]; then
    (
        cd "$ROOT_DIR"
        "$PYTHON" -m habitat_sim.utils.datasets_download \
            --uids habitat_test_scenes habitat_test_pointnav_dataset \
            --data-path "$ROOT_DIR/data" \
            --no-replace
    )
fi

"$PYTHON" "$ROOT_DIR/scripts/validate_install.py"
"$PYTHON" -m unittest discover -s "$ROOT_DIR/tests" -v
