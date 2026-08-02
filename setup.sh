#!/usr/bin/env bash
# 自己評価キットの環境構築（本番採点イメージと同じ手順をローカルに再現する）
#
#   bash setup.sh          # このディレクトリ（キットのルート）で実行
#   source env.sh          # 以後、評価を回すシェルで毎回 source する
#
# やること:
#   1. venv 作成 + 依存インストール（本番と同じピン止めバージョン）
#   2. LIBERO-plus（評価環境）と LIBERO（base assets）の取得
#   3. LIBERO-plus への既知パッチ（__init__.py 追加 / torch.load weights_only）
#   4. タスクアセットのダウンロードと配線（HF assets.zip）
#   5. ~/.libero/config.yaml の生成（既存があれば .bak に退避）
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="$PWD"

PY="${PYTHON:-python3.10}"

echo "[setup] 1/5 venv + 依存"

# 壊れた Python（pkgutil 欠落など）や、mise の別版 pip が混ざるのを防ぐ
if ! "$PY" -c "import pkgutil" 2>/dev/null; then
    echo "error: '$PY' の標準ライブラリが壊れています（pkgutil が無い）。" >&2
    echo "  mise 利用時は python 3.10.20 以上を使い、壊れた 3.10.18 を避けること:" >&2
    echo "    mise install python@3.10.20 && mise use python@3.10.20" >&2
    echo "    rm -rf venv && uv venv venv --python 3.10.20" >&2
    echo "    PYTHON=$PWD/venv/bin/python bash setup.sh" >&2
    exit 1
fi

if [ "${FORCE_RECREATE_VENV:-}" = "1" ] && [ -d venv ]; then
    echo "[setup]   FORCE_RECREATE_VENV=1 → venv を削除して作り直します"
    rm -rf venv
fi

if [ ! -d venv ]; then
    if command -v uv >/dev/null 2>&1; then
        uv venv venv --python "$PY" --seed
    else
        "$PY" -m venv venv
    fi
fi

# shellcheck disable=SC1091
source venv/bin/activate

# activate 後も PATH 汚染で外側の pip が使われないよう、必ず venv の python -m pip
VENV_PY="$ROOT/venv/bin/python"
if [ "$(command -v python)" != "$VENV_PY" ] && [ ! -x "$VENV_PY" ]; then
    echo "error: venv の python が見つかりません: $VENV_PY" >&2
    exit 1
fi
if ! "$VENV_PY" -c "import pkgutil" 2>/dev/null; then
    echo "error: venv の Python が壊れています。次で作り直してください:" >&2
    echo "  rm -rf venv && FORCE_RECREATE_VENV=1 bash setup.sh" >&2
    exit 1
fi

echo "[setup]   python=$("$VENV_PY" -V 2>&1) ($VENV_PY)"
if ! "$VENV_PY" -m pip --version >/dev/null 2>&1; then
    if command -v uv >/dev/null 2>&1; then
        echo "[setup]   pip が無いため uv で seed します"
        uv pip install --python "$VENV_PY" pip setuptools wheel
    else
        echo "error: venv に pip がありません。uv で作り直してください:" >&2
        echo "  rm -rf venv && uv venv venv --python 3.10.20 --seed" >&2
        exit 1
    fi
fi
"$VENV_PY" -m pip install --upgrade pip setuptools wheel -q
# GPU で推論する提出物を作る場合も、評価環境自体は CPU torch で足りる
# Linux 本番相当は +cpu ホイール。macOS は PyPI の通常ビルド（+cpu は提供されない）
if [ "$(uname -s)" = "Darwin" ]; then
    "$VENV_PY" -m pip install -q --timeout 120 --retries 10 "torch==2.11.0"
else
    "$VENV_PY" -m pip install -q --timeout 120 --retries 10 \
        "torch==2.11.0+cpu" --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple
fi
"$VENV_PY" -m pip install -q --timeout 120 \
    mujoco==3.7.0 robosuite==1.4.0 numpy==1.26.4 "gym==0.25.2" bddl==3.6.0 \
    cloudpickle==3.1.2 easydict==1.13 hydra-core==1.3.2 einops==0.8.2 \
    opencv-python-headless==4.11.0.86 \
    scipy pyyaml h5py Pillow termcolor tqdm matplotlib \
    requests msgpack fastapi uvicorn huggingface_hub wand scikit-image pytest

echo "[setup] 2/5 LIBERO-plus / LIBERO の取得"
if [ ! -d LIBERO-plus ]; then
    git clone --depth 1 https://github.com/sylvestf/LIBERO-plus
fi
if [ ! -d LIBERO ]; then
    git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO
fi

echo "[setup] 3/5 LIBERO-plus パッチ"
touch LIBERO-plus/libero/__init__.py LIBERO-plus/libero/libero/__init__.py
# GNU sed: sed -i 's/.../' file  /  BSD sed (macOS): sed -i '' 's/.../' file
PATCH_FILE="LIBERO-plus/libero/libero/benchmark/__init__.py"
if [ "$(uname -s)" = "Darwin" ]; then
    sed -i '' 's/torch.load(init_states_path)/torch.load(init_states_path, weights_only=False)/' \
        "$PATCH_FILE" || true
else
    sed -i 's/torch.load(init_states_path)/torch.load(init_states_path, weights_only=False)/' \
        "$PATCH_FILE" || true
fi
if grep -q "weights_only=False" "$PATCH_FILE" 2>/dev/null; then
    echo "[setup]   torch.load weights_only=False パッチ OK"
else
    echo "[setup]   WARN: weights_only パッチが当たっていない可能性あり（後で確認）"
fi

echo "[setup] 4/5 アセット"
ASSETS="LIBERO-plus/libero/libero/assets"
if [ "$(ls "$ASSETS"/textures 2>/dev/null | wc -l)" -lt 100 ]; then
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download('Sylvest/LIBERO-plus','assets.zip',repo_type='dataset',local_dir='.tmp_assets')"
    unzip -q .tmp_assets/assets.zip -d LIBERO-plus/libero/libero
    rm -rf .tmp_assets
    # assets.zip は作者環境の深いネストパスごと展開されることがあるため対応する
    if [ "$(ls "$ASSETS"/textures 2>/dev/null | wc -l)" -lt 100 ]; then
        rm -rf "$ASSETS"
        NESTED="$(find LIBERO-plus/libero/libero -type d -path '*/assets' -name assets | grep -v '^LIBERO-plus/libero/libero/assets$' | head -1)"
        test -n "$NESTED" && ln -sfn "$(realpath "$NESTED")" "$ASSETS"
    fi
fi
test -e "$ASSETS/scenes/libero_floor_base_style.xml"
echo "[setup]   textures=$(ls "$ASSETS"/textures | wc -l)"

echo "[setup] 5/5 libero 設定"
mkdir -p "$HOME/.libero"
if [ -f "$HOME/.libero/config.yaml" ]; then
    cp "$HOME/.libero/config.yaml" "$HOME/.libero/config.yaml.bak"
    echo "[setup]   既存の ~/.libero/config.yaml を config.yaml.bak に退避しました"
fi
cat > "$HOME/.libero/config.yaml" <<EOF
benchmark_root: $ROOT/LIBERO-plus/libero/libero
bddl_files: $ROOT/LIBERO-plus/libero/libero/bddl_files
init_states: $ROOT/LIBERO-plus/libero/libero/init_files
datasets: $ROOT/LIBERO-plus/libero/libero/datasets
assets: $ROOT/LIBERO/libero/libero/assets
EOF

cat > env.sh <<EOF
source "$ROOT/venv/bin/activate"
export PYTHONPATH="$ROOT/LIBERO-plus:$ROOT:$ROOT/compe"
export LIBERO_ROOT="$ROOT/LIBERO-plus"
EOF

echo "[setup] 動作確認（suite 登録）"
PYTHONPATH="$ROOT/LIBERO-plus:$ROOT:$ROOT/compe" \
    python -c "import libero.libero.benchmark; from compe.t1 import register_t1; register_t1(); print('suite 登録 OK')"

echo
echo "セットアップ完了。評価を回すシェルで次を実行してください:"
echo "  source env.sh"
