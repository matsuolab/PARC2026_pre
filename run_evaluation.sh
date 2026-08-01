#!/bin/bash
# ============================================================
# Docker（GPU）上で提出物のフル評価（evaluate.py）を実行するスクリプト
#
# check_submission.sh が「壊れていないか」の軽量チェックなのに対し、
# こちらは実際に LIBERO タスクを動かしてスコアを算出する本評価。
# 提出直前の最終確認、またはスコアを見たいときに使う。
#
# 使い方:
#   ./run_evaluation.sh                          # submission/*.zip を自動検出、既定設定で評価
#   ./run_evaluation.sh submission/foo.zip       # zip を明示指定
#   ./run_evaluation.sh -- --n-episodes 2        # evaluate.py への追加引数（-- 以降）
#   ./run_evaluation.sh submission/foo.zip -- --n-episodes 2 --max-steps 300
#
# 結果は ./results/<submission_id>.json に出力される（コンテナの
# /workspace/results をホストの ./results にマウントするため、そのまま参照できる）。
#
# 前提: docker build -t parc2026 . を実行済みであること。GPU (nvidia-container-toolkit)
# が使える環境を想定している（Pi0 等の大きいモデルは CPU だと 10 秒タイムアウトに
# 抵触しやすいため）。GPU が無い場合は環境変数 GPU_ARGS="" で無効化できる。
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

SUBMISSION_DIR="submission"
IMAGE="${IMAGE:-parc2026}"
# 空文字（GPU無効化の意図的な指定）と未設定を区別するため := ではなく - を使う
GPU_ARGS="${GPU_ARGS-"--gpus all"}"
ZIP_PATH=""
PIPELINE_ARGS=()
PARSING_PIPELINE_ARGS=false

for arg in "$@"; do
    if [ "$PARSING_PIPELINE_ARGS" = true ]; then
        PIPELINE_ARGS+=("$arg")
        continue
    fi
    case "$arg" in
        --)
            PARSING_PIPELINE_ARGS=true
            ;;
        -*)
            echo "[run_evaluation] エラー: 不明なオプション: $arg（evaluate.py への引数は -- の後ろに書く）" >&2
            exit 1
            ;;
        *)
            ZIP_PATH="$arg"
            ;;
    esac
done

if [ -n "$ZIP_PATH" ]; then
    if [ ! -f "$ZIP_PATH" ]; then
        echo "[run_evaluation] エラー: 指定された zip が見つかりません: $ZIP_PATH" >&2
        exit 1
    fi
else
    if [ ! -d "$SUBMISSION_DIR" ]; then
        echo "[run_evaluation] エラー: '$SUBMISSION_DIR' フォルダが存在しません" >&2
        exit 1
    fi

    mapfile -t ZIP_CANDIDATES < <(find "$SUBMISSION_DIR" -maxdepth 1 -name "*.zip" | sort)

    if [ "${#ZIP_CANDIDATES[@]}" -eq 0 ]; then
        echo "[run_evaluation] エラー: '$SUBMISSION_DIR' に zip ファイルが見つかりません" >&2
        exit 1
    fi

    if [ "${#ZIP_CANDIDATES[@]}" -gt 1 ]; then
        echo "[run_evaluation] エラー: '$SUBMISSION_DIR' に複数の zip があります。対象を1つ指定してください:" >&2
        printf '  %s\n' "${ZIP_CANDIDATES[@]}" >&2
        exit 1
    fi

    ZIP_PATH="${ZIP_CANDIDATES[0]}"
fi

ZIP_ABS="$(cd "$(dirname "$ZIP_PATH")" && pwd)/$(basename "$ZIP_PATH")"
mkdir -p results

# requirements.txt のインストールは毎回コンテナが使い捨てになるため、pip キャッシュを
# ホスト側に永続化して2回目以降を高速化する（無効化したい場合は PIP_CACHE_DIR=""）。
PIP_CACHE_DIR_HOST="${PIP_CACHE_DIR-"$SCRIPT_DIR/.docker_pip_cache"}"
PIP_CACHE_ARGS=()
if [ -n "$PIP_CACHE_DIR_HOST" ]; then
    mkdir -p "$PIP_CACHE_DIR_HOST"
    # コンテナ内は root で実行されるため、キャッシュディレクトリの所有者が
    # ホストユーザーのままだと pip が「所有者不一致」とみなしキャッシュを無効化する。
    if [ "$(stat -c %u "$PIP_CACHE_DIR_HOST" 2>/dev/null)" != "0" ]; then
        docker run --rm -v "$PIP_CACHE_DIR_HOST:/cache" "$IMAGE" chown -R root:root /cache \
            2>/dev/null || true
    fi
    PIP_CACHE_ARGS=(-v "$PIP_CACHE_DIR_HOST:/root/.cache/pip")
fi

echo "[run_evaluation] 対象         : $ZIP_PATH"
echo "[run_evaluation] イメージ     : $IMAGE"
echo "[run_evaluation] GPU          : ${GPU_ARGS:-（無効）}"
echo "[run_evaluation] 追加引数     : ${PIPELINE_ARGS[*]:-（既定: track1 / n_eval_episodes=20 / max_steps=600）}"
echo "[run_evaluation] 結果出力先   : $SCRIPT_DIR/results/"
echo

# evaluate.py / validate_submission.py はイメージのビルド時点のスナップショットが
# 焼き込まれているため、リポジトリを更新した場合はホスト側の最新版を上書きマウントして使う。
DOCKER_CMD=(
    docker run --rm $GPU_ARGS
    -v "$ZIP_ABS:/workspace/submission.zip:ro"
    -v "$SCRIPT_DIR/evaluate.py:/workspace/evaluate.py:ro"
    -v "$SCRIPT_DIR/validate_submission.py:/workspace/validate_submission.py:ro"
    -v "$SCRIPT_DIR/results:/workspace/results"
    "${PIP_CACHE_ARGS[@]}"
    "$IMAGE"
    python evaluate.py /workspace/submission.zip "${PIPELINE_ARGS[@]}"
)

if [ "${DRY_RUN:-0}" = "1" ]; then
    printf '%q ' "${DOCKER_CMD[@]}"
    echo
    exit 0
fi

"${DOCKER_CMD[@]}"
