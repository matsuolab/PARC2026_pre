#!/bin/bash
# ============================================================
# Docker（GPU）上で check_submission.sh（軽量チェック）を実行するホスト側ラッパー
#
# check_submission.sh 自体はコンテナ内で動く前提のスクリプトなので、
# 素の docker run コマンドを毎回手打ちしなくて済むようにする。
# フル評価（実タスク・スコア算出）をしたい場合は run_evaluation.sh を使うこと。
#
# 使い方:
#   ./run_check_submission.sh              # 静的チェック + 動的スモーク（依存は入れない）
#   ./run_check_submission.sh --install    # requirements.txt を入れてから動的スモーク
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE="${IMAGE:-parc2026}"
GPU_ARGS="${GPU_ARGS-"--gpus all"}"

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

echo "[run_check_submission] イメージ: $IMAGE"
echo "[run_check_submission] GPU     : ${GPU_ARGS:-（無効）}"
echo "[run_check_submission] 引数    : $*"
echo

DOCKER_CMD=(
    docker run --rm $GPU_ARGS
    -v "$SCRIPT_DIR/submission:/workspace/submission"
    -v "$SCRIPT_DIR/check_submission.sh:/workspace/check_submission.sh:ro"
    -v "$SCRIPT_DIR/validate_submission.py:/workspace/validate_submission.py:ro"
    "${PIP_CACHE_ARGS[@]}"
    "$IMAGE"
    ./check_submission.sh "$@"
)

if [ "${DRY_RUN:-0}" = "1" ]; then
    printf '%q ' "${DOCKER_CMD[@]}"
    echo
    exit 0
fi

"${DOCKER_CMD[@]}"
