#!/bin/bash
# ============================================================
# 提出前の軽量チェックスクリプト
#
# submission/ フォルダに格納済みの提出 zip に対して、
# 静的チェック → 動的スモークテストの順に validate_submission.py を実行する。
# 実タスク評価（pipeline / evaluate.py）は含まない、素早い健全性確認用。
#
# 使い方:
#   ./check_submission.sh                     # submission/*.zip を自動検出
#   ./check_submission.sh submission/foo.zip  # zip を明示指定
#   ./check_submission.sh --install           # スモーク前に requirements.txt を入れる
#
# --install について:
#   評価環境（Docker イメージなど）には提出物の依存（torch/transformers/lerobot 等）は
#   入っていない。採点時は requirements.txt から専用 venv へインストールされるため、
#   同じことをローカルで再現したい場合に --install を付ける。
#   注意: validate_submission.py の --install は venv 分離をせず「現在の Python 環境」へ
#   インストールするため、評価側スタックのバージョンを書き換える可能性がある。
#   使い捨てのコンテナ（docker run --rm）内で使うこと。
#
#   使用する Python は $PYTHON > ./venv/bin/python > python3 の順で決まる。
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

SUBMISSION_DIR="submission"
INSTALL_ARGS=()
ZIP_PATH=""

for arg in "$@"; do
    case "$arg" in
        --install)
            INSTALL_ARGS=(--install)
            ;;
        -*)
            echo "[check_submission] エラー: 不明なオプション: $arg" >&2
            exit 1
            ;;
        *)
            ZIP_PATH="$arg"
            ;;
    esac
done

if [ -n "$ZIP_PATH" ]; then
    if [ ! -f "$ZIP_PATH" ]; then
        echo "[check_submission] エラー: 指定された zip が見つかりません: $ZIP_PATH" >&2
        exit 1
    fi
else
    if [ ! -d "$SUBMISSION_DIR" ]; then
        echo "[check_submission] エラー: '$SUBMISSION_DIR' フォルダが存在しません" >&2
        exit 1
    fi

    mapfile -t ZIP_CANDIDATES < <(find "$SUBMISSION_DIR" -maxdepth 1 -name "*.zip" | sort)

    if [ "${#ZIP_CANDIDATES[@]}" -eq 0 ]; then
        echo "[check_submission] エラー: '$SUBMISSION_DIR' に zip ファイルが見つかりません" >&2
        exit 1
    fi

    if [ "${#ZIP_CANDIDATES[@]}" -gt 1 ]; then
        echo "[check_submission] エラー: '$SUBMISSION_DIR' に複数の zip があります。対象を1つ指定してください:" >&2
        printf '  %s\n' "${ZIP_CANDIDATES[@]}" >&2
        exit 1
    fi

    ZIP_PATH="${ZIP_CANDIDATES[0]}"
fi

echo "[check_submission] 対象: $ZIP_PATH"

# --- 使用する Python の決定 ---
# 動的スモークテストは提出物のサーバーを「この Python で」起動するため、
# fastapi / uvicorn / msgpack など提出物の依存が入った環境である必要がある。
# 優先順: $PYTHON 環境変数 > setup.sh が作る ./venv > python3
if [ -n "${PYTHON:-}" ]; then
    PY="$PYTHON"
elif [ -x "venv/bin/python" ]; then
    PY="venv/bin/python"
else
    PY="python3"
fi
echo "[check_submission] 使用する Python: $PY"

if [ ${#INSTALL_ARGS[@]} -eq 0 ] \
   && ! "$PY" -c "import fastapi, uvicorn, msgpack, numpy, requests" 2>/dev/null; then
    echo "[check_submission] 警告: '$PY' に提出物の依存（fastapi/uvicorn/msgpack/numpy/requests）が" >&2
    echo "                   揃っていません。動的スモークテストはサーバー起動に失敗します。" >&2
    echo "                   'source env.sh' で venv を有効化するか、PYTHON=... で明示してください。" >&2
    echo "                   提出物の requirements.txt ごと入れて確認するなら --install を付けてください。" >&2
fi

echo "[check_submission] ステップ1/2: 静的チェック"
"$PY" validate_submission.py "$ZIP_PATH" --static

if [ ${#INSTALL_ARGS[@]} -gt 0 ]; then
    echo "[check_submission] ステップ2/2: 依存インストール + 動的スモークテスト"
else
    echo "[check_submission] ステップ2/2: 動的スモークテスト（サーバー起動あり）"
fi
"$PY" validate_submission.py "$ZIP_PATH" "${INSTALL_ARGS[@]}"

echo "[check_submission] 完了: 軽量チェックを通過しました"
