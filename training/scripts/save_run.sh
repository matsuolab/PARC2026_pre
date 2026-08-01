#!/bin/bash
# ============================================================
# Env フォルダの雛形を作成する。
#
# 使い方:
#   ./training/scripts/save_run.sh <説明>
#   例: ./training/scripts/save_run.sh smolvla_lr3e-4
#
# 実行すると training/Env/NNN_YYYYMMDD_<説明>/ を作成し、
# config.template.yaml のコピー・git_hash.txt・run_memo.txt・
# checkpoints_log.jsonl（空）・weights/candidates/ を用意する。
# NNN は既存 Env の最大連番 + 1（3桁ゼロ埋め）。
#
# 仕様の根拠: DESIGN.md §1, §10
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_ROOT="$REPO_ROOT/training/Env"

if [ $# -lt 1 ]; then
    echo "使い方: $0 <説明（例: smolvla_lr3e-4）>" >&2
    exit 1
fi

DESC="$1"
DATE_STR="$(date +%Y%m%d)"

mkdir -p "$ENV_ROOT"

LAST_NUM=0
for d in "$ENV_ROOT"/*/; do
    [ -d "$d" ] || continue
    name="$(basename "$d")"
    num="${name%%_*}"
    if [[ "$num" =~ ^[0-9]{3}$ ]]; then
        num10=$((10#$num))
        if [ "$num10" -gt "$LAST_NUM" ]; then
            LAST_NUM="$num10"
        fi
    fi
done

NEXT_NUM=$(printf "%03d" $((LAST_NUM + 1)))
ENV_ID="${NEXT_NUM}_${DATE_STR}_${DESC}"
ENV_DIR="$ENV_ROOT/$ENV_ID"

if [ -e "$ENV_DIR" ]; then
    echo "[save_run] エラー: '$ENV_DIR' は既に存在します" >&2
    exit 1
fi

mkdir -p "$ENV_DIR/weights/candidates"

sed "s/NNN_YYYYMMDD_説明/$ENV_ID/" "$ENV_ROOT/config.template.yaml" > "$ENV_DIR/config.yaml"

if git -C "$REPO_ROOT" rev-parse HEAD > /dev/null 2>&1; then
    git -C "$REPO_ROOT" rev-parse HEAD > "$ENV_DIR/git_hash.txt"
    if ! git -C "$REPO_ROOT" diff --quiet || ! git -C "$REPO_ROOT" diff --cached --quiet; then
        echo "[save_run] 警告: コミットされていない変更があります（git_hash.txt はHEADのみ記録）" >&2
    fi
else
    echo "(git リポジトリ外での実行のため記録なし)" > "$ENV_DIR/git_hash.txt"
fi

cat > "$ENV_DIR/run_memo.txt" <<EOF
env_id: $ENV_ID
created: $(date -Iseconds)

# ここに実験の目的・変更点・気づいたことを書く
EOF

: > "$ENV_DIR/checkpoints_log.jsonl"

echo "[save_run] 作成しました: $ENV_DIR"
echo "[save_run] config.yaml を編集してから学習を開始してください"
