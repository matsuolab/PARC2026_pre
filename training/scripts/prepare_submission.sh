#!/bin/bash
# ============================================================
# 提出候補（Env + LoRA候補）を submission_template/model_weights/ へ
# マージ・配置し、check_submission.sh で検証するまでを一気通貫で行う。
#
# 学習ループの中では呼ばない。フル評価まで完走した候補を選んだ時点で
# 手動実行する（DESIGN.md §5.3）。
#
# 使い方:
#   ./training/scripts/prepare_submission.sh <env_id> <candidate>
#   例: ./training/scripts/prepare_submission.sh 001_20260801_smolvla_lr3e-4 step_002000
#
# 処理順序:
#   1. config.yaml の base_model.checksum と CHECKSUM.txt の一致を検証
#   2. merge_lora.py で base + LoRA を結合
#   3. VLM processor/tokenizer を同梱し vlm_model_name をローカル化（merge_lora.py内）
#   4. submission_template/model_weights/ へ配置
#   5. SOURCE.json を書き出す
#   6. check_submission.sh 相当の静的チェックを実行
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [ $# -lt 2 ]; then
    echo "使い方: $0 <env_id> <candidate>" >&2
    echo "例:     $0 001_20260801_smolvla_lr3e-4 step_002000" >&2
    exit 1
fi

ENV_ID="$1"
CANDIDATE="$2"
ENV_DIR="$REPO_ROOT/training/Env/$ENV_ID"
MODEL_WEIGHTS_DIR="$REPO_ROOT/submission_template/model_weights"

if [ ! -d "$ENV_DIR" ]; then
    echo "[prepare_submission] エラー: Env が見つかりません: $ENV_DIR" >&2
    exit 1
fi

if [ -n "${PYTHON:-}" ]; then
    PY="$PYTHON"
elif [ -x "venv/bin/python" ]; then
    PY="venv/bin/python"
else
    PY="python3"
fi

echo "[prepare_submission] 対象: env=$ENV_ID candidate=$CANDIDATE"

TMP_MERGE_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_MERGE_DIR"' EXIT

echo "[prepare_submission] ステップ1/3: merge_lora.py 実行"
"$PY" "$SCRIPT_DIR/merge_lora.py" \
    --env-dir "$ENV_DIR" \
    --candidate "$CANDIDATE" \
    --output-dir "$TMP_MERGE_DIR"

echo "[prepare_submission] ステップ2/3: submission_template/model_weights/ へ配置"
# pi0 の重みが残っている場合でも先行削除はしない（DESIGN.md §8:
# SmolVLA が check_submission を通過するまで pi0 を残す方針のため、
# 既存の pi0_* / paligemma_tokenizer ディレクトリには触れない）。
mkdir -p "$MODEL_WEIGHTS_DIR"
DEST_DIR="$MODEL_WEIGHTS_DIR/smolvla_merged"
rm -rf "$DEST_DIR"
mv "$TMP_MERGE_DIR" "$DEST_DIR"

echo "[prepare_submission] ステップ3/3: SOURCE.json を書き出し + 静的チェック"
"$PY" - "$ENV_DIR" "$CANDIDATE" "$MODEL_WEIGHTS_DIR" "$REPO_ROOT" <<'PYEOF'
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

env_dir, candidate, model_weights_dir, repo_root = (Path(p) for p in sys.argv[1:5])

config = yaml.safe_load((env_dir / "config.yaml").read_text(encoding="utf-8"))

git_hash = (env_dir / "git_hash.txt").read_text(encoding="utf-8").strip()

full_eval_score = None
results_path = env_dir / "results.json"
if results_path.is_file():
    try:
        results = json.loads(results_path.read_text(encoding="utf-8"))
        tracks = results.get("tracks") or []
        if tracks:
            full_eval_score = tracks[0].get("overall_score")
    except Exception:
        pass

source = {
    "env_id": config.get("env_id"),
    "arch": config.get("arch"),
    "candidate": candidate,
    "base_model_repo": config["base_model"]["repo_id"],
    "base_model_revision": config["base_model"]["revision"],
    "base_model_checksum": config["base_model"].get("checksum"),
    "vlm_repo": config["vlm"]["repo_id"],
    "git_hash": git_hash,
    "wandb_run_id": config.get("wandb_run_id"),
    "merged_at": datetime.now(timezone.utc).isoformat(),
    "full_eval_score": full_eval_score,
}

(model_weights_dir / "SOURCE.json").write_text(
    json.dumps(source, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
print(f"[prepare_submission] SOURCE.json を書き出しました: {model_weights_dir / 'SOURCE.json'}")
PYEOF

echo "[prepare_submission] 静的チェック（validate_submission.py --static 相当）"
"$PY" "$REPO_ROOT/validate_submission.py" "$REPO_ROOT/submission_template" --static

echo "[prepare_submission] 完了。動的スモークテストは ./check_submission.sh --install で実施してください"
