# training/

PARC2026 の学習実験管理フォルダ。詳細な設計根拠は [DESIGN.md](../DESIGN.md) を参照。

```
training/
├── base_models/      # 基盤モデルのローカル共有キャッシュ（.gitignore 対象）
├── Env/               # 実験管理のメインフォルダ。NNN_YYYYMMDD_説明 の形式
├── data/              # 通常は空。運用ルールにより HF_HOME はリポジトリ外に固定する
├── imitation/         # 模倣学習スクリプト
├── rl/                 # 強化学習スクリプト
└── scripts/
    ├── save_run.sh          # Env フォルダ雛形生成 + git_hash.txt 記録
    ├── screen_candidates.py # 2段階スクリーニング（一次選抜 → 本選）
    ├── merge_lora.py         # base + LoRA → 結合モデル
    └── prepare_submission.sh # マージ〜提出物生成〜検証
```

## 運用ルール

- **`HF_HOME` / `HF_LEROBOT_HOME` はリポジトリ外に固定すること。**
  `training/data/` を追跡対象にしないための保険。
- 学習ループ内ではモデルをマージしない。提出候補を選んだ時点で
  `scripts/prepare_submission.sh` を手動実行する。
- `weights/candidates/` には LoRA アダプタのみを保存する（基盤モデルは複製しない）。
