# PARC2026 実験管理・提出フロー 実装仕様書

本書は PARC2026 に向けた学習実験の管理体制と提出物生成フローの仕様を定める。
設計判断とその根拠を含むため、実装時に「なぜそうなっているか」を本書で確認できる。

- 対象リポジトリ: `PARC2026_pre/`
- 想定読者: 本仕様を実装するエンジニア
- 最終更新: 2026-08-01

---

## 0. 前提となる現状（実装前に把握すべき事実）

実装前に以下を事実として認識しておくこと。すべて現リポジトリで確認済み。

| 項目 | 現状 |
|---|---|
| 現行モデル | pi0（`submission_template/model_weights/pi0_libero_finetuned_v044`、**約6.6GB**） |
| 移行先モデル | SmolVLA + LoRA（`examples/smolvla_libero_spatial_lora.ipynb` に準拠） |
| 推論側 lerobot | **0.4.4 のソースを `submission_template/lerobot/` に直接同梱** |
| 学習側 lerobot | **v0.6.0**（ノートブックが GitHub から clone） |
| 採点環境 | **外部通信を遮断**（`submission_template/requirements.txt` のコメントに明記） |
| 1エピソード平均時間 | **約75.3秒**（`results/server_8000.json` の `mean_avg_episode_time_sec`） |
| 検証スクリプト | `validate_submission.py` / `check_submission.sh` が既に存在 |

### 0.1 wandb 混入リスクの調査結果（対応不要と判断した根拠）

ベンダリング済み `submission_template/lerobot/` には wandb を参照するファイルが7つ存在するが、
**推論経路からは到達しない**ことを確認済み。

- `lerobot/rl/wandb_utils.py:86` の `import wandb` は**関数内の遅延 import**
- `policy_server.py` が import するのは `lerobot.configs.policies` / `lerobot.policies.pi0` /
  `lerobot.processor` / `lerobot.utils.constants` のみで、`lerobot.rl` や
  `lerobot.configs.train` には到達しない

→ **wandb 検出のスコープを `policy_server.py` と `requirements.txt` のみに限定してよい。**
   ベンダリングツリー全体を grep すると誤検知するため、**しない**こと。

---

## 1. フォルダ構成

```
PARC2026_pre/
├── training/
│   ├── base_models/
│   │   └── smolvla_base/                    # 基盤モデルのローカルスナップショット（共有）
│   │       └── CHECKSUM.txt                 # sha256 等でバージョン固定
│   ├── Env/                                 # 実験管理のメインフォルダ
│   │   └── 001_20260801_smolvla_lr3e-4/     # NNN_YYYYMMDD_説明
│   │       ├── weights/
│   │       │   └── candidates/
│   │       │       ├── step_001000/         # LoRAアダプタ（adapter_config.json +
│   │       │       ├── step_002000/         #   adapter_model.safetensors）
│   │       │       └── ...                  # 最大 K 本（既定 K=5）
│   │       ├── checkpoints_log.jsonl        # step毎の記録（後述 §4.3）
│   │       ├── config.yaml                  # 実験設定（後述 §2）
│   │       ├── results.json                 # フル評価の結果
│   │       ├── screening.json               # 一次選抜の結果（後述 §4.2）
│   │       ├── git_hash.txt                 # 実行時のコミットハッシュ
│   │       └── run_memo.txt                 # 実行時のメモ
│   ├── data/                                # 原則空（§6 参照）
│   ├── imitation/                           # 模倣学習スクリプト
│   ├── rl/                                  # 強化学習スクリプト
│   └── scripts/
│       ├── save_run.sh                      # Env フォルダ雛形生成 + git_hash 記録
│       ├── merge_lora.py                    # base + LoRA → 結合モデル（§5）
│       └── prepare_submission.sh            # マージ〜提出物生成〜検証（§5.3）
├── submission_template/
│   ├── policy_server.py
│   ├── requirements.txt
│   └── model_weights/
│       ├── SOURCE.json                      # マージ元の追跡情報（§5.2）
│       └── ...                              # 結合済み重み一式
├── results/                                 # pipeline 標準の出力先
├── pipeline/                                # 公式評価パイプライン（変更不可）
├── validate_submission.py                   # wandb 検出を追加（§3）
└── compe/t1/
```

**命名規則**: `Env/` 配下は `NNN_YYYYMMDD_説明`（例: `001_20260801_smolvla_lr3e-4`）。
`NNN` は連番でゼロ埋め3桁。

---

## 2. `config.yaml` 仕様

各 Env フォルダ直下に配置する。**再現に必要な情報をすべてここに集約する。**

```yaml
# --- 実験の識別 ---
env_id: "001_20260801_smolvla_lr3e-4"
arch: "smolvla"                # "smolvla" | "pi0"（移行期間中は両対応が必要）
wandb_run_id: "abc123xyz"      # W&B の Run ID。Env と W&B Run の紐付けに必須
wandb_run_url: "https://wandb.ai/<entity>/<project>/runs/abc123xyz"

# --- 基盤モデルの固定 ---
base_model:
  repo_id: "lerobot/smolvla_libero_plus"
  revision: "7bb70aa5bc92b82c9239142775d3a173103567ff"
  local_path: "training/base_models/smolvla_base"
  checksum: "<sha256>"         # CHECKSUM.txt と一致すること
vlm:
  repo_id: "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
  local_path: "training/base_models/smolvlm2_processor"

# --- データセットの固定 ---
dataset:
  repo_id: "lerobot/libero_plus"
  revision: "f3f49f426d75030177b18778374005bc12ccd588"

# --- 学習ハイパーパラメータ ---
train:
  steps: 3000
  batch_size: 1
  learning_rate: 3.0e-4
  final_learning_rate: 3.0e-5
  warmup_steps: 100
  lora_r: 16
  lora_alpha: 16
  seed: 42

# --- 評価の設定 ---
eval:
  port: 8000                   # Env 毎にずらす（§7 GPU/ポート競合対策）
  seed: 2026                   # 全候補で共通。相対評価の精度確保に必須

  val_loss_interval: 500       # オフライン validation loss の記録間隔
  task_eval_interval: 1000     # 中間タスク評価の間隔

  # 中間タスク評価（学習中、1000 step 毎）
  intermediate:
    n_tasks: 2
    episodes_per_task: 5

  # 候補保持
  top_k: 5                     # candidates/ に残す本数（オプションで変更可）

  # 一次選抜（K本の候補をふるい落とす。規模は config 化必須）
  screening:
    n_tasks: 10
    n_viewpoints: 1
    episodes_per_task: 2       # 計 20 エピソード / 候補
    prune:
      enabled: true
      after_episodes: 5        # 最初の N エピソード時点で判定
      min_successes: 1         # 成功が min_successes 未満なら打ち切り

  # 本選（フル評価）
  full:
    n_tasks: 10
    n_viewpoints: 3
    episodes_per_task: 5       # 計 150 エピソード / 候補
    n_finalists: 2             # 一次選抜の上位何本をフル評価するか
```

---

## 3. W&B 非送信の担保（`validate_submission.py` への追加）

### 3.1 方針

「提出物に W&B を混入させない」を**運用ルールではなくコードで強制する**。
検出時は warning ではなく **error**（提出ブロック）とする。

### 3.2 実装箇所

既存の関数にロジックを追加する。新規関数を作るより既存フックへ足すこと。

| 関数 | 行 | 追加内容 |
|---|---|---|
| `_check_policy_server_source()` | `validate_submission.py:260` | `policy_server.py` 内の wandb 参照検出 |
| `check_requirements()` | `validate_submission.py:303` | `requirements.txt` 内の wandb 依存検出 |
| `smoke_test()` | `validate_submission.py:513` | サーバー起動時に `WANDB_MODE=disabled` を注入 |

### 3.3 検出ルール

**対象は `policy_server.py` と `requirements.txt` のみ。**
`submission_template/lerobot/` は §0.1 の理由により**対象外**。

`_check_policy_server_source()` では、既に `ast.parse()` 済みなので
**AST を走査して import を検出する**のが望ましい（文字列リテラルやコメントでの誤検知を避けるため）。

- `import wandb` / `from wandb import ...` → `report.error("policy.wandb_import", ...)`
- `wandb.init(` などの呼び出し → `report.error("policy.wandb_call", ...)`
- `WANDB_` で始まる環境変数への代入 → `report.error("policy.wandb_env", ...)`

`check_requirements()` では、既存のパース結果（`Requirement` オブジェクト）を使い、
パッケージ名が `wandb` に正規化されるものがあれば `report.error("req.wandb", ...)`。

エラーコードは既存の命名規則（`policy.*` / `req.*`）に合わせること。

### 3.4 受け入れ条件

- `policy_server.py` に `import wandb` を1行足すと `check_submission.sh` が**失敗する**
- `requirements.txt` に `wandb` を足すと `check_submission.sh` が**失敗する**
- 現状の提出物（pi0）はこの変更後も**従来通り通過する**（誤検知が無いこと）

---

## 4. チェックポイント運用と best 選定

### 4.1 保存対象

**LoRA アダプタのみを保存する。** 基盤モデルは `training/base_models/` で共有するため
Env 毎に複製しない。これにより 1 Env あたりのディスク消費が 6.6GB 級から数十MB級になる。

- 保存先: `weights/candidates/step_XXXXXX/`
- 中身: `adapter_config.json` + `adapter_model.safetensors`
  （**単一ファイルではなくディレクトリ**である点に注意）
- 保持数: `eval.top_k` 本（既定 5）。超過分は中間タスク評価スコアが最も低いものから削除

### 4.2 2段階スクリーニング

**単一の `best` を中間評価で決め打ちしない。** 理由は以下の通り。

中間タスク評価は 3〜10 エピソードしかなく、success rate の分解能が粗い
（8エピソードなら 12.5% 刻み）。実際 `results/server_8000.json` の実績は
`n_total_episodes: 8` に対し `overall_score: 0.125`（8回中1回成功）であり、
この粒度で best を決めると**ノイズを選ぶ**動きになる。

一方でフル評価は 150 エピソード/候補 × 約75.3秒 ≒ **3.1時間/候補**。
K=5 を全部フル評価すると約15.7時間かかり非現実的。

よって以下の2段階とする。

**ステージ1: 一次選抜（screening）**
- 対象: `candidates/` の K 本すべて
- 規模: 10タスク × 1視点 × 2エピソード = 20エピソード/候補（config 化）
- 所要: 約25分/候補 → K=5 で約2時間
- 出力: `screening.json`

**ステージ2: 本選（full）**
- 対象: 一次選抜の上位 `n_finalists` 本（既定 2）
- 規模: 10タスク × 3視点 × 5エピソード = 150エピソード/候補
- 所要: 約3.1時間/候補 → 2本で約6.2時間
- 出力: `results.json`

合計 約8時間。K=5 の保険を維持したまま現実的な時間に収まる。

**同点時の tie-break**: success rate が同点の場合、500 step 毎に記録した
オフライン validation loss が小さい方を上位とする。
（12.5%刻みのため同点は頻発する前提で設計すること。）

### 4.3 seed の共通化（必須）

一次選抜では**全候補で完全に同一のタスク・初期状態**を使う。
少ないエピソード数でもランキング精度が大幅に向上するため。

pipeline は既にこれに対応している。**新規実装は不要で、同じ `--seed` を渡すだけでよい。**

- `pipeline/cli.py:87` — `--seed` が CLI 引数として露出済み
- `pipeline/rollout.py:190` — `episode_seed = self.config.seed + episode_id`（エピソード単位で決定論的）
- `pipeline/environment.py:104` — `env.seed(self.config.seed)`

→ `config.yaml` の `eval.seed` を一次選抜・本選の両方で固定し、全候補に同一値を渡すこと。

### 4.4 早期打ち切り（Prune）

一次選抜の途中で明らかに劣悪な候補を打ち切り、さらに時間を短縮する。
既定は「最初の5エピソードで成功0件なら打ち切り」（`eval.screening.prune`）。

**実装上の必須ルール（ここを誤ると評価が壊れる）:**

1. **Prune 済み候補と完走候補の success rate を数値比較してはならない。**
   5エピソードで打ち切った候補は分母5、完走候補は分母20であり、
   同じ `0.0` でも意味が異なる。Prune 済み候補は**ランキングから除外**し、
   部分スコアで順位付けしないこと。

2. **Prune 判定は固定の絶対閾値で行う。**
   他候補との相対比較で動く閾値にしないこと（評価順序に結果が依存するため）。

3. **エピソード順序も共通 seed で固定する。**
   順序が候補ごとに異なると、たまたま難しいタスクが先に来た候補が不当に Prune される。

4. **監査可能性を確保する。** `checkpoints_log.jsonl` に
   `pruned: true` / `pruned_at_episode: N` を記録すること。

### 4.5 `checkpoints_log.jsonl` 仕様

1行1レコードの JSON Lines。追記のみ。

```jsonl
{"step": 500,  "timestamp": "2026-08-01T10:00:00", "val_loss": 0.412}
{"step": 1000, "timestamp": "2026-08-01T10:20:00", "val_loss": 0.388, "eval_score": 0.20, "saved": "candidates/step_001000"}
{"step": 2000, "timestamp": "2026-08-01T10:40:00", "val_loss": 0.371, "eval_score": 0.10, "saved": "candidates/step_002000"}
{"stage": "screening", "candidate": "step_002000", "score": 0.05, "pruned": true, "pruned_at_episode": 5}
{"stage": "screening", "candidate": "step_001000", "score": 0.25, "pruned": false, "n_episodes": 20}
{"stage": "full", "candidate": "step_001000", "score": 0.22, "n_episodes": 150}
```

---

## 5. マージと提出物生成

### 5.1 `merge_lora.py`

**ゼロから書かないこと。** `examples/smolvla_libero_spatial_lora.ipynb` の
マージ処理（該当セル、`shutil.rmtree(MERGED_MODEL_DIR)` から
`policy_postprocessor*.safetensors` のコピーまで）に完成した実装がある。
これを移植し、後述のパス書き換えを追加する形にすること。

移植元の処理内容:

```python
merged_policy.config.use_peft = False
merged_policy.config.pretrained_path = None
merged_policy.config.push_to_hub = False
merged_policy.config.repo_id = None
merged_policy.config.device = None
merged_policy.config.load_vlm_weights = False
merged_policy.config.vlm_model_name = VLM_REPO
merged_policy.save_pretrained(MERGED_MODEL_DIR)

# save_pretrained だけでは足りない。以下を checkpoint_dir から明示コピーする
for pattern in [
    "policy_preprocessor.json",
    "policy_preprocessor*.safetensors",
    "policy_postprocessor.json",
    "policy_postprocessor*.safetensors",
]:
    ...
```

#### 5.1.1 【重要】VLM 参照のローカル化

**採点環境は外部通信を遮断されているため、上記のままでは提出物が起動しない。**

`load_vlm_weights = False` により VLM の**重み自体**はマージ済み safetensors に含まれるが、
`config.vlm_model_name` には `"HuggingFaceTB/SmolVLM2-500M-Video-Instruct"` という
**Hub の repo-id 文字列がそのまま残る**。ロード時に SmolVLA がこの文字列で
processor / tokenizer / config を解決しようとして失敗する。

対処は、現行 pi0 が `paligemma_tokenizer` を同梱して回避しているのと**まったく同じ**:

1. SmolVLM2 の processor / tokenizer / config 一式を
   `submission_template/model_weights/smolvlm2_processor/` に同梱する
2. `merged_policy.config.vlm_model_name` を**同梱先へのパスに書き換えて**保存する
   （提出物のディレクトリ構成に対して解決可能なパスにすること）

これは pi0 固有の問題ではなくアーキ非依存の制約であり、
**pi0 を削除しても消えない。** pi0 が確立した「同梱パターン」は踏襲対象である。

#### 5.1.2 受け入れ条件

- ネットワークを遮断した状態でマージ済みモデルがロードできること
- マージ後のディレクトリに重み・config・preprocessor/postprocessor・
  VLM processor/tokenizer が**すべて**揃っていること
- `check_submission.sh --install` の動的スモークテストを通過すること

### 5.2 `SOURCE.json`

`submission_template/model_weights/` へコピーした瞬間に「どの Env のどの候補か」の
紐付けが切れるため、追跡情報を書き出す。

```json
{
  "env_id": "001_20260801_smolvla_lr3e-4",
  "arch": "smolvla",
  "candidate": "step_001000",
  "step": 1000,
  "base_model_repo": "lerobot/smolvla_libero_plus",
  "base_model_revision": "7bb70aa5bc92b82c9239142775d3a173103567ff",
  "base_model_checksum": "<sha256>",
  "vlm_repo": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
  "git_hash": "<commit hash>",
  "wandb_run_id": "abc123xyz",
  "merged_at": "2026-08-01T12:00:00",
  "full_eval_score": 0.22
}
```

### 5.3 `prepare_submission.sh`

**学習ループ内ではマージしない。** 提出候補を選定した時点で手動実行する。

処理順序:

1. 引数で Env ID と候補（`step_XXXXXX`）を受け取る
2. `config.yaml` の `base_model.checksum` と `CHECKSUM.txt` の一致を検証（不一致なら中断）
3. `merge_lora.py` を実行し、base + LoRA を結合
4. VLM processor/tokenizer 一式を同梱し、`vlm_model_name` をローカルパスへ書き換え
5. `submission_template/model_weights/` へ配置
6. `SOURCE.json` を書き出す
7. `check_submission.sh` を呼び出して検証

**ディスク注意**: マージ中は結合前後で一時的に2倍の容量を要する。事前確認すること。

---

## 6. `.gitignore` の更新

現状の `.gitignore` は `results/` と `submission_template/model_weights/` を除外済みだが、
**新設する `training/` は一切カバーされていない。** このまま作ると
`git add -A` 一発で数十GBが追跡対象になる。

**ディレクトリ単位で丸ごと除外してはならない。** メタファイル
（`config.yaml` / `checkpoints_log.jsonl` / `git_hash.txt` / `run_memo.txt` /
`results.json` / `screening.json`）は**追跡したい**ため、ホワイトリスト方式にすること。

追加内容（方針）:

- 除外する: `training/base_models/`、`training/Env/**/weights/`、`training/data/`
- 追跡する: 上記メタファイル群

### 6.1 運用ルール: `HF_HOME` はリポジトリ外に固定する

データセットは都度ダウンロードする方針のため、通常 `training/data/` は空のままである
（ノートブックは `HF_HOME=/content/hf_cache` を設定しており、キャッシュはそちらに落ちる）。

ただし `HF_HOME` をリポジトリ内に向けると一発で数十GBが追跡対象になるため、
**「`HF_HOME` および `HF_LEROBOT_HOME` はリポジトリ外に置く」を運用ルールとして明文化する。**
`training/data/` の除外エントリはその保険として残す。

---

## 7. GPU / ポート競合への対処

中間評価は pipeline が HTTP 経由（`--server-url http://localhost:8000`）で
ポリシーサーバーを叩く構成のため、以下の競合が起きる。

- 学習プロセスと評価用サーバーが同一 GPU を奪い合う
- 複数 Env を並行学習させるとポート 8000 が衝突する

対処として `config.yaml` に `eval.port` を持たせ、**Env 毎にポートをずらす**。
（例: Env 001 → 8000、Env 002 → 8001）

---

## 8. pi0 → SmolVLA 移行の順序

**pi0 は現時点で唯一の動作実績があるベースライン**である
（`submission/submission.zip` が存在し、スコア 0.125 の評価実績あり）。
SmolVLA の疎通が取れる前に pi0 を削除すると、フォールバックを失った状態で
ベンダリング作業に入ることになる。

**必須の順序:**

1. pi0 を残したまま SmolVLA 環境を並行構築する
   （`submission_template/model_weights/` は既に `.gitignore` 済みなので、
     pi0 と SmolVLA の重みディレクトリを並置できる）
2. `config.yaml` の `arch` フィールドで切り替える
   （`merge_lora.py` / `prepare_submission.sh` は arch-aware に実装すること）
3. SmolVLA が `check_submission.sh` を通過し、フル評価まで完走したことを確認
4. **その後**に pi0 関連ファイル（`model_weights/pi0_*`、`paligemma_tokenizer`、
   `siglip_patch.py` 等）を削除する

---

## 9. 別タスク（本仕様の範囲外）

以下は独立したタスクとして扱う。本仕様の実装とは分離すること。

### lerobot のバージョン統一

- **学習側**: ノートブックが `LEROBOT_TAG = "v0.6.0"` を GitHub から clone
- **推論側**: `submission_template/lerobot/` は **0.4.4** のソースを直接同梱

この不整合により、lerobot 0.6.0 で保存したマージ済み SmolVLA モデルが
同梱 0.4.4 でロードできない可能性がある。

必要な作業:

1. lerobot 0.6.0 の smolvla サブツリーを、現行と同じ方式で再ベンダリングする
   （現行が部分同梱している理由は `requirements.txt` のコメントに明記されている:
     lerobot のベース依存 `pynput` が Linux で `evdev` を要求し、`evdev` は
     プリビルド wheel が無くソースビルド時に `Python.h` を要求するため、
     採点環境の Docker に `python3-dev` が無いとビルドが失敗する)
2. `torch==2.10.0` / `transformers==4.57.6` のピンを SmolVLA + 0.6.0 向けに張り直す
3. `siglip_patch.py`（pi0 用のモンキーパッチ）の要否を再判断する

「`policy_server.py` のモデルパスを差し替えるだけ」では済まない点に注意。

---

## 10. 実装チェックリスト

- [ ] `.gitignore` を更新（§6）— **他の作業より先に行うこと**
- [ ] `training/` のディレクトリ構造を作成（§1）
- [ ] `config.yaml` テンプレートを作成（§2）
- [ ] `validate_submission.py` に wandb 検出を追加（§3）
- [ ] `check_submission.sh` で現行 pi0 提出物が引き続き通過することを確認（§3.4）
- [ ] `save_run.sh` を作成（Env 雛形生成 + `git_hash.txt` 記録）
- [ ] 学習スクリプトに val loss 記録（500 step）/ 中間評価（1000 step）/
      候補保存（top_k）/ W&B ログを実装（§4）
- [ ] 一次選抜スクリプトを実装（共通 seed + Prune、§4.2〜4.4）
- [ ] 本選（フル評価）スクリプトを実装（§4.2）
- [ ] `merge_lora.py` を実装（ノートブックから移植 + VLM ローカル化、§5.1）
- [ ] `prepare_submission.sh` を実装（§5.3）
- [ ] `SOURCE.json` の書き出しを実装（§5.2）

---

## 付録: 設計判断の要約

| 論点 | 決定 | 根拠 |
|---|---|---|
| wandb 検出スコープ | `policy_server.py` + `requirements.txt` のみ | 推論経路が wandb に到達しないことを確認済み（§0.1）。全体 grep は誤検知する |
| 検出時の重大度 | error（提出ブロック） | 運用ルールでは漏れるため |
| チェックポイント | LoRA アダプタのみ、K本保持 | 1本6.6GB → 数十MB。ディスク制約が消えるため1本に絞る必然性がない |
| best 選定 | 2段階スクリーニング | 中間評価は12.5%刻みでノイズ。フル評価は K=5 で15.7時間かかる |
| seed | 全候補で共通固定 | 少エピソードでも相対評価精度が向上。pipeline が既に対応済み |
| Prune | 固定絶対閾値、部分スコアは順位付けに使わない | 分母が異なる候補の数値比較は無効 |
| 基盤モデル | ローカルキャッシュ + checksum | 採点環境がオフライン。Hub revision 依存を避ける |
| マージ | 手動実行（学習ループ外） | 学習中にマージする必要がない。時間とディスクの無駄 |
| VLM | processor/tokenizer をローカル同梱 | 採点環境が外部通信遮断。pi0 の paligemma_tokenizer と同じ制約 |
| 移行順序 | pi0 通過確認後に削除 | pi0 が唯一の動作実績ベースライン |
