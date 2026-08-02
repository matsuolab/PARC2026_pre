# 参考例（examples）

| ファイル | 内容 |
|---|---|
| [smolvla_libero_plus_track1_mix_lora.ipynb](smolvla_libero_plus_track1_mix_lora.ipynb) | **推奨** Track1 寄りの spatial/object/goal 混成 LoRA |
| [smolvla_libero_spatial_lora.ipynb](smolvla_libero_spatial_lora.ipynb) | Spatial 10 タスクのみ（初期ベースライン。ローカル Track1 では 0%） |

## smolvla_libero_plus_track1_mix_lora.ipynb（推奨）

`lerobot/smolvla_libero_plus` を初期重みとし、公開 Track1 例に近い 4 タスク
（drawer / tomato sauce / milk / bowl→stove）と、同スイートの近傍 6 タスクで
LoRA 追加学習する（計 10 × 5 ep、5,000 steps）。

本番タスクは非公開のため、公開 4 タスクだけに過適合しないよう多様化している。

### 使い方（Colab）

1. GitHub URL から開く（ファイル欄ダブルクリック不可）  
   `https://github.com/yohei1126/PARC2026_pre/blob/main/examples/smolvla_libero_plus_track1_mix_lora.ipynb`
2. ランタイムを GPU（T4）に変更
3. §1〜§9 を実行（マージ zip まで）。§10〜§13 のノート内 Spatial 評価は任意・スキップ可
4. ダウンロードした zip を解凍し  
   `submission_template/model_weights/smolvla_libero_plus_track1_mix_lora_merged/` へ配置
5. ローカルで `PARC_MODEL_DIRNAME=smolvla_libero_plus_track1_mix_lora_merged` を付けて eval

想定所要: T4 でおおよそ 45〜90 分（5,000 steps）。

## smolvla_libero_spatial_lora.ipynb

`lerobot/smolvla_libero_plus` を初期重みとし、LIBERO-plus Spatial の 10 タスクを
LoRA で追加学習する。学習後は LoRA を元の重みへマージし、追加学習の前後を
同一条件で比較する。

### 使い方

1. Google Colab で開き、ランタイムのタイプを GPU（T4 で足りる）に変更する
2. 上から順に実行する。所要時間は T4 で数時間程度である
3. マージ済みモデル一式（zip）と、追加学習前後の成功率の比較（CSV）が出力される

学習条件は 10 タスク × 各 5 エピソード（計 50 エピソード）、3,000 steps、
バッチサイズ 1 で、Colab で完走することを優先した最小構成である。
性能を伸ばす場合はここを出発点に、自身の環境で条件を組み直すとよい。

### 提出物にするまでの作業

出力されるのは LeRobot 形式のモデル重みであり、これ単体では提出できない。
[submission_template/](../submission_template/) の `MyPolicy` にモデルを組み込み、
ポリシーサーバーの形にする。観測と action の仕様は
[submission_template/policy_server.py](../submission_template/policy_server.py)
の docstring にある。

推論は 1 リクエストあたり 10 秒以内に収める必要がある
（[ルートの README](../README.md#タイムアウト仕様)）。

### ノートブック内の評価と、本番の採点の違い

ノートブック内の評価は学習の効果を手早く確認するためのもので、採点とは条件が異なる。
出てくる成功率は本番スコアの目安にはならない。

| 項目 | ノートブック | 本番の採点 |
|---|---|---|
| 評価タスク | LIBERO-plus Spatial の 10 タスク | Track 1（`compe/t1/` のタスクセット） |
| 実行方法 | LeRobot の `lerobot-eval` | `python -m pipeline` + 提出したポリシーサーバー |
| 観測の解像度 | 256×256 | 128×128 |
| 1 タスクあたりの試行数 | 3（`EVAL_EPISODES_PER_TASK` で変更可） | 非公開（配布キットの既定は 20） |

試行数が 3 のままだと 1 エピソードの成否で成功率が約 33 ポイント動くため、
追加学習の前後を比べる場合は `EVAL_EPISODES_PER_TASK` を増やすこと。

### 実行環境

ノートブックの環境構築は Colab 向けで、[setup.sh](../setup.sh) とは独立している。
依存パッケージのバージョンが一致しない箇所があるため、評価と提出前チェックは
リポジトリ側の環境（`setup.sh` + `env.sh`）で行うこと。

ノートブックが利用する第三者製ソフトウェア・モデル・データセットのライセンスは、
各配布元の表記を参照すること。
