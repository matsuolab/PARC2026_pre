# training/base_models/

基盤モデルのローカル共有キャッシュ。`.gitignore` により中身は追跡されない
（`training/README.md` 参照）。全 Env で共有し、Env ごとに複製しない。

## セットアップ

```bash
# 例: smolvla_base
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='lerobot/smolvla_libero_plus',
    revision='7bb70aa5bc92b82c9239142775d3a173103567ff',
    local_dir='training/base_models/smolvla_base',
)
"

# CHECKSUM.txt を作成（config.yaml の base_model.checksum と突き合わせる）
find training/base_models/smolvla_base -type f -name '*.safetensors' \
    -exec sha256sum {} \; | sort | sha256sum | awk '{print $1}' \
    > training/base_models/smolvla_base/CHECKSUM.txt
```

VLM processor（`smolvlm2_processor/`）も同様にローカルへ同梱すること
（`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`）。こちらは `merge_lora.py` が
提出物へコピーする際に使用する（DESIGN.md §5.1.1）。

`config.yaml` の `base_model.checksum` はこの `CHECKSUM.txt` の値と一致させる。
`merge_lora.py` はマージ前にこれを検証し、不一致なら中断する。
