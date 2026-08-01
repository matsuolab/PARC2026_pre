"""base モデル + LoRA アダプタを結合し、提出可能なモデルとして書き出す。

マージ処理そのものは examples/smolvla_libero_spatial_lora.ipynb の実装を移植した
ものであり、そこから独自に発明していない（DESIGN.md §5.1 参照）。

【重要】VLM 参照のローカル化について
    採点環境は外部通信を遮断されているため、merge 後にモデルの config へ
    Hugging Face の repo-id 文字列（例: "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"）
    を残したままにすると、ロード時に解決できず起動に失敗する。
    そのため本スクリプトは VLM の processor/tokenizer 一式をローカルへ同梱し、
    config.vlm_model_name をローカルパスへ書き換えてから保存する。
    これは pi0 が paligemma_tokenizer を同梱しているのと同じ制約への対応であり、
    アーキ非依存（DESIGN.md §5.1.1）。

学習ループの中では呼ばない。提出候補を選んだ時点で
training/scripts/prepare_submission.sh から手動実行される想定。

使い方:
    python training/scripts/merge_lora.py \\
        --env-dir training/Env/001_20260801_smolvla_lr3e-4 \\
        --candidate step_002000 \\
        --output-dir /tmp/merged_model
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import io
import shutil
import sys
from pathlib import Path

import yaml


def load_config(env_dir: Path) -> dict:
    config_path = env_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"config.yaml が見つかりません: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def verify_base_model_checksum(base_model_dir: Path, expected_checksum: str) -> None:
    checksum_path = base_model_dir / "CHECKSUM.txt"
    if not checksum_path.is_file():
        raise FileNotFoundError(
            f"CHECKSUM.txt が見つかりません: {checksum_path}\n"
            f"training/base_models/ の基盤モデルスナップショットが未セットアップです。"
        )
    actual = checksum_path.read_text(encoding="utf-8").strip()
    if expected_checksum and actual != expected_checksum:
        raise ValueError(
            f"基盤モデルの checksum が config.yaml と一致しません。\n"
            f"  config.yaml:  {expected_checksum}\n"
            f"  CHECKSUM.txt: {actual}\n"
            f"別バージョンの基盤モデルで学習された可能性があります。マージを中断します。"
        )


def merge(
    base_model_dir: Path,
    checkpoint_dir: Path,
    output_dir: Path,
) -> None:
    """LoRA アダプタを base モデルへマージし、output_dir へ書き出す。

    examples/smolvla_libero_spatial_lora.ipynb のマージセルを移植したもの。
    """
    import torch
    from peft import PeftModel
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    adapter_file = checkpoint_dir / "adapter_model.safetensors"
    if not adapter_file.is_file():
        raise FileNotFoundError(f"LoRA アダプタが見つかりません: {adapter_file}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    merge_config = PreTrainedConfig.from_pretrained(checkpoint_dir)
    merge_config.device = "cpu"
    merge_config.pretrained_path = str(base_model_dir)
    merge_config.use_peft = False

    quiet_output = io.StringIO()
    with (
        contextlib.redirect_stdout(quiet_output),
        contextlib.redirect_stderr(quiet_output),
    ):
        base_policy = SmolVLAPolicy.from_pretrained(
            str(base_model_dir),
            config=merge_config,
            strict=False,
        )

        peft_policy = PeftModel.from_pretrained(
            base_policy,
            checkpoint_dir,
            is_trainable=False,
            torch_device="cpu",
        )

        merged_policy = peft_policy.merge_and_unload(safe_merge=True)

    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_policy.config.use_peft = False
    merged_policy.config.pretrained_path = None
    merged_policy.config.push_to_hub = False
    merged_policy.config.repo_id = None
    merged_policy.config.device = None
    merged_policy.config.load_vlm_weights = False
    # vlm_model_name は localize_vlm_reference() が後でローカルパスへ書き換える。
    # ここでは一旦 config 由来の repo_id を残しておく（元ノートブックと同じ手順）。

    merged_policy.save_pretrained(output_dir)

    for pattern in [
        "policy_preprocessor.json",
        "policy_preprocessor*.safetensors",
        "policy_postprocessor.json",
        "policy_postprocessor*.safetensors",
    ]:
        for source_path in checkpoint_dir.glob(pattern):
            shutil.copy2(source_path, output_dir / source_path.name)


def localize_vlm_reference(output_dir: Path, vlm_local_path: Path, vlm_repo_id: str) -> None:
    """config.json 内の vlm_model_name を Hub repo-id からローカルパスへ書き換える。

    採点環境は外部通信を遮断されているため、このステップを省略すると
    モデルロード時に processor/tokenizer の解決に失敗する（DESIGN.md §5.1.1）。
    """
    import json

    if not vlm_local_path.is_dir():
        raise FileNotFoundError(
            f"VLM processor のローカルスナップショットが見つかりません: {vlm_local_path}\n"
            f"事前に '{vlm_repo_id}' を training/base_models/ へ同梱してください。"
        )

    dest = output_dir / "smolvlm2_processor"
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(vlm_local_path, dest)

    config_path = output_dir / "config.json"
    if config_path.is_file():
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        # 提出物内での相対パス。policy_server.py 側は自身のディレクトリ基準で解決する。
        cfg["vlm_model_name"] = "./smolvlm2_processor"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    else:
        raise FileNotFoundError(
            f"merge 後の config.json が見つかりません: {config_path}\n"
            f"vlm_model_name のローカル化ができませんでした。"
        )


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-dir", required=True, type=Path,
                         help="training/Env/<env_id> のパス")
    parser.add_argument("--candidate", required=True,
                         help="weights/candidates/ 配下の候補名（例: step_002000）")
    parser.add_argument("--output-dir", required=True, type=Path,
                         help="マージ済みモデルの出力先")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2],
                         help="リポジトリルート（既定: このファイルから自動検出）")
    args = parser.parse_args(argv)

    env_dir = args.env_dir.resolve()
    config = load_config(env_dir)

    if config.get("arch") != "smolvla":
        print(f"[merge_lora] エラー: arch='{config.get('arch')}' は未対応です"
              f"（現状 smolvla のみ）", file=sys.stderr)
        return 1

    base_model_cfg = config["base_model"]
    base_model_dir = args.repo_root / base_model_cfg["local_path"]
    verify_base_model_checksum(base_model_dir, base_model_cfg.get("checksum") or "")

    checkpoint_dir = env_dir / "weights" / "candidates" / args.candidate
    if not checkpoint_dir.is_dir():
        print(f"[merge_lora] エラー: 候補が見つかりません: {checkpoint_dir}", file=sys.stderr)
        return 1

    print(f"[merge_lora] base_model: {base_model_dir}")
    print(f"[merge_lora] checkpoint: {checkpoint_dir}")
    print(f"[merge_lora] output: {args.output_dir}")

    merge(base_model_dir, checkpoint_dir, args.output_dir)

    vlm_cfg = config["vlm"]
    vlm_local_path = args.repo_root / vlm_cfg["local_path"]
    localize_vlm_reference(args.output_dir, vlm_local_path, vlm_cfg["repo_id"])

    print("[merge_lora] マージ完了")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
