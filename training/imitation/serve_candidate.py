"""LoRA候補（マージ前）を推論サーバーとして起動する（学習中の評価専用）。

training/scripts/screen_candidates.py から候補ごとに子プロセスとして起動される。
提出用の submission_template/policy_server.py とは別物 —— こちらは
base モデル + LoRA アダプタを「マージせずに」ロードして高速に評価するための
開発用サーバー。ワイヤプロトコル（/health, /reset, /act の msgpack 形式）は
policy_server.py と同一にしてあるので、pipeline からは同じクライアントで叩ける。

【要編集】
    _load_policy() の中身は SmolVLA + PEFT のロード方法に合わせて実装すること。
    examples/smolvla_libero_spatial_lora.ipynb のマージ前半部分
    （PreTrainedConfig → SmolVLAPolicy.from_pretrained → PeftModel.from_pretrained,
    ただし merge_and_unload() は呼ばない）が土台になる。
    このファイル自体（サーバー部分・シリアライゼーション）は変更不要。

使い方:
    python training/imitation/serve_candidate.py <candidate_dir> \\
        --base-model-dir training/base_models/smolvla_base \\
        --port 8000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import msgpack
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Response


def _load_policy(candidate_dir: Path, base_model_dir: Path):
    """base モデル + LoRA アダプタ（candidate_dir）をマージせずにロードする。

    戻り値は get_action(obs: dict) -> np.ndarray と reset(instruction: str) -> None
    を持つオブジェクトであること。
    """
    raise NotImplementedError(
        "SmolVLA + PEFT のロード処理を実装してください。"
        "examples/smolvla_libero_spatial_lora.ipynb のマージ前半部分を参照。"
    )


def deserialize_obs(data: bytes) -> dict[str, np.ndarray]:
    unpacked = msgpack.unpackb(data, raw=False)
    obs = {}
    for key, val in unpacked.items():
        arr = np.frombuffer(val["data"], dtype=np.dtype(val["dtype"]))
        obs[key] = arr.reshape(val["shape"]).copy()
    return obs


def serialize_action(action: np.ndarray) -> bytes:
    return msgpack.packb(
        {"data": action.astype(np.float32).tobytes()},
        use_bin_type=True,
    )


app = FastAPI(title="LoRA Candidate Eval Server")
_policy = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
async def reset_policy(request: Request):
    body = await request.body()
    instruction = ""
    if body:
        import json
        data = json.loads(body)
        instruction = data.get("instruction", "")
    _policy.reset(instruction=instruction)
    return {"status": "ok"}


@app.post("/act")
async def act(request: Request):
    body = await request.body()
    obs = deserialize_obs(body)
    action = _policy.get_action(obs)
    return Response(
        content=serialize_action(action),
        media_type="application/x-msgpack",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    _policy = _load_policy(args.candidate_dir, args.base_model_dir)
    print(f"[serve_candidate] {args.candidate_dir} を {args.host}:{args.port} で起動")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
