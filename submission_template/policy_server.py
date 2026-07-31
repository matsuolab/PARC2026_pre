"""ポリシーサーバー（提出用テンプレート）

このファイルを編集して、自分のモデルを組み込んでください。
編集が必要なのは MyPolicy クラスの中身だけです。
それ以外のコード（サーバー部分、シリアライゼーション）は変更不可です。

ローカルテスト:
    pip install -r requirements.txt
    python policy_server.py                  # サーバー起動（port 8000）

    # 別ターミナルで評価実行
    python -m pipeline --server-url http://localhost:8000 --dry-run
"""

import argparse
import os
from abc import ABC, abstractmethod

import msgpack
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Response


# ============================================================
# ポリシーのインターフェース定義（変更不可）
# MyPolicy が満たすべき get_action() / reset() の仕様を定める。
# ============================================================


class BasePolicy(ABC):
    """ポリシーの基底クラス。get_action() と reset() を実装してください。"""

    @abstractmethod
    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """観測からアクションを推論する。

        Args:
            obs: 環境からの観測。以下のキーが含まれる:
                - "agentview_image": (128, 128, 3) uint8
                - "robot0_eye_in_hand_image": (128, 128, 3) uint8
                - "robot0_joint_pos": (7,) float
                - "robot0_eef_pos": (3,) float
                - "robot0_eef_quat": (4,) float
                - "robot0_gripper_qpos": (2,) float

        Returns:
            action: (7,) float32 — [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        ...

    @abstractmethod
    def reset(self, instruction: str = "") -> None:
        """エピソード開始時に呼ばれる。内部状態をリセットしてください。

        Args:
            instruction: タスクの言語指示（例: "pick up the red mug and place it on the shelf"）
        """
        ...


# ============================================================
# ここを編集する（MyPolicy の中身だけを自分のモデルに置き換える）
# ============================================================


_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model_weights",
    "pi0_libero_finetuned_v044",
)
# PaliGemma のトークナイザは本来 "google/paligemma-3b-pt-224" から取得するが、
# 同リポジトリは gated（要認証）かつ採点環境は外部通信を遮断するため、
# 同一トークナイザを含む公開リポジトリから取得したファイルをローカルに同梱している。
_TOKENIZER_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model_weights",
    "paligemma_tokenizer",
)


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """クォータニオン (x, y, z, w) を axis-angle（回転ベクトル）へ変換する。

    robosuite の `transform_utils.quat2axisangle` と同一実装
    （lerobot/policies/xvla/utils.py にも同じものがある）。
    角度を 2*acos(w) で求めるため範囲は [0, 2pi] であり、
    ノルムを [0, pi] に畳み込む scipy の `Rotation.as_rotvec()` とは
    w < 0 の場合に結果が異なる。学習データ（HuggingFaceVLA/libero）の
    observation.state はこちらの規約で作られている（統計上、回転成分の
    最大値が 3.67 > pi となっており as_rotvec では表現できない）。
    """
    quat = np.asarray(quat, dtype=np.float64)
    w = float(np.clip(quat[3], -1.0, 1.0))

    den = np.sqrt(1.0 - w * w)
    if np.isclose(den, 0.0):
        # 回転がほぼゼロ
        return np.zeros(3, dtype=np.float32)

    return (quat[:3] * 2.0 * np.arccos(w) / den).astype(np.float32)


class MyPolicy(BasePolicy):
    """Pi0（lerobot/pi0_libero_finetuned_v044）を使って推論するポリシー。

    LIBERO 向けに fine-tune 済みの Pi0 チェックポイントをロードし、
    観測を lerobot の入力形式（observation.images.*, observation.state）に
    変換して推論する。action チャンクのキャッシュは policy.reset() が管理する。
    """

    def __init__(self):
        import torch

        from siglip_patch import apply_pi0_inference_patches, apply_siglip_patch

        # lerobot の Pi0 実装が要求する transformers パッチ（本来は GitHub の特定ブランチ
        # からの導入が必須）を、外部ソースに依存せずローカルで再現する。詳細は
        # siglip_patch.py のモジュール docstring を参照。
        apply_siglip_patch()
        # lerobot 0.4.4 の PI0Pytorch.denoise_step にある推論時バグ
        # （dtype 不一致・KV キャッシュ汚染）の修正。
        apply_pi0_inference_patches()

        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config = PreTrainedConfig.from_pretrained(_MODEL_DIR)
        config.device = str(self.device)
        # 学習時用の torch.compile(mode="max-autotune") は推論単発呼び出しでは不要かつ
        # 単発呼び出しの動的形状でトレースエラーを起こすため無効化する。
        # config.compile_model=False の指定だけでは、内部で保持された古い config
        # 参照が使われる場合があるため、torch.compile 自体を一時的に恒等関数へ
        # 差し替えて確実に無効化する。
        config.compile_model = False
        _original_torch_compile = torch.compile
        torch.compile = lambda fn, *args, **kwargs: fn
        try:
            self.policy = PI0Policy.from_pretrained(_MODEL_DIR, config=config)
        finally:
            torch.compile = _original_torch_compile
        self.policy.to(self.device)
        self.policy.eval()

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=_MODEL_DIR,
            preprocessor_overrides={
                "tokenizer_processor": {"tokenizer_name": _TOKENIZER_DIR},
            },
        )

        self.instruction = ""
        self.policy.reset()

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        from lerobot.policies.utils import prepare_observation_for_inference

        # 学習データ（HuggingFaceVLA/libero）のカメラ向き規約に合わせる。
        # lerobot の LiberoProcessorStep（lerobot/policies/xvla/processor_xvla.py）が
        # 生の robosuite 観測に対して行っているのと同じ変換:
        # メインカメラのみ H/W 両方を反転（180度回転）し、手首カメラは反転しない。
        image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        image2 = np.ascontiguousarray(obs["robot0_eye_in_hand_image"])

        eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
        eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)  # (x, y, z, w)
        eef_axis_angle = _quat2axisangle(eef_quat)
        gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
        state = np.concatenate([eef_pos, eef_axis_angle, gripper_qpos]).astype(np.float32)

        # このモデルの config は empty_camera_0 を入力特徴に含むが、意図的に渡さない。
        # lerobot 側（PI0Policy._preprocess_images）はバッチに無いカメラを
        # 「-1 で埋めた画像 + マスク False」として自前で生成する。学習時もこの扱いなので、
        # ここで零画像を明示的に渡すとマスクが True になり、無効な 256 トークンが
        # 有効扱いで attention に混入してしまう（学習時と不一致になる）。
        raw_obs = {
            "observation.images.image": image,
            "observation.images.image2": image2,
            "observation.state": state,
        }

        with self.torch.inference_mode():
            batch = prepare_observation_for_inference(
                raw_obs, self.device, task=self.instruction,
            )
            batch = self.preprocessor(batch)
            action = self.policy.select_action(batch)
            action = self.postprocessor(action)

        return action.squeeze(0).to("cpu").numpy().astype(np.float32)

    def reset(self, instruction: str = "") -> None:
        # instruction にはタスクの言語指示が渡される
        self.instruction = instruction
        self.policy.reset()


# ============================================================
# 以下は変更不可
# ============================================================


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


app = FastAPI(title="VLA Policy Server")
_policy: BasePolicy | None = None


def set_policy(policy: BasePolicy) -> None:
    global _policy
    _policy = policy


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
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    set_policy(MyPolicy())
    print(f"Policy server starting on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
