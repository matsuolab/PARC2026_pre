"""ポリシーサーバー（提出用テンプレート）

このファイルを編集して、自分のモデルを組み込んでください。
編集が必要なのは MyPolicy クラスの中身だけです。
それ以外のコード（サーバー部分、シリアライゼーション）は変更不可です。

ローカルテスト:
    # 学習時と同じ lerobot==0.6.0 が必要（Python >= 3.12）
    #   uv venv policy_venv --python 3.12 --seed
    #   ./policy_venv/bin/python -m pip install -r requirements.txt
    ./policy_venv/bin/python submission_template/policy_server.py

    # 別ターミナルで評価実行（PARC2026_pre の env.sh / venv）
    python -m pipeline --server-url http://localhost:8000 --dry-run
"""

import argparse
from abc import ABC, abstractmethod
from pathlib import Path

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


class MyPolicy(BasePolicy):
    """SmolVLA（Spatial LoRA マージ済み）を LeRobot 経由で推論する。"""

    MODEL_DIRNAME = "smolvla_libero_plus_spatial_lora_merged"
    IMAGE_SIZE = 256  # 学習時の入力解像度

    def __init__(self):
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self._torch = torch
        weights_root = Path(__file__).resolve().parent / "model_weights"
        model_dir = weights_root / self.MODEL_DIRNAME
        if not model_dir.is_dir():
            raise FileNotFoundError(
                f"モデルディレクトリがありません: {model_dir}\n"
                "Colab のマージ済み重みをここに配置してください。"
            )

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        self.device = device

        # 評価中はネット不可 → ローカル tokenizer があれば優先
        tokenizer_dir = weights_root / "smolvlm_tokenizer"
        preprocessor_overrides: dict = {
            "device_processor": {"device": str(device)},
        }
        if tokenizer_dir.is_dir():
            preprocessor_overrides["tokenizer_processor"] = {
                "tokenizer_name": str(tokenizer_dir),
            }

        self.policy = SmolVLAPolicy.from_pretrained(model_dir)
        self.policy.to(device)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=str(model_dir),
            preprocessor_overrides=preprocessor_overrides,
        )
        self.instruction = ""

    @staticmethod
    def _quat2axisangle(quat):
        """quat (4,) = (x, y, z, w) → axis-angle (3,)"""
        w = quat[3].clamp(-1.0, 1.0)
        den = (1.0 - w * w).clamp(min=0.0).sqrt()
        if den < 1e-10:
            return quat.new_zeros(3)
        return (quat[:3] / den * (2.0 * w.acos())).float()

    def _to_chw(self, img_hwc: np.ndarray):
        torch = self._torch
        # LIBERO 慣習: 180° flip → CHW → [0, 1]
        img = np.ascontiguousarray(img_hwc[::-1, ::-1])
        t = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)
        # 学習は 256x256。競技観測は 128x128 なので揃える
        if t.shape[-2] != self.IMAGE_SIZE or t.shape[-1] != self.IMAGE_SIZE:
            t = torch.nn.functional.interpolate(
                t.unsqueeze(0),
                size=(self.IMAGE_SIZE, self.IMAGE_SIZE),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return t

    def _build_observation(self, obs: dict[str, np.ndarray]) -> dict:
        torch = self._torch
        eef_pos = torch.as_tensor(obs["robot0_eef_pos"], dtype=torch.float32)
        eef_quat = torch.as_tensor(obs["robot0_eef_quat"], dtype=torch.float32)
        gripper = torch.as_tensor(obs["robot0_gripper_qpos"], dtype=torch.float32)
        state = torch.cat(
            [eef_pos, self._quat2axisangle(eef_quat), gripper]
        )  # (8,)
        return {
            "observation.state": state,
            "observation.images.front": self._to_chw(obs["agentview_image"]),
            "observation.images.wrist": self._to_chw(
                obs["robot0_eye_in_hand_image"]
            ),
            "task": self.instruction,
        }

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        torch = self._torch
        batch = self.preprocessor(self._build_observation(obs))
        with torch.inference_mode():
            action = self.policy.select_action(batch)
        action = self.postprocessor(action)
        return (
            action.squeeze(0).detach().cpu().numpy().astype(np.float32)
        )

    def reset(self, instruction: str = "") -> None:
        self.instruction = instruction or ""
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
