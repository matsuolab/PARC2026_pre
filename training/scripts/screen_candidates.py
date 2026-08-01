"""LoRA候補の2段階スクリーニング（一次選抜 → 本選）を実行する。

DESIGN.md §4 の仕様に基づく:
  - 一次選抜: 全候補に対し共通タスク・共通 seed で少数エピソード評価し、
    明らかに劣悪な候補は早期に Prune（打ち切り）する。
  - 本選: 一次選抜の上位 n_finalists 本のみフル評価する。
  - Prune 済み候補と完走候補の success rate は分母が異なるため数値比較しない
    （ランキングから除外する）。

pipeline/ は評価パイプライン本体（変更不可）なので、本スクリプトは
`python -m pipeline --server-url ...` をタスク単位で繰り返し呼び出す
オーケストレーターとして実装している。タスク単位より細かい（エピソード単位の）
早期打ち切りは pipeline が対応していないため、Prune 判定はタスク境界で行う
近似になる点に注意。

候補ごとに config.yaml の eval.server_launch_cmd に従って推論サーバーを
起動する。実体は training/imitation/serve_candidate.py（要実装）。

使い方:
    python training/scripts/screen_candidates.py --env-dir training/Env/001_... --stage screening
    python training/scripts/screen_candidates.py --env-dir training/Env/001_... --stage full
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS_CSV = REPO_ROOT / "compe" / "t1" / "T1_TASKS.csv"


def load_config(env_dir: Path) -> dict:
    with open(env_dir / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_task_ids(n_tasks: int) -> list[str]:
    """compe/t1/T1_TASKS.csv から task_num 順に task_id を読む(全候補で共通の順序)。"""
    with open(TASKS_CSV, encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["task_num"]))
    task_ids = [r["task_id"] for r in rows]
    if n_tasks > len(task_ids):
        print(f"[screen_candidates] 警告: config が要求する n_tasks={n_tasks} は "
              f"{TASKS_CSV} の {len(task_ids)} タスクを超えています。"
              f"利用可能な全タスクを使用します。", file=sys.stderr)
        return task_ids
    return task_ids[:n_tasks]


def discover_candidates(env_dir: Path) -> list[str]:
    cand_dir = env_dir / "weights" / "candidates"
    return sorted(p.name for p in cand_dir.iterdir() if p.is_dir())


def append_log(env_dir: Path, record: dict) -> None:
    record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    with open(env_dir / "checkpoints_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def latest_val_loss(env_dir: Path, candidate: str) -> float | None:
    """tie-break 用。candidate の step 以下で直近に記録された val_loss を返す。"""
    step = _step_of(candidate)
    if step is None:
        return None
    best = None
    log_path = env_dir / "checkpoints_log.jsonl"
    if not log_path.is_file():
        return None
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "val_loss" in rec and rec.get("step") is not None and rec["step"] <= step:
                if best is None or rec["step"] > best[0]:
                    best = (rec["step"], rec["val_loss"])
    return best[1] if best else None


def _step_of(candidate: str) -> int | None:
    if candidate.startswith("step_"):
        try:
            return int(candidate[len("step_"):])
        except ValueError:
            return None
    return None


class ServerHandle:
    def __init__(self, proc: subprocess.Popen, port: int):
        self.proc = proc
        self.port = port

    def stop(self) -> None:
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            self.proc.wait(timeout=15)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass


def launch_server(cmd_template: str, candidate_dir: Path, port: int,
                   health_timeout: int = 180) -> ServerHandle:
    cmd = cmd_template.format(candidate_dir=str(candidate_dir), port=port)
    proc = subprocess.Popen(
        cmd, shell=True, cwd=str(REPO_ROOT), start_new_session=True,
    )
    handle = ServerHandle(proc, port)

    deadline = time.time() + health_timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"推論サーバーが起動直後に終了しました（exit={proc.returncode}）。"
                f"コマンド: {cmd}"
            )
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=2)
            if r.status_code == 200:
                return handle
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)

    handle.stop()
    raise TimeoutError(f"推論サーバーが {health_timeout}秒 以内に /health を返しませんでした")


def run_one_task(port: int, task_id: str, n_episodes: int, seed: int,
                  output_dir: Path) -> dict:
    """1タスクだけ pipeline を実行し、そのタスクの結果 dict を返す。"""
    submission_id = f"server_{port}"
    result_path = output_dir / f"{submission_id}.json"
    if result_path.exists():
        result_path.unlink()

    subprocess.run(
        [
            sys.executable, "-m", "pipeline",
            "--server-url", f"http://localhost:{port}",
            "--tasks", task_id,
            "--n-episodes", str(n_episodes),
            "--seed", str(seed),
            "--output-dir", str(output_dir),
        ],
        cwd=str(REPO_ROOT), check=True,
    )

    with open(result_path, encoding="utf-8") as f:
        result = json.load(f)

    tasks = result.get("tracks", [{}])[0].get("tasks", [])
    if not tasks:
        raise RuntimeError(f"pipeline の出力にタスク結果がありません: {result_path}")
    return tasks[0]


def evaluate_candidate_screening(env_dir: Path, config: dict, candidate: str,
                                  task_ids: list[str]) -> dict:
    eval_cfg = config["eval"]
    screening_cfg = eval_cfg["screening"]
    prune_cfg = screening_cfg["prune"]
    episodes_per_task = screening_cfg["episodes_per_task"]

    candidate_dir = env_dir / "weights" / "candidates" / candidate
    handle = launch_server(eval_cfg["server_launch_cmd"], candidate_dir, eval_cfg["port"])

    cumulative_episodes = 0
    cumulative_successes = 0
    pruned = False
    pruned_at_episode = None

    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            for task_id in task_ids:
                task_result = run_one_task(
                    handle.port, task_id, episodes_per_task, eval_cfg["seed"], tmp_dir,
                )
                successes = round(task_result["success_rate"] * episodes_per_task)
                cumulative_episodes += episodes_per_task
                cumulative_successes += successes

                if (prune_cfg.get("enabled", True)
                        and cumulative_episodes >= prune_cfg["after_episodes"]
                        and cumulative_successes < prune_cfg["min_successes"]):
                    pruned = True
                    pruned_at_episode = cumulative_episodes
                    break
    finally:
        handle.stop()

    score = (cumulative_successes / cumulative_episodes) if cumulative_episodes else 0.0

    record = {
        "stage": "screening",
        "candidate": candidate,
        "score": score,
        "n_episodes": cumulative_episodes,
        "n_successes": cumulative_successes,
        "pruned": pruned,
    }
    if pruned:
        record["pruned_at_episode"] = pruned_at_episode
    append_log(env_dir, record)
    return record


def evaluate_candidate_full(env_dir: Path, config: dict, candidate: str,
                             task_ids: list[str]) -> dict:
    eval_cfg = config["eval"]
    full_cfg = eval_cfg["full"]
    n_viewpoints = full_cfg.get("n_viewpoints", 1)
    # NOTE: pipeline には視点（viewpoint）を切り替える仕組みが現状無い。
    # ここでは暫定的にエピソード数へ乗算する近似で扱っている。
    # 複数視点でのカメラ変更を厳密に評価したい場合は pipeline 側の対応が別途必要
    # （DESIGN.md には無い、実装時に判明した既知の制約）。
    episodes_per_task = full_cfg["episodes_per_task"] * n_viewpoints

    candidate_dir = env_dir / "weights" / "candidates" / candidate
    handle = launch_server(eval_cfg["server_launch_cmd"], candidate_dir, eval_cfg["port"])

    task_results = []
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            for task_id in task_ids:
                task_results.append(
                    run_one_task(handle.port, task_id, episodes_per_task,
                                 eval_cfg["seed"], tmp_dir)
                )
    finally:
        handle.stop()

    total_episodes = episodes_per_task * len(task_ids)
    total_successes = sum(round(t["success_rate"] * episodes_per_task) for t in task_results)
    score = (total_successes / total_episodes) if total_episodes else 0.0

    result = {
        "candidate": candidate,
        "overall_score": score,
        "n_total_episodes": total_episodes,
        "tasks": task_results,
    }

    append_log(env_dir, {
        "stage": "full", "candidate": candidate, "score": score,
        "n_episodes": total_episodes,
    })
    return result


def rank_screening(env_dir: Path, records: list[dict]) -> list[dict]:
    """Prune 済みを除外し、score 降順・同点は val_loss 昇順でランク付けする。"""
    survivors = [r for r in records if not r["pruned"]]
    for r in survivors:
        r["val_loss_tiebreak"] = latest_val_loss(env_dir, r["candidate"])
    survivors.sort(
        key=lambda r: (-r["score"], r["val_loss_tiebreak"]
                        if r["val_loss_tiebreak"] is not None else float("inf"))
    )
    return survivors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-dir", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=["screening", "full"])
    parser.add_argument("--candidates", nargs="*", default=None,
                         help="対象候補（省略時は weights/candidates/ 配下の全候補）")
    args = parser.parse_args(argv)

    env_dir = args.env_dir.resolve()
    config = load_config(env_dir)
    candidates = args.candidates or discover_candidates(env_dir)

    if not candidates:
        print("[screen_candidates] 候補が見つかりません", file=sys.stderr)
        return 1

    if args.stage == "screening":
        n_tasks = config["eval"]["screening"]["n_tasks"]
        task_ids = load_task_ids(n_tasks)

        records = []
        for candidate in candidates:
            print(f"[screen_candidates] 一次選抜: {candidate}")
            record = evaluate_candidate_screening(env_dir, config, candidate, task_ids)
            print(f"  score={record['score']:.3f} "
                  f"n_episodes={record['n_episodes']} pruned={record['pruned']}")
            records.append(record)

        ranked = rank_screening(env_dir, records)
        screening_out = {
            "task_ids": task_ids,
            "seed": config["eval"]["seed"],
            "records": records,
            "ranking": [r["candidate"] for r in ranked],
        }
        with open(env_dir / "screening.json", "w", encoding="utf-8") as f:
            json.dump(screening_out, f, indent=2, ensure_ascii=False)

        n_finalists = config["eval"]["full"]["n_finalists"]
        finalists = [r["candidate"] for r in ranked[:n_finalists]]
        print(f"[screen_candidates] 本選進出: {finalists}")
        print("[screen_candidates] 次のステップ: "
              f"python {Path(__file__).name} --env-dir {env_dir} --stage full "
              f"--candidates {' '.join(finalists)}")

    else:  # full
        n_tasks = config["eval"]["full"]["n_tasks"]
        task_ids = load_task_ids(n_tasks)

        results = []
        for candidate in candidates:
            print(f"[screen_candidates] 本選: {candidate}")
            result = evaluate_candidate_full(env_dir, config, candidate, task_ids)
            print(f"  overall_score={result['overall_score']:.3f}")
            results.append(result)

        results.sort(key=lambda r: -r["overall_score"])
        with open(env_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump({
                "submission_id": env_dir.name,
                "task_ids": task_ids,
                "seed": config["eval"]["seed"],
                "tracks": [{
                    "track": "track1",
                    "overall_score": results[0]["overall_score"] if results else 0.0,
                    "candidates": results,
                }],
            }, f, indent=2, ensure_ascii=False)

        if results:
            print(f"[screen_candidates] 最良候補: {results[0]['candidate']} "
                  f"(score={results[0]['overall_score']:.3f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
