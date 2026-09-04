"""
LIBERO client (LOCAL Mac side): runs + renders LIBERO locally, and asks a remote
GPU policy server for actions. The heavy 7B VLA runs on the GPU; the sim +
rendering run here (Mac has a display -> no headless MuJoCo pain).

Run (on the Mac, after installing LIBERO locally):
  export SERVER_URL="https://<your-runpod-proxy>/act"
  python 04_vla_libero_grpo/libero_client.py --task-suite libero_spatial \
      --num-trials 5

Requires locally: LIBERO installed + `pip install requests pillow numpy`.
"""
from __future__ import annotations

import argparse
import base64
import io
import os

import numpy as np
import requests


def encode_img(arr: np.ndarray) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def get_actions(server_url, full_img, wrist_img, state, instruction):
    payload = {
        "full_image_b64": encode_img(full_img),
        "wrist_image_b64": encode_img(wrist_img) if wrist_img is not None else None,
        "state": state.tolist() if state is not None else None,
        "instruction": instruction,
    }
    r = requests.post(server_url, json=payload, timeout=60)
    r.raise_for_status()
    return np.asarray(r.json()["actions"], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-suite", default="libero_spatial")
    ap.add_argument("--num-trials", type=int, default=5)
    ap.add_argument("--server-url", default=os.environ.get("SERVER_URL", ""))
    args = ap.parse_args()
    assert args.server_url, "set --server-url or SERVER_URL env var"

    # health check
    base = args.server_url.rsplit("/", 1)[0]
    try:
        h = requests.get(base + "/health", timeout=10).json()
        print("server health:", h)
    except Exception as e:  # noqa: BLE001
        print("warning: health check failed:", repr(e))

    # ---- LIBERO env setup (local) ----
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bench = benchmark.get_benchmark_dict()[args.task_suite]()
    n_tasks = bench.n_tasks
    print(f"{args.task_suite}: {n_tasks} tasks")

    successes = 0
    trials = 0
    for task_id in range(n_tasks):
        task = bench.get_task(task_id)
        init_states = bench.get_task_init_states(task_id)
        env_args = {
            "bddl_file_name": os.path.join(
                get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
            "camera_heights": 256, "camera_widths": 256,
        }
        env = OffScreenRenderEnv(**env_args)
        for t in range(min(args.num_trials, len(init_states))):
            env.reset()
            env.set_init_state(init_states[t])
            done = False
            steps = 0
            while not done and steps < 400:
                obs = env._get_observations() if hasattr(env, "_get_observations") else None
                # LIBERO obs keys: agentview_image, robot0_eye_in_hand_image, robot0_* state
                full = obs["agentview_image"][::-1]          # flip if needed
                wrist = obs.get("robot0_eye_in_hand_image")
                state = np.concatenate([obs["robot0_eef_pos"], obs["robot0_eef_quat"],
                                        obs["robot0_gripper_qpos"]]) if "robot0_eef_pos" in obs else None
                actions = get_actions(args.server_url, full, wrist, state,
                                      task.language)
                for a in actions:
                    obs, r, done, info = env.step(a.tolist())
                    steps += 1
                    if done:
                        break
            trials += 1
            ok = bool(info.get("success", False)) if isinstance(info, dict) else False
            successes += ok
            print(f"task {task_id} trial {t}: {'SUCCESS' if ok else 'fail'} "
                  f"({steps} steps)  running {successes}/{trials}")
        env.close()

    print(f"\n=== {args.task_suite} success: {successes}/{trials} "
          f"({100*successes/max(1,trials):.1f}%) ===")


if __name__ == "__main__":
    main()
