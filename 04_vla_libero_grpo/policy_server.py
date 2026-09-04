"""
Policy server (GPU side): wraps OpenVLA-OFT behind a tiny HTTP API so a LOCAL
LIBERO simulator can request actions over the network.

Split architecture:
  LOCAL Mac  ──obs(images+state+instruction)──►  THIS server (GPU)
             ◄──────── action chunk ────────────

The GPU does ONLY inference (no rendering) -> avoids the headless-MuJoCo pain.
The Mac runs + renders LIBERO (it has a display).

Run (on the GPU pod):
  pip install fastapi uvicorn
  python 04_vla_libero_grpo/policy_server.py --checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial --port 8000
Expose the port (RunPod: add an HTTP port / use their proxy URL).

Then set the client's SERVER_URL to this pod's public URL.
"""
from __future__ import annotations

import argparse
import base64
import io

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()
_POLICY = {"vla": None, "cfg": None, "extras": None}


class ObsRequest(BaseModel):
    # images as base64-encoded PNG/raw; state + instruction as plain fields
    full_image_b64: str
    wrist_image_b64: str | None = None
    state: list[float] | None = None
    instruction: str = ""


def _decode_img(b64: str) -> np.ndarray:
    from PIL import Image
    raw = base64.b64decode(b64)
    return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))


@app.on_event("startup")
def _load():
    # Import OpenVLA-OFT helpers (must run inside the openvla-oft repo env).
    from experiments.robot.libero.run_libero_eval import GenerateConfig
    from experiments.robot.openvla_utils import (
        get_action_head, get_processor, get_proprio_projector, get_vla, get_vla_action)
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

    ckpt = app.state.checkpoint
    cfg = GenerateConfig(
        pretrained_checkpoint=ckpt, use_l1_regression=True, use_diffusion=False,
        use_film=False, num_images_in_input=2, use_proprio=True,
        center_crop=True, num_open_loop_steps=NUM_ACTIONS_CHUNK,
        unnorm_key="libero_spatial_no_noops",
    )
    vla = get_vla(cfg)
    processor = get_processor(cfg)
    action_head = get_action_head(cfg, llm_dim=vla.llm_dim)
    proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)
    _POLICY["vla"] = vla
    _POLICY["cfg"] = cfg
    _POLICY["extras"] = (processor, action_head, proprio_projector, get_vla_action)
    print("policy loaded:", ckpt)


@app.post("/act")
def act(req: ObsRequest):
    processor, action_head, proprio_projector, get_vla_action = _POLICY["extras"]
    obs = {
        "full_image": _decode_img(req.full_image_b64),
        "wrist_image": _decode_img(req.wrist_image_b64) if req.wrist_image_b64 else None,
        "state": np.array(req.state, dtype=np.float32) if req.state else None,
        "task_description": req.instruction,
    }
    actions = get_vla_action(_POLICY["cfg"], _POLICY["vla"], processor, obs,
                             req.instruction, action_head, proprio_projector)
    # actions: list/array of [action_dim] per step
    return {"actions": np.asarray(actions, dtype=np.float32).tolist()}


@app.get("/health")
def health():
    return {"ok": _POLICY["vla"] is not None}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="moojink/openvla-7b-oft-finetuned-libero-spatial")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    app.state.checkpoint = args.checkpoint
    uvicorn.run(app, host="0.0.0.0", port=args.port)
