# Phase A Runbook — single-suite GRPO + LoRA expert (FRESH POD)

Assumes a **brand-new pod** from the image `ghcr.io/abishek21/simplevla-rl-libero:cu124`
(all fixes baked: fork→spawn, NVIDIA EGL ICD). `/workspace` starts EMPTY.

Target GPU: **1× A40 / A6000 (48 GB)**. Pod env var REQUIRED at creation:
`NVIDIA_DRIVER_CAPABILITIES=all`. Start command: `sleep infinity`.

---

## 0. Connect (from your Mac)
RunPod proxy ignores `ssh <cmd>`; feed commands via stdin (see `rpod.sh`), or just:
```bash
ssh -tt cy...@ssh.runpod.io -i ~/.ssh/id_ed25519   # then run the blocks below interactively
```

## 1. Sanity: GPU + image
```bash
nvidia-smi --query-gpu=name,memory.total --format=csv
ls /usr/share/glvnd/egl_vendor.d/          # must include 10_nvidia.json (EGL fix)
grep -n "_MP_SPAWN_CTX" /opt/SimpleVLA-RL/verl/workers/rollout/rob_rollout.py  # spawn fix present
```
If `10_nvidia.json` is MISSING (older image), create it:
```bash
mkdir -p /usr/share/glvnd/egl_vendor.d && printf '{\n  "file_format_version":"1.0.0",\n  "ICD":{"library_path":"libEGL_nvidia.so.0"}\n}\n' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
```

## 2. Download the SFT warm-start checkpoint (~15 GB) to persistent volume
```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
pip install -q hf_transfer huggingface_hub 2>/dev/null
mkdir -p /workspace/ckpt_libero10_trajall
huggingface-cli download Haozhan72/Openvla-oft-SFT-libero10-trajall \
  --local-dir /workspace/ckpt_libero10_trajall --local-dir-use-symlinks False
ls /workspace/ckpt_libero10_trajall/*.safetensors | wc -l    # expect 4
```

## 3. LIBERO datasets present?
```bash
ls /workspace/libero_datasets 2>/dev/null || echo "MISSING - see note"
```
(The baseline pod had these baked at `/workspace/libero_datasets`. If missing on this
image, LIBERO will need its datasets — check `/root/.libero/config.yaml` `datasets:` path.)

## 4. Get the Phase-A launch script onto the pod
Option A — pull our repo (has the script):
```bash
cd /workspace && git clone https://github.com/abishek21/VLA-GRPO.git
cp /workspace/VLA-GRPO/05_forgetting_study/run_phaseA_lora.sh /opt/SimpleVLA-RL/examples/
```
Option B — if repo is private / no token, paste the file manually into
`/opt/SimpleVLA-RL/examples/run_phaseA_lora.sh`.

Then verify the 3 EDIT paths inside it:
```bash
grep -nE "SFT_MODEL_PATH=|CKPT_PATH=|ALIGN_PATH=" /opt/SimpleVLA-RL/examples/run_phaseA_lora.sh
# SFT_MODEL_PATH=/workspace/ckpt_libero10_trajall
# CKPT_PATH=/workspace/phaseA_out
# ALIGN_PATH=/opt/SimpleVLA-RL/align.json
```

## 5. align.json (Ray worker env) — ensure render vars, wandb off
```bash
cat > /opt/SimpleVLA-RL/align.json <<'JSON'
{
  "env_vars": {
    "NCCL_DEBUG": "WARN",
    "RAY_memory_monitor_refresh_ms": "0",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "TOKENIZERS_PARALLELISM": "true",
    "MUJOCO_GL": "egl",
    "WANDB_MODE": "disabled"
  },
  "excludes": ["*"]
}
JSON
```

### 5b. (OPTIONAL) Enable W&B live monitoring
The Ray rollout/trainer workers only inherit env vars from `align.json`, so the W&B
key must go there too (do NOT commit it anywhere). Paste YOUR key on the pod:
```bash
export WANDB_API_KEY=xxxxxxxxxxxxxxxx        # <-- your key, pod shell only
python - <<PY
import json,os
p="/opt/SimpleVLA-RL/align.json"; d=json.load(open(p))
d["env_vars"]["WANDB_API_KEY"]=os.environ["WANDB_API_KEY"]
d["env_vars"]["WANDB_MODE"]="online"
json.dump(d,open(p,"w"),indent=2); print("align.json updated with W&B online")
PY
wandb login "$WANDB_API_KEY" 2>/dev/null || pip install -q wandb && wandb login "$WANDB_API_KEY"
```
The launch script auto-detects `WANDB_API_KEY` and switches to
`trainer.logger=['console','wandb']`, `wandb_mode=online`. Dashboard: project
**VLA-MoE-LoRA**, run **phaseA_libero10_lora_r32_grpo** at https://wandb.ai .
⚠️ Rotate this key afterwards (it was shared in plaintext).

## 6. Launch in tmux (survives disconnects)
```bash
mkdir -p /workspace/phaseA_out
cd /opt/SimpleVLA-RL
tmux new -s phaseA -d
tmux send-keys -t phaseA '
cd /opt/SimpleVLA-RL
export MUJOCO_GL=egl NVIDIA_DRIVER_CAPABILITIES=all
export WANDB_API_KEY=xxxxxxxxxxxxxxxx        # <-- your key (omit for console-only)
unset PYOPENGL_PLATFORM
bash examples/run_phaseA_lora.sh 2>&1 | tee /workspace/phaseA_out/train_log.txt
' Enter
tmux attach -t phaseA          # Ctrl-b then d to detach
```

## 7. What to watch (the 4 sanity gates)
```bash
grep -nE "Applying LoRA|trainable params|test_score|env initialization failed|EGL_BAD_ALLOC|Traceback" \
  /workspace/phaseA_out/train_log.txt | tail -40
```
1. `Applying LoRA to actor module` + `trainable params: ~X M` (millions, NOT 7B). ✅ LoRA on.
2. Step-0 `val/test_score/libero_10 ≈ 0.88` (matches SFT baseline). ✅ warm-start OK.
3. `train_verify_score/...` (success) trends UP over steps. ✅ RL learning.
4. `/workspace/phaseA_out/VLA-MoE-LoRA/phaseA_.../ .../lora_adapter/` written at save_freq. ✅ expert saved.

## 8. If it OOMs (1× 48 GB)
Lower in `run_phaseA_lora.sh`:
- `data.train_batch_size=4`, `data.num_trials_per_task=5`
- `actor_rollout_ref.rollout.gpu_memory_utilization=0.5`
- keep `enable_gradient_checkpointing=True`
Or deploy **2× A40** and set `NUM_GPUS=2` in the script.

## 9. If env init loops (`*** env initialization failed ***`)
Reproduce the real error (spawn child) to see the traceback:
```bash
cd /opt/SimpleVLA-RL; export MUJOCO_GL=egl; unset PYOPENGL_PLATFORM
python3 - <<'PY'
import os, multiprocessing as mp
def child():
    import traceback
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        ts=benchmark.get_benchmark_dict()["libero_10"](); t=ts.get_task(0)
        b=os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
        e=OffScreenRenderEnv(bddl_file_name=b,camera_heights=256,camera_widths=256); e.seed(0); e.reset()
        print("CHILD_ENV_OK")
    except Exception: traceback.print_exc()
mp.get_context("spawn").Process(target=child).start()
PY
```
Expect `CHILD_ENV_OK`. If it prints an EGL/Mesa error → re-check step 1's ICD json.

## 10. Save the trained expert off the pod (before terminating)
```bash
cd /workspace && tar czf phaseA_lora.tgz phaseA_out/VLA-MoE-LoRA/*/*/lora_adapter
# then pull phaseA_lora.tgz to your Mac (base64-over-ssh method in DEBUG_HANDOFF,
# or push to HF/your storage). The lora_adapter is small (~tens of MB).
```

---

### Recap of the recipe (what Phase A proves)
- LoRA (rank 32) on the LLaMA control layers, GRPO (KL off), warm-started from
  libero10_trajall → a `lora_adapter/` that improves libero_10.
- This is expert #1. Repeat with spatial/object/goal (their own trajall ckpts) to
  get 4 experts → then Phase B (MoELoRALinear) → Phase C (joint MoE across suites).
