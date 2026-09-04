# M1 Setup Notes — OpenVLA-OFT on LIBERO (RunPod, RTX PRO 4000 Blackwell)

**Status: ✅ M1 COMPLETE** — 95% success on `libero_spatial` (19/20 episodes, 2 trials/task).

This file records the *exact* environment and every fix needed to run OpenVLA-OFT +
LIBERO on a **Blackwell (sm_120)** GPU with **system Python 3.11 (no conda)**.
Reproduce by following top-to-bottom.

---

## Hardware / OS (what we had)
- GPU: **NVIDIA RTX PRO 4000 Blackwell, 24 GB** (compute capability **sm_120**)
- Driver 580.x (CUDA 13 capable), system CUDA toolkit 12.4, nvcc 12.4
- OS: Ubuntu 22.04 container, **system Python 3.11.10**, no conda
- Disk: container root `/` only **20 GB** → put *everything* on `/workspace` (huge NFS)

> Key constraint: **Blackwell sm_120 needs PyTorch built with CUDA 12.8+.**
> The container's preinstalled `torch 2.4.1+cu124` only supports up to sm_90 and
> will NOT run kernels on this GPU.

---

## 0. Environment helper — `env.sh`
Source this at the start of every session (`source env.sh`). It activates the venv
and points all caches at `/workspace` so the 20 GB root doesn't fill up, and sets
headless MuJoCo rendering.

```bash
source /workspace/oft-venv/bin/activate
export PIP_CACHE_DIR=/workspace/.cache/pip
export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub
export TORCH_HOME=/workspace/.cache/torch
export XDG_CACHE_HOME=/workspace/.cache
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

## 1. Create venv (on the big disk)
```bash
python3 -m venv /workspace/oft-venv
source /workspace/VLA-GRPO/04_vla_libero_grpo/env.sh
pip install --upgrade pip setuptools wheel
```

## 2. PyTorch for Blackwell (CUDA 12.8 build)  ← critical
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# gives torch 2.11.0+cu128, arch_list includes sm_120
```

## 3. Clone + install OpenVLA-OFT (preserve our torch)
```bash
cd /workspace
git clone https://github.com/moojink/openvla-oft.git
cd openvla-oft
```
The repo pins `torch==2.2.0` etc., which would DOWNGRADE our Blackwell torch.
Comment those three lines out first:
```bash
cp pyproject.toml pyproject.toml.bak
sed -i 's/^    "torch==2.2.0",/    # "torch==2.2.0",/' pyproject.toml
sed -i 's/^    "torchvision==0.17.0",/    # "torchvision==0.17.0",/' pyproject.toml
sed -i 's/^    "torchaudio==2.2.0",/    # "torchaudio==2.2.0",/' pyproject.toml
pip install -e .
```

## 4. Fix dependency-resolution conflicts (order matters)
The repo's stack is old (tf 2.15, numpy<2) and fights newer transitive deps:
```bash
# protobuf/tensorflow-metadata mismatch (runtime_version ImportError)
pip install "tensorflow-metadata==1.14.0"     # -> protobuf 3.20.3
# wandb 0.28 needs protobuf>4.21 -> use an older wandb compatible with 3.20
pip install "wandb==0.16.6"
```

## 5. Install LIBERO simulator
```bash
cd /workspace
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
# LIBERO's libero/ has NO __init__.py (namespace pkg) -> strict editable install
# hides the top-level package. Use compat mode:
pip install -e LIBERO --config-settings editable_mode=compat
# Use OpenVLA-OFT's MINIMAL libero reqs (NOT LIBERO/requirements.txt which pins
# numpy 1.22 / transformers 4.21 and would break the stack):
pip install -r /workspace/openvla-oft/experiments/robot/libero/libero_requirements.txt
```

## 6. Re-pin numpy stack (libero reqs dragged numpy back to 2.x)
```bash
pip install "numpy==1.26.4" "numba==0.59.1" "llvmlite==0.42.0"
pip install "opencv-python==4.10.0.84"   # opencv 5 wants numpy>=2
# mujoco 3.x breaks robosuite 1.4.1's joint API -> use 2.3.x
pip install "mujoco==2.3.2"
# verify:
pip check      # -> "No broken requirements found."
```

## 7. System EGL libraries (headless rendering)
The container had `libEGL_nvidia.so.0` but not the vendor-neutral dispatch libs
PyOpenGL loads (`libEGL.so.1`). Install them:
```bash
apt-get update
apt-get install -y libglvnd0 libglvnd-dev libegl1 libegl1-mesa-dev libgl1 libgles2 libglib2.0-0
```

## 8. LIBERO config — avoid the interactive prompt
On first import LIBERO asks "Do you want to specify a custom path...(Y/N)" and
blocks on stdin. Pre-create the config (datasets on the big disk):
```bash
BR=/workspace/LIBERO/libero/libero
mkdir -p ~/.libero /workspace/libero_datasets
cat > ~/.libero/config.yaml <<EOF
benchmark_root: $BR
bddl_files: $BR/bddl_files
init_states: $BR/init_files
datasets: /workspace/libero_datasets
assets: $BR/assets
EOF
```

## 9. Patch LIBERO for PyTorch 2.6+ (`weights_only` default flip)
`get_task_init_states` calls `torch.load()` which now defaults to
`weights_only=True` and rejects the numpy init-states file:
```bash
cd /workspace/LIBERO
sed -i 's/init_states = torch.load(init_states_path)/init_states = torch.load(init_states_path, weights_only=False)/' \
  libero/libero/benchmark/__init__.py
```

---

## RUN (the milestone)
```bash
cd /workspace/openvla-oft
source /workspace/VLA-GRPO/04_vla_libero_grpo/env.sh
stdbuf -oL -eL python -u experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 2 </dev/null 2>&1 | tee /workspace/oft_runs/m1_spatial_smoke.log
```
- `stdbuf -oL` + `python -u` = live line-buffered output through the pipe.
- `</dev/null` = never block on an interactive prompt.
- First run downloads the ~14 GB checkpoint into `HF_HOME`.

### Result (2026-09-04)
```
Total episodes: 20 | Total successes: 19 | Overall success rate: 0.9500 (95.0%)
```

---

## Outputs
- **Rollout MP4s** (one per episode, filename has success flag + task):
  `/workspace/openvla-oft/rollouts/<DATE>/`
  Copy to local: `scp -r <pod>:/workspace/openvla-oft/rollouts ./rollouts`
- **Text logs**: `/workspace/oft_runs/m1_spatial_smoke.log`
  and `/workspace/openvla-oft/experiments/logs/EVAL-*.txt`

## Known-benign warnings (safe to ignore)
- `EGLError(EGL_NOT_INITIALIZED)` in `__del__` at interpreter EXIT — cleanup only,
  fires after results print.
- TF cuDNN/cuFFT/cuBLAS "already registered" + "Skipping registering GPU devices"
  — TF isn't used for GPU here; harmless.
- `huggingface/tokenizers: ... forked` — harmless; can silence with
  `export TOKENIZERS_PARALLELISM=false`.

## Pinned versions that make it work
| package | version | why |
|---|---|---|
| torch / vision / audio | 2.11.0 / 0.26.0 / 2.11.0 **+cu128** | Blackwell sm_120 |
| transformers | 4.40.1 (moojink fork) | bidirectional attn / parallel decoding |
| numpy | 1.26.4 | tensorflow 2.15 needs <2 |
| numba / llvmlite | 0.59.1 / 0.42.0 | compatible with numpy 1.26 |
| opencv-python | 4.10.0.84 | opencv 5 requires numpy>=2 |
| tensorflow / -metadata | 2.15.0 / 1.14.0 | protobuf 3.20 compatibility |
| wandb | 0.16.6 | protobuf 3.20 compatibility |
| mujoco | 2.3.2 | robosuite 1.4.1 joint API |
| robosuite | 1.4.1 | (from openvla-oft libero reqs) |

## Model / robot / task recap
- **Model**: OpenVLA-7B-OFT (SigLIP+DINOv2 vision → Llama-2 7B), L1-regression head,
  outputs a **chunk of 8 actions**. bf16, ~14 GB on GPU.
- **Robot**: Franka Emika Panda 7-DoF (robosuite/MuJoCo). Action = 6-DoF EE delta + gripper.
- **Task suite** `libero_spatial`: 10 pick-and-place tasks, same objects in different
  spatial arrangements; language-conditioned; oracle success flag = future GRPO reward.
