# OpenVLA-OFT + LIBERO on Blackwell (sm_120) GPUs — RunPod custom template
#
# IMPORTANT: You do NOT need a GPU to BUILD this image. Building only installs
# packages; CUDA kernels run at RUNTIME (on the RunPod GPU). Build on any machine
# with Docker (CPU-only laptop, CI, or a cheap CPU VM).
#
# If building on Apple Silicon (arm64), you MUST target amd64:
#   docker buildx build --platform linux/amd64 ...
#
# Base: PyTorch 2.11 cu128 (has sm_120 kernels for Blackwell). If this tag ever
# disappears, use nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04 and install torch
# via the pip cu128 index (see commented block below).

FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/workspace/.cache/huggingface \
    PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# 1) System libs: build tools, git, and the EGL/GL dispatch libs LIBERO needs
#    for headless MuJoCo rendering (libEGL.so.1 / libglvnd).
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake \
        libglvnd0 libglvnd-dev libegl1 libegl1-mesa-dev libgl1 libgles2 \
        libglib2.0-0 libosmesa6 ffmpeg wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# If your base image is a plain CUDA image (not the pytorch one), uncomment:
# RUN pip install --upgrade pip setuptools wheel && \
#     pip install torch torchvision torchaudio \
#       --index-url https://download.pytorch.org/whl/cu128

RUN pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# 2) OpenVLA-OFT — clone, drop the torch pins (keep our cu128 torch), install.
# ---------------------------------------------------------------------------
RUN git clone https://github.com/moojink/openvla-oft.git /opt/openvla-oft
WORKDIR /opt/openvla-oft
RUN sed -i 's/^    "torch==2.2.0",/    # "torch==2.2.0",/' pyproject.toml && \
    sed -i 's/^    "torchvision==0.17.0",/    # "torchvision==0.17.0",/' pyproject.toml && \
    sed -i 's/^    "torchaudio==2.2.0",/    # "torchaudio==2.2.0",/' pyproject.toml && \
    pip install -e .

# ---------------------------------------------------------------------------
# 3) Resolve the dependency conflicts (order matters — see SETUP_NOTES.md).
# ---------------------------------------------------------------------------
RUN pip install "tensorflow-metadata==1.14.0" "wandb==0.16.6"

# ---------------------------------------------------------------------------
# 4) LIBERO — compat editable install (namespace pkg), minimal reqs, then
#    re-pin numpy/numba/opencv/mujoco to versions that satisfy TF + robosuite.
# ---------------------------------------------------------------------------
RUN git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/LIBERO && \
    pip install -e /opt/LIBERO --config-settings editable_mode=compat && \
    pip install -r /opt/openvla-oft/experiments/robot/libero/libero_requirements.txt && \
    pip install "numpy==1.26.4" "numba==0.59.1" "llvmlite==0.42.0" \
                "opencv-python==4.10.0.84" "mujoco==2.3.2"

# ---------------------------------------------------------------------------
# 5) Patch LIBERO for PyTorch 2.6+ weights_only default flip.
# ---------------------------------------------------------------------------
RUN sed -i \
    's/init_states = torch.load(init_states_path)/init_states = torch.load(init_states_path, weights_only=False)/' \
    /opt/LIBERO/libero/libero/benchmark/__init__.py

# ---------------------------------------------------------------------------
# 6) Pre-create the LIBERO config so it never shows the interactive prompt.
# ---------------------------------------------------------------------------
RUN mkdir -p /root/.libero /workspace/libero_datasets && \
    printf 'benchmark_root: %s\nbddl_files: %s/bddl_files\ninit_states: %s/init_files\ndatasets: /workspace/libero_datasets\nassets: %s/assets\n' \
      /opt/LIBERO/libero/libero /opt/LIBERO/libero/libero /opt/LIBERO/libero/libero /opt/LIBERO/libero/libero \
      > /root/.libero/config.yaml

# Sanity check at build time (import only; no GPU needed).
RUN python -c "import torch, transformers, robosuite, mujoco; \
from libero.libero import benchmark; \
print('build-time imports OK; suites:', list(benchmark.get_benchmark_dict().keys()))"

WORKDIR /opt/openvla-oft
CMD ["/bin/bash"]
