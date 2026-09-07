# SimpleVLA-RL + OpenVLA-OFT + LIBERO — RunPod custom template
#
# Purpose: reproducible env for the "skill interference / catastrophic forgetting"
# study. Provides SimpleVLA-RL's veRL-based RL + eval stack on LIBERO.
#
# Build on ANY machine (no GPU needed to build; CUDA kernels run at runtime).
# On Apple Silicon you MUST target amd64:  docker buildx build --platform linux/amd64 ...
# Built for free on GitHub Actions -> GHCR (see the workflow).
#
# WHY THIS BASE: SimpleVLA-RL/veRL 0.2.x pins **torch 2.4.0 + cu124** (their SETUP
# warns newer torch breaks veRL). The pytorch official image ships exactly that,
# uses CONDA python (so pip works -> avoids the PEP 668 "externally-managed"
# error from Debian system python), and is a -devel image (has nvcc for
# flash-attn).

FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    MUJOCO_GL=egl \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/workspace/.cache/huggingface \
    MAX_JOBS=4 \
    PYTHONUNBUFFERED=1
# NOTE: do NOT hard-set PYOPENGL_PLATFORM=egl here — it blocks the osmesa
# software-render fallback used when the pod lacks NVIDIA_DRIVER_CAPABILITIES
# =graphics. Set MUJOCO_GL (egl or osmesa) at runtime instead. For fast GPU
# rendering, run the pod with NVIDIA_DRIVER_CAPABILITIES=all.

# ---------------------------------------------------------------------------
# 1) System libs: build tools + EGL/GL dispatch libs for headless MuJoCo render.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake \
        libglvnd0 libglvnd-dev libegl1 libegl1-mesa-dev libgl1 libgles2 \
        libglib2.0-0 libosmesa6 ffmpeg wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

# Their required layout: verl / openvla-oft / LIBERO / SimpleVLA-RL as siblings.
WORKDIR /opt

# ---------------------------------------------------------------------------
# 2) veRL 0.2.x — the RL backbone. Keep the torch 2.4 already in the base.
# ---------------------------------------------------------------------------
RUN git clone -b v0.2.x https://github.com/volcengine/verl.git /opt/verl && \
    cd /opt/verl && \
    sed -i 's/^torch==.*/# torch pin removed (base provides 2.4.0)/' requirements.txt 2>/dev/null || true && \
    pip install -e .

# ---------------------------------------------------------------------------
# 3) OpenVLA-OFT — drop its torch pins (keep base torch 2.4), install, flash-attn.
#    NOTE: flash-attn compiles CUDA kernels; slow on CI. If the CI build OOMs on
#    this step, comment it out and install flash-attn at runtime on the GPU.
# ---------------------------------------------------------------------------
RUN git clone https://github.com/moojink/openvla-oft.git /opt/openvla-oft && \
    cd /opt/openvla-oft && \
    sed -i 's/^    "torch==2.2.0",/    # "torch==2.2.0",/' pyproject.toml && \
    sed -i 's/^    "torchvision==0.17.0",/    # "torchvision==0.17.0",/' pyproject.toml && \
    sed -i 's/^    "torchaudio==2.2.0",/    # "torchaudio==2.2.0",/' pyproject.toml && \
    pip install -e .
RUN pip install packaging ninja && \
    (pip install flash-attn --no-build-isolation || \
     echo "WARN: flash-attn build failed at image time; install at runtime on GPU")

# ---------------------------------------------------------------------------
# 4) LIBERO — compat editable install (namespace pkg) + reqs + version pins.
# ---------------------------------------------------------------------------
RUN git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/LIBERO && \
    pip install -e /opt/LIBERO --config-settings editable_mode=compat && \
    pip install -r /opt/openvla-oft/experiments/robot/libero/libero_requirements.txt && \
    pip install "numpy==1.26.4" "numba==0.59.1" "llvmlite==0.42.0" \
                "opencv-python==4.10.0.84" "mujoco==2.3.2"

# Pin huggingface-hub < 1.0: transformers requires >=0.19.3,<1.0. A later
# `pip install -U huggingface_hub` (e.g. to speed checkpoint downloads) pulls
# 1.30+ and breaks transformers. Pin so the image stays consistent.
RUN pip install "huggingface-hub==0.25.2"

# LIBERO patch for PyTorch weights_only default flip.
RUN sed -i \
    's/init_states = torch.load(init_states_path)/init_states = torch.load(init_states_path, weights_only=False)/' \
    /opt/LIBERO/libero/libero/benchmark/__init__.py || true

# Patch robosuite EGL: some RunPod hosts expose 0 EGL devices via
# eglQueryDevicesEXT even though the GPU renders fine via EGL_DEFAULT_DISPLAY.
# Add a fallback so robosuite (LIBERO) uses the working default-display path.
RUN python - <<'PY'
import re
p="/opt/conda/lib/python3.11/site-packages/robosuite/renderers/context/egl_context.py"
src=open(p).read()
patch='''def create_initialized_egl_device_display(device_id=0):
    """Creates an initialized EGL display directly on a device."""
    from mujoco.egl import egl_ext as EGL
    from OpenGL import error, EGL as _EGL
    try:
        all_devices = EGL.eglQueryDevicesEXT()
    except Exception:
        all_devices = []
    if not all_devices:
        display = _EGL.eglGetDisplay(_EGL.EGL_DEFAULT_DISPLAY)
        if display != _EGL.EGL_NO_DISPLAY:
            try:
                if _EGL.eglInitialize(display, None, None):
                    return display
            except Exception:
                pass
        return _EGL.EGL_NO_DISPLAY
'''
src2=re.sub(r'def create_initialized_egl_device_display\(device_id=0\):\n    """Creates an initialized EGL display directly on a device\."""\n    all_devices = EGL\.eglQueryDevicesEXT\(\)\n', patch, src, count=1)
open(p,"w").write(src2)
print("robosuite EGL patched" if src2!=src else "WARN: robosuite EGL patch not applied")
PY

# Pre-create LIBERO config so it never shows the interactive prompt.
RUN mkdir -p /root/.libero /workspace/libero_datasets && \
    printf 'benchmark_root: %s\nbddl_files: %s/bddl_files\ninit_states: %s/init_files\ndatasets: /workspace/libero_datasets\nassets: %s/assets\n' \
       /opt/LIBERO/libero/libero /opt/LIBERO/libero/libero /opt/LIBERO/libero/libero /opt/LIBERO/libero/libero \
       > /root/.libero/config.yaml

# ---------------------------------------------------------------------------
# 5) SimpleVLA-RL itself.
# ---------------------------------------------------------------------------
RUN git clone https://github.com/PRIME-RL/SimpleVLA-RL.git /opt/SimpleVLA-RL

# Build-time sanity import (no GPU needed).
RUN python -c "import torch; print('torch', torch.__version__); \
import robosuite, mujoco; from libero.libero import benchmark; \
print('suites:', list(benchmark.get_benchmark_dict().keys()))" || \
    echo "WARN: sanity import had issues; verify on the GPU pod."

WORKDIR /opt/SimpleVLA-RL
# Keep the container alive on RunPod (a bare bash CMD exits immediately and the
# pod's container stops). sleep infinity holds it open so SSH/web-terminal work.
CMD ["sleep", "infinity"]
