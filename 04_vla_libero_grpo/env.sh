# Source this at the start of every session:  source env.sh
# Activates the OpenVLA-OFT venv and points all caches at the big /workspace disk.

source /workspace/oft-venv/bin/activate

# Keep pip / HuggingFace / torch caches OFF the tiny 20 GB container root.
export PIP_CACHE_DIR=/workspace/.cache/pip
export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub
export TORCH_HOME=/workspace/.cache/torch
export XDG_CACHE_HOME=/workspace/.cache

# Headless MuJoCo rendering (LIBERO). If EGL fails, switch to: export MUJOCO_GL=osmesa
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

echo "[env] venv: $(which python)  |  MUJOCO_GL=$MUJOCO_GL  |  HF_HOME=$HF_HOME"
