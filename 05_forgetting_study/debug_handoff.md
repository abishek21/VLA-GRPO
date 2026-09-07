# SimpleVLA-RL LIBERO-Long Eval — Debugging Handoff

## Goal
Run SimpleVLA-RL's **eval-only** (`val_only=True`) on **LIBERO-Long (libero_10)** with the
checkpoint `Haozhan72/Openvla-oft-SFT-libero10-trajall` (~86.5% baseline) to produce a
**per-task success table** (T1..T10). This is the baseline for a *catastrophic forgetting /
skill interference* study (later: RL one task, re-eval all 10, measure interference).

## Current status: BLOCKED on `EGL_BAD_ALLOC` when Ray worker creates the LIBERO render context

Everything works EXCEPT the sim render context creation inside the vLLM rollout worker.

---

## Environment (all working)
- **Image:** `ghcr.io/abishek21/simplevla-rl-libero:cu124` (built via GitHub Actions, public)
- **Repo Dockerfile:** `05_forgetting_study/Dockerfile`
- **Pod:** RunPod, **A40 48GB** (sm_86 — NOT Blackwell; Blackwell/sm_120 is incompatible with torch 2.4)
- **Pod env var REQUIRED:** `NVIDIA_DRIVER_CAPABILITIES=all` (else no NVIDIA EGL libs)
- **Start command:** `sleep infinity` (else container exits)
- **Persistent volume** at `/workspace` (checkpoint lives here, survives restarts)
- torch 2.4.0 + cu124, veRL 0.2.x, OpenVLA-OFT, LIBERO, SimpleVLA-RL all at `/opt/...`

## Verified working (in isolation)
1. `nvidia-smi` → A40 visible ✅
2. `torch 2.4.0`, `x.cuda()` compute OK ✅
3. `from libero.libero import benchmark` → 6 suites incl. `libero_10` ✅
4. **Standalone** `OffScreenRenderEnv(...)` + `env.reset()` → **`ENV INIT + RESET OK`** ✅
   (i.e. rendering works in a FRESH process with no model loaded)
5. `mujoco.Renderer` + EGL default-display → renders ✅

## The blocker (reproduced, root cause identified)
- **Error:** `OpenGL.raw.EGL._errors.EGLError: EGLError(err = EGL_BAD_ALLOC, baseOperation = eglCreateContext)`
- **Where:** `robosuite/utils/binding_utils.py:78` → `egl_context.py:152` `eglCreateContext`,
  called from `robosuite/environments/base.py:299`
  `render_context = MjRenderContextOffscreen(self.sim, device_id=self.render_gpu_device_id)`
- **Swallowed by:** `verl/workers/rollout/rob_rollout.py:349` bare `except:` in a `while True:`
  loop that prints `*** env initialization failed ***` forever. (We added
  `import traceback; traceback.print_exc()` after line 350 to expose it.)
- **Root cause:** The Ray **rollout worker** already holds vLLM's CUDA context on the GPU.
  Creating a SECOND GPU context (EGL render) in the same process on the same device fails
  with `EGL_BAD_ALLOC`. Standalone env creation works because no model context exists there.
- vLLM `gpu_memory_utilization=0.9` (in `examples/*.sh:72`) makes it worse (reserves ~41GB).

---

## Fixes ALREADY applied / baked into the image
1. **PEP 668** → use pytorch conda base + `PIP_BREAK_SYSTEM_PACKAGES=1`. ✅
2. **huggingface-hub pin** `==0.25.2` (else `-U` pulls 1.30 → breaks transformers). ✅ baked
3. **LIBERO** `editable_mode=compat` install + `weights_only=False` patch + pre-created
   `/root/.libero/config.yaml`. ✅ baked
4. **robosuite EGL default-display fallback** patch in
   `egl_context.py::create_initialized_egl_device_display` (handles hosts where
   `eglQueryDevicesEXT()` returns 0 devices). ✅ baked
5. **tmux** installed; **CMD sleep infinity**. ✅ baked
6. **align.json** `env_vars` → added `MUJOCO_GL=egl`, `WANDB_MODE=disabled`,
   removed placeholder `WANDB_API_KEY`. (Ray workers only inherit align.json env, NOT the shell.)

## Fixes TRIED that did NOT resolve EGL_BAD_ALLOC
- Lowering `gpu_memory_utilization` 0.9 → 0.4 (still EGL_BAD_ALLOC).
- osmesa (CPU render): **broken in image** (`'NoneType' has no attribute 'glGetError'`).
- Forcing `__EGL_VENDOR_LIBRARY_FILENAMES`, `MUJOCO_EGL_DEVICE_ID=0`.

---

## The eval script (`/opt/SimpleVLA-RL/examples/eval_smoke.sh`, copy of run_openvla_oft_rl_libero.sh)
Configured via sed:
- `SFT_MODEL_PATH="/workspace/ckpt_libero10_trajall"`
- `CKPT_PATH="/workspace/eval_out"`
- `NUM_GPUS=1`
- `DATASET_NAME="libero_10"`
- `trainer.val_only=True`, `trainer.logger=['console']`, `trainer.wandb_mode=offline`
- `data.num_trials_per_task=10` (smoke; full = 50)
- `actor_rollout_ref.rollout.gpu_memory_utilization=0.9` (line 72)

Run with:
```bash
cd /opt/SimpleVLA-RL
export MUJOCO_GL=egl WANDB_MODE=disabled
unset PYOPENGL_PLATFORM
bash examples/eval_smoke.sh 2>&1 | tee /workspace/eval_out/smoke_log.txt
```

---

## NEXT THINGS TO TRY (in priority order)

### A. Check SimpleVLA-RL issues / docs (agent with web access should do this FIRST)
Search https://github.com/PRIME-RL/SimpleVLA-RL/issues and veRL issues for:
`EGL_BAD_ALLOC`, `env initialization failed`, `single GPU`, `MjRenderContextOffscreen`,
`gpu_memory_utilization`, `enforce_eager`. Determine if single-GPU eval is even supported.

### B. Try `enforce_eager=True` for vLLM (disables CUDA graphs that reserve big GPU blocks)
```bash
grep -rn "enforce_eager\|free_cache_engine" /opt/SimpleVLA-RL/
# add to the python cmd in eval_smoke.sh:
#   actor_rollout_ref.rollout.enforce_eager=True
```

### C. Force robosuite render off the GPU device (device_id=-1)
```bash
python - <<'PY'
p="/opt/conda/lib/python3.11/site-packages/robosuite/environments/base.py"
s=open(p).read()
import re
s2=re.sub(r'self\.render_gpu_device_id\s*=\s*.*', 'self.render_gpu_device_id = -1', s, count=1)
open(p,"w").write(s2); print("forced -1" if s2!=s else "not found")
PY
```

### D. Use 2 GPUs (STRONG CANDIDATE — likely the intended config)
SimpleVLA-RL examples use **8 GPUs**. vLLM rollout + EGL render likely need separate GPUs.
Deploy **2× A40**, set `NUM_GPUS=2`. The context conflict may vanish when render and
inference are on different devices.

### E. Fix osmesa (CPU render) as a fallback
osmesa is broken in the image. Would need proper `libOSMesa` + PyOpenGL osmesa platform wired.
If fixed, set `MUJOCO_GL=osmesa` in align.json → render on CPU, avoid GPU context conflict.

---

## Key file references
- `verl/workers/rollout/rob_rollout.py:333` `env_worker()`, `:347` `get_libero_env()`, `:349` bare except
- `verl/utils/libero_utils.py:20` `get_libero_env()` (uses `OffScreenRenderEnv`, imports tensorflow)
- `robosuite/environments/base.py:299` render context creation
- `robosuite/renderers/context/egl_context.py:152` `eglCreateContext` (EGL_BAD_ALLOC)
- `examples/eval_smoke.sh:72` `gpu_memory_utilization`

## The checkpoint (verified present, ~15GB)
`/workspace/ckpt_libero10_trajall/` → 4 safetensors shards + config.json + dataset_statistics.json ✅

## Repro of the REAL error (bypasses Ray, shows EGL_BAD_ALLOC directly is context conflict)
Standalone env works; only fails inside the vLLM worker → confirms 2-context-per-GPU conflict.
```

I've written the full handoff above. **Copy that block into a file** — I'd suggest `05_forgetting_study/DEBUG_HANDOFF.md` in your repo so it's version-controlled and openable in any future session.

## What this doc captures

- ✅ **The exact goal** and where you are
- ✅ **Everything verified working** (so no re-testing)
- ✅ **The precise root cause** (EGL_BAD_ALLOC = 2 GPU contexts in one worker process)
- ✅ **All fixes already baked in** (so they're not re-attempted)
- ✅ **What was tried and failed** (so those aren't repeated)
- ✅ **Prioritized next steps** — with **2× GPU (option D)** flagged as the strong candidate
- ✅ **Exact file:line references** for fast navigation

## My honest recommendation for next session

Start with **A** (read their issues — a web-capable agent can do this) and **D** (try 2 GPUs). My strong bet is **D — deploy 2× A40 with `NUM_GPUS=2`** — because:
- Their examples all use 8 GPUs.
- The error is fundamentally "vLLM + EGL can't share one GPU's context."
- Separate GPUs for rollout vs. render likely sidesteps it entirely.

## To save it now

Create the file and paste the block:
```bash
# on your Mac, in the repo:
# create 05_forgetting_study/DEBUG_HANDOFF.md, paste the markdown, then:
git add 05_forgetting_study/DEBUG_HANDOFF.md
git commit -m "Add SimpleVLA-RL eval debugging handoff doc"
git push
```

Want me to also add the **2-GPU quick-start commands** (deploy config + the `NUM_GPUS=2` script edits) to the doc, so next session you can try option D immediately without re-deriving it? That's the most likely path to finally getting your baseline numbers.# SimpleVLA-RL LIBERO-Long Eval — Debugging Handoff

## Goal
Run SimpleVLA-RL's **eval-only** (`val_only=True`) on **LIBERO-Long (libero_10)** with the
checkpoint `Haozhan72/Openvla-oft-SFT-libero10-trajall` (~86.5% baseline) to produce a
**per-task success table** (T1..T10). This is the baseline for a *catastrophic forgetting /
skill interference* study (later: RL one task, re-eval all 10, measure interference).

## Current status: BLOCKED on `EGL_BAD_ALLOC` when Ray worker creates the LIBERO render context

Everything works EXCEPT the sim render context creation inside the vLLM rollout worker.

---

## Environment (all working)
- **Image:** `ghcr.io/abishek21/simplevla-rl-libero:cu124` (built via GitHub Actions, public)
- **Repo Dockerfile:** `05_forgetting_study/Dockerfile`
- **Pod:** RunPod, **A40 48GB** (sm_86 — NOT Blackwell; Blackwell/sm_120 is incompatible with torch 2.4)
- **Pod env var REQUIRED:** `NVIDIA_DRIVER_CAPABILITIES=all` (else no NVIDIA EGL libs)
- **Start command:** `sleep infinity` (else container exits)
- **Persistent volume** at `/workspace` (checkpoint lives here, survives restarts)
- torch 2.4.0 + cu124, veRL 0.2.x, OpenVLA-OFT, LIBERO, SimpleVLA-RL all at `/opt/...`

## Verified working (in isolation)
1. `nvidia-smi` → A40 visible ✅
2. `torch 2.4.0`, `x.cuda()` compute OK ✅
3. `from libero.libero import benchmark` → 6 suites incl. `libero_10` ✅
4. **Standalone** `OffScreenRenderEnv(...)` + `env.reset()` → **`ENV INIT + RESET OK`** ✅
   (i.e. rendering works in a FRESH process with no model loaded)
5. `mujoco.Renderer` + EGL default-display → renders ✅

## The blocker (reproduced, root cause identified)
- **Error:** `OpenGL.raw.EGL._errors.EGLError: EGLError(err = EGL_BAD_ALLOC, baseOperation = eglCreateContext)`
- **Where:** `robosuite/utils/binding_utils.py:78` → `egl_context.py:152` `eglCreateContext`,
  called from `robosuite/environments/base.py:299`
  `render_context = MjRenderContextOffscreen(self.sim, device_id=self.render_gpu_device_id)`
- **Swallowed by:** `verl/workers/rollout/rob_rollout.py:349` bare `except:` in a `while True:`
  loop that prints `*** env initialization failed ***` forever. (We added
  `import traceback; traceback.print_exc()` after line 350 to expose it.)
- **Root cause:** The Ray **rollout worker** already holds vLLM's CUDA context on the GPU.
  Creating a SECOND GPU context (EGL render) in the same process on the same device fails
  with `EGL_BAD_ALLOC`. Standalone env creation works because no model context exists there.
- vLLM `gpu_memory_utilization=0.9` (in `examples/*.sh:72`) makes it worse (reserves ~41GB).

---

## Fixes ALREADY applied / baked into the image
1. **PEP 668** → use pytorch conda base + `PIP_BREAK_SYSTEM_PACKAGES=1`. ✅
2. **huggingface-hub pin** `==0.25.2` (else `-U` pulls 1.30 → breaks transformers). ✅ baked
3. **LIBERO** `editable_mode=compat` install + `weights_only=False` patch + pre-created
   `/root/.libero/config.yaml`. ✅ baked
4. **robosuite EGL default-display fallback** patch in
   `egl_context.py::create_initialized_egl_device_display` (handles hosts where
   `eglQueryDevicesEXT()` returns 0 devices). ✅ baked
5. **tmux** installed; **CMD sleep infinity**. ✅ baked
6. **align.json** `env_vars` → added `MUJOCO_GL=egl`, `WANDB_MODE=disabled`,
   removed placeholder `WANDB_API_KEY`. (Ray workers only inherit align.json env, NOT the shell.)

## Fixes TRIED that did NOT resolve EGL_BAD_ALLOC
- Lowering `gpu_memory_utilization` 0.9 → 0.4 (still EGL_BAD_ALLOC).
- osmesa (CPU render): **broken in image** (`'NoneType' has no attribute 'glGetError'`).
- Forcing `__EGL_VENDOR_LIBRARY_FILENAMES`, `MUJOCO_EGL_DEVICE_ID=0`.

---

## The eval script (`/opt/SimpleVLA-RL/examples/eval_smoke.sh`, copy of run_openvla_oft_rl_libero.sh)
Configured via sed:
- `SFT_MODEL_PATH="/workspace/ckpt_libero10_trajall"`
- `CKPT_PATH="/workspace/eval_out"`
- `NUM_GPUS=1`
- `DATASET_NAME="libero_10"`
- `trainer.val_only=True`, `trainer.logger=['console']`, `trainer.wandb_mode=offline`
- `data.num_trials_per_task=10` (smoke; full = 50)
- `actor_rollout_ref.rollout.gpu_memory_utilization=0.9` (line 72)

Run with:
```bash
cd /opt/SimpleVLA-RL
export MUJOCO_GL=egl WANDB_MODE=disabled
unset PYOPENGL_PLATFORM
bash examples/eval_smoke.sh 2>&1 | tee /workspace/eval_out/smoke_log.txt
```

---

## NEXT THINGS TO TRY (in priority order)

### A. Check SimpleVLA-RL issues / docs (agent with web access should do this FIRST)
Search https://github.com/PRIME-RL/SimpleVLA-RL/issues and veRL issues for:
`EGL_BAD_ALLOC`, `env initialization failed`, `single GPU`, `MjRenderContextOffscreen`,
`gpu_memory_utilization`, `enforce_eager`. Determine if single-GPU eval is even supported.

### B. Try `enforce_eager=True` for vLLM (disables CUDA graphs that reserve big GPU blocks)
```bash
grep -rn "enforce_eager\|free_cache_engine" /opt/SimpleVLA-RL/
# add to the python cmd in eval_smoke.sh:
#   actor_rollout_ref.rollout.enforce_eager=True
```

### C. Force robosuite render off the GPU device (device_id=-1)
```bash
python - <<'PY'
p="/opt/conda/lib/python3.11/site-packages/robosuite/environments/base.py"
s=open(p).read()
import re
s2=re.sub(r'self\.render_gpu_device_id\s*=\s*.*', 'self.render_gpu_device_id = -1', s, count=1)
open(p,"w").write(s2); print("forced -1" if s2!=s else "not found")
PY
```

### D. Use 2 GPUs (STRONG CANDIDATE — likely the intended config)
SimpleVLA-RL examples use **8 GPUs**. vLLM rollout + EGL render likely need separate GPUs.
Deploy **2× A40**, set `NUM_GPUS=2`. The context conflict may vanish when render and
inference are on different devices.

### E. Fix osmesa (CPU render) as a fallback
osmesa is broken in the image. Would need proper `libOSMesa` + PyOpenGL osmesa platform wired.
If fixed, set `MUJOCO_GL=osmesa` in align.json → render on CPU, avoid GPU context conflict.

---

## Key file references
- `verl/workers/rollout/rob_rollout.py:333` `env_worker()`, `:347` `get_libero_env()`, `:349` bare except
- `verl/utils/libero_utils.py:20` `get_libero_env()` (uses `OffScreenRenderEnv`, imports tensorflow)
- `robosuite/environments/base.py:299` render context creation
- `robosuite/renderers/context/egl_context.py:152` `eglCreateContext` (EGL_BAD_ALLOC)
- `examples/eval_smoke.sh:72` `gpu_memory_utilization`

## The checkpoint (verified present, ~15GB)
`/workspace/ckpt_libero10_trajall/` → 4 safetensors shards + config.json + dataset_statistics.json ✅

## Repro of the REAL error (bypasses Ray, shows EGL_BAD_ALLOC directly is context conflict)
Standalone env works; only fails inside the vLLM worker → confirms 2-context-per-GPU conflict.
```

I've written the full handoff above. **Copy that block into a file** — I'd suggest `05_forgetting_study/DEBUG_HANDOFF.md` in your repo so it's version-controlled and openable in any future session.

## What this doc captures

- ✅ **The exact goal** and where you are
- ✅ **Everything verified working** (so no re-testing)
- ✅ **The precise root cause** (EGL_BAD_ALLOC = 2 GPU contexts in one worker process)
- ✅ **All fixes already baked in** (so they're not re-attempted)
- ✅ **What was tried and failed** (so those aren't repeated)
- ✅ **Prioritized next steps** — with **2× GPU (option D)** flagged as the strong candidate
- ✅ **Exact file:line references** for fast navigation

## My honest recommendation for next session

Start with **A** (read their issues — a web-capable agent can do this) and **D** (try 2 GPUs). My strong bet is **D — deploy 2× A40 with `NUM_GPUS=2`** — because:
- Their examples all use 8 GPUs.
- The error is fundamentally "vLLM + EGL can't share one GPU's context."
- Separate GPUs for rollout vs. render likely sidesteps it entirely.

## To save it now

Create the file and paste the block:
```bash
# on your Mac, in the repo:
# create 05_forgetting_study/DEBUG_HANDOFF.md, paste the markdown, then:
git add 05_forgetting_study/DEBUG_HANDOFF.md
git commit -m "Add SimpleVLA-RL eval debugging handoff doc"
git push
```

Want me to also add the **2-GPU quick-start commands** (deploy config + the `NUM_GPUS=2` script edits) to the doc, so next session you can try option D immediately without re-deriving it? That's the most likely path to finally getting your baseline