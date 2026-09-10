# Study 05 — Skill Interference / Catastrophic Forgetting in Targeted VLA RL

## Research question
When a multi-task VLA already knows tasks T1..T10, if we use RL to improve one
weak task T_i, does performance on the *other* previously-learned tasks
**degrade (forgetting), stay stable (retention), or improve (positive transfer)?**

This is the RLHF "alignment tax" question, transposed to embodied multi-task
policies — underexplored and directly measurable via per-task before/after.

## Phase 1 (current): establish the SFT baseline — NO RL yet
Reproduce the SFT-only per-task success on **LIBERO-Long (libero-10, 10 tasks)**
and produce a per-task table (not just suite average — forgetting is a
*distributional* effect the average hides).

## Model / checkpoint (locked)
- **Checkpoint:** `OpenVLA-OFT, libero-10 (LIBERO-Long), trajall SFT` from the
  **SimpleVLA-RL HuggingFace collection** (`Haozhan72/simplevla-rl-...`), NOT the
  official Stanford `moojink/...`.
  - "OpenVLA-OFT" = architecture; the *weights* are SimpleVLA-RL's.
  - `trajall` (~86.5%), NOT `traj1` (~17.3%) — a forgetting study needs a policy
    that already knows all 10 tasks well (something to forget).
- **Why SimpleVLA-RL's checkpoint:** our later RL uses SimpleVLA-RL's veRL/GRPO
  pipeline; baseline and post-RL must be measured through the IDENTICAL pipeline
  for the before/after deltas to be valid.

## Evaluation (locked)
- Use **SimpleVLA-RL's own eval** (`trainer.val_only=True` in
  `examples/run_openvla_oft_rl_libero.sh`), not the vanilla OFT script — same
  pipeline the RL will use.
- Extract **per-task** success (patch `rob_rollout.py` logging if it only reports
  suite average).
- Fixed episode count per task (e.g. 50/task = 500 total). Sanity check: average
  ≈ **86.5%**; report ACTUAL numbers.

## Deliverable (Phase 1)

**Status: ✅ pipeline reproduced.** Smoke run (n=10 trials/task, 100 episodes) on
a single RTX A6000 gives suite average **88.0%** — matches the ~86.5% target.
(Full 50/task = 500-episode run still TODO for publication-grade per-task numbers.)

| Task | Success (smoke, n=10) | Full (n=50) |
|------|----------------------:|------------:|
| task_0 |  90% | … |
| task_1 | 100% | … |
| task_2 |  80% | … |
| task_3 |  80% | … |
| task_4 |  90% | … |
| task_5 | 100% | … |
| task_6 |  80% | … |
| task_7 |  **70% (weakest)** | … |
| task_8 |  90% | … |
| task_9 | 100% | … |
| **Average** | **88.0%** | **… (target ~86.5%)** |

- **task_7** = _"put both the alphabet soup and the cream cheese box in the basket"_
  — a two-object task, natural Phase 2 RL candidate.
- To run the full baseline: `data.num_trials_per_task=50`, `data.val_batch_size=500`,
  `actor_rollout_ref.rollout.val_micro_batch_size=10` (500/10 divides cleanly).

### Fixes required to make eval run (now baked into `Dockerfile`)
1. **fork → spawn** for LIBERO env workers in `rob_rollout.py` (fixes the original
   `EGL_BAD_ALLOC` — forking after CUDA init poisons the child's GPU/EGL state).
2. **NVIDIA EGL ICD** `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` (else GLVND
   falls back to Mesa → `/dev/dri` permission denied → no PLATFORM_DEVICE).
3. **val batch divisibility** (`val_micro_batch_size` divides `val_batch_size`).

Run-time env: `MUJOCO_GL=egl`, pod `NVIDIA_DRIVER_CAPABILITIES=all`,
`unset PYOPENGL_PLATFORM`.

## Infra (one-time)
- Docker image: `05_forgetting_study/Dockerfile` -> GHCR via
  `.github/workflows/build-simplevla-image.yml` (built free on CI, no GPU).
  Base `pytorch:2.4.0-cuda12.4` (matches veRL 0.2.x torch pin).
- Use as a **RunPod custom template**:
  `ghcr.io/abishek21/simplevla-rl-libero:cu124` (make the package public first).

## Known gotchas / open items
1. **Single-GPU eval** — veRL/Ray assumes multi-GPU; first test `NUM_GPUS=1`
   `val_only=True` runs at all. (Their runs used 8× A800 80GB.)
2. **Per-task breakdown** — may need to pull from `rob_rollout.py` logs/records.
3. **flash-attn** — compiles CUDA kernels; may fail at image build (CI OOM). The
   Dockerfile falls back to a warning; install at runtime on the GPU if needed.
4. **Single-task RL cost (Phase 2 gate)** — verify single-task GRPO fits 1-2 GPUs
   BEFORE committing to many experiments. Single-task should be far lighter than
   their full multi-task scaling runs.

## Phase 2 (later, after baseline)
- Identify naturally weak task(s) from the per-task table.
- RL-post-train on ONE weak task via SimpleVLA-RL.
- Re-eval ALL 10 tasks (identical pipeline) -> measure transfer / retention /
  interference. Repeat for a few tasks. This loop is cheap once infra works.

## References
- SimpleVLA-RL: arXiv 2509.09674 · github.com/PRIME-RL/SimpleVLA-RL (ICLR 2026)
- OpenVLA-OFT: arXiv 2502.19645 · github.com/moojink/openvla-oft
- LIBERO: github.com/Lifelong-Robot-Learning/LIBERO
