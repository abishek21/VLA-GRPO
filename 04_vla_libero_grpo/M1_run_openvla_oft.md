# M1 — Run & Understand OpenVLA-OFT on LIBERO

Goal: run a **working** VLA in the LIBERO simulator and *understand the loop*
(observation -> VLA -> action chunk -> sim steps -> success flag). No training.

Pod: RTX PRO 4000 (24 GB) is plenty — eval needs ~16 GB. Headless => set MUJOCO_GL.

Source: OpenVLA-OFT by **Stanford** (Kim, Finn, Liang), not OpenAI.
Repo: https://github.com/moojink/openvla-oft  ·  Paper: arxiv.org/abs/2502.19645

---

## Step 0 — headless rendering (do this first each session)
```bash
export MUJOCO_GL=egl        # if EGL errors later, try: export MUJOCO_GL=osmesa
```

## Step 1 — conda env
```bash
conda create -n oft python=3.10.14 -y
conda activate oft
pip3 install torch torchvision torchaudio        # CUDA build for the pod
```

## Step 2 — clone OpenVLA-OFT + install
```bash
git clone https://github.com/moojink/openvla-oft.git
cd openvla-oft
pip install -e .
# Flash-Attention 2 (for the model)
pip install packaging ninja
ninja --version; echo $?                          # should print 0
pip install "flash-attn==2.5.5" --no-build-isolation
```
> If flash-attn build fails, it's not fatal for eval; can fall back to eager
> attention. Tell me the error.

## Step 3 — install LIBERO simulator
```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt
```
> **Known gotcha — `import libero` fails after install.** LIBERO's `libero/`
> folder has no `__init__.py` (PEP-420 namespace package); modern setuptools does
> a "strict" editable install that hides the top-level import. Fix:
> ```bash
> pip install -e LIBERO --config-settings editable_mode=compat
> ```
> This drops LIBERO onto the path via a `.pth` file. Verify: `python -c "import libero; print('ok')"`.

## Step 4 — RUN a working VLA on LIBERO (the milestone)
This downloads the released LIBERO checkpoint and runs the robot in the sim:
```bash
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 5          # small first, to verify (default is 50)
```
- `--center_crop True` is REQUIRED (they trained with random crops).
- Watch the printed per-task success rates approach ~97%.
- **This is a WORKING VLA controlling the sim robot.** 🎉

## What to understand while it runs (the learning goal)
Read `experiments/robot/libero/run_libero_eval.py` and trace the loop:
```
for each task:
  reset LIBERO env -> initial observation o (images + state + instruction)
  while not done:
      action_chunk = vla.predict(o)          # <- the VLA: obs -> actions
      for a in action_chunk:                 # action chunking!
          o, reward, done, info = env.step(a)   # sim advances
      success = info['success']              # LIBERO's ground-truth flag
```
Key things to internalize:
- **Observation**: what exactly the VLA sees (image keys, resolution, proprio).
- **Action chunk**: the VLA outputs H future actions at once (not one).
- **The success flag**: LIBERO's oracle -> this is our future GRPO reward.
- **Where actions come from**: OpenVLA-OFT uses L1-regression (continuous,
  deterministic) -> relevant later for the GRPO stochastic-head fork.

## Milestone check (M1 done when)
- [ ] eval runs without crashing (rendering works)
- [ ] you see a non-trivial success rate (~90%+ on libero_spatial)
- [ ] you can point to the obs->action->step->success loop in the code

## Next (later, don't do yet)
- M2: full eval (num_trials_per_task 50) for a clean baseline number.
- M3: wire GRPO (needs bigger GPU; attach stochastic action head).
- M4: reward-hacking study (design rewards, observe exploits).

## Notes
- The 4 LIBERO suites: `libero_spatial`, `libero_object`, `libero_goal`,
  `libero_10` (a.k.a. LIBERO-Long). Start with spatial.
- Checkpoints (one per suite):
  `moojink/openvla-7b-oft-finetuned-libero-{spatial,object,goal,10}`.
