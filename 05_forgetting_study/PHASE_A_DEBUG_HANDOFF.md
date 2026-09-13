# Phase A — Debug Handoff (2026-09-13)

Status: **infra blockers solved; one memory-config tweak left before a clean run.**

## ✅ Solved today
1. **Placement-group `already exists` (the day-eater).**
   Root cause: veRL's PG name was **deterministic** (`global_poolverl_group_1:0`);
   `ray.init()` reconnects to a surviving cluster (Ray keeps raylet+GCS alive after a
   driver crash) → collision. **Fix (permanent):** uuid-suffixed PG name in
   `verl/single_controller/ray/base.py::get_placement_groups` (committed to fork
   `4183486`). It printed `PATCHED OK` on the pod — prior runs were NEVER patched,
   which is why it kept recurring.
2. **GPU OOM** in training forward (LLaMA attention): fixed by `TRAIN_BS=2`,
   `gpu_memory_utilization=0.4`, `enable_gradient_checkpointing=True`.
3. **Ray env-worker OOM kill** — see below (the remaining item).

## 🔴 The remaining issue (understood, fix ready)
Log line (5621): `Memory on the node was 74.19GB / 77.30GB (0.96)` → **Ray killed 3 workers**.
KEY INSIGHT: **the container RAM limit is ~77 GB (cgroup), NOT the 503 GB that
`free -g` shows** (free shows the host). So setting **`param_offload=True`** pushed the
7B base (~14 GB) to CPU RAM, and the many spawned env workers (each re-imports
TF/robosuite, ~2–4 GB) pushed host RAM to the 77 GB cap → Ray OOM-killer killed the
driver.

## ✅ Fix for tomorrow (do these before launching)
1. **Turn OFF param_offload** (keep 7B on GPU; we have room now with TRAIN_BS=2 + util=0.4):
   ```bash
   F=/opt/SimpleVLA-RL/examples/run_phaseA_lora.sh
   sed -i 's/actor.fsdp_config.param_offload=True/actor.fsdp_config.param_offload=False/' $F
   grep -nE "param_offload|^TRAIN_BS=|gpu_memory_utilization" $F   # want False, 2, 0.4
   ```
2. **Ensure the uuid PG fix is on the pod** (fresh pod = re-apply):
   ```bash
   grep -n "uuid4().hex" /opt/SimpleVLA-RL/verl/single_controller/ray/base.py || \
   python3 - <<'PY'
   p="/opt/SimpleVLA-RL/verl/single_controller/ray/base.py"; s=open(p).read()
   old="            f\"{self.name_prefix}verl_group_{'_'.join([str(count) for count in self._store])}:\""
   new="            f\"{self.name_prefix}verl_group_{'_'.join([str(count) for count in self._store])}_{__import__('uuid').uuid4().hex[:8]}:\""
   open(p,"w").write(s.replace(old,new,1)); print("PATCHED" if old in s else "CHECK PATTERN")
   PY
   ```
   (Better: image should ship the fork with this baked in — see below.)
3. **Disable Ray OOM killer in the LAUNCHING shell** (align.json only reaches workers,
   not the raylet that runs the monitor):
   ```bash
   export RAY_memory_monitor_refresh_ms=0
   export RAY_memory_usage_threshold=0.98
   ```
4. **Launch once** (tmux), then watch:
   ```bash
   tmux kill-session -t phaseA 2>/dev/null; ray stop --force 2>/dev/null; rm -rf /tmp/ray; sleep 4
   cd /opt/SimpleVLA-RL; tmux new -s phaseA -d
   tmux send-keys -t phaseA 'cd /opt/SimpleVLA-RL; export MUJOCO_GL=egl NVIDIA_DRIVER_CAPABILITIES=all WANDB_API_KEY=<KEY>; export RAY_memory_monitor_refresh_ms=0 RAY_memory_usage_threshold=0.98; unset PYOPENGL_PLATFORM; bash examples/run_phaseA_lora.sh 2>&1 | tee /workspace/phaseA_out/train_log.txt' Enter
   ```
   Watch: `grep -nE "placement group|Applying LoRA|val/test_score|killed due to memory|OutOfMemory|step:" /workspace/phaseA_out/train_log.txt | tail`

## Memory tradeoff cheat-sheet (this pod: 48 GB GPU, ~77 GB container RAM)
- `param_offload=True`  → saves GPU HBM, COSTS ~14 GB host RAM → risks 77 GB cap. **Avoid here.**
- `param_offload=False` → 7B on GPU, frees host RAM. Use with small batch + low gpu_mem_util.
- If GPU OOM returns with offload OFF → go **2× GPU** (`NUM_GPUS=2`) instead of offloading.

## If it still host-OOMs with offload OFF
- Fewer parallel env spawns (root cause of the RAM spike = many workers each importing TF).
  Lower `TRAIN_BS` further (→ fewer concurrent env workers) or investigate limiting
  concurrent env_worker processes in `_generate_minibatch_libero`.

## Housekeeping done
- Fork `abishek21/SimpleVLA-RL` @ `4183486` has the uuid PG fix.
- `run_phaseA_lora.sh` (repo) has Ray clean-start guard + LoRA + (revert param_offload to
  False for this pod — repo default may still say the pod-specific value).

## TODO before next pod (optional, saves time)
- Bake the **uuid PG fix** + **NVIDIA EGL ICD** into the Docker image so fresh pods work
  out of the box (rebuild via CI).
- Decide 1-GPU (offload off, small batch) vs 2-GPU for Phase A convergence.
