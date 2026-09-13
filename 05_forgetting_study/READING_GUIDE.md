# Reading Guide — SimpleVLA-RL (veRL + GRPO + Ray + FSDP + LoRA)

A learning path to *understand* the large-scale VLA post-training stack, not just run it.
Paths are relative to `05_forgetting_study/SimpleVLA-RL/`.

---

## 0. The mental model first (read this before any code)

RL post-training = a loop of **4 stages**, each mapped to code:

```
        ┌───────────────────────────────────────────────────────────┐
        │  1. SAMPLE: policy rolls out trajectories in the env       │  rob_rollout.py
        │  2. REWARD: env verifier gives 0/1 success per trajectory  │  reward / verify
        │  3. ADVANTAGE: GRPO normalizes rewards within a group      │  core_algos (grpo)
        │  4. UPDATE: policy-gradient step on the VLA (FSDP/LoRA)    │  dp_actor / fsdp_workers
        └───────────────────────────────────────────────────────────┘
                     orchestrated by Ray across GPUs (ray_trainer.py)
```

Keep asking, for every file: **which of the 4 stages is this?**

---

## 1. Start here — the orchestrator (the "main loop")
**`verl/trainer/main_ppo.py`** — entrypoint. Builds the Ray cluster, config (Hydra),
and launches `main_task`. Skim: how config → trainer.

**`verl/trainer/ppo/ray_trainer.py`** — THE control loop. Read `fit()` and `_validate()`.
- L295–310: builds `LIBERO_Dataset` (train/val).  ← data
- `fit()`: the epoch loop → get batch → `generate_sequences` (sample) → compute
  advantage → actor update. This is stages 1→3→4 in one function.
- `_validate()`: eval-only path (what we ran for the baseline).
👉 *Goal: trace one training step end-to-end at a high level.*

---

## 2. Stage 1 — SAMPLE (rollout in the environment)
**`verl/workers/rollout/rob_rollout.py`** — the heart of VLA RL (differs from LLM RL).
- `env_worker()` (~L334): a **subprocess** that builds a LIBERO env, sets the init
  state, steps the sim, returns obs/reward. (We patched fork→spawn here.)
- `_generate_minibatch_libero()` (~L773): spawns N env workers, feeds actions,
  collects trajectories. The image→action→step loop.
- `generate_sequences()` (~L500): chunks the batch, calls the model to produce action
  tokens per step. ← the "policy acts" part.
👉 *Goal: see how a VLA rollout interleaves model inference with env stepping
(unlike an LLM that just generates text).*

**`verl/utils/libero_utils.py`** — `get_libero_env()` builds the sim env + success check.

**`verl/utils/dataset/rob_dataset.py`** — `LIBERO_Dataset._read_files_and_tokenize()`
builds the (task_id, trial_id) scenarios. ← where we add the single-suite filter.

---

## 3. Stage 2+3 — REWARD & ADVANTAGE (GRPO math)
**`verl/trainer/ppo/core_algos.py`** (search `grpo`, `advantage`) — the GRPO objective:
group-relative advantage `A_i = (R_i − mean)/std`, PPO clip, (KL term = off here).
👉 *Goal: connect the paper's Eq. (GRPO) to ~20 lines of code.*

**Reward** = binary success from the env (see rollout records / verify function).
GRPO propagates the trajectory reward to every action token.

---

## 4. Stage 4 — UPDATE (the model + distributed training)
**`verl/workers/fsdp_workers.py`** — how the VLA is loaded, sharded, LoRA'd, saved.
- L129–150: registers the OpenVLA-OFT model classes.
- L195–248: `from_pretrained` + **LoRA** injection (`get_peft_model`, `target_modules`).
- L275: `get_fsdp_wrap_policy_vla` → FSDP sharding.
- L581–620: **LoRA save** (adapter-only + merged full model). ← how experts persist.
👉 *Goal: understand FSDP `summon_full_params` for saving, and the PEFT-LoRA path.*

**`verl/workers/actor/dp_actor.py`** (or similar) — `update_policy()`: the actual
loss.backward()/optimizer step. The gradient update on the policy.

---

## 5. The model itself (what's being trained)
**`verl/utils/vla_utils/openvla_oft/modeling_prismatic.py`** — the VLA network.
- `PrismaticVisionBackbone` (~L88): SigLIP + DINOv2 fused (perception).
- `PrismaticProjector` (~L250): vision→LLM bridge.
- `self.language_model = AutoModelForCausalLM` (~L381): LLaMA-2 stack. Layers at
  `language_model.model.layers[i].self_attn.{q,k,v,o}_proj` + `.mlp.{gate,up,down}_proj`.
  ← where LoRA / future MoE-LoRA attach.
- `lm_head` → action **tokens** (discrete bins), de-tokenized to continuous actions.
👉 *Goal: locate the exact linears our experts modify; see there's no separate action
MLP (actions = LM tokens).*

---

## 6. Configs & launch scripts (glue)
- **`verl/trainer/config/ppo_trainer.yaml`** — every knob (lr, ppo_mini_batch,
  `lora_rank/lora_alpha/target_modules`, kl_ctrl, val_batch, etc.). Read top-to-bottom once.
- **`examples/run_openvla_oft_rl_libero.sh`** — the training launcher; maps CLI overrides
  onto the yaml. Our `eval_smoke.sh` is a val_only copy.
- **`align.json`** — Ray runtime env vars (MUJOCO_GL, etc.).

---

## Suggested reading order (1 sitting each)
1. `main_ppo.py` + `ray_trainer.py::fit` (the loop).
2. `rob_rollout.py` (`env_worker`, `_generate_minibatch_libero`) (sampling).
3. `core_algos.py` grpo/advantage (the math).
4. `fsdp_workers.py` (load/LoRA/FSDP/save).
5. `modeling_prismatic.py` (the network).
6. `ppo_trainer.yaml` + the launch script (glue).

## Exercises to cement understanding
- Draw the 4-stage loop and annotate each with the file+function that implements it.
- Trace a single `task_id`/`trial_id` from `rob_dataset.py` → `rob_rollout.py` → reward.
- Find where `kl_coef` would re-enable the KL term (core_algos) — the knob SimpleVLA set 0.
- Find the 224 LLaMA linears LoRA targets (`q,k,v,o,gate,up,down` × 32 layers).

## Concepts to look up alongside (just enough theory)
- GRPO (group-relative advantage) vs PPO — why no value network.
- FSDP (fully-sharded data parallel) — params/grads/optimizer sharded across GPUs.
- LoRA / PEFT — low-rank adapters; `merge_and_unload`.
- Ray actors — each GPU worker is a Ray actor; `WorkerDict` wraps them.
- vLLM rollout vs HF generate — why VLA uses closed-loop env stepping.
