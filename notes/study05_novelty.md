# Study 05 — Novelty Memo: RL-Induced Forgetting in Single-Task VLA Specialization

*Grounded literature check + defensible thesis. Last updated: 2026-09-10.*

---

## 0. TL;DR

- **Our question:** When a multi-task VLA already masters tasks T1..T10, does using
  RL to improve **one** weak task cause **forgetting on the other, already-mastered
  tasks** — and is that forgetting **structured** (predictable from task similarity)
  and **controllable** (via a KL-to-SFT anchor)?
- **Status of the gap:** **Survives** a check of the 3 closest papers. None measure
  per-task *retention* of already-mastered seen tasks after **single-task** RL
  specialization on a multi-skill policy.
- **Extra leverage:** the SimpleVLA-RL pipeline we use **removes the KL penalty by
  design** (β=0), i.e. the classic anti-forgetting safety belt is OFF — a perfect
  built-in mitigation experiment.

---

## 1. What we mean (precise definitions)

- **Regime — single-task specialization:** run GRPO/RL on exactly ONE task `T_i`
  (its 50 init states), NOT multi-task RL.
- **Dependent variable — retention/interference vector:** the per-task success
  delta `ΔT_j = SR_after(T_j) − SR_before(T_j)` for every **already-mastered, seen**
  task `j ≠ i`. The whole 10-dim vector, not the suite average.
- **Forgetting** = negative `ΔT_j` on seen tasks (alignment-tax analogue).
  **Positive transfer** = positive `ΔT_j`. **Retention** = `ΔT_j ≈ 0`.
- Contrast with **generalization** (the term the literature uses): success on
  **UNSEEN / OOD** tasks or perturbed versions of a task. Different DV.

---

## 2. Closest prior work — and why our gap survives

| Paper | What they measure | Regime | Eval target | Overlap with us |
|-------|-------------------|--------|-------------|-----------------|
| **SimpleVLA-RL** (2509.09674) §5.2 "Generalization Analysis" | RL vs SFT on a **held-out UNSEEN** task | **multi-task** RL (train on 9 seen) | 1 unseen/OOD task | Claims "RL retains capabilities" — but in the *friendly* multi-task+OOD regime. Does NOT test single-task specialization's effect on the other **seen** tasks. |
| **Liu et al. 2025a** (2505.19789, NeurIPS'25) "What can RL bring to VLA generalization?" | RL vs SFT robustness across **visual / semantic / execution** distribution shifts; PPO > GRPO/DPO | fine-tune then perturb | perturbed/OOD variants | About robustness of a task under perturbation, NOT retention of *other* tasks. |
| **RIPT-VLA** (2505.17016) "Interactive Post-Training" | Sparse-reward post-training; boosts single task massively (OpenVLA-OFT→97.5%) | single/low-data | the trained task (+ some cross-task/scenario claims as a *positive*) | Claims cross-task generalization as a capability; no per-task **before/after forgetting audit** on a multi-skill policy. |

**Conclusion:** the specific measurement — *"specialize `T_i` → per-task success
change on the other already-mastered seen tasks"* — is **not reported** by any of
these. The RLHF "alignment tax" (InstructGPT, Llama-2) and continual-learning
"catastrophic forgetting" are old, but their **transposition to single-task RL of a
multi-task VLA with verifiable sim rewards** is open.

⚠️ **Residual risk to keep checking:** RLinf, VLA-RL (2505.18719), ConRFT, GRAPE,
iRe-VLA (Guo 2025b), TGRPO. Skim each for a per-task retention table before final
claims. (Priors: they focus on gains/efficiency, not seen-task retention.)

---

## 3. Why this is a *contribution*, not a re-confirmation

1. **Qualifies a published claim.** SimpleVLA-RL asserts *"RL retains previously
   acquired capabilities."* We test whether that holds under **narrow single-task
   specialization** (very plausibly it does NOT — the classic tax reappears).
   Refining/bounding a specific published claim is a clean, defensible result.
2. **Structure/predictability.** Hypothesis: `ΔT_j` correlates with **task
   similarity** to `T_i` (shared objects/verbs/scene). A similarity→interference
   correlation is a positive result, not just "forgetting happens."
3. **Mechanism.** Link interference to **action-distribution drift**: measure
   `KL(π_after ‖ π_before)` over action-token logits on fixed off-target states;
   does it predict the success drop? Ties to SimpleVLA-RL's **"pushcut"** — RL
   rewriting grasp→push on `T_i` could break a `T_j` that needs grasping.
4. **Mitigation they discarded.** SimpleVLA-RL sets **β=0 (no KL)** for exploration.
   We sweep `kl_coef` (β>0) to trace the **gain↔retention Pareto frontier** — the
   RLHF anti-tax knob, tested in the embodied setting.

---

## 4. Defensible thesis statement

> **"The 'RL retains capabilities' property reported for *multi-task* VLA RL does
> not transfer to *single-task* specialization: RL-improving one LIBERO-Long task
> induces *structured* forgetting on other already-mastered tasks — predictable
> from task/skill similarity and mirrored by action-distribution drift — and a
> KL-to-SFT anchor traces the gain↔retention trade-off."**

Contributions: (a) first per-task **retention audit** of single-task VLA RL;
(b) **similarity→interference** structure; (c) **action-drift** mechanism linked to
pushcut; (d) **KL** mitigation for the embodied alignment tax.

---

## 5. Minimal experiment plan (cost-aware)

Baseline (DONE, smoke): LIBERO-Long SFT `trajall` per-task =
90/100/80/80/90/100/80/**70**/90/100, avg 88.0% (n=10/task). Weakest = task_7
("put both the alphabet soup and the cream cheese box in the basket").

1. **Lock baseline:** full 50/task (500 eps) before/after eval harness.
2. **Specialize-one, eval-all** for ~3 chosen `T_i`:
   - weak (**task_7**), mid (task_2/3/6 @80%), strong (task_1 @100% — reward-hacking
     probe). = 3 RL runs, each re-eval all 10.
3. **Dynamics:** checkpoint through RL steps → is off-target decay monotonic or
   spike-then-recover? (turns 1 run into a time series).
4. **Structure:** build a task-similarity matrix (shared object nouns / verb / scene)
   vs measured `ΔT_j` → correlation.
5. **Mechanism:** action-token `KL(π_after‖π_before)` on fixed off-target states vs
   `ΔT_j`.
6. **Mitigation:** sweep `kl_coef ∈ {0, small, large}` on task_7 → Pareto (target
   gain vs mean/worst-case seen-task drop).

Phase-2 gate: confirm single-task GRPO fits 1–2 GPUs before scaling the grid
(their runs used 8× A800; single-task should be far lighter).

---

## 6. Venue framing

- **Workshop / short paper:** items 1–2 (existence + per-task retention audit) with
  the similarity structure is already a crisp, novel contribution.
- **Full paper:** add mechanism (5) + mitigation (6) for a complete
  "structured, explained, controllable" story.
- **Positioning:** continual/lifelong RL × VLA × RLHF alignment tax; the
  **verifiable simulator reward** removes the reward-model noise that confounds
  LLM-RLHF forgetting studies — a genuine methodological edge.

---

## 7. References (for the memo)
- SimpleVLA-RL — arXiv 2509.09674 (esp. §3.4 KL removed; §5.2 generalization; §6.1 pushcut)
- Liu et al. — "What can RL bring to VLA generalization?" arXiv 2505.19789 (NeurIPS 2025)
- RIPT-VLA — arXiv 2505.17016
- OpenVLA-OFT — arXiv 2502.19645 · LIBERO — Liu et al. 2023 (lifelong benchmark)
- Background: InstructGPT (Ouyang 2022) alignment tax; catastrophic forgetting (continual learning)

---

## 8. Mechanisms & Mitigations (how to *solve* forgetting, not just show it)

The thesis contribution is **mechanism → matched fix**, not just "RL forgets."

### 8.1 Diagnosed mechanism (to measure first)
Hypothesis: single-task GRPO forgets via **action-distribution drift on off-target
task states**. Measure `KL(π_after ‖ π_before)` over action-token logits on a fixed
set of *other* tasks' states; correlate with each task's success drop `ΔT_j`. Link
to SimpleVLA-RL's **"pushcut"** (RL rewriting grasp→push on T_i can break a T_j that
needs grasping). If drift predicts the drop, the mechanism is established.

### 8.2 The 5 historical anti-forgetting families → VLA/GRPO instantiation
| Family | Idea | Classic methods | Our instantiation |
|--------|------|-----------------|-------------------|
| 1. Regularization | penalize moving important weights | EWC, SI, MAS, L2-SP, **KL-to-ref** | restore GRPO KL term (β>0) |
| 2. Replay/rehearsal | mix old-task data in | ER, generative replay, **PPO-ptx** | mix off-target task trials into GRPO batch |
| 3. Param isolation | new skill = new params | LoRA/adapters, PackNet, Prog-Nets | **LoRA-only RL**; per-task adapters |
| 4. Gradient projection | project update off old-task subspace | **GEM/A-GEM, OGD, PCGrad, null-space** | **off-target-aware projection** (hero) |
| 5. Functional distillation | keep old outputs stable | LwF, policy distillation | distill π_SFT action logits on other tasks |

RLHF-specific tax fixes: **KL-to-SFT** (β; SimpleVLA removed it), **PPO-ptx**
(family 2), **weight interpolation / model soup** `θ=(1−α)θ_SFT+αθ_RL` (WiSE-FT,
reward-soups, Lin et al. "Mitigating the Alignment Tax of RLHF"), **reference reset**
(ProRL).

### 8.3 KL leash vs gradient projection (the core design choice)
- **KL-to-SFT = global, isotropic leash.** Penalizes *all* drift equally → cannot
  separate "improves T_i" from "breaks T_j". Small β → little retention; large β →
  retention but kills target gain + exploration (pushcut). **Moves *along* the
  gain↔retention Pareto frontier; does not expand it.** Necessary baseline, not hero.
- **Gradient projection = directional, task-targeted.** Estimate off-target gradient
  from a tiny replay set; project the GRPO update onto its orthogonal complement →
  keep the part that helps T_i and is neutral to others. **Can expand the frontier**
  and directly targets the §8.1 mechanism. Higher engineering cost; conservative when
  T_i and T_j genuinely share a needed direction (an informative failure).
- **Weight soup (α) = cheap dark horse.** Post-hoc, zero training change; if a 2-line
  interpolation matches projection, that itself is a finding.

### 8.4 Ladder of fixes (run in this order)
1. **KL sweep** (β ∈ {0, small, large}) — baseline; expect frontier-sliding only.
2. **Weight soup** (α sweep) — cheap post-hoc win.
3. **LoRA-RL** — constrains *where* drift can happen; cheaper compute; "where does
   forgetting live" probe.
4. **Off-target gradient projection** — hero; the only one that targets the mechanism
   and can push the frontier outward.

Punchline: *"KL/soup move along the gain–retention frontier; off-target gradient
projection pushes it outward, because forgetting is caused by a specific, removable
gradient direction (action drift), not by generic distance from SFT."*

⚠️ **Novelty to verify before committing to the hero:** has gradient-projection /
model-soup already been applied to **VLA RL** specifically? (20-min lit check:
A-GEM/OGD/PCGrad + VLA; model-soup + robot policy.)

---

## 9. Experimental Setup — how RL training actually works here

### 9.1 What "training data" is in GRPO RL (no demos, no labels)
The learning signal is generated on-the-fly:
- **Task def** (`.bddl`) = objects + goal + **success predicate** (the verifier).
- **Init states** (`init_files` scenarios) = starting arrangements to roll out from.
- **Simulator** = produces the binary reward (1 success / 0 fail, §3.2).
Per init state, sample `G=n_samples=8` rollouts at `temperature=1.6`; GRPO advantage =
reward vs the group mean for that state → nudge toward better rollouts. "Dataset" =
(init states) + (verifier); trajectories are self-generated.

### 9.2 Stock (multi-task) config — `run_openvla_oft_rl_libero.sh`
```
data.task_suite_name=libero_10        # all 10 tasks at once
data.num_trials_per_task=50           # init states/task for RL rollouts
data.n_samples=8                      # G: group size
data.filter_accuracy=True             # dynamic sampling: drop all-succeed/all-fail states
data.accuracy_lower_bound=0.1 / upper_bound=0.9
actor_rollout_ref.actor.optim.lr=5e-6
actor_rollout_ref.rollout.temperature=1.6
algorithm.adv_estimator=grpo
algorithm.kl_ctrl.kl_coef=0.00        # KL to SFT OFF (the knob)
trainer.val_only=False                # training mode
trainer.test_freq=4                   # eval all tasks every 4 steps
```

### 9.3 Single-task specialization = the core fork change (TO IMPLEMENT)
The stock pipeline trains on the whole suite. To specialize on ONE task `T_i`:
- **Where:** task/trial selection enters `rob_rollout.py` via
  `prompts.batch['task_id']`, `['trial_id']`, `non_tensor_batch['task_suite_name']`.
  These are produced by a **data-scenario builder** (a parquet/dataset layer NOT in
  our captured file — inspect on the fork). Filter that builder to `task_id == i`
  (or point at a 1-task subset). Keep eval on all 10 (test_freq already evals suite).
- **Disjoint splits:** RL rollouts should use an init-state split **disjoint** from
  eval, so a post-RL drop isn't eval-set overfitting (ties to the init-overlap task).
- Keep everything else identical to the baseline for apples-to-apples ΔT_j.

### 9.4 Before/after flow
```
[SFT trajall] --eval_smoke.sh(all 10)--> BEFORE table (88%, task_7=70%)  [DONE]
      |  RL specialize on ONE task (task_7 only), GRPO, kl_coef swept:
      |  run_openvla_oft_rl_libero.sh (val_only=False, rollouts=task_7 only)
      |        -> [RL-specialized ckpt]
      +--eval_smoke.sh(all 10) on RL ckpt--> AFTER table
                 ΔT_j = AFTER - BEFORE  = interference / forgetting vector
```

### 9.5 Scripts involved
| Script | Role |
|--------|------|
| `examples/run_openvla_oft_rl_libero.sh` | GRPO **training** launcher (val_only=False) |
| `examples/eval_smoke.sh` (our copy) | **before/after eval** (val_only=True) |
| `examples/overwrite_vla_ckpt_utils.sh` | copies OFT model utils into ckpt dir at launch |
| `verl/workers/rollout/rob_rollout.py` | rollout worker: sample traj, step sim, reward |
| `verl/utils/libero_utils.py` | builds LIBERO env, dummy action, success check |
| `align.json` | Ray runtime env (MUJOCO_GL, wandb) |

### 9.6 Open items before running
1. Single-task scenario selection (the fork change, §9.3 / §10).
2. KL sweep values.
3. Verify RL vs eval init-state disjointness + train/eval init overlap (per §5/§6 memo).
4. Confirm single-task GRPO fits 1–2 GPUs before scaling to several T_i.

---

## 10. Fork-change map (verified against source) — `abishek21/SimpleVLA-RL`

Fork created: **github.com/abishek21/SimpleVLA-RL** (from PRIME-RL/SimpleVLA-RL).

### 10.1 Where scenarios are built (the single-task filter point)
`verl/utils/dataset/rob_dataset.py` → `LIBERO_Dataset._read_files_and_tokenize()`
(lines ~84–99): loops over **all** tasks and trials:
```python
for task_id in range(num_tasks_in_suite):        # <-- iterate ALL 10 tasks
    if self.train_val == "train":
        trials_range = list(range(0, int(self.num_trials_per_task)))
    elif self.train_val == "valid":
        trials_range = list(range(0, int(self.num_trials_per_task)))   # SAME range!
    for i in trials_range:
        data = {"task_suite_name":..., "task_id":task_id, "trial_id":i, ...}
```
Instantiated in `verl/trainer/ppo/ray_trainer.py` ~L297–302 (`train_dataset` /
`val_dataset` = `LIBERO_Dataset(..., train_val="train"|"valid")`).

**Single-task change (minimal):** add an optional task filter and read it from config.
```python
# LIBERO_Dataset.__init__: add  rl_task_ids=None  and store it
# in _read_files_and_tokenize:
task_ids = self.rl_task_ids if self.rl_task_ids is not None else range(num_tasks_in_suite)
for task_id in task_ids:
    ...
```
Wire `data.rl_task_ids=[7]` through config; pass to the **train** dataset only, keep
**val** dataset over all 10 (so eval stays full-suite for the interference vector).

### 10.2 🔴 CRITICAL METHOD FINDING — train and eval init states are IDENTICAL
In `LIBERO_Dataset`, **`train` and `valid` use the same `trials_range =
range(0, num_trials_per_task)`** (lines ~86 vs ~88 are identical). So in this
pipeline **RL rolls out on the same initial states it is evaluated on** — there is
**no held-out init-state split** at the dataset level. Implications:
- SimpleVLA-RL's "50 held-out test scenarios" (§4.1) is **not** enforced by this
  code path for LIBERO; train and eval init states overlap fully.
- For a clean forgetting study we **must create a disjoint split ourselves**, e.g.:
  ```python
  # train:  trials_range = list(range(0, num_trials_per_task))            # e.g. 0..39
  # valid:  trials_range = list(range(TRAIN_N, TRAIN_N + EVAL_N))         # e.g. 40..49
  ```
  so a post-RL drop reflects **skill degradation**, not overfitting to the RL init
  states. (Contrast: RoboTwin path DOES use separate train/eval seed files —
  `envs/robotwin2/seeds/*` — so the LIBERO overlap is a LIBERO-specific gap.)
- Also verify whether these eval init states coincide with the SFT **demo** init
  states (separate question; needs the `.hdf5` check from §6).

### 10.3 Other edits already needed on the fork (currently Dockerfile seds)
- fork→spawn + traceback in `rob_rollout.py` (make it a real commit; drop the
  Dockerfile python-patch).
- Per-task success logging is already implicit via rollout MP4 filenames +
  `results_manifest`; consider adding an explicit per-task metric to the val loop
  (`ray_trainer.py::_validate`) so numbers appear in the log, not just filenames.

### 10.4 Suggested repo integration
Add the fork as a **git submodule** under `05_forgetting_study/SimpleVLA-RL`
(pinned commit), make the §10.1–10.3 edits as commits on the fork, and change the
`Dockerfile` to `git clone` the fork@pinned-commit instead of upstream + seds.

---

## 11. LIBERO's own metrics (FWT / NBT / AUC) — use this vocabulary

From the LIBERO paper (arXiv 2306.03310, §5.1). Let `c_{i,j}` = success on task `j`
after the agent has learned up to task `i`; `c_{k,k}` = best success on task `k`
right after learning it. Over `K` tasks:

- **FWT (Forward Transfer)** — how *fast* a new task is learned (higher = better):
  `FWT_k = (1/11) Σ_{e∈{0,5,…,50}} c_{k,k,e}` (avg success across eval epochs while
  learning task k).
- **NBT (Negative Backward Transfer)** — **the forgetting metric** (lower = better):
  `NBT_k = 1/(K−k) Σ_{τ=k+1..K} (c_{k,k} − c_{τ,k})`
  i.e. average drop in task k's success *after* learning each later task τ.
- **AUC** — area under the success-rate curve; combines FWT + retention.

**Mapping to Study 05:** our **interference vector** `ΔT_j = SR_after(T_i) − SR_before`
is exactly a **backward-transfer** measurement. Adopt LIBERO's language:
- Report **per-task backward transfer** `BT_{i→j} = SR_after(j) − SR_before(j)` for
  j ≠ i (negative = forgetting = their NBT with sign flipped).
- Our design is a **one-step** version of NBT (specialize ONE task, not a full
  sequence), which is *cleaner* to attribute than sequential NBT. Frame it as
  "single-task RL backward transfer."
- Target-task gain relates to **FWT** (learning/improving the trained task).

### 11.1 LIBERO *does* report a multitask (joint) baseline — but small-policy
- LIBERO defines **MTL (multitask learning)** as the **upper bound** and **SeqL
  (sequential)** as the **lower bound** for lifelong learning (§4.3, App. D).
- **App. E.3 "Multitask Success Rate"** gives a **single joint model's per-task
  success** on LIBERO-90/100 — for the **small ResNet-T/ViT-T policies**, not a VLA.
  Many per-task entries are **0.0–0.3**, i.e. even the joint (MTL) model is weak on
  long-horizon/entangled tasks → supports "a single all-expert is genuinely hard."
- ⚠️ So a **joint LIBERO model IS reported** (small policies), but **no VLA-scale
  joint LIBERO-all expert** exists in the literature we've seen. Our "why no
  all-expert" narrative should say: *joint (MTL) is the acknowledged upper-bound but
  is hard and under-delivers even for small policies; VLA papers sidestep it with
  per-suite specialists.*
- LIBERO's key qualitative findings we can cite: **SeqL beats LL algorithms on FWT**
  (anti-forgetting methods hurt forward transfer — the tax!), **EWC (regularization)
  often underperforms SeqL**, **ER (replay) is robust**, **PackNet (isolation) best
  at preventing forgetting but capacity-limited**. These *directly* motivate our
  ladder (KL≈EWC-like reg; replay; LoRA≈isolation; projection).

### 11.2 Bonus: task_7 appears in LIBERO's own forgetting analysis
LIBERO's attention-forgetting figures (Fig. 22/25) use the LIBERO-Long task
**"put both the alphabet soup and the tomato sauce in the basket"** — a sibling of
our weakest **task_7** ("…alphabet soup and the **cream cheese box**…"). The
two-object basket tasks are exactly where they visualize forgetting → good anchor
for our case study.

---

## 12. Part 2 — MoE-LoRA LIBERO generalist (design, verified against code)

**Goal:** compose per-suite specialists into ONE VLA good across LIBERO suites
(scope = 40 tasks: spatial+object+goal+long; we have `trajall` specialists for all
four). Two-part thesis: **Part 1 measures** single-task RL forgetting; **Part 2
overcomes** it by building a generalist that sidesteps the multi-task tax.

### 12.1 OpenVLA-OFT architecture (SimpleVLA variant) — from `modeling_prismatic.py`
```
image ─► vision_backbone (SigLIP + DINOv2 fused, timm)      # perception
      ─► projector (MLP fc1/fc2/fc3: vision_dim → llm_dim)   # perception→LLM bridge
      ─► language_model = LLaMA2-7B (AutoModelForCausalLM)   # the policy "brain"
      ─► lm_head (Linear → vocab_size)                       # ACTION HEAD
      ─► action **tokens** (discrete bins) → detokenized to continuous
      ─► 7-DoF × 8-step action chunk
```
- **No separate action MLP** in SimpleVLA's variant: actions are **special tokens via
  the LLM `lm_head`** (hence `vocab_size`, `bin_centers`; paper = "LLaMA2 output head +
  cross-entropy"). The official OFT *continuous* MLP+L1 head is NOT used here.
- **SFT = full-parameter** (SimpleVLA §4.1, 8×A800): vision + projector + LLM + lm_head
  all trained. LoRA exists but is **off by default** (`lora_rank: 0`).

### 12.2 Which matrices LoRA touches (LLaMA2 decoder)
LLaMA2-7B = 32 decoder layers; each has **7 linear matrices** (no bias):
- attention: `q_proj, k_proj, v_proj, o_proj` (4096→4096)
- MLP (SwiGLU): `gate_proj, up_proj (4096→11008)`, `down_proj (11008→4096)`
- RMSNorm has no LoRA-able linear; `embed_tokens`/`lm_head` are separate.

SimpleVLA config default `target_modules: all-linear` (ppo_trainer.yaml) → PEFT wraps
**every** `nn.Linear`: the 7×32=224 LLaMA matrices **+ vision backbone + projector**.
LoRA plumbing already exists: `verl/workers/fsdp_workers.py` L234–248
(`peft`, `get_peft_model`, `LoraConfig`, keys `lora_rank/lora_alpha/target_modules`).

**For our design → restrict experts to the LLM control layers, share perception:**
```yaml
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```
Include the MLP projections (gate/up/down): much task-specific procedural knowledge
lives there — exactly what specializes and interferes.

### 12.3 MoE-LoRA placement
| Module | Placement | Why |
|--------|-----------|-----|
| vision_backbone | **shared** (freeze / 1 common LoRA) | perception common; forces robust features (attacks LIBERO's "spurious attention" overfitting) |
| projector | shared | common perception→LLM bridge |
| LLaMA attn+MLP linears | **per-suite LoRA experts + router** ← core | control/interference lives here; shapes action tokens |
| lm_head | **shared** | action-token vocab common |

### 12.4 MoE-LoRA vs multi-adapter LoRA (the key distinction)
- **Multi-adapter LoRA:** load N adapters, **activate exactly one** per forward (hard
  external switch by known suite id). No blending. = special case of MoE-LoRA with a
  fixed one-hot router.
- **MoE-LoRA:** a **router** produces input-dependent weights over experts and
  **blends** them (soft or top-k). Task id NOT required; can compose skills.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class LoRAExpert(nn.Module):
    def __init__(self, d_in, d_out, r=16, alpha=32):
        super().__init__()
        self.A = nn.Linear(d_in, r, bias=False); self.B = nn.Linear(r, d_out, bias=False)
        nn.init.normal_(self.A.weight, std=1/r); nn.init.zeros_(self.B.weight)
        self.scale = alpha / r
    def forward(self, x): return self.B(self.A(x)) * self.scale

class MoELoRALinear(nn.Module):
    """Frozen base Linear W + N LoRA experts + router that blends them."""
    def __init__(self, base: nn.Linear, num_experts=4, r=16, alpha=32, top_k=None):
        super().__init__()
        self.base = base
        for p in self.base.parameters(): p.requires_grad_(False)   # base frozen
        d_in, d_out = base.in_features, base.out_features
        self.experts = nn.ModuleList([LoRAExpert(d_in, d_out, r, alpha) for _ in range(num_experts)])
        self.router = nn.Linear(d_in, num_experts, bias=False)     # the gate
        self.top_k = top_k
    def forward(self, x, route_feat=None):
        y = self.base(x)
        gates = F.softmax(self.router(x if route_feat is None else route_feat), dim=-1)
        if self.top_k:
            val, idx = gates.topk(self.top_k, dim=-1)
            gates = torch.zeros_like(gates).scatter_(-1, idx, val)
            gates = gates / gates.sum(-1, keepdim=True).clamp_min(1e-9)
        outs = torch.stack([e(x) for e in self.experts], dim=-1)   # (...,d_out,N)
        delta = (outs * gates.unsqueeze(-2)).sum(-1)
        return y + delta

# Multi-adapter == MoE-LoRA with a fixed one-hot router:
# gates = F.one_hot(suite_id, num_experts).float()
```

### 12.5 Router granularity & plan
- **Per-sequence routing** (one gate from the **instruction embedding**) — cheap,
  natural for VLA; start here.
- **Per-token routing** — classic MoE, more expressive/costly; extension.
- **Oracle router** (one-hot by known suite) = upper-bound baseline == multi-adapter.
- **Learned router** (from instruction/hidden state) = the real contribution
  (handles unknown/compositional tasks; can blend skills).

### 12.6 Feasibility ladder (Part 2)
1. **Model merging** of the 4 `trajall` specialists (task arithmetic / TIES / DARE) —
   ~free; probes interference in weight space. Do first.
2. **MoE-LoRA (oracle router)** — upper bound; reuses existing LoRA plumbing.
3. **MoE-LoRA (learned router)** — the contribution.
4. **Frozen strong encoder** (share perception) — attacks spurious-attention overfitting.
5. **Balanced multi-task SFT / replay** — necessary MTL baselines.

⚠️ **Novelty to verify:** has MoE-LoRA / model-merging been applied to **VLA across
LIBERO suites**? (Check before committing.) Also verify all 4 `trajall` specialists
share identical base/config/tokenizer (required for merging & shared base).

---

## 14. Phase A — single-suite GRPO + LoRA expert (LAUNCH RECIPE)

**Goal:** train ONE LoRA expert with GRPO on ONE suite → produces one
`lora_adapter/`. Proves the LoRA+GRPO path end-to-end; the atom for the 4-expert
MoE. Config keys verified against `verl/trainer/config/ppo_trainer.yaml`.

### 14.1 Pick the first expert
Start with **libero_10 (Long)** — we already have the SFT baseline (88%) and the
checkpoint, so we can sanity-check improvement. Warm-start:
`SFT_MODEL_PATH=/workspace/ckpt_libero10_trajall`.

### 14.2 Enable LoRA (config or CLI overrides)
Defaults today: `lora_rank: 0` (off), `target_modules: all-linear`. Change to:
```
actor_rollout_ref.model.lora_rank=32
actor_rollout_ref.model.lora_alpha=32
# restrict experts to LLM control layers (share perception); PEFT matches name suffix
actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]
```
(If passing a list via Hydra CLI is fiddly, edit `ppo_trainer.yaml` directly.)
LoRA save is automatic → `CKPT_PATH/.../lora_adapter/` + a merged full model
(`fsdp_workers.py` L581–620).

### 14.3 Single-suite / single-task data filter (fork edit, §10.1)
For a *suite* expert on libero_10, the stock loop over all 10 tasks is what we want
(the "suite" = its 10 tasks). For a *single-task* expert (forgetting study), add the
`rl_task_ids` filter from §10.1. Phase A can start at **suite-level** (no code change)
to derisk, then add the task filter.

### 14.4 Disjoint train/eval init states (fork edit, §10.2) — IMPORTANT
Currently train==valid trials (`range(0, num_trials_per_task)`). For a clean result:
```python
# rob_dataset.py LIBERO_Dataset._read_files_and_tokenize
if self.train_val == "train":
    trials_range = list(range(0, 40))      # 40 for RL rollouts
elif self.train_val == "valid":
    trials_range = list(range(40, 50))     # 10 held-out for eval
```
(Phase A can skip this to first confirm training runs; add before reporting numbers.)

### 14.5 GRPO / RL knobs (match SimpleVLA recipe)
```
algorithm.adv_estimator=grpo
algorithm.kl_ctrl.kl_coef=0.0            # SimpleVLA default (KL off)
actor_rollout_ref.rollout.temperature=1.6
data.n_samples=8                         # GRPO group size G
data.filter_accuracy=True                # dynamic sampling
data.accuracy_lower_bound=0.1
data.accuracy_upper_bound=0.9
actor_rollout_ref.actor.optim.lr=5e-6    # small; LoRA can tolerate a bit higher
trainer.val_only=False
trainer.test_freq=<eval every N steps>
```
NOTE: with LoRA, `param_offload`/`optimizer_offload` needs are lower; can likely fit
**1–2 A40/A6000** (vs 8×A800 for full-param). Verify on a short run first.

### 14.6 Sanity checks (in order)
1. Launch prints `Applying LoRA to actor module` + `print_trainable_parameters()`
   showing only LoRA params trainable (~millions, not 7B). ✅ LoRA active.
2. First eval (step 0) ≈ SFT baseline (88% on libero_10). ✅ warm-start OK.
3. After a few GRPO steps, train-suite success ↑ (RL improving). ✅ RL path works.
4. `CKPT_PATH/.../lora_adapter/` written. ✅ expert persisted.

### 14.7 KL-sweep variant (for the forgetting story, later)
Re-run with `algorithm.kl_ctrl.kl_coef ∈ {0, small, large}` to trace gain↔retention.
`kl_coef` re-enables the reference-KL term in `core_algos.py` (the knob SimpleVLA
zeroed). This is also the cheapest "mitigation" baseline.

### 14.8 After Phase A works
- Repeat §14 for spatial / object / goal (each warm-started from its own `trajall`)
  → 4 `lora_adapter/`s = the 4 experts.
- Then Phase B: build `MoELoRALinear` (experts+router) and unit-test forward/backward.
- Then Phase C: GRPO across suites with the MoE module + load-balancing loss.
