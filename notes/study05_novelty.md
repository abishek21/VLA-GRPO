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
