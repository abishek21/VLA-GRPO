# VLA-GRPO: From GRPO Fundamentals to VLA Post-Training

A structured learning + research repo. Goal: build strong fundamentals in
RL post-training (GRPO in particular), then reproduce **OpenVLA-OFT + GRPO**
on a few manipulation tasks, en route to a paper.

## Roadmap

| Stage | Folder | Goal | Status |
|-------|--------|------|--------|
| 0 | `00_grpo_fundamentals/` | Understand the math: policy gradients -> PPO -> GRPO | start here |
| 1 | `01_tiny_llm_grpo/` | Implement GRPO from scratch on a toy task, then a tiny LLM | next |
| 2 | `02_openvla_oft_grpo/` | Reproduce OpenVLA-OFT, add GRPO on manipulation tasks | later |
| - | `notes/` | Paper reading notes, experiment log, ideas | ongoing |

## Why this order

You cannot debug RL-on-VLA if you don't first *feel* how GRPO behaves. The
same failure modes (reward hacking, advantage collapse, KL blow-up) show up
in a 5-minute toy run and in a 5-day GPU run. Master them cheaply first.

## The core idea of GRPO in one sentence

> Sample a **group** of G answers for the same prompt, score each with a
> reward, and push the policy toward answers that are **better than the
> group average** -- using the group itself as the baseline, so no value
> network is needed.

## Stages

### Stage 0 - Math (today)
- Derive the policy gradient and why we need a baseline.
- PPO clipped surrogate + why it stabilizes updates.
- Exactly what GRPO changes vs PPO (group baseline, no critic).
- Read `notes/grpo_math.md`, reproduce derivations by hand.

### Stage 1 - Tiny implementation (today / this week)
- `toy_grpo.py`: GRPO on a "count the 1s" verifiable task (no LLM, seconds).
- `tiny_llm_grpo.py`: GRPO on a small HF model on an arithmetic task.

The one equation to keep in your head (group-relative advantage):

$$
A_i = \frac{r_i - \operatorname{mean}(r_1, \dots, r_G)}{\operatorname{std}(r_1, \dots, r_G) + \varepsilon}
$$

### Stage 2 - VLA (the research)
- Get OpenVLA-OFT running + evaluated (SFT baseline) on LIBERO.
- Define a verifiable/environment reward (task success, shaped distance).
- Apply GRPO over sampled action chunks. Measure success-rate deltas.
