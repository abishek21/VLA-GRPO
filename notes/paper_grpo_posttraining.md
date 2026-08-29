# GRPO Post-Training with Interaction Rewards — Method & Experiments (paper draft)

*Draft of Section 4. Pairs with Algorithm 2 in `notes/algorithms.md`, the GRPO
code in `01_tiny_llm_grpo/toy_grpo.py`, and the Gaussian-head prototype in
`02_openvla_oft_grpo/proto_gaussian_head.py`.*

---

## 4. Reinforcement Fine-Tuning of a VLA with the Learned Reward

### 4.1 Overview

Given the interaction-event reward model $R_\phi$ (Section 3), we fine-tune a
pretrained vision–language–action policy with Group Relative Policy Optimization
(GRPO) [Shao et al., 2024], using the *dense, per-timestep* interaction events as
a process reward. The central hypothesis is that rewarding the correct sequence
of physical interaction events — and, in particular, penalizing failures and
rewarding recoveries — yields policies that both succeed more often and recover
from mistakes more reliably than policies trained with sparse task-success
rewards alone. We evaluate entirely in simulation (LIBERO [Liu et al., 2023]),
which provides a ground-truth success/failure oracle against which the learned
reward can be measured.

### 4.2 Policy and the action-likelihood problem

Our base policy is **SmolVLA** [Shukor et al., 2025], a compact VLA that maps an
observation $o$ (multi-view images + language instruction) to a chunk of $H$
future continuous actions $A=a_{1:H}$. SmolVLA generates actions via
**flow matching**, which does not admit a closed-form action likelihood. GRPO,
however, requires the probability ratio
$\rho = \pi_\theta(A\mid o)/\pi_{\theta_\text{old}}(A\mid o)$, and hence a
tractable $\log\pi_\theta(A\mid o)$.

We resolve this by attaching a lightweight **stochastic action head** to the
frozen SmolVLA backbone: a diagonal Gaussian (optionally tanh-squashed to
respect action bounds) over the action chunk, whose log-probability is
closed-form. To preserve the competence of the pretrained policy, we
**initialize the head's mean to reproduce the flow policy's output**
($\mu(o)\approx \text{flow-action}(o)$); the KL reference $\pi_\text{ref}$ is the
initialized policy, so early updates stay close to the SFT behavior. Only the
action head (and optionally a low-rank adapter on the backbone) is updated during
RL. We retain the flow head for deterministic evaluation.

We keep the action chunk in its native $[H,d]$ form rather than flattening, and
compute the log-ratio per control step, aggregating over the horizon; this avoids
the ratio blow-up that arises when summing $H\times d$ independent Gaussian
log-ratios and enables per-step diagnostics.

### 4.3 Rollouts and the group baseline

For each update we draw a start state $s_0$ and sample a **group of $G$ complete
episodes** $\{\tau^i\}_{i=1}^G$ from the current policy. Unlike the token-level
groups used for language GRPO, robot rollouts *diverge* after the first action
chunk: each $\tau^i$ is a full multi-decision trajectory whose observations are
distinct. The group provides the critic-free baseline central to GRPO.

### 4.4 From events to reward

For each rollout we render its frames and read the simulated gripper pose, and
apply the reward model to obtain per-timestep events
$\hat e^i_t = R_\phi(\text{frames}(\tau^i), \text{grip}(\tau^i))$. We convert
these into a scalar process reward per step,
$$
r^i_t = w_g\,\Delta\text{grasp}_t + w_c\,\Delta\text{contact}_t
       - w_f\,\text{failure}_t + w_r\,\text{recovery}_t,
$$
where $\Delta$ denotes the *onset* of a state event (rewarding the transition,
not its persistence, to prevent trivial farming such as holding contact
indefinitely). The recovery term supplies the dense signal that is unavailable to
both imitation learning and sparse-reward RL. The episode return combines the
process reward with the simulator's ground-truth success flag,
$$
R^i = \sum_t r^i_t + w_s\,\text{success}(\tau^i).
$$

Group-relative advantages are
$A^i = (R^i - \mathrm{mean}(R))/(\mathrm{std}(R)+\epsilon)$, with degenerate
(zero-variance) groups — all-success or all-failure — contributing no gradient,
which is the dominant pathology of sparse binary robot rewards.

### 4.5 Objective

We optimize the standard clipped surrogate with an explicit KL leash to the
reference policy:
$$
\mathcal{L}(\theta) = \mathbb{E}\Big[\min\big(\rho A,\ \mathrm{clip}(\rho,1{-}\varepsilon,1{+}\varepsilon)A\big)\Big]
- \beta\,\widehat{\mathrm{KL}}(\pi_\theta\,\Vert\,\pi_\text{ref}),
$$
using a per-token (per-control-step) unbiased KL estimator. The full procedure is
given in Algorithm 2.

---

**Algorithm 2 — GRPO VLA Post-Training with the Learned Reward**

```text
Require: SFT policy pi_ref; reward model R_phi; sim env; group size G; clip eps; KL beta
Ensure:  post-trained policy pi_theta
 1: pi_theta <- pi_ref + Gaussian head; init mu ~ flow action   # keep SFT skill
 2: for iteration = 1, 2, ... do
 3:     pi_theta_old <- pi_theta
 4:     sample start state s0
 5:     for i = 1..G do                       # G FULL rollouts (obs diverge)
 6:         roll out tau^i with A^i_j ~ pi_theta_old(.|o^i_j)
 7:         e_hat^i_t <- R_phi(frames(tau^i), gripper_pose(tau^i))
 8:         r^i_t <- w_g Δgrasp + w_c Δcontact - w_f failure + w_r recovery
 9:         R^i   <- sum_t r^i_t + w_s success(tau^i)
10:     end for
11:     A^i <- (R^i - mean(R)) / (std(R) + eps)      # 0 if zero-variance group
12:     for ppo_epoch = 1..E do
13:         rho^i_j <- pi_theta(A^i_j|o^i_j) / pi_theta_old(A^i_j|o^i_j)
14:         L_clip  <- mean min(rho A^i, clip(rho,1±eps) A^i)
15:         theta   <- theta + eta * grad(L_clip - beta * per_token_KL(pi_theta||pi_ref))
16:     end for
17: end for
```

---

## 5. Experiments

### 5.1 Setup

We evaluate on LIBERO manipulation suites, reporting task success rate over a
fixed set of held-out initial states and seeds. Baselines: (i) the SFT policy
(no RL); (ii) **sparse-RL** — GRPO with only the simulator success flag; and
(iii) **ours** — GRPO with the interaction-event reward (success + process
terms). Unless noted, all methods start from the same SFT checkpoint and share
GRPO hyperparameters.

### 5.2 Research questions and result tables

**Q1 — Does the interaction reward improve success?** *(Table A)*
Success rate per suite for SFT vs sparse-RL vs ours.

| Method | Spatial | Object | Goal | Long | Avg |
|--------|--------:|-------:|-----:|-----:|----:|
| SFT (no RL)                 | — | — | — | — | — |
| Sparse-RL (success only)    | — | — | — | — | — |
| **Ours (interaction reward)** | — | — | — | — | — |

**Q2 — Does it improve failure recovery?** *(Table B, our headline claim)*
After inducing a perturbation/fumble mid-episode, we measure the rate of
successful recovery. The dense recovery signal should help here where sparse
rewards cannot.

| Method | Recovery rate | Δ vs SFT |
|--------|--------------:|--------:|
| SFT                       | — | — |
| Sparse-RL                 | — | — |
| **Ours**                  | — | — |
| Ours − recovery term (ablation) | — | — |

**Q3 — How well does the human-derived reward transfer?** *(Table C)*
Correlation between the reward model's predicted events and the simulator's
ground-truth interaction/success signals on robot rollouts, quantifying the
human$\rightarrow$robot embodiment gap.

| Signal | Precision | Recall | F1 | Corr. w/ oracle |
|--------|----------:|-------:|---:|----------------:|
| grasp     | — | — | — | — |
| contact   | — | — | — | — |
| failure   | — | — | — | — |
| recovery  | — | — | — | — |

**Q4 — Does the policy hack the learned reward?** *(Table D)*
Frequency and taxonomy of episodes where the process reward is high but the
oracle reports failure (reward-model exploitation), and the effect of mitigations
(KL weight, $\Delta$-transition vs persistence rewards, term balancing).

| Configuration | High-reward-but-failed (%) | Success | Notes |
|---------------|---------------------------:|--------:|-------|
| Persistence reward (no Δ)    | — | — | expect farming |
| Δ-transition reward (ours)   | — | — | — |
| + higher KL leash            | — | — | — |

### 5.3 Ablations

- **Advantage type:** outcome (episode-level) vs process (per-chunk) advantages.
- **Reward terms:** removing recovery / failure / contact terms.
- **Encoder:** R3M (manipulation-pretrained CNN) vs DINOv2 (general ViT).
- **Action head:** Gaussian head vs flow-through surrogate.

---

### Notes for later (not paper prose)
- Fill all tables from runs; Table B is the priority result.
- Confirm SmolVLA action-chunk dims / control rate for the reward-alignment
  resampling (human 10 Hz vs robot control rate).
- Define the "induced fumble" protocol precisely for Table B (perturbation type,
  timing).
- Cite: GRPO (DeepSeekMath), SmolVLA, LIBERO, R3M, DINOv2, PPO.
