# GRPO Math: Policy Gradients → PPO → GRPO

Read top to bottom. Each section builds on the previous. Reproduce every
boxed equation by hand once; that is how it sticks.

> **Rendering note.** This file uses LaTeX math. In VS Code press
> `Cmd+Shift+V` to open the Markdown preview — inline math `$...$` and
> display math `$$...$$` render via the built-in KaTeX support.

---

## 0. Setup and notation

- Policy $\pi_\theta(a \mid s)$: probability of action $a$ in state $s$, with
  parameters $\theta$.
- For LLMs a "trajectory" is a token sequence $o = (o_1, \dots, o_T)$ given a
  prompt $q$. The policy factorizes autoregressively:

$$
\pi_\theta(o \mid q) = \prod_{t=1}^{T} \pi_\theta\!\left(o_t \mid q, o_{<t}\right)
$$

- Reward $r(q, o)$: a scalar score for the whole output (e.g. $1$ if the final
  answer is correct, else $0$). A **verifiable reward** is one we can check with
  a program (math answer, unit test, task success) — no reward model needed.

**Objective:** maximize expected reward

$$
J(\theta) = \mathbb{E}_{o \sim \pi_\theta(\cdot \mid q)}\big[\, r(q, o) \,\big]
$$

---

## 1. The policy gradient (REINFORCE)

We want $\nabla_\theta J$. The trick is the **log-derivative identity**:

$$
\nabla_\theta \pi = \pi \, \nabla_\theta \log \pi
$$

Applying it:

$$
\boxed{\;\nabla_\theta J = \mathbb{E}_{o \sim \pi_\theta}\big[\, r(q,o)\, \nabla_\theta \log \pi_\theta(o \mid q) \,\big]\;} \tag{1}
$$

Intuition: increase the log-probability of sequences that got high reward,
decrease it for low reward. That's it. Equation $(1)$ is the entire foundation.

Monte-Carlo estimate with samples $o^{(1)}, \dots, o^{(G)}$:

$$
\nabla_\theta J \approx \frac{1}{G} \sum_{i=1}^{G} r(q, o^i)\, \nabla_\theta \log \pi_\theta(o^i \mid q)
$$

**Problem:** the variance is huge. If all rewards are positive (say
$r \in \{0, 1\}$), every gradient pushes *up*, and only the *relative* magnitude
separates good from bad. Slow and noisy.

---

## 2. Baselines: subtract something to reduce variance

Key fact: for any baseline $b$ that does **not** depend on the sampled action,

$$
\mathbb{E}\big[\, b \, \nabla_\theta \log \pi_\theta(o \mid q) \,\big]
= b \, \nabla_\theta\, \mathbb{E}[1] = 0
$$

**Proof:**

$$
\mathbb{E}\big[\nabla_\theta \log \pi\big]
= \sum_o \pi \, \nabla_\theta \log \pi
= \sum_o \nabla_\theta \pi
= \nabla_\theta \sum_o \pi
= \nabla_\theta 1 = 0.
$$

So we can subtract $b$ for free (unbiased) while reducing variance:

$$
\nabla_\theta J = \mathbb{E}\big[\, (r(q,o) - b)\, \nabla_\theta \log \pi_\theta(o \mid q) \,\big] \tag{2}
$$

Now $(r - b)$ is the **advantage**: how much better than "typical" this sample
was. A good baseline has $b \approx \mathbb{E}[r]$. Two ways to get $b$:

- **(a)** Learn a value function $V_\phi(s) \approx \mathbb{E}[r]$ → this is
  PPO's critic.
- **(b)** Use the empirical mean over a group of samples → this is GRPO.

GRPO's whole selling point: option **(b)** needs **no second network**.

---

## 3. PPO (what GRPO simplifies away)

PPO uses an advantage $A_t$ (often via GAE from a learned critic $V_\phi$) and
optimizes a **clipped surrogate** to allow multiple gradient steps on the same
batch without the policy moving too far.

Let the probability ratio be

$$
\rho_t(\theta) = \frac{\pi_\theta(o_t \mid q, o_{<t})}{\pi_{\theta_\text{old}}(o_t \mid q, o_{<t})}
$$

PPO objective (maximize):

$$
L_\text{PPO} = \mathbb{E}\Big[\, \min\big(\rho_t A_t,\; \operatorname{clip}(\rho_t,\, 1-\epsilon,\, 1+\epsilon)\, A_t\big) \,\Big] \tag{3}
$$

- If $A_t > 0$ (good action): we want to raise its probability, but the clip
  stops us once $\rho_t > 1+\epsilon$, so we don't over-commit on one batch.
- If $A_t < 0$ (bad action): the clip at $1-\epsilon$ limits how hard we push
  down.
- The $\min$ makes it a pessimistic (lower) bound → conservative, stable.

PPO also adds a KL penalty / entropy term. Cost: you must train and store a
value network $V_\phi$ of roughly the same size as the policy. Expensive for
LLMs / VLAs.

---

## 4. GRPO: group-relative advantage, no critic

GRPO (from DeepSeekMath / DeepSeek-R1) replaces the learned baseline with the
**mean reward over a group** of $G$ samples for the *same* prompt $q$.

### 4.1 Sampling

For prompt $q$, sample a group of $G$ outputs from the old policy:

$$
o^1, \dots, o^G \;\sim\; \pi_{\theta_\text{old}}(\cdot \mid q)
$$

Score each: $r_i = r(q, o^i)$.

### 4.2 Group-relative advantage (the key equation)

$$
\boxed{\;A_i = \frac{r_i - \operatorname{mean}(r_1, \dots, r_G)}{\operatorname{std}(r_1, \dots, r_G) + \varepsilon}\;} \tag{4}
$$

- Subtracting the mean = the baseline from Section 2 (variance reduction).
- Dividing by the std = normalization so the update scale is stable across
  prompts. Easy prompts where all $r_i$ are similar contribute little; hard
  prompts with spread contribute more informative signal.
- Every token in output $i$ shares the same sequence-level advantage $A_i$
  (outcome supervision). A process-supervised variant assigns per-step $A$.

Note: if all $G$ samples get the same reward, $\operatorname{std} \to 0$ and
$A_i \to 0$ — that prompt gives no learning signal. This is **advantage
collapse**; watch for it (prompts too easy or too hard, or temperature too low).

### 4.3 GRPO objective

Same clipped surrogate as PPO, but with $A_i$ from $(4)$ and an explicit KL
term to a reference policy $\pi_\text{ref}$ (usually the SFT model):

$$
L_\text{GRPO}(\theta) =
\mathbb{E}_{q,\, \{o^i\}} \left[
\frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o^i|} \sum_{t=1}^{|o^i|}
\min\!\big(\rho_{i,t} A_i,\; \operatorname{clip}(\rho_{i,t},\, 1-\epsilon,\, 1+\epsilon)\, A_i\big)
\;-\; \beta\, \mathrm{KL}\!\big(\pi_\theta \,\|\, \pi_\text{ref}\big)
\right] \tag{5}
$$

where

$$
\rho_{i,t} = \frac{\pi_\theta(o^i_t \mid q, o^i_{<t})}{\pi_{\theta_\text{old}}(o^i_t \mid q, o^i_{<t})}.
$$

KL estimator used by DeepSeek (unbiased, always positive), per token:

$$
\widehat{\mathrm{KL}} = \frac{\pi_\text{ref}}{\pi_\theta} - \log\frac{\pi_\text{ref}}{\pi_\theta} - 1 \tag{6}
$$

$\beta$ controls how far you drift from the reference (prevents reward hacking
and catastrophic forgetting).

### 4.4 Algorithm (one iteration)

1. Set $\pi_{\theta_\text{old}} \leftarrow \pi_\theta$.
2. For each prompt $q$ in the batch: sample $G$ outputs from
   $\pi_{\theta_\text{old}}$.
3. Compute rewards $r_i$, then advantages $A_i$ via $(4)$.
4. For a few inner epochs: compute $L_\text{GRPO}$ $(5)$ and take gradient
   ascent steps.
5. (Optionally update $\pi_\text{ref}$ occasionally.)

That's it. No value network, no GAE. The "group" is your baseline.

---

## 5. PPO vs GRPO cheat-sheet

| Aspect            | PPO                        | GRPO                          |
|-------------------|----------------------------|-------------------------------|
| Baseline          | learned critic $V_\phi$    | group mean reward             |
| Extra network     | yes (value head/model)     | no                            |
| Advantage         | GAE (per-token)            | $(r - \text{mean})/\text{std}$, shared/seq |
| Memory/compute    | higher                     | lower                         |
| Needs             | reward + critic            | reward + $G$ samples per prompt |
| Clipped surrogate | yes                        | yes                           |
| KL control        | penalty / early-stop       | explicit KL to $\pi_\text{ref}$ |

---

## 6. How this maps to VLA (preview for Stage 2)

- "Prompt $q$" = image observation(s) + language instruction.
- "Output $o$" = an action chunk (a sequence of predicted actions / tokens).
- "Reward $r$" = task success ($0/1$) or shaped (e.g. progress, $-\text{distance}$).
- Group = $G$ sampled action chunks / rollouts from the same start state.
- Same advantage $(4)$ and surrogate $(5)$. The hard parts are: (a) getting a
  usable reward from a simulator (LIBERO), (b) sampling diverse rollouts,
  (c) credit assignment across a long-horizon episode.

---

## 7. Things to verify by hand (exercises)

1. Prove $\mathbb{E}[b \, \nabla_\theta \log \pi] = 0$ (done above — redo it
   closed-book).
2. Show that $(1)$ reduces to the supervised cross-entropy gradient when $r$ is
   a constant $1$ for the demonstrated trajectory (this links RL $\leftrightarrow$ SFT).
3. For $G = 2$ with rewards $\{1, 0\}$, compute $A_1, A_2$ from $(4)$.
   *(Answer: $\operatorname{mean} = 0.5$, $\operatorname{std} = 0.5 \Rightarrow A_1 = +1,\; A_2 = -1$ approximately.)*
4. Explain why $\operatorname{std} \to 0$ kills the signal, and two ways to
   avoid it.
