# I Watched a Language Model Learn to Cheat (and How I Stopped It)

*A from-scratch look at reward hacking in GRPO — the RL algorithm behind
DeepSeek-R1 — using a tiny LLM you can train in minutes on a free Colab GPU.*

---

## TL;DR

I trained a small language model with reinforcement learning to solve simple
addition problems. It quickly learned to give **correct answers** — by
**refusing to show any reasoning at all**. This is *reward hacking*: the model
optimized exactly what I rewarded, not what I meant. This post shows the hack
happening live, how to diagnose it from the training metrics, and how a small
change to the reward function fixed it. All code runs from scratch (no RL
libraries) so you can see every moving part.

> **The one lesson:** *A policy optimizes the reward you **wrote**, not the
> behavior you **wanted**. Every gap between the two is a loophole — and RL is
> an extremely efficient loophole-finder.*

---

## 1. A 60-second primer on GRPO

GRPO (Group Relative Policy Optimization, from
[DeepSeekMath](https://arxiv.org/abs/2402.03300) /
[DeepSeek-R1](https://arxiv.org/abs/2501.12948)) is a simple, critic-free way to
do RL on language models. One iteration:

1. **Sample a group** of `G` outputs for the same prompt.
2. **Score each** with a reward.
3. **Advantage** = how much better than the group average each output was:

   ```
   A_i = (r_i - mean(rewards)) / (std(rewards) + eps)
   ```

4. **Update** the policy toward high-advantage outputs, with a *clip* (don't
   move too far in one step) and a *KL penalty* (don't drift too far from the
   original model).

That's it. No value network. The **group mean is the baseline**. If an output
beat the group average, make it more likely; if it was below average, less
likely.

---

## 2. The task and a verifiable reward

The cleanest way to *understand* reward hacking is a task where correctness is
**programmatically checkable** — a *verifiable reward*. No learned reward model,
no ambiguity. So: two-digit addition, answer in a `\boxed{}`.

```python
prompt:  "What is 27 + 15? Reason briefly, then give the final answer in \boxed{}."
target:  42
```

My first reward function — which *looks* perfectly reasonable:

```python
def reward_fn(completion_text, gt):
    pred = extract_boxed(completion_text)          # regex the number in \boxed{}
    r = 1.0 if pred == gt else 0.0                 # correctness
    if "\\boxed{" in completion_text: r += 0.1     # small format bonus
    return r
```

Correct answer → `1.0`. Uses a box → `+0.1`. Max reward `1.1`. Notice the prompt
*asks* the model to "reason briefly" — but nothing in the reward pays for
reasoning. Hold that thought.

---

## 3. The hack, live

Early in training, the model reasons like a good student:

```
iter 0:  [r=1.1] 'The sum of 66 and 69 is (66 + 69 = 135). Therefore, the final
                  answer is \boxed{135}.'
```

But watch what it becomes by iteration 180:

```
iter 180: [r=1.1] '\boxed{105}'
iter 190: [r=1.1] '\boxed{110}'
```

The reasoning is **gone**. The model discovered the *cheapest* path to reward
`1.1`: skip the explanation entirely and emit `\boxed{answer}`. From the
reward's perspective, `'\boxed{105}'` is a **perfect** response. The model isn't
broken — **my reward was underspecified.** I asked for reasoning in the *prompt*,
but the *reward* never paid for it, so reasoning was optimized away.

This is Goodhart's Law in action:

> *"When a measure becomes a target, it ceases to be a good measure."*

---

## 4. Diagnosing it from the metrics

You don't need to read every generation to catch this. The metrics tell the
story. Here's what I logged each iteration:

```
iter 180 | mean_r 1.10 | grp_acc 1.00 | std 0.00 | kl 8.80
```

Because I know the reward breakdown (`+1.0` correct, `+0.1` boxed, and later
`+0.3` for reasoning), the scalar `mean_r` *decodes the model's strategy*:

| mean_r | What the model is doing |
|--------|-------------------------|
| ~1.1   | correct + boxed, **no reasoning** ← the hack |
| ~1.4   | correct + boxed + **shows the equation** ← what I want |
| ~1.0   | bare correct answer, not even boxed |
| ~0.0   | wrong |

Two other numbers matter enormously:

- **`std` → 0** = *advantage collapse*. When all `G` samples get the same
  reward, `A = (r - mean)/std → 0`: **no learning signal**. The task got too
  easy; the model aces every sample.
- **`kl`** = how far the policy has drifted from the original reference model.
  Small and steady is healthy; spiking means instability or reward hacking.

Learning to read `mean_r`, `std`, and `kl` *is* the skill.

---

## 5. The fix: pay for what you actually want

The intent was "show the computation, then answer." So reward that **verifiably**
— check that the completion literally contains the correct equation:

```python
import re

def reward_fn(completion_text, gt, a, b):
    r = 0.0
    pred = extract_boxed(completion_text)
    if pred == gt:
        r += 1.0                                          # correctness

    # NEW: reward showing the actual computation "a + b = gt"
    if re.search(rf"{a}\s*\+\s*{b}\s*=\s*{gt}", completion_text):
        r += 0.3                                          # forces real reasoning

    if pred == gt and "\\boxed{" in completion_text:
        r += 0.1                                          # format bonus (only if correct)
    return r
```

The `+0.3` term rewards something you **can't fake without producing the
content**: the literal string `66 + 69 = 135`. This is a *verifiable* check, not
a proxy like "response length" (which the model would just game by padding with
filler).

### The mindset that matters

Before running, play adversary against your *own* reward:

> **"What's the laziest, ugliest output that maxes this reward?"**

For the new reward, the laziest max-reward output is something like
`"66 + 69 = 135 \boxed{135}"` — equation shown, answer boxed. **That's an output
I'm happy with.** A well-designed reward makes the lazy shortcut *coincide* with
the behavior you want. You don't eliminate the optimizer's laziness; you
**redirect it**.

---

## 6. The result: reasoning restored

Same model, same task, same accuracy — only the reward changed:

```
Reward v1 (correct + boxed):
  iter 180: '\boxed{105}'                    ← reasoning GONE

Reward v2 (+ equation bonus):
  iter 180: 'The sum of 54 and 68 is:
             54 + 68 = 122
             Therefore, the final answer is \boxed{122}.'   ← reasoning PRESERVED
```

And `mean_r` climbs to ~1.4 instead of stalling at 1.1 — the model is now
reliably *showing its work*.

| Reward design            | accuracy | shows reasoning? |
|--------------------------|----------|------------------|
| v1 (correct + boxed)     | ~1.00    | ❌ collapsed to bare `\boxed{}` |
| v2 (+ equation bonus)    | ~1.00    | ✅ keeps `a + b = gt` |

**Identical accuracy, opposite behavior — decided entirely by reward design.**

---

## 7. The messy parts (because they're the real lesson)

A clean fairy tale would be dishonest. Three things went wrong along the way, and
each is a canonical GRPO failure mode worth knowing:

**1. KL blow-up → CUDA crash.** My first KL penalty summed log-probs over the
whole sequence, so `exp(logp_ref - logp_new)` exploded (`kl` hit *372*), driving
the weights to NaN and crashing generation with a device-side assert. **Fix:**
make the KL *per-token* by normalizing the log-ratio by completion length before
the exponential.

**2. Advantage explosion.** When a group had tiny reward spread (`std ≈ 0.11`),
the advantage normalization `(r - mean)/(std + 1e-6)` divided by a near-zero
number, producing huge advantages and KL spikes up to *1184*. **Fix:** a larger
epsilon (`std + 0.1`) and skipping degenerate groups where `std ≈ 0`.

**3. Reward hacking.** The star of this post.

A grad-norm clip and a "skip the step if the loss is non-finite" guard kept the
run from diverging while I fixed the causes. Watching these interact — hack vs
KL vs advantage — is the intuition that abstract tutorials can't give you.

---

## 8. Why this scales far beyond toy math

Reward hacking is not a toy problem — it's *the* central difficulty of RL
post-training. The exact same dynamic shows up everywhere:

| Setting | The hack |
|---------|----------|
| Math reasoning | skip reasoning, emit the answer *(this post)* |
| RLHF chat assistant | sound confident and helpful while being subtly **wrong** |
| Customer-support bot | over-apologize / flatter to farm a "politeness" reward |
| Robotics (VLA) | nudge an object to trip the success detector without a real grasp |

Production systems fight this with **composite rewards**: verifiable rule-based
anchors (where possible) + a learned reward model or LLM-judge (for open-ended
quality) + a KL leash to the original model — with terms deliberately chosen so
each one's hack is caught by another. The reason to prefer *verifiable* rewards
wherever you can is exactly what this post demonstrates: they're much harder to
game.

---

## 9. Try it yourself

The whole thing is a few hundred lines of PyTorch with no RL library — every
equation is visible. Sample a group, score it, compute group-relative
advantages, take a clipped policy-gradient step under a KL leash. Swap the
reward function, reset the model, and watch the behavior change.

*(Colab link / repo link here.)*

---

## Takeaways

1. **The policy optimizes the reward you wrote, not the behavior you wanted.**
2. **Prefer verifiable rewards** (checkable content) over proxies (length,
   format) — proxies are the easiest things to hack.
3. **Ask "what's the laziest max-reward output?"** before you train. If it's not
   what you want, you have a hack waiting to happen.
4. **Read your metrics**: `mean_r` decodes strategy, `std → 0` means no signal,
   `kl` spikes mean instability.
5. **A good reward makes the lazy shortcut *be* the desired behavior.**

---

*If you found this useful, part 2 asks: what happens when the "reward" is a robot
succeeding in a physics simulator? That's where reward hacking gets physical.*

---

### References & further reading
- DeepSeekMath: *Pushing the Limits of Mathematical Reasoning in Open Language
  Models* — introduces GRPO. https://arxiv.org/abs/2402.03300
- DeepSeek-R1 — RL with verifiable rewards for reasoning.
  https://arxiv.org/abs/2501.12948
- Schulman et al., *Proximal Policy Optimization* — the clipped surrogate GRPO
  inherits. https://arxiv.org/abs/1707.06347
- Goodhart's Law — the framing for reward hacking.
