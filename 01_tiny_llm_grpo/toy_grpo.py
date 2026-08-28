"""
GRPO from scratch on a toy *verifiable* task -- NO LLM, runs in seconds.

Task: the policy emits a binary string of length N (one "token" per position).
Reward = number of positions that match a hidden target pattern.
This is the smallest possible setting that still exercises EVERY moving part
of GRPO exactly as in the LLM/VLA case:

  - autoregressive-style per-token log-probs   (here: independent Bernoulli)
  - group sampling of G outputs per "prompt"
  - group-relative advantage  A_i = (r_i - mean)/std      <-- eq (4)
  - PPO clipped surrogate with rho = pi_new/pi_old         <-- eq (5)
  - KL penalty to a frozen reference policy                <-- eq (6)

Map to the math notes (notes/grpo_math.md):
  "prompt q"  -> there is a single prompt (the task itself)
  "output o"  -> the N-bit string
  "token o_t" -> bit t
  pi_theta(o_t=1) = sigmoid(logit_t)

Run:  python toy_grpo.py
"""
import torch
import torch.nn.functional as F

torch.manual_seed(0)

# ---------------- task config ----------------
N        = 20          # sequence length ("number of tokens")
TARGET   = torch.randint(0, 2, (N,)).float()   # hidden pattern to match
G        = 16          # group size (samples per update)  -- the GRPO "group"
ITERS    = 300         # outer GRPO iterations
INNER    = 4           # inner PPO epochs per iteration (reuse the same batch)
EPS      = 0.2         # clip epsilon
BETA     = 0.01        # KL penalty coefficient
LR       = 0.1

# ---------------- policy ----------------
# theta = per-position logits. pi(bit_t = 1) = sigmoid(theta_t).
theta      = torch.zeros(N, requires_grad=True)   # start at p=0.5 everywhere
theta_ref  = torch.zeros(N)                        # frozen reference (SFT stand-in)
opt        = torch.optim.SGD([theta], lr=LR)


def sample(logits, g):
    """Sample g binary sequences. Returns (samples[g,N], no grad)."""
    p = torch.sigmoid(logits)
    return torch.bernoulli(p.expand(g, N))


def seq_logprob(logits, samples):
    """Sum of per-token log-probs for each sequence -> log pi(o|q). Shape [g]."""
    # Bernoulli log-prob per bit, then sum over the sequence (product in prob).
    p = torch.sigmoid(logits)                       # [N]
    logp_bit = samples * torch.log(p + 1e-9) + (1 - samples) * torch.log(1 - p + 1e-9)
    return logp_bit.sum(dim=1)                        # [g]


def reward(samples):
    """Verifiable reward: number of matching bits. Shape [g]."""
    return (samples == TARGET).float().sum(dim=1)


for it in range(ITERS):
    # ---- 1. freeze old policy, sample a group ----
    with torch.no_grad():
        logits_old = theta.detach().clone()
        samples    = sample(logits_old, G)            # [G, N]
        logp_old   = seq_logprob(logits_old, samples) # [G]
        r          = reward(samples)                  # [G]

        # ---- 2. group-relative advantage  (eq 4) ----
        A = (r - r.mean()) / (r.std() + 1e-6)         # [G]

    # ---- 3. inner PPO epochs on the same batch ----
    for _ in range(INNER):
        logp_new = seq_logprob(theta, samples)        # [G], grad flows
        rho      = torch.exp(logp_new - logp_old)     # ratio pi_new/pi_old

        unclipped = rho * A
        clipped   = torch.clamp(rho, 1 - EPS, 1 + EPS) * A
        surrogate = torch.min(unclipped, clipped).mean()   # maximize this

        # KL(pi_theta || pi_ref) -- simple analytic form for Bernoulli, summed
        p_new = torch.sigmoid(theta)
        p_ref = torch.sigmoid(theta_ref)
        kl = (p_new * (torch.log(p_new + 1e-9) - torch.log(p_ref + 1e-9)) +
              (1 - p_new) * (torch.log(1 - p_new + 1e-9) - torch.log(1 - p_ref + 1e-9))
             ).sum()

        loss = -(surrogate - BETA * kl)               # minimize negative objective
        opt.zero_grad(); loss.backward(); opt.step()

    if it % 25 == 0 or it == ITERS - 1:
        # greedy decode accuracy = how many bits the argmax policy gets right
        greedy = (torch.sigmoid(theta) > 0.5).float()
        acc = (greedy == TARGET).float().mean().item()
        print(f"iter {it:4d} | mean_reward {r.mean():5.2f}/{N} "
              f"| greedy_acc {acc:4.2f} | kl {kl.item():5.2f}")

print("\nTarget :", TARGET.int().tolist())
print("Learned:", (torch.sigmoid(theta) > 0.5).int().tolist())
print("\nIf greedy_acc -> 1.00, GRPO successfully recovered the hidden pattern")
print("using ONLY a scalar reward + the group baseline (no critic network).")
