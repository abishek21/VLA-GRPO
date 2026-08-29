"""
Prototype: a stochastic action head that gives GRPO a tractable log-prob for
CONTINUOUS actions (the SmolVLA / VLA case).

READ THIS FRAMING FIRST (addresses common misconceptions):

(1) This defines a NEW policy. The log-prob here is
        log p_{our-Gaussian-policy}(A | o)
    NOT
        log p_{SmolVLA-flow}(A | o).
    We are *replacing* SmolVLA's flow-matching action head with a stochastic
    Gaussian head for RL, because flow matching has no closed-form log-prob and
    GRPO needs one to form  rho = pi_new/pi_old.

(2) IMPORTANT for real use: a randomly-initialized head throws away SmolVLA's
    pretrained action ability. Before RL, initialize the mean to match the flow
    policy (distill:  mu(features) ~= predict_action_chunk(o)) or use a residual
    (action = flow_action + gaussian_noise). Otherwise the KL-leash "reference"
    is garbage and you're training from scratch, not post-training.

(3) We keep the action-chunk shape [B, T, D] (NOT flattened to [B, T*D]).
    Summing T*D per-dim log-ratios makes rho explode (e.g. +0.01 over 350 dims
    -> log-ratio +3.5 -> rho ~= 33 -> clip saturates). Keeping [B, T, D] lets us
    inspect per-timestep log-ratios and choose how to aggregate. (Same lesson as
    the sequence-summed-KL blow-up in the tiny-LLM GRPO.)

(4) Optional tanh-squashing: robots clip actions to [-1, 1]. If you sample
    a ~ N(mu, sigma) and then clamp, the executed action != sampled action and
    log_prob is WRONG. A tanh-squashed Gaussian keeps sample and log_prob
    consistent via the change-of-variables Jacobian. Enabled by squash=True.

(5) The GRPO "group" here is a single-step BANDIT demo (G samples of one chunk
    from one observation). Real manipulation: G FULL rollouts from the same
    initial state; observations DIVERGE after the first chunk; each trajectory
    has multiple action decisions. See notes in demo() and research_proposal.md.

Run (local, CPU/MPS):
  conda activate vla
  python 02_openvla_oft_grpo/proto_gaussian_head.py
"""
import torch
import torch.nn as nn


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ----------------------------------------------------------------------
# The stochastic action head  (diagonal Gaussian, optional tanh squash)
# ----------------------------------------------------------------------
class GaussianActionHead(nn.Module):
    """
    features [B, feat_dim] -> diagonal Gaussian over an action chunk [B, T, D].

    We DO NOT flatten T and D away; log_prob_per_step returns [B, T] so the
    caller can inspect / choose the aggregation over the horizon.

    squash=True applies a=tanh(z) with the Jacobian correction, so sampled and
    executed actions match when the env expects actions in [-1, 1].
    """

    def __init__(self, feat_dim: int, horizon: int, action_dim: int,
                 log_std_init: float = -0.5, squash: bool = False,
                 log_std_min: float = -5.0, log_std_max: float = 2.0):
        super().__init__()
        self.T, self.D = horizon, action_dim
        self.squash = squash
        self.log_std_min, self.log_std_max = log_std_min, log_std_max

        # state-dependent mean over the whole chunk (T*D outputs, reshaped)
        self.mu = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.GELU(),
            nn.Linear(feat_dim, horizon * action_dim),
        )
        # per-(t,d) log-std as a global learned parameter (simple start)
        self.log_std = nn.Parameter(torch.full((horizon, action_dim), log_std_init))

    def _base_dist(self, feats: torch.Tensor) -> torch.distributions.Normal:
        B = feats.shape[0]
        mu = self.mu(feats).view(B, self.T, self.D)          # [B, T, D]
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        sigma = log_std.exp().unsqueeze(0).expand_as(mu)     # [B, T, D]
        return torch.distributions.Normal(mu, sigma)

    @torch.no_grad()
    def sample(self, feats: torch.Tensor) -> torch.Tensor:
        """Draw an action chunk [B, T, D] (no grad; rollout time)."""
        z = self._base_dist(feats).rsample()                 # [B, T, D]
        return torch.tanh(z) if self.squash else z

    def log_prob_per_step(self, feats: torch.Tensor,
                          action: torch.Tensor) -> torch.Tensor:
        """
        Per-timestep log-prob [B, T]  (summed over the D action dims only).
        Keeping the T axis lets us watch where the ratio explodes and decide
        how to aggregate over the horizon.
        """
        dist = self._base_dist(feats)
        if self.squash:
            # invert a=tanh(z) -> z=atanh(a); add change-of-variables Jacobian.
            a = action.clamp(-0.999999, 0.999999)
            z = torch.atanh(a)
            logp = dist.log_prob(z)                          # [B, T, D]
            # tanh Jacobian: log|da/dz| = log(1 - tanh(z)^2)
            logp = logp - torch.log(1 - torch.tanh(z) ** 2 + 1e-6)
        else:
            logp = dist.log_prob(action)                     # [B, T, D]
        return logp.sum(dim=-1)                              # sum D -> [B, T]

    def log_prob(self, feats: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Full-chunk log-prob [B] = sum over the horizon of the per-step values."""
        return self.log_prob_per_step(feats, action).sum(dim=-1)  # [B]


# ----------------------------------------------------------------------
# Demo: GRPO ratio for continuous actions (SINGLE-STEP BANDIT — see note 5)
# ----------------------------------------------------------------------
def group_advantages(r: torch.Tensor) -> torch.Tensor:
    """
    Group-relative advantage with correct normalization (feedback #6):
    - biased std (population) for a small group,
    - explicit zero-variance handling (all-success or all-fail -> no signal).
    """
    std = r.std(unbiased=False)
    if std < 1e-6:
        return torch.zeros_like(r)          # no relative info in the group
    return (r - r.mean()) / (std + 1e-6)


def demo():
    torch.manual_seed(0)
    device = pick_device()
    print("device:", device)

    FEAT_DIM = 64
    T, D = 50, 7                 # SmolVLA-like horizon; 7-DoF action (LIBERO)
    G = 8                        # group size

    head = GaussianActionHead(FEAT_DIM, T, D, squash=True).to(device)

    # NOTE (feedback #5): this repeats ONE observation G times = a single-step
    # bandit. Real robotics: G full env rollouts whose observations DIVERGE
    # after the first chunk. This demo only validates the log-prob/ratio math.
    feats = torch.randn(1, FEAT_DIM, device=device).expand(G, FEAT_DIM)

    with torch.no_grad():
        actions = head.sample(feats)                         # [G, T, D]
        logp_old = head.log_prob(feats, actions)             # [G]
        logp_old_step = head.log_prob_per_step(feats, actions)  # [G, T]
    print("actions        :", tuple(actions.shape), "(G, T, D)  -- kept unflattened")
    print("logp_old       :", tuple(logp_old.shape))
    print("logp_old_step  :", tuple(logp_old_step.shape), "-- per-timestep, inspectable")

    r = torch.randn(G, device=device)                        # placeholder rewards
    A = group_advantages(r)                                  # [G]

    logp_new = head.log_prob(feats, actions)                 # [G], grad flows
    log_ratio = logp_new - logp_old                          # [G]
    rho = torch.exp(log_ratio)
    EPS = 0.2
    surrogate = torch.min(rho * A, torch.clamp(rho, 1 - EPS, 1 + EPS) * A).mean()
    loss = -surrogate
    loss.backward()

    # per-step ratio inspection (feedback #2): watch where ratio would explode
    with torch.no_grad():
        step_logratio = (head.log_prob_per_step(feats, actions) - logp_old_step)
        print("\nrho (full chunk):", rho.detach().cpu().numpy().round(3),
              " (≈1 before any update)")
        print("per-step log-ratio abs max:", float(step_logratio.abs().max()),
              " (near 0 at init; watch this grow during training)")
    print("surrogate:", float(surrogate.detach()))
    print("grad on mu? ->", head.mu[0].weight.grad is not None)

    print("\nTakeaways:")
    print(" - NEW Gaussian policy (not SmolVLA's flow log-prob).")
    print(" - shape kept [B,T,D]; per-step log-ratio is inspectable (avoids")
    print("   hidden ratio blow-up from flattening T*D dims).")
    print(" - tanh-squash keeps sampled==executed action for [-1,1] envs.")
    print(" - advantage handles zero-variance groups (all-success/all-fail).")
    print(" - real GRPO group = G FULL rollouts with diverging obs (see note 5).")


if __name__ == "__main__":
    demo()
