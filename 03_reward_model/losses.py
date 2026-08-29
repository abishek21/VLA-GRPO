"""
Loss functions for the interaction-event reward model.

Composite objective (see notes/research_proposal.md 4b.B):

    L = L_event + lambda_tmp * L_temporal + lambda_trn * L_transition

where
  L_event      : class-weighted BCE over the 5 events (handles the heavy
                 imbalance we measured: contact ~51%, grasp ~20%, release ~3%,
                 failure ~4%, recovery ~1%).  Optionally focal to focus on hard,
                 rare positives.
  L_temporal   : penalizes frame-to-frame jitter on the STATE events
                 (contact/grasp/release are piecewise-constant); we do NOT
                 smooth the transition events (failure/recovery) since those ARE
                 supposed to change quickly.
  L_transition : soft grammar prior — grasp with no (current or recent) contact
                 is unlikely; penalize p(grasp) that appears without contact
                 support. A light, differentiable stand-in for a CRF/HMM.

Class weights: we pass pos_weight per event = neg/pos frequency (inverse-
frequency), which is the standard fix for imbalanced BCE.

Run (local): a smoke test on dummy predictions/labels.
  conda activate vla
  python 03_reward_model/losses.py
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# event order must match reward_model / build_event_labels
EVENTS = ["contact", "grasp", "release", "failure", "recovery"]
STATE_EVENTS = ["contact", "grasp", "release"]          # smoothed
TRANSITION_EVENTS = ["failure", "recovery"]             # NOT smoothed
IDX = {n: i for i, n in enumerate(EVENTS)}


def pos_weight_from_freq(active_frac: torch.Tensor,
                         clamp_max: float = 200.0) -> torch.Tensor:
    """
    Given per-event active-frame fraction p (e.g. [0.51,0.20,0.03,0.04,0.01]),
    return pos_weight = (1-p)/p  for BCEWithLogits, clamped for stability.
    Rare events -> large weight (recovery ~ (0.99/0.01)=99x).
    """
    p = active_frac.clamp(min=1e-4)
    return ((1 - p) / p).clamp(max=clamp_max)


def event_bce(logits: torch.Tensor, labels: torch.Tensor,
              pos_weight: torch.Tensor, focal_gamma: float = 0.0) -> torch.Tensor:
    """
    Class-weighted BCE with logits. logits/labels: [B, T, 5].
    focal_gamma>0 adds a focal modulation (1-p_t)^gamma to focus on hard cases.
    """
    # per-element weighted BCE (no reduction yet)
    bce = F.binary_cross_entropy_with_logits(
        logits, labels, pos_weight=pos_weight, reduction="none")   # [B,T,5]
    if focal_gamma > 0:
        # p_t = prob assigned to the true class
        p = torch.sigmoid(logits)
        p_t = torch.where(labels > 0.5, p, 1 - p)
        bce = ((1 - p_t) ** focal_gamma) * bce
    return bce.mean()


def temporal_smoothness(logits: torch.Tensor) -> torch.Tensor:
    """
    Penalize |p_t - p_{t-1}|^2 on STATE events only (piecewise-constant).
    Transition events are excluded so we don't discourage their fast changes.
    """
    p = torch.sigmoid(logits)                                   # [B,T,5]
    state_idx = [IDX[e] for e in STATE_EVENTS]
    ps = p[..., state_idx]                                      # [B,T,3]
    dif = ps[:, 1:, :] - ps[:, :-1, :]                          # [B,T-1,3]
    return (dif ** 2).mean()


def transition_prior(logits: torch.Tensor, support_window: int = 5) -> torch.Tensor:
    """
    Soft grammar: grasp should be supported by contact within a recent window.
    Penalize p(grasp) that lacks nearby contact support.
        penalty = mean( p_grasp * relu(1 - max_recent_contact) )
    Differentiable, light-weight stand-in for a CRF transition model.
    """
    p = torch.sigmoid(logits)
    p_grasp = p[..., IDX["grasp"]]                              # [B,T]
    p_contact = p[..., IDX["contact"]]                         # [B,T]
    # max contact prob over a trailing window via max-pool
    B, T = p_contact.shape
    pad = p_contact.unsqueeze(1)                               # [B,1,T]
    recent_contact = F.max_pool1d(pad, kernel_size=support_window,
                                  stride=1, padding=support_window // 2)
    recent_contact = recent_contact.squeeze(1)[:, :T]          # [B,T]
    unsupported = torch.relu(1.0 - recent_contact)             # high if no contact
    return (p_grasp * unsupported).mean()


def total_loss(logits, labels, pos_weight,
               lambda_tmp: float = 0.1, lambda_trn: float = 0.05,
               focal_gamma: float = 0.0):
    """Composite loss + a dict of the components for logging."""
    l_event = event_bce(logits, labels, pos_weight, focal_gamma)
    l_tmp = temporal_smoothness(logits)
    l_trn = transition_prior(logits)
    total = l_event + lambda_tmp * l_tmp + lambda_trn * l_trn
    return total, {"event": float(l_event), "temporal": float(l_tmp),
                   "transition": float(l_trn), "total": float(total)}


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
def smoke_test():
    torch.manual_seed(0)
    B, T, E = 2, 40, len(EVENTS)

    # measured class balance from build_event_labels (approx)
    active_frac = torch.tensor([0.51, 0.20, 0.03, 0.04, 0.01])
    pw = pos_weight_from_freq(active_frac)
    print("pos_weight per event:")
    for n, w in zip(EVENTS, pw.tolist()):
        print(f"  {n:9s}: {w:7.1f}")

    logits = torch.randn(B, T, E, requires_grad=True)
    labels = (torch.rand(B, T, E) > 0.9).float()               # sparse positives

    total, parts = total_loss(logits, labels, pw,
                              lambda_tmp=0.1, lambda_trn=0.05, focal_gamma=2.0)
    total.backward()

    print("\nloss components:")
    for k, v in parts.items():
        print(f"  {k:11s}: {v:.4f}")
    print("grad flows to logits? ->", logits.grad is not None)
    print("\nOK: composite loss (weighted BCE + temporal + transition) runs.")
    print("Rare events (failure/recovery) get large pos_weight -> not ignored.")


if __name__ == "__main__":
    smoke_test()
