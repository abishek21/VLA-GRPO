"""
Overfit smoke test: prove the reward model can LEARN before spending GPU money.

Idea: build a tiny SYNTHETIC dataset where the event labels are a known function
of the (dummy) inputs, then train the model to memorize it. If the loss drops
and the rare events (failure/recovery) become predictable, the whole learning
machinery (model + composite loss + weighting) is sound. Real frames/hand-pose
are swapped in later on the GPU.

We use a *learnable* stub encoder here (unlike the frozen one in reward_model)
so the tiny synthetic signal can flow end-to-end -- this isolates "does the
training loop learn?" from "is the frozen encoder good?".

Run (local, CPU/MPS):
  conda activate vla
  python 03_reward_model/train_smoke.py
"""
from __future__ import annotations

import torch
import torch.nn as nn

from losses import EVENTS, pos_weight_from_freq, total_loss
from reward_model import InteractionRewardModel, VisualEncoder, pick_device


class LearnableStubEncoder(VisualEncoder):
    """Like StubEncoder but TRAINABLE, so a synthetic signal can propagate.
    (Only for the overfit test; real runs use a frozen pretrained encoder.)"""

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.out_dim = out_dim
        self.proj = nn.Linear(3, out_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.proj(images.mean(dim=(-1, -2)))          # [N, out_dim]


def make_synthetic_batch(B, T, H, W, hand_dim, n_events, device):
    """
    Create inputs and labels where labels are a deterministic function of the
    inputs, so the model *can* fit them. We encode each event as a threshold on
    a fixed linear projection of (pooled frame + hand) -- a learnable pattern.
    """
    frames = torch.randn(B, T, 3, H, W, device=device)
    hand = torch.randn(B, T, hand_dim, device=device)

    pooled = frames.mean(dim=(-1, -2))                        # [B,T,3]
    feat = torch.cat([pooled, hand[..., :5]], dim=-1)         # [B,T,8]
    proj = torch.randn(feat.shape[-1], n_events, device=device)
    score = feat @ proj                                       # [B,T,n_events]
    # per-event thresholds tuned to reproduce the REAL class balance
    # (contact common ... recovery very rare)
    q = torch.tensor([0.49, 0.80, 0.97, 0.96, 0.99], device=device)
    thresh = torch.quantile(score.reshape(-1, n_events), q, dim=0).diagonal()
    labels = (score > thresh).float()                        # [B,T,n_events]
    return frames, hand, labels


def main():
    torch.manual_seed(0)
    device = pick_device()
    print("device:", device)

    B, T, H, W = 4, 60, 32, 32
    HAND_DIM, N = 63, len(EVENTS)

    frames, hand, labels = make_synthetic_batch(B, T, H, W, HAND_DIM, N, device)
    active = labels.mean(dim=(0, 1))                          # per-event fraction
    print("synthetic class balance:",
          {e: round(float(a), 3) for e, a in zip(EVENTS, active)})
    pw = pos_weight_from_freq(active.cpu()).to(device)

    enc = LearnableStubEncoder(out_dim=64).to(device)
    model = InteractionRewardModel(enc, hand_dim=HAND_DIM, n_events=N,
                                   d_model=128).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    print("\ntraining to overfit one synthetic batch:")
    for step in range(300):
        logits = model(frames, hand)
        loss, parts = total_loss(logits, labels, pw,
                                 lambda_tmp=0.05, lambda_trn=0.02, focal_gamma=2.0)
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 50 == 0 or step == 299:
            with torch.no_grad():
                p = torch.sigmoid(logits)
                pred = (p > 0.5).float()
                # per-event recall on positives (did we catch the rare events?)
                rec = {}
                for i, e in enumerate(EVENTS):
                    pos = labels[..., i] > 0.5
                    rec[e] = float((pred[..., i][pos] > 0.5).float().mean()) \
                        if pos.any() else float("nan")
            print(f"  step {step:3d} | total {parts['total']:.3f} "
                  f"| event {parts['event']:.3f} "
                  f"| recall failure {rec['failure']:.2f} recovery {rec['recovery']:.2f}")

    print("\nIf total loss dropped and failure/recovery recall climbed toward 1.0,")
    print("the model + weighted loss can learn the rare events. Machinery is sound.")
    print("Next (GPU): swap in frozen R3M + real HoloAssist frames/hand pose.")


if __name__ == "__main__":
    main()
