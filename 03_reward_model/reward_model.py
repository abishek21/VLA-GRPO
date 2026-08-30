"""
Interaction-event reward model.

Architecture (see notes/research_proposal.md 4b.A and the flow diagram):

    RGB window [t-k..t] --> VISUAL ENCODER (frozen: R3M ResNet)  --> [T, d_vis]
    Hand3D at t         --> HAND MLP (trained)                   --> [T, d_hand]
                              concat/fuse                        --> [T, d]
                          TEMPORAL HEAD (GRU, trained)           --> [T, d_h]
                          Linear + sigmoid                       --> [T, 5]
    outputs  e_hat_t = [contact, grasp, release, failure, recovery] in [0,1]

Design:
  - The visual encoder is FROZEN and SWAPPABLE. Locally we use a StubEncoder
    (random projection) so we can validate the whole pipeline on Mac/MPS with
    no downloads. On the GPU we swap in R3M (a ResNet CNN pretrained on Ego4D
    human manipulation video) or DINOv2 (a ViT) behind the same interface.
  - Only the hand MLP + GRU + classifier are trained (few params -> cheap).

Run (local, CPU/MPS): forward+backward smoke test on dummy tensors.
  conda activate vla
  python 03_reward_model/reward_model.py
"""
from __future__ import annotations

import torch
import torch.nn as nn


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ----------------------------------------------------------------------
# Visual encoder interface (frozen, swappable)
# ----------------------------------------------------------------------
class VisualEncoder(nn.Module):
    """Interface: images [N, 3, H, W] -> features [N, out_dim]. Always frozen."""
    out_dim: int

    def forward(self, images: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class StubEncoder(VisualEncoder):
    """Local stand-in: a fixed random linear map on pooled pixels. No download.
    Lets us validate shapes / training logic on Mac before using real R3M."""

    def __init__(self, out_dim: int = 512):
        super().__init__()
        self.out_dim = out_dim
        self.proj = nn.Linear(3, out_dim)
        for p in self.parameters():
            p.requires_grad_(False)     # frozen, like a real pretrained encoder

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # global-average-pool over H,W -> [N, 3] -> project -> [N, out_dim]
        pooled = images.mean(dim=(-1, -2))          # [N, 3]
        return self.proj(pooled)                    # [N, out_dim]


def load_r3m_encoder():  # used on GPU only
    """Load a frozen R3M ResNet encoder. Requires `pip install r3m` + its deps.
    Returns a VisualEncoder wrapper. Not called during local stub testing."""
    from r3m import load_r3m                        # noqa: WPS433 (GPU-only import)
    net = load_r3m("resnet18")                      # pretrained on Ego4D
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)

    class _R3M(VisualEncoder):
        def __init__(self):
            super().__init__()
            self.net = net
            self.out_dim = 512                       # resnet18 R3M embedding

        @torch.no_grad()
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            # R3M expects uint8-like [0,255] RGB; caller must match its transform
            return self.net(images)

    return _R3M()


# ----------------------------------------------------------------------
# The reward model
# ----------------------------------------------------------------------
class InteractionRewardModel(nn.Module):
    def __init__(self, encoder: VisualEncoder, hand_dim: int = 63,
                 d_hand: int = 128, d_model: int = 256,
                 n_events: int = 5, gru_layers: int = 1, bidirectional: bool = False,
                 dropout: float = 0.0):
        super().__init__()
        self.encoder = encoder                       # frozen
        self.n_events = n_events

        # hand-pose encoder (trained). hand_dim default 63 = 21 keypoints x 3.
        self.hand_mlp = nn.Sequential(
            nn.Linear(hand_dim, d_hand), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hand, d_hand), nn.GELU(), nn.Dropout(dropout),
        )
        # fuse visual + hand -> d_model per timestep
        self.fuse = nn.Linear(encoder.out_dim + d_hand, d_model)
        self.drop = nn.Dropout(dropout)
        # temporal head (trained): sees the whole sequence
        self.gru = nn.GRU(d_model, d_model, num_layers=gru_layers,
                          batch_first=True, bidirectional=bidirectional,
                          dropout=dropout if gru_layers > 1 else 0.0)
        # event classifier (trained): 5 independent sigmoids (multi-label)
        gru_out = d_model * (2 if bidirectional else 1)
        self.classifier = nn.Linear(gru_out, n_events)

    def forward(self, frames: torch.Tensor, hand: torch.Tensor) -> torch.Tensor:
        """
        frames: [B, T, 3, H, W]   RGB per timestep
        hand:   [B, T, hand_dim]  3D hand keypoints per timestep
        returns logits [B, T, n_events]  (apply sigmoid for probabilities)
        """
        B, T = frames.shape[:2]
        # encode each frame with the frozen encoder
        flat = frames.reshape(B * T, *frames.shape[2:])       # [B*T,3,H,W]
        vis = self.encoder(flat).reshape(B, T, -1)            # [B,T,d_vis]
        return self.forward_features(vis, hand)

    def forward_features(self, vis: torch.Tensor, hand: torch.Tensor) -> torch.Tensor:
        """Same as forward but takes PRECOMPUTED visual features vis [B,T,d_vis]
        (skips the frozen encoder). Used with cached features for fast training."""
        h = self.hand_mlp(hand)                               # [B,T,d_hand]
        fused = self.drop(self.fuse(torch.cat([vis, h], dim=-1)))  # [B,T,d_model]
        temporal, _ = self.gru(fused)                         # [B,T,d_model]
        return self.classifier(temporal)                     # [B,T,n_events]

    @torch.no_grad()
    def predict(self, frames, hand) -> torch.Tensor:
        return torch.sigmoid(self.forward(frames, hand))


# ----------------------------------------------------------------------
# Local smoke test: forward + backward on dummy tensors
# ----------------------------------------------------------------------
def smoke_test():
    torch.manual_seed(0)
    device = pick_device()
    print("device:", device)

    B, T, H, W = 2, 30, 64, 64        # 2 clips, 30 frames (3 s @ 10Hz)
    HAND_DIM, N_EVENTS = 63, 5

    enc = StubEncoder(out_dim=512).to(device)
    model = InteractionRewardModel(enc, hand_dim=HAND_DIM, n_events=N_EVENTS).to(device)

    frames = torch.randn(B, T, 3, H, W, device=device)
    hand = torch.randn(B, T, HAND_DIM, device=device)
    labels = (torch.rand(B, T, N_EVENTS, device=device) > 0.8).float()

    logits = model(frames, hand)
    print("logits shape:", tuple(logits.shape), "(B, T, n_events)")
    assert logits.shape == (B, T, N_EVENTS)

    # plain BCE just to confirm gradients flow (real weighted loss = losses.py)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"loss: {loss.item():.4f}")
    print(f"trainable params: {trainable:,}   frozen params: {frozen:,}")
    print("grad on classifier? ->", model.classifier.weight.grad is not None)
    print("grad on encoder?    ->",
          any(p.grad is not None for p in model.encoder.parameters()),
          "(should be False -- encoder is frozen)")
    print("\nOK: full pipeline (encode->fuse->GRU->classify->loss->backward) runs.")
    print("On GPU: swap StubEncoder for load_r3m_encoder() + real frames/hand.")


if __name__ == "__main__":
    smoke_test()
