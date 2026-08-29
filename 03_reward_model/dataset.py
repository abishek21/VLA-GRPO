"""
Dataset / DataLoader for the interaction-event reward model.

Yields fixed-length clips:
    frames [T, 3, H, W]   RGB at 10 Hz
    hand   [T, hand_dim]  3D hand keypoints at 10 Hz
    labels [T, 5]         event targets from build_event_labels

Design:
  - The label side is REAL (from the HoloAssist JSON) and works locally now.
  - The frame/hand side is abstracted behind loaders so we can run STRUCTURALLY
    on Mac with a synthetic provider (no video), then swap to real decoders on
    the GPU (video reader + hand-pose parser) with zero change to training code.

Run (local): structural test with the synthetic frame/hand provider.
  conda activate vla
  python 03_reward_model/dataset.py
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from build_event_labels import build_labels, RATE_HZ, EVENT_NAMES

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = os.path.join(HERE, "data", "data-annotation-trainval-v1_1.json")
MANIFEST = os.path.join(HERE, "subset_manifest.json")


# ----------------------------------------------------------------------
# Frame / hand providers (swappable: synthetic now, real decoders on GPU)
# ----------------------------------------------------------------------
@dataclass
class SyntheticProvider:
    """Returns random frames/hand for a (video, frame-range). Local structural
    testing only -- no video needed."""
    H: int = 64
    W: int = 64
    hand_dim: int = 63

    def frames(self, video_name: str, f0: int, f1: int) -> torch.Tensor:
        return torch.randn(f1 - f0, 3, self.H, self.W)

    def hand(self, video_name: str, f0: int, f1: int) -> torch.Tensor:
        return torch.randn(f1 - f0, self.hand_dim)


# On GPU you implement a provider with the same two methods, e.g.:
#   class RealProvider:
#       def frames(self, name, f0, f1): decode video[name] frames f0:f1 @10Hz
#       def hand(self,   name, f0, f1): read hand-pose stream[name] f0:f1 @10Hz
# ...and pass it to HoloAssistClips instead of SyntheticProvider.


# ----------------------------------------------------------------------
# Dataset: cut each video's timeline into fixed-length clips
# ----------------------------------------------------------------------
class HoloAssistClips(Dataset):
    def __init__(self, provider, clip_len: int = 60, stride: Optional[int] = None,
                 use_manifest: bool = True, max_videos: Optional[int] = None):
        self.provider = provider
        self.clip_len = clip_len
        self.stride = stride or clip_len              # non-overlapping by default

        data = json.load(open(LABELS))
        wanted = None
        if use_manifest and os.path.exists(MANIFEST):
            wanted = set(json.load(open(MANIFEST)).get("video_names", []))
        vids = [v for v in data if (wanted is None or v.get("video_name") in wanted)]
        if max_videos:
            vids = vids[:max_videos]

        # precompute label tensors + clip index (video, start_frame)
        self.labels = {}          # video_name -> [T,5] float tensor
        self.index = []           # list of (video_name, f0)
        for v in vids:
            name = v.get("video_name")
            e = build_labels(v)                        # [T,5] numpy
            if e.shape[0] < clip_len:
                continue
            self.labels[name] = torch.from_numpy(e)
            T = e.shape[0]
            for f0 in range(0, T - clip_len + 1, self.stride):
                self.index.append((name, f0))

        print(f"HoloAssistClips: {len(self.labels)} videos -> "
              f"{len(self.index)} clips of {clip_len} frames "
              f"({clip_len/RATE_HZ:.1f}s each)")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        name, f0 = self.index[i]
        f1 = f0 + self.clip_len
        frames = self.provider.frames(name, f0, f1)    # [T,3,H,W]
        hand = self.provider.hand(name, f0, f1)        # [T,hand_dim]
        labels = self.labels[name][f0:f1]              # [T,5]
        return frames, hand, labels


def class_balance(ds: HoloAssistClips) -> torch.Tensor:
    """Aggregate per-event active fraction across all clips (for pos_weight)."""
    tot = torch.zeros(len(EVENT_NAMES))
    n = 0
    for name, lab in ds.labels.items():
        tot += lab.sum(dim=0)
        n += lab.shape[0]
    return tot / max(1, n)


# ----------------------------------------------------------------------
# Structural test (local, synthetic frames)
# ----------------------------------------------------------------------
def main():
    provider = SyntheticProvider(H=64, W=64, hand_dim=63)
    ds = HoloAssistClips(provider, clip_len=60, max_videos=5)
    frac = class_balance(ds)
    print("aggregate class balance:",
          {e: round(float(a), 3) for e, a in zip(EVENT_NAMES, frac)})

    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    frames, hand, labels = next(iter(dl))
    print("batch frames:", tuple(frames.shape), "(B,T,3,H,W)")
    print("batch hand  :", tuple(hand.shape), "(B,T,hand_dim)")
    print("batch labels:", tuple(labels.shape), "(B,T,5)")
    print("\nOK: dataset yields (frames, hand, labels) batches.")
    print("On GPU: replace SyntheticProvider with a real video+handpose provider.")


if __name__ == "__main__":
    main()
