"""
Precompute frozen-encoder features for every session (the big speedup).

Since the visual encoder (R3M/DINOv2) is FROZEN, its output per frame never
changes. We run it over all frames ONCE and cache per-session arrays:
    <out>/<session>.npz  ->  vis [T, D] float16, hand [T,156] f16, labels [T,5] f16
Then training reads these arrays (no video decode) and trains only the small
GRU+classifier -> ~10-50x faster per epoch.

Run (GPU):
  python 03_reward_model/precompute_features.py \
      --root /workspace/holoassist --encoder r3m --H 224 --W 224 \
      --out /workspace/feats_r3m --batch 256
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

from build_event_labels import build_labels, RATE_HZ, EVENT_NAMES
from real_provider import _video_meta, _load_hand_full, session_dir, PER_HAND_DIM, HAND_DIM

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = os.path.join(HERE, "data", "data-annotation-trainval-v1_1.json")
MANIFEST = os.path.join(HERE, "subset_manifest.json")


def build_encoder(kind, device):
    from train_reward_model import build_encoder as be
    return be(kind, device)


@torch.no_grad()
def encode_session(mp4, encoder, device, H, W, batch=256):
    """Decode all 10 Hz frames sequentially, encode in batches -> [T, D] f16."""
    import cv2
    fps, nframes = _video_meta(mp4)
    if nframes <= 0 or fps <= 0:
        return np.zeros((0, encoder.out_dim), np.float16)
    T10 = max(1, int(round(nframes / fps * RATE_HZ)))
    src_idx = [int(round(i / RATE_HZ * fps)) for i in range(T10)]  # increasing
    cap = cv2.VideoCapture(mp4)
    feats = []
    buf = []
    ptr = 0          # pointer into src_idx
    k = 0            # current source frame index
    ok, fr = cap.read()
    while ok and ptr < T10:
        # this source frame k may satisfy one or more targets
        while ptr < T10 and src_idx[ptr] == k:
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            buf.append(cv2.resize(rgb, (W, H)))
            ptr += 1
            if len(buf) >= batch:
                feats.append(_encode_batch(buf, encoder, device))
                buf = []
        # advance
        k += 1
        ok, fr = cap.read()
    cap.release()
    if buf:
        feats.append(_encode_batch(buf, encoder, device))
    if not feats:
        return np.zeros((0, encoder.out_dim), np.float16)
    return np.concatenate(feats, axis=0)     # [T10, D] f16


def _encode_batch(buf, encoder, device):
    x = torch.from_numpy(np.stack(buf)).float().permute(0, 3, 1, 2) / 255.0
    x = x.to(device)
    z = encoder(x)                            # [N, D]
    return z.detach().cpu().half().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/holoassist")
    ap.add_argument("--encoder", choices=["r3m", "dinov2", "stub"], default="r3m")
    ap.add_argument("--H", type=int, default=224)
    ap.add_argument("--W", type=int, default=224)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", default="/workspace/feats")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    print("device:", device, "| encoder:", args.encoder, "| out:", args.out)

    encoder = build_encoder(args.encoder, device)
    print("encoder out_dim:", encoder.out_dim)

    data = json.load(open(LABELS))
    wanted = set(json.load(open(MANIFEST)).get("video_names", [])) \
        if os.path.exists(MANIFEST) else None
    recs = [v for v in data if wanted is None or v.get("video_name") in wanted]
    print(f"sessions to process: {len(recs)}")

    for n, rec in enumerate(recs):
        name = rec["video_name"]
        outp = os.path.join(args.out, f"{name}.npz")
        if os.path.exists(outp):
            print(f"[{n+1}/{len(recs)}] {name} (exists, skip)")
            continue
        try:
            d = session_dir(args.root, name)
        except FileNotFoundError:
            print(f"[{n+1}/{len(recs)}] {name} (no data dir, skip)")
            continue
        mp4 = os.path.join(d, "Export_py", "Video_compress.mp4")
        hands = os.path.join(d, "Export_py", "Hands")

        vis = encode_session(mp4, encoder, device, args.H, args.W, args.batch)  # [Tv,D]
        hand_full = _load_hand_full(hands)                                       # [Th,156]
        lab = build_labels(rec)                                                  # [Tl,5]

        # align hand to the vis 10 Hz length by index
        fps, _ = _video_meta(mp4)
        Tv = vis.shape[0]
        src = np.clip((np.arange(Tv) / RATE_HZ * fps).round().astype(int),
                      0, max(0, hand_full.shape[0] - 1))
        hand = hand_full[src] if hand_full.shape[0] else np.zeros((Tv, HAND_DIM), np.float32)
        T = min(Tv, hand.shape[0], lab.shape[0])
        np.savez_compressed(outp, vis=vis[:T].astype(np.float16),
                            hand=hand[:T].astype(np.float16),
                            labels=lab[:T].astype(np.float16))
        print(f"[{n+1}/{len(recs)}] {name}  T={T} D={vis.shape[1]}")

    print("\nDone. Train fast with: train_on_features.py --feats", args.out)


if __name__ == "__main__":
    main()
