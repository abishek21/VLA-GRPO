"""
Fast reward-model training on PRECOMPUTED features (no video decode).

Reads the .npz files from precompute_features.py and trains only the small
hand-MLP + GRU + classifier via model.forward_features. Epochs take seconds.

Run (GPU):
  python 03_reward_model/train_on_features.py \
      --feats /workspace/feats_r3m --epochs 60 --clip-len 60 --batch-size 32 \
      --out runs/reward_v1 --wandb --wandb-run reward_v1
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from build_event_labels import EVENT_NAMES
from reward_model import InteractionRewardModel, StubEncoder, pick_device
from losses import total_loss, pos_weight_from_freq


class FeatureClips(Dataset):
    """Slices fixed-length clips from cached per-session feature arrays."""

    def __init__(self, feat_dir, clip_len=60, stride=None):
        self.clip_len = clip_len
        self.stride = stride or clip_len
        self.sessions = {}       # name -> dict(vis, hand, labels)
        self.index = []          # (name, f0)
        for f in sorted(glob.glob(os.path.join(feat_dir, "*.npz"))):
            name = os.path.splitext(os.path.basename(f))[0]
            d = np.load(f)
            vis, hand, lab = d["vis"], d["hand"], d["labels"]
            T = min(vis.shape[0], hand.shape[0], lab.shape[0])
            if T < clip_len:
                continue
            self.sessions[name] = {"vis": vis[:T], "hand": hand[:T], "labels": lab[:T]}
            for f0 in range(0, T - clip_len + 1, self.stride):
                self.index.append((name, f0))
        self.vis_dim = next(iter(self.sessions.values()))["vis"].shape[1]
        print(f"FeatureClips: {len(self.sessions)} sessions -> {len(self.index)} clips "
              f"(vis_dim={self.vis_dim})")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        name, f0 = self.index[i]
        s = self.sessions[name]
        sl = slice(f0, f0 + self.clip_len)
        vis = torch.from_numpy(s["vis"][sl].astype(np.float32))
        hand = torch.from_numpy(s["hand"][sl].astype(np.float32))
        lab = torch.from_numpy(s["labels"][sl].astype(np.float32))
        return vis, hand, lab


def class_balance(ds):
    tot = torch.zeros(len(EVENT_NAMES)); n = 0
    for s in ds.sessions.values():
        lab = torch.from_numpy(s["labels"].astype(np.float32))
        tot += lab.sum(dim=0); n += lab.shape[0]
    return tot / max(1, n)


@torch.no_grad()
def evaluate(model, loader, device, thr=0.5):
    model.eval()
    tp = torch.zeros(len(EVENT_NAMES)); fp = torch.zeros(len(EVENT_NAMES))
    fn = torch.zeros(len(EVENT_NAMES))
    for vis, hand, labels in loader:
        vis, hand, labels = vis.to(device), hand.to(device), labels.to(device)
        p = torch.sigmoid(model.forward_features(vis, hand))
        pred = (p > thr).float()
        tp += ((pred == 1) & (labels == 1)).sum(dim=(0, 1)).cpu()
        fp += ((pred == 1) & (labels == 0)).sum(dim=(0, 1)).cpu()
        fn += ((pred == 0) & (labels == 1)).sum(dim=(0, 1)).cpu()
    prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return {EVENT_NAMES[i]: {"P": float(prec[i]), "R": float(rec[i]),
                             "F1": float(f1[i])} for i in range(len(EVENT_NAMES))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--clip-len", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--out", default="runs/reward_v1")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="vla-grpo-reward")
    ap.add_argument("--wandb-run", default=None)
    args = ap.parse_args()

    device = pick_device(); os.makedirs(args.out, exist_ok=True)
    print("device:", device)

    wb = None
    if args.wandb:
        try:
            import wandb as wb
            wb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))
        except Exception as e:  # noqa: BLE001
            print(f"[wandb] disabled ({e!r})"); wb = None

    ds = FeatureClips(args.feats, clip_len=args.clip_len)
    sessions = sorted(ds.sessions.keys()); random.Random(0).shuffle(sessions)
    n_val = max(1, int(len(sessions) * args.val_frac))
    val_s = set(sessions[:n_val])
    tr_idx = [i for i, (n, _) in enumerate(ds.index) if n not in val_s]
    va_idx = [i for i, (n, _) in enumerate(ds.index) if n in val_s]
    train_dl = DataLoader(Subset(ds, tr_idx), batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(Subset(ds, va_idx), batch_size=args.batch_size)
    print(f"clips: {len(tr_idx)} train / {len(va_idx)} val")

    frac = class_balance(ds); pw = pos_weight_from_freq(frac).to(device)
    print("class balance:", {e: round(float(a), 3) for e, a in zip(EVENT_NAMES, frac)})

    # model with a stub encoder placeholder (unused: we call forward_features)
    enc = StubEncoder(out_dim=ds.vis_dim).to(device)
    model = InteractionRewardModel(enc, hand_dim=ds.sessions[sessions[0]]["hand"].shape[1],
                                   n_events=len(EVENT_NAMES)).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)

    best = -1.0; hist = []; gstep = 0
    for epoch in range(args.epochs):
        model.train(); running = 0.0
        for vis, hand, labels in train_dl:
            vis, hand, labels = vis.to(device), hand.to(device), labels.to(device)
            logits = model.forward_features(vis, hand)
            loss, parts = total_loss(logits, labels, pw, lambda_tmp=0.05,
                                     lambda_trn=0.02, focal_gamma=2.0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
            running += parts["total"]; gstep += 1
            if wb is not None:
                wb.log({"train/loss": parts["total"], "epoch": epoch}, step=gstep)

        m = evaluate(model, val_dl, device)
        key = 0.5 * (m["failure"]["F1"] + m["recovery"]["F1"])
        print(f"[epoch {epoch}] loss {running/max(1,len(train_dl)):.3f} "
              f"| failure F1 {m['failure']['F1']:.3f} | recovery F1 {m['recovery']['F1']:.3f} "
              f"| grasp {m['grasp']['F1']:.3f} contact {m['contact']['F1']:.3f} "
              f"release {m['release']['F1']:.3f} | key {key:.3f}")
        hist.append({"epoch": epoch, "metrics": m, "key_f1": key})
        json.dump(hist, open(os.path.join(args.out, "history.json"), "w"), indent=2)
        if wb is not None:
            log = {"val/key_f1": key, "epoch": epoch}
            for e, mm in m.items():
                for k, v in mm.items():
                    log[f"val/{e}_{k}"] = v
            wb.log(log, step=gstep)
        if key > best:
            best = key
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": m,
                        "args": vars(args)}, os.path.join(args.out, "best.pt"))
            print(f"  -> saved best (key F1 {best:.3f})")

    if wb is not None:
        wb.summary["best_key_f1"] = best; wb.finish()
    print(f"\nDone. Best failure/recovery F1: {best:.3f}")


if __name__ == "__main__":
    main()
