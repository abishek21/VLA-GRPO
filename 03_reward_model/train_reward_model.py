"""
Train the interaction-event reward model on real HoloAssist data (Gate G3).

Wires:  RealProvider (video+hand)  ->  HoloAssistClips  ->  DataLoader
        InteractionRewardModel (frozen R3M encoder + hand MLP + GRU + classifier)
        composite loss (weighted focal BCE + temporal + transition)

Produces per-event precision/recall/F1 on a held-out session split, with focus
on failure/recovery, and checkpoints the best model.

Typical GPU run (after pulling data from Azure blob):
  python 03_reward_model/train_reward_model.py \
      --root /workspace/holoassist --encoder r3m \
      --epochs 30 --clip-len 60 --batch-size 8 --H 224 --W 224 \
      --out runs/reward_v1

Local dry run (no data/R3M): uses the stub encoder + synthetic provider.
  python 03_reward_model/train_reward_model.py --encoder stub --dry-run
"""
from __future__ import annotations

import argparse
import os
import json
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from build_event_labels import EVENT_NAMES
from reward_model import InteractionRewardModel, StubEncoder, pick_device
from losses import total_loss, pos_weight_from_freq
from dataset import HoloAssistClips, class_balance, SyntheticProvider


# ----------------------------------------------------------------------
# encoder factory
# ----------------------------------------------------------------------
def build_encoder(kind: str, device: str):
    if kind == "stub":
        return StubEncoder(out_dim=512).to(device)

    from reward_model import VisualEncoder

    if kind == "dinov2":
        # Easy, robust frozen encoder via torch.hub (no r3m dependency hell).
        try:
            net = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] dinov2 load failed ({e!r}); using StubEncoder.")
            return StubEncoder(out_dim=512).to(device)
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)

        class _DINO(VisualEncoder):
            def __init__(self):
                super().__init__()
                self.net = net
                self.out_dim = 384          # dinov2_vits14 embedding dim

            @torch.no_grad()
            def forward(self, images):       # [N,3,H,W] in [0,1]
                # DINOv2 wants 14-divisible size; 224 works. ImageNet norm.
                x = F.interpolate(images, size=(224, 224), mode="bilinear",
                                  align_corners=False)
                mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
                x = (x - mean) / std
                return self.net(x)           # [N, 384] CLS embedding

        return _DINO().to(device)

    # kind == "r3m": try real R3M; fall back to stub if it fails to import.
    try:
        from r3m import load_r3m
    except Exception as e:  # noqa: BLE001
        print(f"[warn] r3m import failed ({e!r}); falling back to StubEncoder.")
        print("       tip: use --encoder dinov2 for an easy frozen encoder.")
        return StubEncoder(out_dim=512).to(device)

    net = load_r3m("resnet18")
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)

    class _R3M(VisualEncoder):
        def __init__(self):
            super().__init__()
            self.net = net
            self.out_dim = 512

        @torch.no_grad()
        def forward(self, images):                       # [N,3,H,W] in [0,1]
            x = F.interpolate(images, size=(224, 224), mode="bilinear",
                              align_corners=False)
            x = x * 255.0                                # R3M expects 0-255 RGB
            return self.net(x)

    return _R3M().to(device)


# ----------------------------------------------------------------------
# evaluation: per-event precision / recall / F1 at threshold
# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, thr=0.5):
    model.eval()
    tp = torch.zeros(len(EVENT_NAMES))
    fp = torch.zeros(len(EVENT_NAMES))
    fn = torch.zeros(len(EVENT_NAMES))
    for frames, hand, labels in loader:
        frames, hand, labels = frames.to(device), hand.to(device), labels.to(device)
        p = torch.sigmoid(model(frames, hand))           # [B,T,5]
        pred = (p > thr).float()
        tp += ((pred == 1) & (labels == 1)).sum(dim=(0, 1)).cpu()
        fp += ((pred == 1) & (labels == 0)).sum(dim=(0, 1)).cpu()
        fn += ((pred == 0) & (labels == 1)).sum(dim=(0, 1)).cpu()
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return {EVENT_NAMES[i]: {"P": float(prec[i]), "R": float(rec[i]),
                             "F1": float(f1[i])} for i in range(len(EVENT_NAMES))}


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/holoassist")
    ap.add_argument("--encoder", choices=["r3m", "dinov2", "stub"], default="dinov2")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--clip-len", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--H", type=int, default=224)
    ap.add_argument("--W", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--out", default="runs/reward_v1")
    ap.add_argument("--dry-run", action="store_true",
                    help="use SyntheticProvider (no real data) to test the loop")
    ap.add_argument("--wandb", action="store_true",
                    help="log to Weights & Biases (pip install wandb; wandb login)")
    ap.add_argument("--wandb-project", default="vla-grpo-reward")
    ap.add_argument("--wandb-run", default=None)
    args = ap.parse_args()

    device = pick_device()
    os.makedirs(args.out, exist_ok=True)
    print("device:", device, "| encoder:", args.encoder, "| out:", args.out)

    # ---- optional Weights & Biases ----
    wb = None
    if args.wandb:
        try:
            import wandb as wb
            wb.init(project=args.wandb_project, name=args.wandb_run,
                    config=vars(args))
            print("[wandb] logging to project:", args.wandb_project)
        except Exception as e:  # noqa: BLE001
            print(f"[wandb] disabled ({e!r}); continuing without it.")
            wb = None

    # ---- data ----
    if args.dry_run:
        provider = SyntheticProvider(H=args.H, W=args.W, hand_dim=156)
        ds = HoloAssistClips(provider, clip_len=args.clip_len, use_manifest=False,
                             max_videos=5)
    else:
        from real_provider import RealProvider, HAND_DIM
        provider = RealProvider(args.root, H=args.H, W=args.W)
        ds = HoloAssistClips(provider, clip_len=args.clip_len, use_manifest=True)

    # ---- session-level train/val split (avoid clip leakage across split) ----
    sessions = sorted(ds.labels.keys())
    random.Random(0).shuffle(sessions)
    n_val = max(1, int(len(sessions) * args.val_frac))
    val_sessions = set(sessions[:n_val])
    train_idx = [i for i, (name, _) in enumerate(ds.index) if name not in val_sessions]
    val_idx = [i for i, (name, _) in enumerate(ds.index) if name in val_sessions]
    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)
    print(f"clips: {len(train_ds)} train / {len(val_ds)} val "
          f"({len(sessions)-n_val} / {n_val} sessions)")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, drop_last=True,
                          persistent_workers=args.num_workers > 0,
                          pin_memory=True,
                          prefetch_factor=4 if args.num_workers > 0 else None)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=max(1, args.num_workers // 2),
                        pin_memory=True)

    # ---- class weights from the FULL label set ----
    frac = class_balance(ds)
    pw = pos_weight_from_freq(frac).to(device)
    print("class balance:", {e: round(float(a), 3) for e, a in zip(EVENT_NAMES, frac)})
    print("pos_weight   :", {e: round(float(w), 1) for e, w in zip(EVENT_NAMES, pw)})

    # ---- model ----
    hand_dim = 156 if not args.dry_run else 156
    encoder = build_encoder(args.encoder, device)
    model = InteractionRewardModel(encoder, hand_dim=hand_dim,
                                   n_events=len(EVENT_NAMES)).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    print("trainable params:", sum(p.numel() for p in trainable))

    # ---- training loop ----
    best_f1 = -1.0
    hist = []
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for step, (frames, hand, labels) in enumerate(train_dl):
            frames, hand, labels = frames.to(device), hand.to(device), labels.to(device)
            logits = model(frames, hand)
            loss, parts = total_loss(logits, labels, pw,
                                     lambda_tmp=0.05, lambda_trn=0.02, focal_gamma=2.0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            running += parts["total"]
            global_step += 1
            if wb is not None:
                wb.log({"train/loss": parts["total"],
                        "train/loss_event": parts["event"],
                        "train/loss_temporal": parts["temporal"],
                        "train/loss_transition": parts["transition"],
                        "epoch": epoch}, step=global_step)
            if step % 20 == 0:
                print(f"  e{epoch} s{step} loss {parts['total']:.3f} "
                      f"(evt {parts['event']:.3f})")

        metrics = evaluate(model, val_dl, device)
        # headline: mean F1 over failure+recovery
        key_f1 = 0.5 * (metrics["failure"]["F1"] + metrics["recovery"]["F1"])
        print(f"[epoch {epoch}] train_loss {running/max(1,len(train_dl)):.3f} "
              f"| failure F1 {metrics['failure']['F1']:.3f} "
              f"| recovery F1 {metrics['recovery']['F1']:.3f} "
              f"| key {key_f1:.3f}")
        hist.append({"epoch": epoch, "metrics": metrics, "key_f1": key_f1})
        json.dump(hist, open(os.path.join(args.out, "history.json"), "w"), indent=2)

        if wb is not None:
            log = {"val/key_f1": key_f1,
                   "train/epoch_loss": running / max(1, len(train_dl)), "epoch": epoch}
            for e, m in metrics.items():
                for k, val in m.items():
                    log[f"val/{e}_{k}"] = val
            wb.log(log, step=global_step)

        if key_f1 > best_f1:
            best_f1 = key_f1
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "metrics": metrics, "args": vars(args)},
                       os.path.join(args.out, "best.pt"))
            print(f"  -> saved best (key F1 {best_f1:.3f})")

    if wb is not None:
        wb.summary["best_key_f1"] = best_f1
        wb.finish()
    print(f"\nDone. Best failure/recovery F1: {best_f1:.3f}")
    print(f"Checkpoint: {os.path.join(args.out, 'best.pt')}")
    print("Push the checkpoint to Azure/HF before stopping the pod!")


if __name__ == "__main__":
    main()
