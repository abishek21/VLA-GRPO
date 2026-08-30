# Reward Model — Interaction Events from Egocentric Human Video

Predicts per-timestep events `e_t = [contact, grasp, release, failure, recovery]`
(10 Hz) from `RGB window + Hand3D`, trained on HoloAssist. This reward drives
VLA post-training (see `notes/research_proposal.md`).

## Status: architecture COMPLETE + validated locally ($0 GPU)

| File | What | Validated |
|------|------|-----------|
| `inspect_holoassist_labels.py` | audit annotation JSON | ✅ failure/recovery pre-labeled |
| `select_subset.py` | pick failure-rich videos + size estimate | ✅ 60 vids ≈ 23 GB |
| `build_event_labels.py` | events JSON → `[T,5]` labels @10 Hz | ✅ visualized |
| `reward_model.py` | frozen encoder + hand MLP + GRU + classifier | ✅ fwd/bwd |
| `losses.py` | weighted BCE + temporal + transition | ✅ runs |
| `train_smoke.py` | overfit test | ✅ **recovery 1% → recall 1.0** |
| `dataset.py` | clips loader (real labels + provider) | ✅ batches |

## The local → GPU swap (only 2 things change)

1. **Encoder:** `StubEncoder(...)` → `load_r3m_encoder()` in `reward_model.py`.
2. **Frame/hand data:** `SyntheticProvider` → a real provider in `dataset.py`
   that decodes video frames + reads the hand-pose stream at 10 Hz.

Everything else (labels, model, loss, training loop) is unchanged.

## GPU setup

```bash
# env (same as local)
conda create -n vla python=3.10 -y && conda activate vla
pip install torch torchvision
pip install "lerobot[smolvla]"          # for the VLA side (Stage 2)
pip install r3m                          # frozen visual encoder (Ego4D-pretrained)
pip install av opencv-python-headless    # video decoding

git clone <your-repo> && cd vla-grpo
```

## Download ONLY the subset (not 370 GB)

`subset_manifest.json` lists the 60 `video_names`. Download, for each, the
**compressed RGB video** + **hand pose** streams from the HoloAssist release
onto the GPU disk / network volume (≈23 GB total). Skip depth/gaze/IMU.

HoloAssist streams (from https://holoassist.github.io/#download):
- Compressed videos (width 256)
- Hand pose
- (Camera calibration if projecting hand pose into image space)

## Build the real provider (the one new component on GPU)

Implement two methods matching `SyntheticProvider` in `dataset.py`:

```python
class RealProvider:
    def frames(self, video_name, f0, f1):
        # decode video[video_name], return frames [f0:f1] at 10 Hz -> [T,3,H,W]
    def hand(self, video_name, f0, f1):
        # read hand-pose stream[video_name], slice [f0:f1] @10 Hz -> [T,hand_dim]
```

Then: `ds = HoloAssistClips(RealProvider(), clip_len=60)`.

## Train

Wire `dataset.py` + `reward_model.py` (with `load_r3m_encoder()`) + `losses.py`
into a loop (mirror `train_smoke.py`, but iterate the DataLoader and use
`class_balance(ds)` for `pos_weight`). Target: failure/recovery F1 on a held-out
video split.

Ready-made trainer:
```bash
# 1. (optional) monitoring: pip install wandb && wandb login
# 2. train on real data with R3M + live W&B:
python 03_reward_model/train_reward_model.py \
    --root /workspace/holoassist --encoder r3m \
    --epochs 30 --clip-len 60 --batch-size 8 --H 224 --W 224 \
    --out runs/reward_v1 --wandb --wandb-run reward_v1
```
- Live dashboard: loss (event/temporal/transition) per step, and per-event
  precision/recall/F1 per epoch. Headline metric: `val/key_f1`
  (mean of failure + recovery F1).
- `--wandb` is optional; without it, metrics still print + save to
  `runs/.../history.json`. Best checkpoint → `runs/.../best.pt`.
- **Push `best.pt` to Azure/HF before stopping the pod** (ephemeral disk!).

Local dry run (no data/R3M, validates wiring):
```bash
python 03_reward_model/train_reward_model.py --encoder stub --dry-run --epochs 2
```

## Gates (from the proposal)
- [ ] **G3:** reward model reaches usable failure/recovery F1 on held-out human clips.
- [ ] **G4:** reward correlates with sim ground-truth success on robot rollouts.
- [ ] **G5:** GRPO with this reward improves VLA success / recovery.

## Design decisions (recorded)
- 10 Hz (native HoloAssist rate). Recovery = "corrected by student" (self-only).
- Frozen encoder (R3M ResNet, Ego4D) — only hand MLP + GRU + classifier train.
- Class weights ≈ inverse frequency (recovery ~99×, failure ~24×).
- Events kept as `[B,T,5]`; state events smoothed, transition events not.
