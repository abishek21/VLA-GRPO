# Stage 2 (revised): VLA + LIBERO + GRPO — Reward-Hacking Study

**Pivot rationale (2026-09-04):** the human-video reward model (`03_reward_model/`)
hit a fundamental wall — its hand-pose input has no robot equivalent, and even
in-domain event detection was weak. That work is parked as **future work**.

**New direction (lower risk, plays to our strengths):**
> Run an open-source VLA (**OpenVLA-OFT**, LIBERO-ready) on LIBERO, apply GRPO
> with simulator rewards, and rigorously **characterize how/when VLA policies
> reward-hack** — while genuinely understanding how a VLA works end-to-end.

**Model choice: OpenVLA-OFT (Stanford)** — LIBERO-ready with released 97%
checkpoints, so we run a *working* VLA immediately (no SFT detour). SmolVLA was
"ready" for real SO-100 arms, not LIBERO, so it would need LIBERO fine-tuning
first. OpenVLA-OFT eval needs only ~16 GB VRAM (fits a 24 GB consumer GPU).

The novelty is the **analysis** (reward-hacking taxonomy + triggers + mitigations
in the embodied setting), not a leaderboard number. Reuses the GRPO stack from
`01_tiny_llm_grpo/`, the SmolVLA knowledge and Gaussian-head prototype from
`02_openvla_oft_grpo/`, and the reward-hacking framing from
`notes/blog_reward_hacking.md`.

---

## Milestones (each a checkpoint)

### M1 — Understand the VLA: one LIBERO rollout  ← START HERE
- Load SmolVLA, load a LIBERO task, run ONE episode (robot acts, sim steps),
  read the success flag, save frames.
- **Deliverable:** we can run + evaluate a VLA and understand its obs→action loop.
- Script: `step1_smolvla_libero_rollout.py`

### M2 — Reproduce an SFT baseline eval
- Evaluate SmolVLA (or a LIBERO-finetuned checkpoint) over N episodes on 1 suite.
- **Deliverable:** a baseline success rate to improve/compare against.

### M3 — GRPO with the sim success reward
- Wire the GRPO loop onto SmolVLA over action chunks (resolve the flow-head
  log-prob fork: attach Gaussian head, init mean from flow output).
- Reward = LIBERO success flag (+ optional shaping).
- **Deliverable:** RL post-training that changes success rate; stable KL.

### M4 — The reward-hacking study (the paper)
- Design several rewards: sparse success, shaped progress, distance-to-goal, etc.
- Observe + measure how the policy exploits each (e.g. nudging the object to trip
  the success detector without a real grasp).
- **Deliverables (paper tables):**
  - Taxonomy of VLA reward-hacking modes observed.
  - Which reward designs trigger which hacks (reproducible).
  - Mitigations (KL strength, reward shaping choices) and their measured effect.
  - Ideally a surprising finding (e.g. shaping that helps LLMs hurts VLAs).

---

## What runs where
| | Local Mac | GPU |
|---|---|---|
| Write/validate code | ✅ | ✅ |
| Load SmolVLA (fwd pass) | ✅ (MPS, slow) | ✅ |
| LIBERO rollouts (MuJoCo) | ⚠️ painful | ✅ preferred |
| GRPO training | ❌ | ✅ |

## Setup (GPU)
```bash
conda create -n vla python=3.10 -y && conda activate vla
pip install torch torchvision "lerobot[smolvla]"
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git && pip install -e LIBERO
# (LIBERO deps: robosuite, mujoco, etc.)
```

## Open design questions (resolve as we go)
- SmolVLA action head is flow-matching (no log-prob) → attach Gaussian head for
  GRPO, init mean ≈ flow action (keeps SFT skill). See
  `02_openvla_oft_grpo/proto_gaussian_head.py`.
- Does SmolVLA have a LIBERO-finetuned checkpoint, or do we SFT it first?
- LIBERO action/obs spec ↔ SmolVLA I/O mapping.
