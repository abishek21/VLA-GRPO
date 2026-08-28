# Stage 2 — SmolVLA + GRPO (Human-Derived Interaction Rewards)

Implementation of the research proposal in `notes/research_proposal.md`.
Model: **SmolVLA** (now), π0.5 (later). Sim: **LIBERO**.

---

## What runs where (be realistic)

| Task | Local Mac (arm64, MPS) | Rented GPU (CUDA) |
|------|------------------------|-------------------|
| Load SmolVLA, inspect action head | ✅ yes (CPU/MPS) | ✅ |
| Single forward pass (dummy obs) → action | ✅ yes (slow) | ✅ |
| Prototype reward-model code (shapes, losses) | ✅ yes | ✅ |
| LIBERO sim rollouts (MuJoCo render) | ⚠️ maybe (headless, fiddly) | ✅ preferred |
| GRPO training (G rollouts/update) | ❌ no (too slow) | ✅ required |
| Reward-model training on video | ❌ no | ✅ required |

**Local goal:** write + smoke-test all the *code* (model loading, action head,
reward-model modules, GRPO glue) so that on the GPU we only debug scale/CUDA,
not logic. Push from here, pull on the GPU box.

---

## The #1 question to answer locally (Gate G2 blocker)

**Does SmolVLA's action head give a tractable log-prob for the GRPO ratio?**

GRPO needs `ratio = πθ(a|o) / πθ_old(a|o)`, which requires a *sampling
distribution with a log-prob*. If SmolVLA uses flow-matching / deterministic
regression, we must either use a native stochastic head or attach a Gaussian
head for RL. `step1_inspect_smolvla.py` inspects this **without a GPU**.

---

## Local setup (do once)

Python 3.13 (your base) is too new for LeRobot. Make a 3.10 env:

```bash
conda create -n vla python=3.10 -y
conda activate vla

# PyTorch (Apple Silicon build with MPS)
pip install torch torchvision

# LeRobot + SmolVLA
pip install "lerobot[smolvla]"      # or: pip install lerobot && pip install smolvla deps

# misc
pip install -r 02_openvla_oft_grpo/requirements-local.txt
```

Then:
```bash
python 02_openvla_oft_grpo/step1_inspect_smolvla.py
```

## Milestones (from the proposal)
- [ ] **G1**: load SmolVLA, forward pass, read action shape (local).
- [ ] **G1.5**: determine action-head type + log-prob availability (local).
- [ ] **G2**: GRPO loop over action chunks with sim success flag (GPU).
- [ ] **G3**: interaction-event detector on human video subset (GPU).
- [ ] **G4**: reward correlates with sim oracle (GPU).
- [ ] **G5**: GRPO with interaction reward; measure success/recovery/hacking (GPU).
