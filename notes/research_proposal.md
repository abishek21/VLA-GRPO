# Research Proposal: Human-Derived Interaction Rewards for VLA Post-Training

*Living document. Last updated: 2026-08-27. Refine milestones as we hit them.*

---

## 0. One-paragraph plain-language summary

Today's robot foundation models (VLAs) are trained either by **imitation
learning** (blindly copy human teleoperated demos) or by **hand-written reward
functions** (someone manually defines "+1 if object in bin"). Imitation can't
teach recovery from mistakes (demos are too clean), and hand-written rewards are
sparse and easy to game. We propose a third path: **learn the reward from
egocentric human videos** by detecting universal *physical interaction events*
— grasp, contact, release, and failure/recovery — and use that as a **dense,
embodiment-agnostic process reward** to RL-fine-tune (GRPO) a VLA. Because
"a grasp happened" is meaningful for a human hand *and* a robot gripper, a reward
learned from cheap, abundant human video can potentially guide many robots and,
crucially, teach the **failure-and-recovery** behaviors that current VLAs lack.

---

## 1. Research question

> **Can task-conditioned physical interaction events learned from egocentric
> human demonstrations provide robust, cross-embodiment process rewards for VLA
> reinforcement fine-tuning — particularly for manipulation failure and
> recovery?**

Sub-questions:
1. Does an interaction-event reward improve VLA task success over the SFT
   baseline in simulation?
2. How much does the **human→robot embodiment gap** degrade the reward's
   usefulness? (Quantify it.)
3. Where and how does the policy **reward-hack** the learned reward (exploit sim
   artifacts / trigger event detectors without real task progress)?
4. Does the dense event reward specifically improve **failure detection and
   recovery** compared to sparse success-only rewards?

---

## 2. Why now / why this is in-trend (positioning)

Adjacent threads are all active *simultaneously* (examples from arXiv cs.RO,
late Aug 2026 — replace with full citations when writing):
- **Human-video → action/world models:** e.g. "Zero-WAM: In-Context
  World-Action Modeling from Human Videos."
- **RL post-training of robot policies:** e.g. "R³: Training Robots to Reason
  via RL."
- **Contact as a first-class signal:** e.g. "VISTA: Visually Inferred Spatial
  Contact Attention," "TacForcing."
- **Event/progress process signals:** e.g. "LM-X: Progress, Event, and
  Uncertainty Prediction."
- **Cross-embodiment:** e.g. "One Policy, Many Embodiments."
- **Reward learning for robots:** e.g. "Listwise VL Supervision for
  Preference-Based Reward Learning."
- **Human-video representation/reward priors (older, foundational):** R3M, MVP,
  VIP, LIV, TCN.

**Our defensible gap:** none of the above combine (interaction-*event* reward)
× (used as a *process* reward for VLA *RL post-training*) × (studied for
*cross-embodiment transfer*) × (with an explicit focus on *failure/recovery* and
*reward-hacking analysis*). The failure/recovery + reward-hacking-characterization
lens is the differentiator that is hard to scoop because it is an *insight*, not
a leaderboard number.

---

## 3. Method overview

```
[Egocentric human video]  --(RGB + Hand3D)-->  Interaction Event Detector
                                                (contact / grasp / release / failure)
                                                        |
                                                        v
                                              Interaction Reward r_event
                                                        |
[VLA policy πθ (SmolVLA/π0.5-class)]  --sample action chunks-->  rollouts in SIM
                                                        |
                                          GRPO update using r_event  (+ KL leash)
                                                        |
                              Evaluate vs SIM ground-truth success/failure oracle
```

Key design choices (de-risking):
- **Off-the-shelf small VLA** (SmolVLA or comparable) — do NOT build a model.
- **Reuse existing components** for event detection where possible (hand-pose /
  contact estimators) rather than training from scratch.
- **Evaluate in simulation** with a **ground-truth success/failure oracle**, so
  we can measure the *gap* between "reward model says success/failure" and
  "actually succeeded/failed." That gap = transfer quality AND reward-hacking,
  quantified. This oracle is the empirical heart of the paper — it must not be
  hand-wavy.

---

## 4. The five-step plan (with milestones and go/no-go gates)

### Step 1 — Get a small VLA running + fine-tuning (Weeks 1–2)
- Pick **SmolVLA** (or π0.5-class). Reproduce a basic SFT / eval loop in sim.
- **Gate G1:** can load the model, run a rollout in sim, and read a
  success flag. *If not, stop and fix before anything else.*

### Step 2 — Reproduce a VLA RL post-training method (Weeks 2–4)
- Plug our **GRPO loop** (already built from scratch) onto the small VLA over
  sampled action chunks, using the **sim's built-in success flag** as a
  placeholder reward.
- **Gate G2:** GRPO produces *any* positive success-rate delta over SFT on one
  task suite, with stable KL (no divergence). Confirms the RL machinery works
  *before* we introduce the learned reward.

### Step 3 — Build the interaction-event detector (Weeks 4–6)
- Use a small, well-annotated egocentric subset (**HoloAssist** or similar) —
  NOT the full 100K set yet.
- Train/adapt: `RGB + Hand3D -> {contact, grasp, release, failure}`.
- **Gate G3:** detector achieves usable precision/recall on held-out human
  clips. **Risk flag:** verify which interaction events are actually *labeled*
  in the dataset before committing; budget any hand-labeling early.

### Step 4 — Turn the detector into a reward + validate on robot data (Weeks 6–8)
- Define `r_event` from detected events (dense, per-step process reward).
- Sanity-check it on *robot* rollouts against the sim oracle: does high r_event
  correlate with true success? Measure the **embodiment gap**.
- **Gate G4:** r_event is positively (even if imperfectly) correlated with
  ground-truth success on robot rollouts.

### Step 5 — GRPO with the interaction reward, in sim (Weeks 8–11)
- Replace the placeholder reward with `r_event`. Run GRPO.
- Measure: (a) success-rate delta vs SFT and vs sparse-reward RL, (b) the
  **failure/recovery** improvement, (c) **reward-hacking** cases (r_event high
  but oracle says fail).
- **Gate G5:** a measurable, defensible result on at least one axis
  (success ↑, recovery ↑, or a characterized hacking/transfer finding).

### (Later, only if Steps 1–5 succeed)
- Scale human data, add OXE / cross-embodiment experiments, more task suites.

---

## 4b. Technical Design (detailed — no oversimplification)

This section spells out (A) the reward model, (B) how it is trained, and (C) how
its output drives VLA post-training. It deliberately surfaces the hard decisions.

### A. The reward model

**A.1 What an "interaction event" is (operational definitions).**
The reward model is a *per-timestep structured classifier* over physical contact
states, not a vague progress score. At each timestep `t` it predicts:

```
e_t = [ contact, grasp, release, failure ]   ∈ [0,1]^4
```

- **contact**: hand/gripper touching the target object (distance ≈ 0 + a
  force/penetration proxy).
- **grasp**: object secured and load-bearing (moves *with* the hand; stable
  enclosure over ≥ N frames).
- **release**: *intended* transition grasp → no-grasp with the object placed at
  a goal.
- **failure**: *unintended* loss of grasp / missed contact / slip (grasp →
  no-grasp WITHOUT goal placement).

The **release-vs-failure distinction is the crux of the paper**: both are "grasp
ended," but one is success and one is a mistake. Detecting that difference from
video is the novel, hard part.

**A.2 Inputs.**
```
x_t = ( RGB_{t-k:t}, Hand3D_t, [optional: object track] )
```
- RGB over a short window (contact is a *transition*, needs temporal context).
- Hand3D = 3D hand keypoints from an off-the-shelf estimator (HaMeR-class). This
  is the bridge to embodiment invariance: reasoning happens over *contact
  geometry* (fingertips → contact points), not raw appearance.
- Object track optional but disambiguates "contact with target" vs "with
  anything."

**A.3 Architecture (chosen: discriminative event head).**
```
x_{t-k:t} → frozen visual encoder (R3M/DINO/ViT) → temporal head (GRU/small TF) → e_t
```
- Keep the encoder **frozen** initially (standard human-video move; we lack data
  to train it). Interpretable event outputs → interpretable reward.
- Optional hybrid later: add a VIP/LIV-style time-contrastive *value* head for
  dense shaping between events, if events prove too sparse.

### B. Training the reward model

**B.1 Labels (the crux risk).** Sources, in order of preference:
1. **Existing annotations** (HoloAssist / EgoExo4D / Epic-Kitchens have some
   hand-object interaction / contact labels). **First action: audit which of
   {contact, grasp, release, failure} actually exist.** contact/grasp often do;
   *failure almost never does* — that gap is both our contribution and our
   labeling burden.
2. **Weak/auto-labels**: contact from hand-object distance + object motion;
   grasp from "object moves with hand ≥ N frames"; failure from "grasp lost AND
   object not at goal." Noisy but scalable.
3. **Hand-label a small failure/recovery set** (hundreds–low thousands of
   clips). Expensive; scope early (Gate G3).

**B.2 Losses (explicit).** Given predictions `ê_t` and labels `e_t`:

```
L_event      = Σ_t Σ_c  w_c · BCE( ê_{t,c}, e_{t,c} )     # w_c up-weights rare 'failure'
L_temporal   = Σ_t || ê_t − ê_{t−1} ||²  on state events   # events are piecewise-constant
L_transition = penalty on illegal transitions              # grammar: nocontact→contact→grasp→(release|failure)
L_value      = (optional) time-contrastive goal-distance monotonicity  (VIP/LIV)

L = L_event + λ1·L_temporal + λ2·L_transition (+ λ3·L_value)
```
Class imbalance is a real threat: **failure is rare**, so `w_failure` must be
large or a CRF/HMM layer used, else the model never predicts it.

**B.3 Embodiment gap, handled at training time (not assumed away).**
- Feed pose in a **shared canonical contact space** (fingertips/gripper tips →
  contact points), so the model reasons about geometry that transfers.
- Optionally domain-adapt on a small set of robot rollouts with pseudo-labels.
- **Measure the residual gap** as a result (Table C) — we quantify transfer, we
  don't assume it.

### C. Using the reward model for VLA post-training (GRPO)

**C.1 RL setup.**
- Policy: SmolVLA `πθ`. Obs `o = (images, instruction)`. Output: an **action
  chunk** `a_{t:t+H}`.
- Rollout `G` episodes from the *same start state* with stochastic action
  sampling → group `τ^1..τ^G` (the GRPO group).

**C.2 Reward model → scalar process reward (design choice, justify).**
Run the reward model on rendered frames + robot gripper pose each step:
```
ê_t = RewardModel(frames_{t-k:t}, GripperPose_t)

r_t =  w_c·Δcontact_t          # reward the ONSET (transition), not persistence
     + w_g·Δgrasp_t
     − w_f·failure_t           # penalty for unintended loss
     + w_r·recovery_t          # failure followed by re-grasp within a window  ← HEADLINE term
```
Use `Δ` (transition) rewards to prevent farming (e.g. "hold contact forever" or
"tap repeatedly"). `recovery_t` is the dense signal SFT and sparse-RL lack.

**C.3 Credit assignment (a real decision).**
- **Outcome GRPO (start here):** one scalar per rollout
  `R(τ^i) = Σ_t r_t + w_success·oracle_success`; then
  `A_i = (R_i − mean)/(std+eps)`, shared across all chunks — *identical* to the
  tiny-LLM GRPO already built.
- **Process GRPO (the contribution):** use per-step `r_t` for **per-chunk
  advantages** (chunks near grasp/recovery get credit; near failure get
  penalized). Harder (needs per-step returns/baselines); second-half research.

**C.4 The post-training loss (mirrors `01_tiny_llm_grpo`).**
```
A_i    = (R_i − mean(R)) / (std(R) + eps)        # or per-chunk
ratio  = πθ(a_i|o_i) / πθ_old(a_i|o_i)           # importance ratio over action chunks
L_surr = mean( min( ratio·A , clip(ratio,1±ε)·A ) )
L      = −( L_surr − β·KL(πθ || π_SFT) )         # KL leash to the SFT VLA
```
Same skeleton as `toy_grpo.py`. What changes: `πθ` is SmolVLA; "output" is a
continuous action chunk, so the ratio is over the **action distribution**
(Gaussian/flow head), NOT a token softmax — see D.1.

### D. Honest hard problems (open forks)

1. **Continuous-action log-probs (first engineering fork).** GRPO needs a
   sampling distribution with a tractable log-prob to form `ratio`. If SmolVLA's
   head is flow-matching/regression (deterministic-ish), we must use a native
   stochastic head or attach a Gaussian action head for RL. Resolve at Gate G2.
2. **Failure labels are scarce** — the recovery story depends on the least
   annotated event. Front-load the labeling audit (Gate G3).
3. **Embodiment gap is empirically unknown** — measured (Table C), not assumed.
4. **Reward hacking of the learned reward** — expected; it's a *feature* (Table
   D), quantified against the sim oracle.
5. **Rollout cost** — sampling `G` full episodes per update is far costlier than
   LLM generation. Keep `G` small (4–8), episodes short initially.

### One-diagram summary
```
HUMAN VIDEO (RGB + Hand3D)
   │ labels: contact/grasp/release/failure  (audit + weak + some hand-label)
   ▼
REWARD MODEL (frozen encoder + temporal event head)
   │ losses: weighted BCE + temporal + transition (+ value)
   ▼  validate vs SIM oracle → embodiment gap (Table C)
PROCESS REWARD r_t  (Δ-transition; failure penalty; recovery bonus)
   │
   ▼
SmolVLA πθ  --sample G action-chunk rollouts in LIBERO-->  GRPO (+KL leash to SFT)
   │
   ▼  eval: success↑ (A), recovery↑ (B), transfer gap (C), hacking (D)
```

---

## 5. What "a result" looks like (the paper's tables)

- **Table A — Success rate:** SFT vs sparse-RL vs interaction-reward-RL, per
  task suite.
- **Table B — Failure/recovery:** rate of successful recovery after an induced
  fumble; interaction reward vs baselines. *(Our headline claim.)*
- **Table C — Embodiment gap:** correlation of r_event with ground-truth
  success on human-like vs robot embodiments.
- **Table D — Reward hacking:** frequency and taxonomy of cases where r_event is
  high but the oracle says failure; effect of mitigations (KL, term balancing).

Any *one* of B/C/D being clean and honest is a contribution. All four is a
strong paper.

---

## 6. Risks and mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| Interaction events not labeled in dataset | high | verify labels first (Step 3 gate); scope hand-labeling early; start with contact/grasp which are easiest |
| Embodiment gap kills transfer | medium | measure it as a *result*, not a failure; frame paper around characterizing it |
| Reward hacking dominates | medium | it's a *feature* — analyze it (your strength); use sim oracle to quantify |
| 3-month timeline slips | high | front-load Gates G1/G2; use off-the-shelf model + reward components; keep full-data scaling as "future work" |
| Scooped (fast-moving field) | medium | lead with failure/recovery + hacking analysis (insight, not leaderboard); move fast on the differentiator |

---

## 7. Immediate next actions

### Findings log

**2026-08-28 — Gate G1 / G1.5 (local, Mac arm64 / MPS):**
- ✅ **G1:** SmolVLA (`lerobot/smolvla_base`) loads and runs on Mac via MPS,
  no GPU needed. Import path (lerobot 0.4.4):
  `lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy`. `pi05` is also
  available as a policy for the later π0.5 comparison.
- ⚠️ **G1.5 — action head is FLOW-MATCHING (key fork confirmed):**
  config shows `num_steps=10` (denoising), `chunk_size=50`, `n_action_steps=50`,
  `max_action_dim=32`. Only action method is `predict_action_chunk`; there is
  **no `sample()` / `log_prob()`**. => The GRPO importance ratio
  `πθ(a|o)/πθ_old(a|o)` is **not** available out of the box. (Earlier
  "gaussian" hint was a false positive matching the `GELUTanh` activation.)

**Implication / decision needed (D.1):** to run GRPO on SmolVLA we must obtain a
tractable action likelihood. Options to evaluate at Gate G2:
1. **Attach a small stochastic (Gaussian/tanh) action head** for RL fine-tuning
   on top of SmolVLA features; use its `log_prob` for the ratio. Simplest,
   well-understood; slight departure from the native flow policy.
2. **Flow/diffusion-policy RL**: use a surrogate ratio over the sampled noise /
   ELBO-style bound, or treat the denoising chain as the stochastic policy
   (à la diffusion-policy RL methods). More faithful, more complex.
3. **Reparam/consistency**: distill the flow head to a 1-step stochastic head
   for RL, keep the flow head for eval.

**Recommendation:** start with Option 1 at Gate G2 (fastest path to a working
GRPO loop over action chunks); revisit Option 2 if the Gaussian head degrades
the base policy too much.

### To-do

1. ☑ Confirm model choice — **SmolVLA** (done); π0.5 available for later.
2. ☑ Step 1 (G1): load model, run on device — done locally on Mac.
3. ☐ **G2 (GPU):** resolve D.1 (start Option 1), wire GRPO over action chunks
   using the LIBERO success flag as the placeholder reward.
4. ☐ Confirm SmolVLA observation schema + `select_action` obs dict (local).
5. ☐ In parallel, audit HoloAssist (or chosen dataset) for interaction-event
   labels (de-risks Step 3 early).

## 8. Relationship to earlier work in this repo

- `notes/grpo_math.md` — the RL math foundation (done).
- `01_tiny_llm_grpo/` — GRPO built from scratch; failure modes lived (done).
- `notes/blog_reward_hacking.md` — reward-hacking demo; the intellectual prequel
  to Table D here.
- **This proposal = Stage 2** of the repo roadmap, made concrete.
