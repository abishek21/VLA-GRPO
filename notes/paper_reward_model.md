# Interaction-Event Reward Model — Method (paper draft)

*Draft of the reward-model section. Written in paper prose so you can read it
end-to-end. Pairs with Algorithm 1 in `notes/algorithms.md` and the code in
`03_reward_model/`.*

---

## 3. Learning Interaction-Event Rewards from Human Video

### 3.1 Motivation and overview

Reinforcement fine-tuning of vision–language–action (VLA) policies is
bottlenecked by the reward signal. Sparse task-success rewards provide almost no
guidance within an episode, and hand-engineered dense rewards are brittle and
easily exploited. We instead **learn a dense, structured reward from egocentric
human manipulation video**, exploiting the observation that the *physical
interaction events* that define good manipulation — making contact, forming a
grasp, releasing, and, critically, *failing and recovering* — are (i) common in
human video, (ii) semantically embodiment-agnostic, and (iii) partially
pre-annotated in existing datasets.

Concretely, we train a model $R_\phi$ that maps a short window of egocentric RGB
and 3D hand pose to a vector of per-timestep interaction-event probabilities

$$
\hat e_t = R_\phi\big(V_{t-k:t},\, H_t\big) \in [0,1]^5,\qquad
\hat e_t = [\text{contact},\text{grasp},\text{release},\text{failure},\text{recovery}].
$$

At post-training time (Section 4), these event predictions are converted into a
dense process reward for a VLA policy acting in simulation. This section
describes the reward model itself: its inputs and label construction (3.2), its
architecture (3.3), and its training objective (3.4).

### 3.2 Data and label construction

We build $R_\phi$ from **HoloAssist** [Wang et al., 2023], a 169-hour egocentric
dataset of two-person collaborative physical-manipulation tasks. HoloAssist is
uniquely suited to our goal for two reasons. First, it ships **synchronized 3D
hand pose** alongside RGB, giving us $H_t$ directly without a pose estimator.
Second, every *fine-grained action* segment is annotated with an **Action
Correctness** label — one of *correct*, *wrong (corrected by performer)*,
*wrong (corrected by instructor)*, or *wrong (not corrected)* — together with a
free-text explanation and a verb/noun description. This provides direct,
time-stamped supervision for the two events that are otherwise the hardest to
obtain: **failure** and **recovery**.

**Event labels.** Each annotation is a time interval $[s,e]$ with attributes. We
rasterize annotations onto a fixed $10\,\mathrm{Hz}$ grid (HoloAssist's native
rate) to obtain per-timestep binary targets $e_{t}\in\{0,1\}^5$:

- *State events* (contact, grasp, release) are derived from the action **verb**
  (e.g. `grab`$\rightarrow$grasp; `insert`,`screw`,`place`$\rightarrow$contact;
  `withdraw`,`remove`$\rightarrow$release) and are set for the full duration of
  the segment.
- *Transition events* (failure, recovery) are derived from **Action
  Correctness**: `Wrong Action*`$\rightarrow$failure, and specifically
  *corrected by performer*$\rightarrow$recovery (self-recovery). Because these
  denote onsets rather than sustained states, we activate them over a short
  window around the transition.

This yields a heavily imbalanced multi-label target: in our training subset,
per-frame positive rates are approximately contact $51\%$, grasp $20\%$, release
$3\%$, failure $4\%$, recovery $1\%$. The rarity of failure/recovery motivates
the weighting scheme in Section 3.4.

**Subset selection.** As the raw streams total $\sim$370 GB, we select a
$60$-session subset (RGB + hand pose, $\sim$23 GB) ranked by wrong-action
density, concentrating the rare failure/recovery supervision. The selected
sessions are assembly/disassembly tasks (utility carts, stools, camera rigs),
which are contact-rich and exhibit frequent grasp failures (notably object
*drops*), the failure mode most relevant to robot manipulation.

### 3.3 Architecture

$R_\phi$ has four components; only the last three are trained.

**Frozen visual encoder.** Each RGB frame is encoded by a pretrained,
**frozen** encoder $E$. We use R3M [Nair et al., 2022], a ResNet trained with
time-contrastive and language objectives on egocentric human manipulation video
(Ego4D), so its features are pre-tuned for hand–object interaction. Freezing $E$
removes the need to learn vision from scratch and keeps training cheap; the
encoder is swappable (we also consider DINOv2 as a general-purpose ViT
alternative in ablations).

**Hand encoder.** The per-frame 3D hand keypoints $H_t$ are embedded by a small
MLP. Reasoning over hand geometry (fingertip/contact-point configuration) rather
than raw appearance is the mechanism by which the learned reward is intended to
transfer across embodiments (human hand $\rightarrow$ robot gripper).

**Temporal head.** Visual and hand features are fused per timestep and passed
through a recurrent head (GRU) that models the sequence. Interaction events are
inherently temporal — a grasp is a hand closing *followed by* the object moving
with the hand — so a single frame is insufficient; the temporal head provides
the necessary context window.

**Event classifier.** A linear layer with per-event sigmoids produces
$\hat e_t\in[0,1]^5$. We use independent sigmoids rather than a softmax because
events are not mutually exclusive (e.g. contact and grasp co-occur).

Formally, for a clip of length $T$:
$$
z^{\text{vis}}_t = E(V_{t-k:t}),\quad
z^{\text{hand}}_t = \mathrm{MLP}(H_t),\quad
u_t = \mathrm{Fuse}([z^{\text{vis}}_t; z^{\text{hand}}_t]),
$$
$$
h_{1:T} = \mathrm{GRU}(u_{1:T}),\qquad
\hat e_t = \sigma(\mathrm{Linear}(h_t)).
$$

### 3.4 Training objective

We train $\phi$ with a composite loss combining a class-balanced classification
term with two light structural priors:

$$
\mathcal{L} = \mathcal{L}_{\text{evt}}
 + \lambda_1\,\mathcal{L}_{\text{tmp}}
 + \lambda_2\,\mathcal{L}_{\text{trn}}.
$$

**Class-weighted focal BCE.** To prevent the rare events from being ignored, we
use per-event positive weights $w_c=(1-p_c)/p_c$ (inverse frequency; e.g.
$w_{\text{recovery}}\!\approx\!99$) inside a focal binary cross-entropy:
$$
\mathcal{L}_{\text{evt}} = \sum_{t,c} w_c\,(1-\hat p_{t,c})^{\gamma}\,
\mathrm{BCE}(\hat e_{t,c}, e_{t,c}),
$$
where $\hat p_{t,c}$ is the model's probability of the true class and $\gamma$
the focal factor. This is the component that makes learning the $1\%$ recovery
event tractable.

**Temporal smoothness.** State events are piecewise-constant, so we penalize
frame-to-frame change on those channels only (transition events are excluded so
their fast onsets are not suppressed):
$$
\mathcal{L}_{\text{tmp}} = \sum_t \big\lVert \hat e^{\text{state}}_t - \hat e^{\text{state}}_{t-1} \big\rVert_2^2 .
$$

**Transition prior.** A grasp should be supported by recent contact. We add a
soft, differentiable grammar penalty that discourages grasp probability without
nearby contact support:
$$
\mathcal{L}_{\text{trn}} = \sum_t \hat e^{\text{grasp}}_t \cdot
\mathrm{relu}\!\Big(1 - \max_{t-w\le t'\le t}\hat e^{\text{contact}}_{t'}\Big).
$$

The full training procedure is summarized in Algorithm 1.

---

**Algorithm 1 — Interaction-Event Reward Model Training**

$$
\begin{aligned}
&\textbf{Require: } \text{human clips } \{(V_i, H_i)\}\ (\text{RGB},\text{Hand3D}),\ \text{event labels } \{e^i_{1:T}\},\ \text{frozen encoder } E \\
&\textbf{Ensure: } \text{reward model } R_\phi=\{\mathrm{MLP},\mathrm{Fuse},\mathrm{GRU},\mathrm{Linear}\} \\
&1:\ p_c \leftarrow \mathrm{mean}_t\, e_{t,c};\quad w_c \leftarrow (1-p_c)/p_c \qquad \triangleright\ \text{inverse-freq class weights} \\
&2:\ \textbf{for } \text{minibatch } (V,H,e_{1:T}) \textbf{ do} \\
&3:\quad z^{\text{vis}}_t \leftarrow E(V_{t-k:t})\ \ \forall t \qquad \triangleright\ \text{frozen} \\
&4:\quad z^{\text{hand}}_t \leftarrow \mathrm{MLP}_\phi(H_t) \\
&5:\quad u_t \leftarrow \mathrm{Fuse}_\phi([z^{\text{vis}}_t; z^{\text{hand}}_t]);\quad h_{1:T} \leftarrow \mathrm{GRU}_\phi(u_{1:T}) \\
&6:\quad \hat e_t \leftarrow \sigma(\mathrm{Linear}_\phi(h_t)) \\
&7:\quad \mathcal{L}_{\text{evt}} \leftarrow \textstyle\sum_{t,c} w_c\,(1-\hat p_{t,c})^{\gamma}\,\mathrm{BCE}(\hat e_{t,c}, e_{t,c}) \\
&8:\quad \mathcal{L}_{\text{tmp}} \leftarrow \textstyle\sum_t \lVert \hat e^{\text{state}}_t-\hat e^{\text{state}}_{t-1}\rVert_2^2 \qquad \triangleright\ \text{state events only} \\
&9:\quad \mathcal{L}_{\text{trn}} \leftarrow \textstyle\sum_t \hat e^{\text{grasp}}_t\,\mathrm{relu}(1-\max_{t'\le t}\hat e^{\text{contact}}_{t'}) \\
&10:\quad \phi \leftarrow \phi - \eta\,\nabla_\phi(\mathcal{L}_{\text{evt}} + \lambda_1\mathcal{L}_{\text{tmp}} + \lambda_2\mathcal{L}_{\text{trn}}) \\
&11:\ \textbf{end for}
\end{aligned}
$$

```text
Algorithm 1: Interaction-Event Reward Model Training
Require: human clips {(V_i,H_i)} (RGB, Hand3D); event labels {e^i_{1:T}};
         frozen visual encoder E (R3M)
Ensure:  reward model R_phi = {MLP, Fuse, GRU, Linear}
 1: p_c <- mean_t e_{t,c};  w_c <- (1 - p_c)/p_c        # inverse-freq weights
 2: for minibatch (V, H, e_{1:T}) do
 3:     z_vis_t  <- E(V_{t-k:t})  for all t             # frozen encoder
 4:     z_hand_t <- MLP_phi(H_t)
 5:     u_t      <- Fuse_phi([z_vis_t ; z_hand_t]); h_{1:T} <- GRU_phi(u_{1:T})
 6:     e_hat_t  <- sigmoid(Linear_phi(h_t))
 7:     L_evt <- sum_{t,c} w_c (1 - p_hat_{t,c})^gamma BCE(e_hat_{t,c}, e_{t,c})
 8:     L_tmp <- sum_t || e_hat_state_t - e_hat_state_{t-1} ||^2   # state only
 9:     L_trn <- sum_t e_hat_grasp_t * relu(1 - max_recent contact)
10:     phi <- phi - eta * grad_phi (L_evt + l1 L_tmp + l2 L_trn)
11: end for
```

---

### 3.5 Validation protocol (bridge to Section 4)

We evaluate $R_\phi$ on held-out human sessions using per-event precision/recall
and F1, with particular attention to failure/recovery. Crucially, because these
events will drive VLA post-training in a simulator that provides a **ground-truth
success/failure oracle**, we can later measure the *gap* between what the reward
model predicts and what actually occurs on robot rollouts (Section 4). This gap
quantifies both cross-embodiment transfer and susceptibility to reward hacking,
and is a central empirical contribution of this work.

---

### Notes for later (not paper prose)
- Numbers above (51/20/3/4/1 %) are from `build_event_labels.py` on the subset —
  update with the final full-subset stats once trained.
- Cite exact HoloAssist / R3M / DINOv2 references; add Ego4D for R3M pretraining.
- Fill architecture dims (encoder out-dim, GRU width) from the final config.
- Figure: the one-diagram pipeline from `research_proposal.md` §4b.
