# Method — Paper-style Algorithm Skeletons

Pseudocode for the two components. Written in the style of a paper's Algorithm
blocks (Require/Ensure, numbered steps). LaTeX renders in VS Code preview
(`Cmd+Shift+V`); the fenced versions are drop-in for an `algorithm` environment.

Notation:
- $\pi_\theta$ : VLA policy (SmolVLA + Gaussian action head). $\pi_\text{ref}$ : frozen SFT policy.
- $o$ : observation (images + instruction). $A=a_{1:H}$ : action chunk (horizon $H$).
- $R_\phi$ : interaction-event reward model. $e_t\in[0,1]^5$ : events
  $[\text{contact},\text{grasp},\text{release},\text{failure},\text{recovery}]$.
- $G$ : group size. $\varepsilon$ : PPO clip. $\beta$ : KL weight.

---

## Algorithm 1 — Interaction-Event Reward Model Training

$$
\begin{aligned}
&\textbf{Require: } \text{human clips } \{(V_i, H_i)\}\ (\text{RGB}, \text{Hand3D}),\ \text{event labels } \{e^i_{1:T}\},\ \text{frozen encoder } E \\
&\textbf{Ensure: } \text{reward model } R_\phi \\
&1:\ \text{compute class freqs } p_c \leftarrow \text{mean}_t\, e_{t,c};\quad w_c \leftarrow (1-p_c)/p_c \quad \triangleright\ \text{inverse-freq weights} \\
&2:\ \textbf{for } \text{minibatch } (V, H, e_{1:T}) \textbf{ do} \\
&3:\quad z^{\text{vis}}_t \leftarrow E(V_{t-k:t})\ \ \forall t \qquad \triangleright\ \text{frozen visual features} \\
&4:\quad z^{\text{hand}}_t \leftarrow \text{MLP}_\phi(H_t) \\
&5:\quad u_t \leftarrow \text{Fuse}_\phi([z^{\text{vis}}_t; z^{\text{hand}}_t]);\quad h_{1:T} \leftarrow \text{GRU}_\phi(u_{1:T}) \\
&6:\quad \hat{e}_t \leftarrow \sigma(\text{Linear}_\phi(h_t)) \\
&7:\quad \mathcal{L}_{\text{evt}} \leftarrow \textstyle\sum_{t,c} w_c\,\text{BCE}(\hat{e}_{t,c}, e_{t,c})\,(1-\hat p_{t,c})^\gamma \quad \triangleright\ \text{weighted focal} \\
&8:\quad \mathcal{L}_{\text{tmp}} \leftarrow \textstyle\sum_{t}\lVert \hat e^{\text{state}}_t - \hat e^{\text{state}}_{t-1}\rVert^2 \quad \triangleright\ \text{state events only} \\
&9:\quad \mathcal{L}_{\text{trn}} \leftarrow \textstyle\sum_t \hat e^{\text{grasp}}_t\,\text{relu}(1-\max_{t'\le t}\hat e^{\text{contact}}_{t'}) \\
&10:\quad \phi \leftarrow \phi - \eta\,\nabla_\phi(\mathcal{L}_{\text{evt}} + \lambda_1\mathcal{L}_{\text{tmp}} + \lambda_2\mathcal{L}_{\text{trn}}) \\
&11:\ \textbf{end for}
\end{aligned}
$$

```text
Algorithm 1: Interaction-Event Reward Model Training
Require: human clips {(V_i, H_i)} (RGB, Hand3D); event labels {e^i_{1:T}};
         frozen visual encoder E (R3M)
Ensure:  reward model R_phi = {MLP, Fuse, GRU, Linear}
 1: p_c   <- mean_t e_{t,c}              # per-event active fraction
 2: w_c   <- (1 - p_c) / p_c             # inverse-frequency class weights
 3: for minibatch (V, H, e_{1:T}) do
 4:     z_vis_t  <- E(V_{t-k:t})   for all t      # frozen
 5:     z_hand_t <- MLP_phi(H_t)
 6:     u_t      <- Fuse_phi([z_vis_t ; z_hand_t])
 7:     h_{1:T}  <- GRU_phi(u_{1:T})
 8:     e_hat_t  <- sigmoid(Linear_phi(h_t))
 9:     L_evt <- sum_{t,c} w_c * focal_BCE(e_hat_{t,c}, e_{t,c})
10:     L_tmp <- sum_t || e_hat_state_t - e_hat_state_{t-1} ||^2   # state events
11:     L_trn <- sum_t e_hat_grasp_t * relu(1 - max_recent contact) # grammar
12:     phi <- phi - eta * grad_phi (L_evt + l1*L_tmp + l2*L_trn)
13: end for
```

---

## Algorithm 2 — GRPO VLA Post-Training with the Learned Reward

$$
\begin{aligned}
&\textbf{Require: } \text{SFT policy } \pi_\text{ref},\ \text{reward model } R_\phi,\ \text{sim env},\ G,\ \varepsilon,\ \beta \\
&\textbf{Ensure: } \text{post-trained policy } \pi_\theta \\
&1:\ \pi_\theta \leftarrow \text{attach Gaussian head to } \pi_\text{ref};\ \ \text{init } \mu\approx\text{flow action} \quad \triangleright\ \text{keep SFT skill} \\
&2:\ \textbf{for } \text{iteration} = 1,2,\dots \textbf{ do} \\
&3:\quad \pi_{\theta_\text{old}} \leftarrow \pi_\theta \\
&4:\quad \text{sample start state } s_0 \\
&5:\quad \textbf{for } i=1..G \textbf{ do} \qquad \triangleright\ G\ \text{full rollouts, obs diverge} \\
&6:\qquad \text{rollout } \tau^i=(o^i_0,A^i_0,o^i_1,\dots)\ \text{with } A^i_j\sim\pi_{\theta_\text{old}}(\cdot\mid o^i_j) \\
&7:\qquad \hat e^i_t \leftarrow R_\phi(\text{frames}(\tau^i), \text{grip}(\tau^i)) \\
&8:\qquad r^i_t \leftarrow w_g\Delta\text{grasp} + w_c\Delta\text{contact} - w_f\text{failure} + w_r\text{recovery} \\
&9:\qquad R^i \leftarrow \textstyle\sum_t r^i_t + w_s\,\text{success}(\tau^i) \\
&10:\quad \textbf{end for} \\
&11:\quad A^i \leftarrow (R^i - \text{mean}(R))/(\text{std}(R)+\epsilon) \quad \triangleright\ \text{group-relative; } 0\ \text{if degenerate} \\
&12:\quad \textbf{for } \text{PPO epoch} = 1..E \textbf{ do} \\
&13:\qquad \rho^i_j \leftarrow \pi_\theta(A^i_j\mid o^i_j)/\pi_{\theta_\text{old}}(A^i_j\mid o^i_j) \\
&14:\qquad \mathcal{L}_\text{clip} \leftarrow \text{mean}_{i,j}\min(\rho^i_j A^i,\ \text{clip}(\rho^i_j,1{\pm}\varepsilon)A^i) \\
&15:\qquad \mathcal{L}_\text{KL} \leftarrow \widehat{\text{KL}}(\pi_\theta\,\Vert\,\pi_\text{ref}) \quad \triangleright\ \text{per-token} \\
&16:\qquad \theta \leftarrow \theta + \eta\,\nabla_\theta(\mathcal{L}_\text{clip} - \beta\mathcal{L}_\text{KL}) \\
&17:\quad \textbf{end for} \\
&18:\ \textbf{end for}
\end{aligned}
$$

```text
Algorithm 2: GRPO VLA Post-Training with Learned Interaction Reward
Require: SFT policy pi_ref; reward model R_phi; sim env; group size G;
         clip eps; KL weight beta
Ensure:  post-trained policy pi_theta
 1: pi_theta <- attach Gaussian head to pi_ref; init mu ~ flow action  # keep SFT skill
 2: for iteration = 1, 2, ... do
 3:     pi_theta_old <- pi_theta                      # frozen sampling anchor
 4:     sample start state s0
 5:     for i = 1..G do                               # G FULL rollouts (obs diverge)
 6:         roll out tau^i with A^i_j ~ pi_theta_old(.|o^i_j)
 7:         e_hat^i_t <- R_phi(frames(tau^i), gripper_pose(tau^i))
 8:         r^i_t <- w_g*Δgrasp + w_c*Δcontact - w_f*failure + w_r*recovery
 9:         R^i   <- sum_t r^i_t + w_s * success(tau^i)   # + sim oracle
10:     end for
11:     A^i <- (R^i - mean(R)) / (std(R) + eps)        # 0 if zero-variance group
12:     for ppo_epoch = 1..E do
13:         rho^i_j <- pi_theta(A^i_j|o^i_j) / pi_theta_old(A^i_j|o^i_j)
14:         L_clip  <- mean_{i,j} min(rho*A^i, clip(rho,1±eps)*A^i)
15:         L_kl    <- per_token_KL(pi_theta || pi_ref)
16:         theta   <- theta + eta * grad(L_clip - beta*L_kl)
17:     end for
18: end for
```

---

## Notes / correspondence to code

- **Alg. 1** ↔ `03_reward_model/{reward_model,losses,dataset,build_event_labels}.py`.
  Lines 9–11 are `losses.total_loss`; lines 4–8 are `InteractionRewardModel.forward`.
- **Alg. 2** ↔ `01_tiny_llm_grpo/toy_grpo.py` skeleton, with:
  `seq_logprob` → Gaussian-head `log_prob` (`02_openvla_oft_grpo/proto_gaussian_head.py`),
  reward → `R_phi` (Alg. 1), group = G full rollouts (not G completions).
- **Ablations the algorithms expose (paper):** outcome vs process advantage
  (line 8–9 vs per-chunk), reward terms $w_\cdot$ (Table D reward hacking),
  encoder R3M vs DINOv2 (Alg. 1 line 4), recovery term on/off (Table B).
