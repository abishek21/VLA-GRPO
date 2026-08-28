"""
Step 1 (local, no GPU needed): Load SmolVLA and inspect its action head.

Goals:
  G1   : confirm we can load the model and run a forward pass.
  G1.5 : determine HOW the policy produces actions, because that decides whether
         GRPO's importance ratio  rho = pi_theta(a|o) / pi_theta_old(a|o)  is
         tractable out of the box.

Why this matters (the first real research fork):
  GRPO needs a *sampling distribution with a log-prob*. If SmolVLA's action head
  is:
    (a) a flow-matching / diffusion head  -> no simple closed-form log-prob
    (b) deterministic regression (L1/MSE) -> no stochasticity to form a ratio
    (c) a Gaussian / tokenized head       -> log-prob is available -> GRPO clean
  The result tells us whether we use the native head or must attach a stochastic
  (e.g. Gaussian) action head for RL.

Run:
  conda activate vla
  python 02_openvla_oft_grpo/step1_inspect_smolvla.py

This script is intentionally defensive: it prints guidance instead of crashing
if SmolVLA/LeRobot isn't installed yet, so you can read the plan first.
"""

import sys
import platform


def banner(msg):
    print("\n" + "=" * 70 + f"\n{msg}\n" + "=" * 70)


def report_env():
    banner("Environment")
    print("python :", sys.version.split()[0])
    print("machine:", platform.machine(), "|", platform.platform())
    try:
        import torch
        print("torch  :", torch.__version__)
        print("cuda   :", torch.cuda.is_available())
        print("mps    :", getattr(torch.backends, "mps", None) is not None
              and torch.backends.mps.is_available())
    except ImportError:
        print("torch  : NOT INSTALLED  -> pip install torch torchvision")
        return None
    # pick device: prefer cuda, then mps, then cpu
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def try_load_smolvla(device):
    """Attempt to load SmolVLA via LeRobot. Returns the policy or None."""
    banner("Loading SmolVLA")
    try:
        # LeRobot's SmolVLA policy. The module path changed across versions;
        # try the current (0.4.x) path first, then the older `common` path.
        SmolVLAPolicy = None
        for modpath in (
            "lerobot.policies.smolvla.modeling_smolvla",       # lerobot >= 0.4
            "lerobot.common.policies.smolvla.modeling_smolvla",  # older
        ):
            try:
                mod = __import__(modpath, fromlist=["SmolVLAPolicy"])
                SmolVLAPolicy = getattr(mod, "SmolVLAPolicy")
                print("imported SmolVLAPolicy from", modpath)
                break
            except Exception:  # noqa: BLE001
                continue
        if SmolVLAPolicy is None:
            raise ImportError("SmolVLAPolicy not found on known module paths")
    except Exception as e:  # noqa: BLE001
        print("Could not import SmolVLAPolicy from lerobot.")
        print("  reason:", repr(e))
        print("\nInstall with:  pip install \"lerobot[smolvla]\"")
        print("Then re-run. (You can still read the rest of this file as a plan.)")
        return None

    # Public pretrained checkpoint (base SmolVLA). Update if the id differs.
    ckpt = "lerobot/smolvla_base"
    try:
        print(f"from_pretrained({ckpt}) ...")
        policy = SmolVLAPolicy.from_pretrained(ckpt)
        policy.to(device)
        policy.eval()
        print("loaded OK on", device)
        return policy
    except Exception as e:  # noqa: BLE001
        print("Load failed:", repr(e))
        print("Check the exact checkpoint id on HF: https://huggingface.co/lerobot")
        return None


def inspect_action_head(policy):
    """Introspect modules to classify the action head + log-prob availability."""
    banner("Action-head inspection (the GRPO fork)")

    # 1) list submodule class names -> look for diffusion/flow/gaussian hints
    names = [type(m).__name__ for _, m in policy.named_modules()]
    joined = " ".join(names).lower()

    hints = {
        "flow/diffusion": any(k in joined for k in
                              ["flow", "diffusion", "ddpm", "ddim", "vf", "velocity"]),
        "gaussian/normal": any(k in joined for k in ["gaussian", "normal", "tanh"]),
        "tokenized/discrete": any(k in joined for k in
                                  ["tokenizer", "vocab", "categorical", "logits"]),
    }
    print("module-name hints:")
    for k, v in hints.items():
        print(f"  {k:20s}: {v}")

    # 2) look for a log-prob-like method on the policy
    api = [a for a in dir(policy) if any(
        s in a.lower() for s in ["log_prob", "logprob", "sample", "distribution", "predict_action"]
    )]
    print("\nrelevant methods/attrs on policy:", api or "(none obvious)")

    # 3) verdict
    banner("Verdict / next step")
    if hints["gaussian/normal"] or hints["tokenized/discrete"]:
        print("Likely a distribution WITH a log-prob -> GRPO ratio is tractable.")
        print("Next: confirm sample() + log_prob() exist; wire them into GRPO.")
    elif hints["flow/diffusion"]:
        print("Likely FLOW/DIFFUSION head -> NO simple log-prob.")
        print("Options for RL:")
        print("  (i)  attach a Gaussian action head for RL fine-tuning, or")
        print("  (ii) use a surrogate ratio (e.g. on the sampled noise / a")
        print("       stochastic wrapper). This is the first design decision.")
    else:
        print("Head type unclear from names. Inspect policy.forward / config next;")
        print("print(policy) and read the action-head module directly.")


def dummy_forward(policy, device):
    """Run a single forward pass with a synthetic observation, if we can."""
    banner("Dummy forward pass")
    try:
        import torch
        # SmolVLA expects an observation dict: image(s) + state + task string.
        # Exact keys depend on the version; we probe policy.config if present.
        cfg = getattr(policy, "config", None)
        print("policy.config type:", type(cfg).__name__ if cfg else "(none)")
        print("Skipping a real forward until obs schema is confirmed from config.")
        print("TODO(G1): build obs dict per SmolVLA schema and call select_action().")
    except Exception as e:  # noqa: BLE001
        print("forward probe failed:", repr(e))


def main():
    device = report_env()
    if device is None:
        return
    policy = try_load_smolvla(device)
    if policy is None:
        return
    inspect_action_head(policy)
    dummy_forward(policy, device)
    banner("Done")
    print("Record findings in notes/research_proposal.md (Gate G1 / G1.5).")


if __name__ == "__main__":
    main()
