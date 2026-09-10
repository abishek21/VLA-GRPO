# Pod Artifacts — Study 05 SFT Baseline (smoke run)

Pulled from RunPod pod `cy2g8xo165mslz` (RTX A6000) before termination, 2026-09-10.
Checkpoint: `Haozhan72/Openvla-oft-SFT-libero10-trajall` on LIBERO-Long (libero_10),
eval-only (`val_only=True`), **10 trials/task = 100 episodes** (smoke).

## Contents
- `eval_smoke.sh` — the exact eval script used (copy of `run_openvla_oft_rl_libero.sh`
  with: `val_only=True`, `num_trials_per_task=10`, `val_batch_size=100`,
  `val_micro_batch_size=10`, `logger=['console']`, wandb disabled).
- `align.json` — Ray runtime env (MUJOCO_GL=egl, wandb disabled).
- `smoke_log.txt` — full console log. Suite metric: `val/test_score/libero_10 = 0.88`.
- `results_manifest.csv` — `task,trial,success` for all 100 episodes (source of truth).
- `rob_rollout.patched.py` — the fork→spawn + traceback patched file (reference;
  also baked into `../Dockerfile`).
- `rollouts/` — all 100 rollout MP4s (filenames encode task/trial/success).

## Per-task result (recomputed from results_manifest.csv)
| task | success | task | success |
|------|--------:|------|--------:|
| task_0 | 90% | task_5 | 100% |
| task_1 | 100% | task_6 | 80% |
| task_2 | 80% | task_7 | **70% (weakest)** |
| task_3 | 80% | task_8 | 90% |
| task_4 | 90% | task_9 | 100% |
| **Average** | | | **88.0% (88/100)** |

Reproduce the table:
```bash
python3 -c "import csv,collections;\
t=collections.Counter();s=collections.Counter();\
[ (t.update([int(r['task'])]), s.update({int(r['task']):(r['success']=='True')})) for r in csv.DictReader(open('results_manifest.csv'))];\
[print(f'task_{k}: {s[k]}/{t[k]}') for k in sorted(t)]"
```

Note: smoke numbers have ±~15% per-task noise (n=10). Full baseline = 50/task
(500 eps): set `num_trials_per_task=50`, `val_batch_size=500`, `val_micro_batch_size=10`.
