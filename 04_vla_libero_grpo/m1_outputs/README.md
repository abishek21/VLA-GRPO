# M1 Outputs — OpenVLA-OFT on LIBERO-Spatial (2026-09-04)

Result: **19/20 = 95.0%** success (`libero_spatial`, 2 trials/task, 10 tasks).

## Contents
- `rollouts/2026_09_04/` — 20 MP4 replays, one per episode. Filenames encode the
  outcome: `...episode=<i>--success=<True|False>--task=<instruction>.mp4`.
- `logs/EVAL-libero_spatial-openvla-2026_09_04-21_30_11.txt` — the eval script's
  structured per-episode log.
- `logs/m1_spatial_smoke.log` — full console output of the run.

## Per-episode summary
| Ep | Task (spatial reference) | Success |
|----|--------------------------|:-------:|
| 1–2  | between the plate and the ramekin | ✅ ✅ |
| 3–4  | next to the ramekin | ✅ ✅ |
| 5–6  | from table center | ✅ ✅ |
| 7–8  | on the cookie box | ✅ ✅ |
| 9–10 | in the top drawer of the wooden cabinet | ✅ ✅ |
| 11–12| on the ramekin | ✅ ✅ |
| 13–14| next to the cookie box | ✅ ✅ |
| 15–16| on the stove | ✅ ✅ |
| 17–18| next to the plate | ✅ ✅ |
| 19–20| on the wooden cabinet | ❌ ✅ |

All tasks share the same pick-and-place ("...and place it on the plate"); only the
spatial reference changes — that is what `libero_spatial` tests. The single failure
(ep 19) was the "on the wooden cabinet" variant.
