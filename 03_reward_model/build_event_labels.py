"""
Convert HoloAssist video `events` into a per-timestep interaction-event label
tensor  e_t = [contact, grasp, release, failure, recovery]  at 10 Hz.

Design decisions (methodological — see notes/research_proposal.md):
  - RATE = 10 Hz (matches HoloAssist native 10fps annotation; no resampling).
  - Events are timestamped segments [start, end] (type="range") -> we FILL each
    segment's frames with the corresponding event bit.
  - VERB -> event mapping (from the 51 HoloAssist verbs):
        grasp   <- grab, pick, hold, grasp, grip
        contact <- insert, screw, unscrew, place, press, push, plug, tighten,
                   connect, attach, tap, touch
        release <- withdraw, remove, release, unplug, detach, drop
    (A single fine-grained action can set multiple bits, e.g. 'insert' implies
     contact; we keep them independent for now.)
  - failure  <- Action Correctness startswith "Wrong Action"
  - recovery <- Action Correctness contains "corrected by student"  (self-only)
    failure/recovery are TRANSITION events -> we mark a short WINDOW so the
    temporal model can learn the onset (not just a single frame).

This runs locally on the labels JSON only (no video needed). It validates the
label logic and prints an ASCII timeline so we can SEE the events.

Run:
  conda activate vla
  python 03_reward_model/build_event_labels.py                 # first subset video
  python 03_reward_model/build_event_labels.py --video-index 3
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = os.path.join(HERE, "data", "data-annotation-trainval-v1_1.json")
MANIFEST = os.path.join(HERE, "subset_manifest.json")

RATE_HZ = 10
EVENT_NAMES = ["contact", "grasp", "release", "failure", "recovery"]
RECOVERY_WINDOW_S = 0.5   # mark +/- this around a transition onset
FAILURE_WINDOW_S = 0.5

# verb -> event bit(s)
GRASP_VERBS = {"grab", "pick", "hold", "grasp", "grip", "pick_up"}
CONTACT_VERBS = {"insert", "screw", "unscrew", "place", "press", "push", "plug",
                 "tighten", "connect", "attach", "tap", "touch", "align", "fit"}
RELEASE_VERBS = {"withdraw", "remove", "release", "unplug", "detach", "drop",
                 "put_down", "putdown", "let_go"}


def load_video(video_index):
    data = json.load(open(LABELS))
    # prefer a video from our subset manifest if present
    if os.path.exists(MANIFEST):
        names = json.load(open(MANIFEST)).get("video_names", [])
        if names:
            target = names[video_index % len(names)]
            for v in data:
                if v.get("video_name") == target:
                    return v
    return data[video_index]


def duration_s(vid):
    try:
        return float(vid["videoMetadata"]["duration"]["seconds"])
    except Exception:  # noqa: BLE001
        return max((float(e.get("end", 0)) for e in vid.get("events", [])),
                   default=0.0)


def build_labels(vid):
    """Return e[T, 5] float array of event bits at RATE_HZ."""
    dur = duration_s(vid)
    T = max(1, int(round(dur * RATE_HZ)))
    e = np.zeros((T, len(EVENT_NAMES)), dtype=np.float32)
    idx = {n: i for i, n in enumerate(EVENT_NAMES)}

    def frame(t_sec):
        return int(np.clip(round(t_sec * RATE_HZ), 0, T - 1))

    for ev in vid.get("events", []):
        if ev.get("label") != "Fine grained action":
            continue
        at = ev.get("attributes", {})
        s, en = float(ev.get("start", 0)), float(ev.get("end", 0))
        f0, f1 = frame(s), frame(en)
        verb = str(at.get("Verb", "")).lower().strip()
        corr = str(at.get("Action Correctness", ""))

        # ---- state events: fill the whole segment ----
        if verb in GRASP_VERBS:
            e[f0:f1 + 1, idx["grasp"]] = 1.0
        if verb in CONTACT_VERBS:
            e[f0:f1 + 1, idx["contact"]] = 1.0
        if verb in RELEASE_VERBS:
            e[f0:f1 + 1, idx["release"]] = 1.0

        # ---- transition events: mark a window around the segment start ----
        if corr.startswith("Wrong Action"):
            w = int(FAILURE_WINDOW_S * RATE_HZ)
            e[max(0, f0 - w): f0 + w + 1, idx["failure"]] = 1.0
            if "corrected by student" in corr:
                # recovery onset ~ end of the wrong action (re-attempt after)
                w2 = int(RECOVERY_WINDOW_S * RATE_HZ)
                e[max(0, f1 - w2): min(T, f1 + w2 + 1), idx["recovery"]] = 1.0

    return e


def print_timeline(e, width=100):
    """ASCII timeline: one row per event, downsampled to `width` columns."""
    T = e.shape[0]
    step = max(1, T // width)
    print(f"\ntimeline (T={T} frames @ {RATE_HZ}Hz, each col ~{step/RATE_HZ:.1f}s):")
    for i, name in enumerate(EVENT_NAMES):
        row = e[:, i]
        cols = []
        for c in range(0, T, step):
            seg = row[c:c + step]
            cols.append("#" if seg.max() > 0 else "·")
        print(f"  {name:9s} |{''.join(cols)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-index", type=int, default=0)
    args = ap.parse_args()

    vid = load_video(args.video_index)
    print(f"video_name : {vid.get('video_name')}")
    print(f"task_type  : {vid.get('taskType')}")
    print(f"duration_s : {duration_s(vid):.1f}")

    e = build_labels(vid)
    T = e.shape[0]
    print(f"\nlabel tensor: shape {e.shape}  (T x {len(EVENT_NAMES)} events)")
    # per-event coverage (fraction of frames active) -> shows class imbalance
    print("per-event active-frame fraction (class balance):")
    for i, name in enumerate(EVENT_NAMES):
        frac = e[:, i].mean()
        print(f"  {name:9s}: {frac*100:5.2f}%  ({int(e[:, i].sum())} / {T} frames)")

    print_timeline(e)

    print("\nNote: failure/recovery are sparse (few %) -> weighted BCE needed.")
    print("This label tensor is the training target y; the model input will be")
    print("RGB_{t-k:t} + Hand3D_t (added on the GPU with real streams).")


if __name__ == "__main__":
    main()
