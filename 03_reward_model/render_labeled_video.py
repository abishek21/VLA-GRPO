"""
Render a HoloAssist session video with the interaction-event labels overlaid,
to eyeball whether our label construction looks right.

Draws, per frame:
  - 5 event indicators (contact/grasp/release/failure/recovery), lit when active
  - a scrolling timeline strip showing event activity over the whole clip
  - the current timestamp

Uses the labels from the annotation JSON (build_event_labels) + a local mp4.

Run:
  python 03_reward_model/render_labeled_video.py \
      --mp4 /path/to/Video_compress.mp4 \
      --session z057-june-28-22-rashult_assemble \
      --out /tmp/labeled.mp4 --seconds 60
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from build_event_labels import build_labels, EVENT_NAMES, RATE_HZ

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = os.path.join(HERE, "data", "data-annotation-trainval-v1_1.json")

# BGR colors per event (for OpenCV)
COLORS = {
    "contact":  (0, 200, 255),   # amber
    "grasp":    (0, 255, 0),     # green
    "release":  (255, 200, 0),   # cyan-ish
    "failure":  (0, 0, 255),     # red
    "recovery": (255, 0, 255),   # magenta
}


def find_video_record(session_substr):
    data = json.load(open(LABELS))
    for v in data:
        if session_substr in str(v.get("video_name", "")):
            return v
    raise SystemExit(f"no annotation record matching '{session_substr}'")


def main():
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--mp4", required=True)
    ap.add_argument("--session", required=True, help="video_name substring")
    ap.add_argument("--out", default="/tmp/labeled.mp4")
    ap.add_argument("--start", type=float, default=0.0, help="start time (sec)")
    ap.add_argument("--seconds", type=float, default=60, help="limit output length")
    args = ap.parse_args()

    rec = find_video_record(args.session)
    e = build_labels(rec)                       # [T, 5] at RATE_HZ (10 Hz)
    print(f"session {rec['video_name']} | task {rec.get('taskType')} | labels {e.shape}")

    cap = cv2.VideoCapture(args.mp4)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.mp4}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"video {W}x{H} @ {src_fps:.1f}fps")

    # output at source fps; we look up the 10 Hz label row by timestamp
    panel_h = 130
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out, fourcc, src_fps, (W, H + panel_h))

    max_frames = int(args.seconds * src_fps)
    start_frame = int(args.start * src_fps)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fno = 0
    while True:
        ok, frame = cap.read()
        if not ok or fno > max_frames:
            break
        t = (start_frame + fno) / src_fps
        row = int(np.clip(round(t * RATE_HZ), 0, e.shape[0] - 1))
        ev = e[row]                              # [5]

        panel = np.zeros((panel_h, W, 3), np.uint8)
        # event indicators (top row of panel)
        x = 10
        for i, name in enumerate(EVENT_NAMES):
            active = ev[i] > 0.5
            col = COLORS[name] if active else (60, 60, 60)
            cv2.circle(panel, (x + 8, 22), 8, col, -1)
            cv2.putText(panel, name, (x + 22, 27), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, col if active else (120, 120, 120), 1, cv2.LINE_AA)
            x += 150
        cv2.putText(panel, f"t={t:5.1f}s", (W - 120, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # scrolling timeline (bottom of panel): one thin row per event
        strip_top = 45
        strip_h = 14
        Tlab = e.shape[0]
        for i, name in enumerate(EVENT_NAMES):
            y0 = strip_top + i * (strip_h + 2)
            # map full label timeline to width W
            for xpix in range(W):
                lab_i = int(xpix / W * Tlab)
                if e[lab_i, i] > 0.5:
                    panel[y0:y0 + strip_h, xpix] = COLORS[name]
            # playhead
            px = int(row / Tlab * W)
            cv2.line(panel, (px, strip_top), (px, strip_top + 5 * (strip_h + 2)),
                     (255, 255, 255), 1)

        canvas = np.vstack([frame, panel])
        out.write(canvas)
        fno += 1

    cap.release(); out.release()
    print(f"wrote {args.out}  ({fno} frames, {fno/src_fps:.1f}s)")


if __name__ == "__main__":
    main()
