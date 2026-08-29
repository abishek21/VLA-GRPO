"""
RealProvider: reads HoloAssist RGB video + hand pose for the dataset.

Implements the same interface as SyntheticProvider (frames/hand) so it drops
into HoloAssistClips with no other change.

File layout (verified against the HoloAssist tars):
    <root>/<session>/Export_py/Video_compress.mp4      # RGB (width 256)
    <root>/<session>/Export_py/Hands/Left_sync.txt     # left hand, synced to RGB
    <root>/<session>/Export_py/Hands/Right_sync.txt    # right hand, synced to RGB

Frame timing: HoloAssist is ~10 fps but not exactly; we RESAMPLE to a fixed
TARGET_HZ (10) by timestamp so frames and hand rows align to our label grid.

The hand `_sync.txt` column format is parsed generically (whitespace/comma
separated numeric rows). We take the per-row keypoint coordinates and flatten to
a fixed-width vector; the exact joint count is auto-detected and padded/trimmed
to HAND_DIM. Adjust parse_hand_line() once we confirm the real columns.

Requires (GPU/VM): pip install av  (PyAV) for video decoding.
"""
from __future__ import annotations

import os
import glob
from functools import lru_cache

import numpy as np
import torch

from build_event_labels import RATE_HZ  # 10 Hz

TARGET_HZ = RATE_HZ
NUM_JOINTS = 26            # HoloLens hand model joints
JOINT_STRIDE = 17         # per joint: [valid, 16 matrix values (row-major 4x4)]
PER_HAND_DIM = NUM_JOINTS * 3          # xyz per joint = 78
HAND_DIM = PER_HAND_DIM * 2            # left + right = 156


def session_dir(root: str, session_name: str) -> str:
    """Find the session folder under root that matches session_name (substring)."""
    # sessions were extracted as <name>/Export_py/...
    cands = glob.glob(os.path.join(root, f"*{session_name}*"))
    if not cands:
        raise FileNotFoundError(f"no session dir matching {session_name} under {root}")
    return cands[0]


# ----------------------------------------------------------------------
# Video -> frames at TARGET_HZ
# ----------------------------------------------------------------------
@lru_cache(maxsize=8)
def _decode_video_10hz(mp4_path: str, out_h: int, out_w: int):
    """Decode the whole mp4, resample to TARGET_HZ by timestamp.
    Returns a uint8 numpy array [T, H, W, 3]. Cached per file (few big videos)."""
    import av  # PyAV
    container = av.open(mp4_path)
    stream = container.streams.video[0]
    frames = []
    times = []
    for frame in container.decode(stream):
        t = float(frame.pts * stream.time_base) if frame.pts is not None else None
        img = frame.to_ndarray(format="rgb24")   # [H,W,3] uint8
        frames.append(img)
        times.append(t if t is not None else len(frames) / (float(stream.average_rate or 30)))
    container.close()
    if not frames:
        return np.zeros((0, out_h, out_w, 3), np.uint8)
    times = np.array(times)
    dur = times[-1] if times[-1] and times[-1] > 0 else len(frames) / 30.0
    T = max(1, int(round(dur * TARGET_HZ)))
    # nearest-frame resample onto the 10 Hz grid
    grid = (np.arange(T) / TARGET_HZ)
    idx = np.searchsorted(times, grid).clip(0, len(frames) - 1)
    import cv2
    out = np.empty((T, out_h, out_w, 3), np.uint8)
    for i, j in enumerate(idx):
        out[i] = cv2.resize(frames[j], (out_w, out_h))
    return out


# ----------------------------------------------------------------------
# Hand pose -> vector per frame at TARGET_HZ
# ----------------------------------------------------------------------
def parse_hand_line(line: str) -> np.ndarray:
    """Parse one row of a HoloAssist *_sync.txt hand file into 26 joint XYZ
    positions (78 numbers).

    Verified: row = [timestamp, frame_id, <26 joint blocks>, <trailing flags>].
    Each joint block encodes a 4x4 row-major pose; translation = matrix[3,7,11].

    TODO(GPU): the exact per-joint stride needs numpy verification — 469 body
    numbers is not divisible by 17, so there is likely one extra value per joint
    or a trailing validity block to account for. Joint 0 parses correctly with
    stride 17; confirm/adjust JOINT_STRIDE and the trailing offset on the GPU
    (inspect one row with numpy, checking positions are contiguous & metric).
    Returns [78] float32; zeros if malformed."""
    toks = line.replace(",", " ").split()
    vals = []
    for t in toks:
        try:
            vals.append(float(t))
        except ValueError:
            vals.append(0.0)
    vals = np.asarray(vals, np.float32)
    need = 2 + NUM_JOINTS * JOINT_STRIDE
    if vals.size < need:
        return np.zeros(PER_HAND_DIM, np.float32)
    body = vals[2:need].reshape(NUM_JOINTS, JOINT_STRIDE)
    # per joint: [valid, m0..m15]; translation = m3, m7, m11 (row-major 4x4)
    xyz = body[:, [1 + 3, 1 + 7, 1 + 11]]         # [26, 3]
    return xyz.reshape(-1).astype(np.float32)     # [78]


@lru_cache(maxsize=8)
def _load_hand_10hz(hands_dir: str, n_frames: int):
    """Load Left+Right sync files, produce [n_frames, HAND_DIM].
    The _sync files are already aligned to RGB frames, so row i ~ frame i; we
    still resample by index onto the TARGET_HZ grid length n_frames."""
    def load_one(path):
        if not os.path.exists(path):
            return None
        rows = []
        with open(path) as f:
            for ln in f:
                v = parse_hand_line(ln)
                if v.size:
                    rows.append(v)
        if not rows:
            return None
        # pad/trim each row to a common width
        w = max(r.size for r in rows)
        M = np.zeros((len(rows), w), np.float32)
        for i, r in enumerate(rows):
            M[i, :r.size] = r
        return M

    L = load_one(os.path.join(hands_dir, "Left_sync.txt"))
    R = load_one(os.path.join(hands_dir, "Right_sync.txt"))
    # combine available hands; if one missing, zero-fill
    mats = [m for m in (L, R) if m is not None]
    if not mats:
        return np.zeros((n_frames, HAND_DIM), np.float32)
    # align lengths by index-resample to n_frames
    def resample(M):
        src = np.linspace(0, M.shape[0] - 1, n_frames).round().astype(int)
        return M[src]
    combined = np.concatenate([resample(m) for m in mats], axis=1)  # [n_frames, sumW]
    # fix to HAND_DIM (trim/pad)
    out = np.zeros((n_frames, HAND_DIM), np.float32)
    k = min(HAND_DIM, combined.shape[1])
    out[:, :k] = combined[:, :k]
    return out


# ----------------------------------------------------------------------
# The provider
# ----------------------------------------------------------------------
class RealProvider:
    def __init__(self, root: str, H: int = 128, W: int = 128):
        self.root = root
        self.H, self.W = H, W

    def _paths(self, session_name):
        d = session_dir(self.root, session_name)
        mp4 = os.path.join(d, "Export_py", "Video_compress.mp4")
        hands = os.path.join(d, "Export_py", "Hands")
        return mp4, hands

    def frames(self, session_name: str, f0: int, f1: int) -> torch.Tensor:
        mp4, _ = self._paths(session_name)
        vid = _decode_video_10hz(mp4, self.H, self.W)      # [T,H,W,3] uint8
        clip = vid[f0:f1]
        if clip.shape[0] < (f1 - f0):                       # pad short tail
            pad = np.zeros((f1 - f0 - clip.shape[0], self.H, self.W, 3), np.uint8)
            clip = np.concatenate([clip, pad], 0)
        t = torch.from_numpy(clip).float().permute(0, 3, 1, 2) / 255.0  # [T,3,H,W]
        return t

    def hand(self, session_name: str, f0: int, f1: int) -> torch.Tensor:
        mp4, hands = self._paths(session_name)
        vid = _decode_video_10hz(mp4, self.H, self.W)
        h = _load_hand_10hz(hands, vid.shape[0])            # [T,HAND_DIM]
        clip = h[f0:f1]
        if clip.shape[0] < (f1 - f0):
            clip = np.concatenate(
                [clip, np.zeros((f1 - f0 - clip.shape[0], HAND_DIM), np.float32)], 0)
        return torch.from_numpy(clip)


# ----------------------------------------------------------------------
# quick check (run on the VM/GPU once data is present)
# ----------------------------------------------------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/azureuser/holoassist")
    ap.add_argument("--session", default=None, help="substring; default first found")
    args = ap.parse_args()

    root = args.root
    sess = args.session
    if sess is None:
        dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        if not dirs:
            print("no sessions found under", root); return
        sess = dirs[0]
    print("session:", sess)

    prov = RealProvider(root, H=128, W=128)
    fr = prov.frames(sess, 0, 30)
    hd = prov.hand(sess, 0, 30)
    print("frames:", tuple(fr.shape), "dtype", fr.dtype, "range", float(fr.min()), float(fr.max()))
    print("hand  :", tuple(hd.shape), "dtype", hd.dtype)
    print("OK: RealProvider decodes video + hand for a clip.")


if __name__ == "__main__":
    main()
