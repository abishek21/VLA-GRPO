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
MATRIX_START = 3          # v[0]=timestamp, v[1]=frame id, v[2]=leading valid flag
JOINT_STRIDE = 16         # per joint: a 4x4 row-major pose matrix (16 values)
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
# Video -> frames at TARGET_HZ  (decode only the requested clip window)
# ----------------------------------------------------------------------
@lru_cache(maxsize=64)
def _video_meta(mp4_path: str):
    """Cheap metadata only (fps, frame count) — never caches pixels."""
    import cv2
    cap = cv2.VideoCapture(mp4_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return fps, nframes


def _decode_clip(mp4_path: str, f0: int, f1: int, out_h: int, out_w: int):
    """Decode ONLY 10 Hz frames [f0, f1) from the mp4 (memory-light).
    Seeks to the source frame for f0, reads contiguously, subsamples to 10 Hz,
    resizes to (out_h, out_w). Returns uint8 [f1-f0, H, W, 3]."""
    import cv2
    fps, _ = _video_meta(mp4_path)
    T = f1 - f0
    src0 = int(round(f0 / TARGET_HZ * fps))
    n_src = max(T, int(round(T / TARGET_HZ * fps)))   # contiguous src frames to read
    cap = cv2.VideoCapture(mp4_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, src0))
    buf = []
    for _ in range(n_src):
        ok, fr = cap.read()
        if not ok:
            break
        buf.append(fr)                                # BGR
    cap.release()
    out = np.zeros((T, out_h, out_w, 3), np.uint8)
    if buf:
        # subsample the contiguous src frames down to T output frames
        idx = np.linspace(0, len(buf) - 1, T).round().astype(int)
        for i, j in enumerate(idx):
            rgb = cv2.cvtColor(buf[j], cv2.COLOR_BGR2RGB)
            out[i] = cv2.resize(rgb, (out_w, out_h))
    return out


# ----------------------------------------------------------------------
# Hand pose -> vector per frame at TARGET_HZ
# ----------------------------------------------------------------------
def parse_hand_line(line: str) -> np.ndarray:
    """Parse one row of a HoloAssist *_sync.txt hand file into 26 joint XYZ
    positions (78 numbers).

    Verified format (471 fields):
        v[0]=timestamp, v[1]=frame_id, v[2]=leading valid flag,
        then 26 joints x 16 (a row-major 4x4 pose matrix each; bottom row 0 0 0 1
        confirmed at indices 15,31,47,... => stride 16), then trailing 1-flags.
        Translation (position) = matrix elements [3,7,11] (row-major tx,ty,tz).
    Returns [78] float32; zeros if malformed."""
    toks = line.replace(",", " ").split()
    vals = []
    for t in toks:
        try:
            vals.append(float(t))
        except ValueError:
            vals.append(0.0)
    vals = np.asarray(vals, np.float32)
    need = MATRIX_START + NUM_JOINTS * JOINT_STRIDE
    if vals.size < need:
        return np.zeros(PER_HAND_DIM, np.float32)
    mats = vals[MATRIX_START:need].reshape(NUM_JOINTS, JOINT_STRIDE)  # [26,16]
    xyz = mats[:, [3, 7, 11]]                      # translation per joint [26,3]
    return xyz.reshape(-1).astype(np.float32)     # [78]


@lru_cache(maxsize=32)
def _load_hand_full(hands_dir: str):
    """Load Left+Right sync files once -> [N_rows, HAND_DIM] (cached; small).
    Row i ~ source frame i (files are synced to RGB)."""
    def load_one(path):
        if not os.path.exists(path):
            return None
        rows = [parse_hand_line(ln) for ln in open(path)]
        rows = [r for r in rows if r.size == PER_HAND_DIM]
        return np.stack(rows) if rows else None

    L = load_one(os.path.join(hands_dir, "Left_sync.txt"))
    R = load_one(os.path.join(hands_dir, "Right_sync.txt"))
    parts = []
    n = max((m.shape[0] for m in (L, R) if m is not None), default=0)
    for m in (L, R):
        if m is None:
            parts.append(np.zeros((n, PER_HAND_DIM), np.float32))
        else:
            parts.append(m if m.shape[0] == n else m[:n])
    if not parts or n == 0:
        return np.zeros((1, HAND_DIM), np.float32)
    return np.concatenate(parts, axis=1).astype(np.float32)   # [N, 156]


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
        clip = _decode_clip(mp4, f0, f1, self.H, self.W)     # [T,H,W,3] uint8
        t = torch.from_numpy(clip).float().permute(0, 3, 1, 2) / 255.0  # [T,3,H,W]
        return t

    def hand(self, session_name: str, f0: int, f1: int) -> torch.Tensor:
        _, hands = self._paths(session_name)
        full = _load_hand_full(hands)                        # [N, 156] at ~src fps
        fps, _ = _video_meta(self._paths(session_name)[0])
        # map 10 Hz clip indices [f0,f1) to hand rows (rows ~ source frames)
        src = (np.arange(f0, f1) / TARGET_HZ * fps).round().astype(int)
        src = np.clip(src, 0, full.shape[0] - 1)
        clip = full[src]                                     # [T, 156]
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
