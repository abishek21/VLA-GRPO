"""
Select a HoloAssist SUBSET for the interaction-reward model, prioritizing
failure/recovery richness, and emit a download manifest + size estimate.

Why: the full RGB + hand-pose streams are ~370 GB. We don't need all of it.
Our contribution is failure/recovery, and 'Wrong Action' segments are ~5% and
concentrated in some videos. So we pick the videos richest in wrong-action /
corrected-by-student segments, and download ONLY those.

Outputs (in 03_reward_model/):
  - subset_manifest.json : the chosen video_names + per-video stats
  - prints total estimated download size (RGB compressed + hand pose)

No GPU, no big downloads. Uses only the 111 MB labels JSON already present.

Run:
  conda activate vla
  python 03_reward_model/select_subset.py --num-videos 60
  python 03_reward_model/select_subset.py --task-substr assemble --num-videos 40
"""
import argparse
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = os.path.join(HERE, "data", "data-annotation-trainval-v1_1.json")
MANIFEST = os.path.join(HERE, "subset_manifest.json")

# Size estimates derived from the HoloAssist download page totals:
#   compressed video (width 256): 144.62 GB / 608400 s  ~= 0.238 MB per second
#   hand pose:                    219.24 GB / 608400 s  ~= 0.360 MB per second
# (608400 s = 169 h total across 1758 videos). We estimate per-video size from
# its duration. These are approximate but good enough to size the download.
MB_PER_SEC_RGB = 0.238
MB_PER_SEC_HAND = 0.360


def video_duration_s(vid):
    try:
        return float(vid["videoMetadata"]["duration"]["seconds"])
    except Exception:  # noqa: BLE001
        # fall back to the max event end time
        ev = vid.get("events", [])
        return max((float(e.get("end", 0)) for e in ev), default=0.0)


def video_stats(vid):
    dur = video_duration_s(vid)
    n_fine = n_wrong = n_rec_student = n_rec_instr = n_notcorr = n_drop = 0
    verbs = Counter()
    for e in vid.get("events", []):
        if e.get("label") != "Fine grained action":
            continue
        n_fine += 1
        at = e.get("attributes", {})
        corr = str(at.get("Action Correctness", ""))
        verb = str(at.get("Verb", "")).lower()
        if verb:
            verbs[verb] += 1
        if corr.startswith("Wrong Action"):
            n_wrong += 1
            expl = str(at.get("Incorrect Action Explanation", "")).lower()
            if "drop" in expl:
                n_drop += 1
            if "corrected by student" in corr:
                n_rec_student += 1
            elif "corrected by instructor" in corr:
                n_rec_instr += 1
            elif "not corrected" in corr:
                n_notcorr += 1
    size_mb = dur * (MB_PER_SEC_RGB + MB_PER_SEC_HAND)
    return {
        "video_name": vid.get("video_name"),
        "task_type": vid.get("taskType"),
        "duration_s": round(dur, 1),
        "n_fine_actions": n_fine,
        "n_wrong": n_wrong,
        "n_recovery_student": n_rec_student,
        "n_recovery_instructor": n_rec_instr,
        "n_not_corrected": n_notcorr,
        "n_drop_failures": n_drop,
        "est_size_mb": round(size_mb, 1),
        "top_verbs": verbs.most_common(5),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-videos", type=int, default=60,
                    help="how many videos to select")
    ap.add_argument("--task-substr", type=str, default=None,
                    help="only consider tasks whose taskType contains this")
    ap.add_argument("--rank-by", type=str, default="n_wrong",
                    choices=["n_wrong", "n_recovery_student", "n_drop_failures",
                             "n_fine_actions"],
                    help="metric to prioritize (failure/recovery richness)")
    args = ap.parse_args()

    data = json.load(open(LABELS))
    stats = [video_stats(v) for v in data]

    # optional task filter
    if args.task_substr:
        s = args.task_substr.lower()
        stats = [v for v in stats if s in str(v["task_type"]).lower()]
        print(f"filtered to task contains '{args.task_substr}': {len(stats)} videos")

    # rank by the chosen richness metric (desc), then by fine-action count
    stats.sort(key=lambda v: (v[args.rank_by], v["n_fine_actions"]), reverse=True)
    subset = stats[: args.num_videos]

    # totals
    tot_gb = sum(v["est_size_mb"] for v in subset) / 1024
    tot_wrong = sum(v["n_wrong"] for v in subset)
    tot_rec = sum(v["n_recovery_student"] + v["n_recovery_instructor"] for v in subset)
    tot_drop = sum(v["n_drop_failures"] for v in subset)
    tot_dur_h = sum(v["duration_s"] for v in subset) / 3600

    print("\n" + "=" * 68)
    print(f"SUBSET: {len(subset)} videos  (ranked by {args.rank_by})")
    print("=" * 68)
    print(f"  est. download (RGB+hand) : {tot_gb:6.1f} GB")
    print(f"  total duration           : {tot_dur_h:6.1f} h")
    print(f"  wrong-action segments     : {tot_wrong}")
    print(f"    of which recoveries     : {tot_rec}")
    print(f"    of which drop-failures  : {tot_drop}")
    print("\n  top 10 selected videos:")
    print(f"    {'video_name':32s} {'task':22s} {'wrong':>5s} {'rec':>4s} {'MB':>7s}")
    for v in subset[:10]:
        rec = v["n_recovery_student"] + v["n_recovery_instructor"]
        print(f"    {str(v['video_name'])[:32]:32s} "
              f"{str(v['task_type'])[:22]:22s} "
              f"{v['n_wrong']:5d} {rec:4d} {v['est_size_mb']:7.0f}")

    manifest = {
        "rank_by": args.rank_by,
        "task_substr": args.task_substr,
        "num_videos": len(subset),
        "est_download_gb": round(tot_gb, 1),
        "total_wrong_actions": tot_wrong,
        "total_recoveries": tot_rec,
        "video_names": [v["video_name"] for v in subset],
        "videos": subset,
    }
    json.dump(manifest, open(MANIFEST, "w"), indent=2)
    print(f"\nwrote manifest -> {MANIFEST}")
    print("Next: use video_names to download ONLY these videos' RGB + hand pose")
    print("onto the GPU/network volume (not your Mac).")


if __name__ == "__main__":
    main()
