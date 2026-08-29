"""
Inspect the HoloAssist annotation JSON to decide if its labels are usable for
our interaction-event reward model (Gate G3 prerequisite).

What we need to learn (the whole point):
  1. Top-level structure: how are videos / clips / segments organized?
  2. What annotation TYPES exist (actions, mistakes, interventions, ...)?
  3. GRANULARITY: are labels per-frame, per-timestamp-segment, or per-clip?
     -> decides whether we can train a PER-TIMESTEP event detector directly.
  4. The mistake / intervention taxonomy -> our `failure` / `recovery` events.
  5. Any fields we can map to contact / grasp / release.

This script only READS json (no GPU, no torch). It downloads the 111 MB labels
file once into 03_reward_model/data/ (gitignored) and prints a schema summary.

Run:
  conda activate vla
  python 03_reward_model/inspect_holoassist_labels.py
"""
import json
import os
import sys
import urllib.request
from collections import Counter, defaultdict

LABELS_URL = (
    "https://hl2data.z5.web.core.windows.net/holoassist-data-release/"
    "data-annotation-trainval-v1_1.json"
)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
LOCAL = os.path.join(DATA_DIR, "data-annotation-trainval-v1_1.json")


def download_if_needed():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(LOCAL) and os.path.getsize(LOCAL) > 1_000_000:
        print(f"labels already present: {LOCAL} "
              f"({os.path.getsize(LOCAL)/1e6:.1f} MB)")
        return
    print(f"downloading labels (~111 MB) ->\n  {LABELS_URL}")
    try:
        def _progress(n, bs, total):
            done = n * bs
            pct = min(100, 100 * done / total) if total > 0 else 0
            sys.stdout.write(f"\r  {pct:5.1f}%  ({done/1e6:6.1f} MB)")
            sys.stdout.flush()
        urllib.request.urlretrieve(LABELS_URL, LOCAL, _progress)
        print("\n  done.")
    except Exception as e:  # noqa: BLE001
        print("\nDownload failed:", repr(e))
        print("You can download manually from the HoloAssist site and place it at:")
        print("  ", LOCAL)
        sys.exit(1)


def describe(obj, depth=0, max_depth=3, path="root"):
    """Recursively print the shape/type of a nested JSON structure (shallow)."""
    pad = "  " * depth
    if depth > max_depth:
        print(f"{pad}...")
        return
    if isinstance(obj, dict):
        keys = list(obj.keys())
        print(f"{pad}{path}: dict with {len(keys)} keys: {keys[:12]}"
              + (" ..." if len(keys) > 12 else ""))
        # recurse into a couple of representative keys
        for k in keys[:4]:
            describe(obj[k], depth + 1, max_depth, path=k)
    elif isinstance(obj, list):
        print(f"{pad}{path}: list len={len(obj)}")
        if obj:
            describe(obj[0], depth + 1, max_depth, path=f"{path}[0]")
    else:
        val = repr(obj)
        if len(val) > 60:
            val = val[:60] + "..."
        print(f"{pad}{path}: {type(obj).__name__} = {val}")


def deep_key_scan(obj, keys_of_interest, found=None, path="root"):
    """Find where interesting keys appear anywhere in the structure."""
    if found is None:
        found = defaultdict(list)
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = str(k).lower()
            for want in keys_of_interest:
                if want in lk:
                    sample = v
                    if isinstance(sample, (dict, list)):
                        sample = f"<{type(sample).__name__} len={len(sample)}>"
                    found[want].append((path + "." + str(k), sample))
            deep_key_scan(v, keys_of_interest, found, path + "." + str(k))
    elif isinstance(obj, list) and obj:
        deep_key_scan(obj[0], keys_of_interest, found, path + "[0]")
    return found


def main():
    download_if_needed()
    print("\nloading json (may take ~10s for 111 MB) ...")
    with open(LOCAL) as f:
        data = json.load(f)

    print("\n" + "=" * 70 + "\nTOP-LEVEL SCHEMA\n" + "=" * 70)
    describe(data, max_depth=3)

    # HoloAssist labels are usually a list of videos, each with events/segments.
    # Try to locate the per-video annotation records and summarize event types.
    print("\n" + "=" * 70 + "\nANNOTATION-TYPE / LABEL SCAN\n" + "=" * 70)
    interest = ["mistake", "interven", "correct", "action", "verb", "noun",
                "attribute", "event", "start", "end", "label", "coarse", "fine"]
    hits = deep_key_scan(data, interest)
    for want in interest:
        if hits.get(want):
            print(f"\n[{want}]  ({len(hits[want])} occurrences) e.g.:")
            for pathv, sample in hits[want][:3]:
                print(f"   {pathv}  =  {sample}")

    # Try to tabulate the distribution of any 'mistake'/'action' label values.
    print("\n" + "=" * 70 + "\nLABEL VALUE DISTRIBUTIONS (best-effort)\n" + "=" * 70)
    counters = defaultdict(Counter)

    def collect(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                lk = str(k).lower()
                if isinstance(v, (str, int, bool)) and any(
                    w in lk for w in ["mistake", "interven", "verb", "noun",
                                      "attribute", "coarse", "fine", "label", "type"]
                ):
                    counters[lk][str(v)] += 1
                collect(v)
        elif isinstance(obj, list):
            for it in obj:
                collect(it)

    collect(data)
    for field, c in list(counters.items())[:12]:
        top = c.most_common(8)
        print(f"\n{field}: {len(c)} unique -> top: {top}")

    print("\n" + "=" * 70 + "\nWHAT TO CONCLUDE\n" + "=" * 70)
    print("Look above for:")
    print(" - Do segments have start/end TIMESTAMPS? -> per-segment granularity")
    print("   (good: we can build per-timestep event labels by filling segments).")
    print(" - Is there a 'mistake' field with values? -> our `failure` signal.")
    print(" - Is there an 'intervention'/'correction' field? -> our `recovery`.")
    print(" - action verb/noun -> can weak-derive contact/grasp/release.")
    print("\nRecord findings in notes/research_proposal.md (Gate G3).")


if __name__ == "__main__":
    main()
