"""
Emit wanted_sessions.txt (one session name per line) from subset_manifest.json,
for use by download_subset.sh.

IMPORTANT — naming verification:
  The annotation `video_name` (e.g. 'z057-june-28-22-rashult_assemble') may or may
  not exactly match the session FOLDER names inside the tar. Before a full
  download, run:
        bash 03_reward_model/download_subset.sh list
  and compare the printed folder names to these. If they differ, adjust the
  transform below (e.g. strip a suffix / change case) so the patterns match.

Run:
  python 03_reward_model/make_wanted_sessions.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "subset_manifest.json")
OUT = os.path.join(HERE, "wanted_sessions.txt")


def session_from_video_name(video_name: str) -> str:
    """Map an annotation video_name to the tar session folder name.
    Start with identity; adjust after verifying with `download_subset.sh list`."""
    return video_name.strip()


def main():
    m = json.load(open(MANIFEST))
    names = m.get("video_names", [])
    sessions = sorted({session_from_video_name(n) for n in names if n})
    with open(OUT, "w") as f:
        f.write("\n".join(sessions) + "\n")
    print(f"wrote {len(sessions)} session names -> {OUT}")
    print("First 5:")
    for s in sessions[:5]:
        print("  ", s)
    print("\nNEXT: verify these match the tar's folders:")
    print("  bash 03_reward_model/download_subset.sh list")
    print("If names differ, edit session_from_video_name() and re-run.")


if __name__ == "__main__":
    main()
