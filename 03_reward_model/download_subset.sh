#!/usr/bin/env bash
# Download ONLY the subset sessions from HoloAssist's monolithic tars by
# streaming + selectively extracting. Downloads full bandwidth but stores only
# the matching sessions (fits a ~100 GB disk).
#
# HoloAssist ships two big tars (no per-video files):
#   video_compress.tar  (~144 GB)  -> <session>/Export_py/Video/Video_compress.mp4
#   hands.tar           (~219 GB)  -> <session>/Export_py/Hands/{Left,Right}_sync.txt
#
# Usage:
#   1) FIRST verify session names match the manifest:
#        bash download_subset.sh list        # prints first tar member names
#   2) then extract the subset:
#        bash download_subset.sh video
#        bash download_subset.sh hands
#
# Requires: curl, tar, and 03_reward_model/wanted_sessions.txt (one name/line).
set -euo pipefail

BASE="https://hl2data.z5.web.core.windows.net/holoassist-data-release"
VIDEO_TAR="$BASE/video_compress.tar"
HANDS_TAR="$BASE/hands.tar"
DEST="${DEST:-./holoassist_subset}"
WANTED="${WANTED:-03_reward_model/wanted_sessions.txt}"
MODE="${1:-help}"

mkdir -p "$DEST"

case "$MODE" in
  list)
    # Stream just the header listing; Ctrl-C after you've seen enough names.
    echo "listing member names from video_compress.tar (Ctrl-C to stop) ..."
    curl -sL "$VIDEO_TAR" | tar -tv | head -40
    ;;
  video)
    [ -f "$WANTED" ] || { echo "missing $WANTED"; exit 1; }
    # Build --wildcards patterns: match any path containing a wanted session.
    PATTERNS=()
    while IFS= read -r s; do
      [ -z "$s" ] && continue
      PATTERNS+=( "--wildcards" "--no-anchored" "*${s}*/Video/Video_compress.mp4" )
    done < "$WANTED"
    echo "streaming video_compress.tar, extracting $(wc -l < "$WANTED") sessions ..."
    curl -sL "$VIDEO_TAR" | tar -xv -C "$DEST" "${PATTERNS[@]}"
    ;;
  hands)
    [ -f "$WANTED" ] || { echo "missing $WANTED"; exit 1; }
    PATTERNS=()
    while IFS= read -r s; do
      [ -z "$s" ] && continue
      PATTERNS+=( "--wildcards" "--no-anchored" "*${s}*/Hands/*_sync.txt" )
    done < "$WANTED"
    echo "streaming hands.tar, extracting hand pose for the subset ..."
    curl -sL "$HANDS_TAR" | tar -xv -C "$DEST" "${PATTERNS[@]}"
    ;;
  *)
    echo "usage: bash download_subset.sh {list|video|hands}"
    echo "  set DEST=/path  WANTED=path/to/wanted_sessions.txt to override"
    ;;
esac
