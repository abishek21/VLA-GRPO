#!/usr/bin/env bash
# Helper: run remote commands on the RunPod pod via the interactive SSH proxy.
# Usage: ./rpod.sh 'command; command2'
#    or: ./rpod.sh < script.sh
set -euo pipefail
HOST="cy2g8xo165mslz-64410d71@ssh.runpod.io"
KEY="$HOME/.ssh/id_ed25519"
if [ $# -ge 1 ]; then
  BODY="$*"
else
  BODY="$(cat)"
fi
printf '%s\nexit\n' "$BODY" | ssh -tt -o StrictHostKeyChecking=no -o ConnectTimeout=20 "$HOST" -i "$KEY" 2>&1 | tr -d '\r'
