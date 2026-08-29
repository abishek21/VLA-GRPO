#!/usr/bin/env bash
# One-shot: download+extract the HoloAssist subset, then upload to Azure Blob.
# Run on the West-Europe VM inside tmux. AZ_SAS must be exported before running.
set -x
cd /home/azureuser/VLA-GRPO
git pull -q || true
python3 03_reward_model/make_wanted_sessions.py

DEST=/home/azureuser/holoassist bash 03_reward_model/download_subset.sh video
DEST=/home/azureuser/holoassist bash 03_reward_model/download_subset.sh hands

echo "=== extracted files ==="
find /home/azureuser/holoassist -type f | wc -l
du -sh /home/azureuser/holoassist

DEST=/home/azureuser/holoassist bash 03_reward_model/sync_azure.sh upload
echo "=== ALL DONE ==="
