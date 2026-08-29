#!/usr/bin/env bash
# Persist the HoloAssist subset to Azure Blob (durable), so ephemeral RunPod
# disks only need to pull ~23 GB per session instead of re-streaming 363 GB.
#
# Workflow:
#   FIRST TIME (once):
#     bash download_subset.sh video && bash download_subset.sh hands   # ~23 GB local
#     export AZ_SAS="https://<acct>.blob.core.windows.net/<container>?<sas>"
#     bash sync_azure.sh upload
#   EVERY NEW POD:
#     export AZ_SAS="https://<acct>.blob.core.windows.net/<container>?<sas>"
#     bash sync_azure.sh download
#
# AZ_SAS = a container-level Blob SAS URL with Read/Write/List/Create perms.
# DEST   = local subset dir (default ./holoassist_subset).
set -euo pipefail

DEST="${DEST:-./holoassist_subset}"
MODE="${1:-help}"

install_azcopy() {
  if command -v azcopy >/dev/null 2>&1; then return; fi
  echo "installing azcopy ..."
  curl -sL https://aka.ms/downloadazcopy-v10-linux -o /tmp/azcopy.tgz
  tar -xf /tmp/azcopy.tgz -C /tmp
  cp /tmp/azcopy_linux_amd64_*/azcopy /usr/local/bin/azcopy
  chmod +x /usr/local/bin/azcopy
  azcopy --version
}

# split the SAS url into base + query so we can insert a path
sas_with_path() {  # $1 = subpath under the container
  local path="$1"
  local base="${AZ_SAS%%\?*}"     # https://acct.blob.../container
  local query="${AZ_SAS#*\?}"     # sv=...&sig=...
  echo "${base}/${path}?${query}"
}

case "$MODE" in
  upload)
    : "${AZ_SAS:?set AZ_SAS to your container Blob SAS URL}"
    install_azcopy
    echo "uploading $DEST -> Azure blob ..."
    # recursive upload; preserves the session folder structure under the container
    azcopy copy "$DEST/*" "$(sas_with_path '')" --recursive=true
    echo "done. Data persisted in Azure."
    ;;
  download)
    : "${AZ_SAS:?set AZ_SAS to your container Blob SAS URL}"
    install_azcopy
    mkdir -p "$DEST"
    echo "downloading subset from Azure blob -> $DEST ..."
    azcopy copy "$(sas_with_path '*')" "$DEST" --recursive=true
    echo "done. Subset restored on this pod."
    ;;
  *)
    echo "usage: AZ_SAS=<container-sas-url> bash sync_azure.sh {upload|download}"
    echo "  DEST=/path to override local subset dir (default ./holoassist_subset)"
    ;;
esac
