#!/usr/bin/env bash
# Build & push the OpenVLA-OFT + LIBERO image for RunPod.
# You do NOT need a GPU to build. Run this on your laptop / any Docker host.
#
# Usage:
#   export IMAGE=docker.io/<your-dockerhub-user>/openvla-oft-libero:cu128
#   ./build_and_push.sh
set -euo pipefail

IMAGE="${IMAGE:-docker.io/CHANGE_ME/openvla-oft-libero:cu128}"

if [[ "$IMAGE" == *CHANGE_ME* ]]; then
  echo "Set IMAGE first, e.g.:"
  echo "  export IMAGE=docker.io/youruser/openvla-oft-libero:cu128"
  exit 1
fi

# RunPod GPUs are amd64. If you're on Apple Silicon this cross-builds (slower).
# buildx handles both native and emulated amd64 builds and can push directly.
docker buildx create --use --name oftbuilder 2>/dev/null || docker buildx use oftbuilder

docker buildx build \
  --platform linux/amd64 \
  -t "$IMAGE" \
  --push \
  .

echo "Pushed: $IMAGE"
echo "Now in RunPod: New Pod -> Custom template -> Container Image = $IMAGE"
