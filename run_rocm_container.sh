#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-sam-bench:rocm-nightly}"

exec docker run --rm -it \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  --ipc=host \
  --shm-size=16G \
  -v "$PWD:/workspace" \
  -v "$HOME/Downloads:/downloads:ro" \
  -w /workspace \
  "$IMAGE"
