#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-sam-bench:rocm-nightly}"

group_add_args() {
  local group
  local gid

  for group in video render; do
    gid="$(getent group "$group" | cut -d: -f3 || true)"
    if [[ -n "$gid" ]]; then
      printf '%s\n' --group-add "$gid"
    else
      echo "Warning: host group '$group' was not found; skipping Docker group add" >&2
    fi
  done
}

mapfile -t group_args < <(group_add_args)

exec docker run --rm -it \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device=/dev/kfd \
  --device=/dev/dri \
  "${group_args[@]}" \
  --ipc=host \
  --shm-size=16G \
  -v "$PWD:/workspace" \
  -v "$HOME/Downloads:/downloads:ro" \
  -w /workspace \
  "$IMAGE"
