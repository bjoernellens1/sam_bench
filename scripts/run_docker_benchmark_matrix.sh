#!/usr/bin/env bash
set -euo pipefail

# Build ROCm Docker images if they are missing and run a fixed SAM1 AMG
# benchmark matrix on both stable and nightly PyTorch/ROCm stacks.
#
# Usage:
#   scripts/run_docker_benchmark_matrix.sh
#   scripts/run_docker_benchmark_matrix.sh --rebuild
#   scripts/run_docker_benchmark_matrix.sh --only nightly
#
# Assumptions:
#   - Docker is installed.
#   - AMD ROCm devices are available through /dev/kfd and /dev/dri.
#   - The SAM ViT-B checkpoint is in ~/Downloads or can be downloaded.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STABLE_IMAGE="sam-bench:rocm-stable"
NIGHTLY_IMAGE="sam-bench:rocm-nightly"
STABLE_DOCKERFILE="Dockerfile.rocm-stable"
NIGHTLY_DOCKERFILE="Dockerfile.rocm-nightly"
CHECKPOINT_HOST="${CHECKPOINT_HOST:-$HOME/Downloads/sam_vit_b_01ec64.pth}"
CHECKPOINT_CONTAINER="/downloads/sam_vit_b_01ec64.pth"
NUM_WARMUP="${NUM_WARMUP:-2}"
NUM_RUNS="${NUM_RUNS:-10}"
RESULT_ROOT="${RESULT_ROOT:-results}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RESULT_DIR="$RESULT_ROOT/$STAMP"
REBUILD=0
ONLY="all"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rebuild)
      REBUILD=1
      shift
      ;;
    --only)
      ONLY="${2:-}"
      if [[ "$ONLY" != "stable" && "$ONLY" != "nightly" && "$ONLY" != "all" ]]; then
        echo "--only must be one of: stable, nightly, all" >&2
        exit 2
      fi
      shift 2
      ;;
    --help|-h)
      sed -n '1,80p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$RESULT_DIR/logs"

ensure_checkpoint() {
  if [[ -f "$CHECKPOINT_HOST" ]]; then
    return 0
  fi
  mkdir -p "$(dirname "$CHECKPOINT_HOST")"
  echo "Checkpoint missing; downloading to $CHECKPOINT_HOST"
  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
    -O "$CHECKPOINT_HOST"
}

image_exists() {
  docker image inspect "$1" >/dev/null 2>&1
}

build_image_if_needed() {
  local image="$1"
  local dockerfile="$2"

  if [[ ! -f "$dockerfile" ]]; then
    echo "Missing Dockerfile: $dockerfile" >&2
    exit 1
  fi

  if [[ "$REBUILD" -eq 1 ]] || ! image_exists "$image"; then
    echo "Building $image from $dockerfile"
    docker build -f "$dockerfile" -t "$image" .
  else
    echo "Using existing image $image"
  fi
}

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

run_in_container() {
  local image="$1"
  shift
  local -a group_args
  mapfile -t group_args < <(group_add_args)

  docker run --rm \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device=/dev/kfd \
    --device=/dev/dri \
    "${group_args[@]}" \
    --ipc=host \
    -v "$ROOT_DIR:/workspace" \
    -v "$HOME/Downloads:/downloads:ro" \
    -w /workspace \
    "$image" "$@"
}

run_bench() {
  local stack_name="$1"
  local image="$2"
  local bench_name="$3"
  shift 3

  local log_file="$RESULT_DIR/logs/${stack_name}_${bench_name}.log"
  echo
  echo "=== Running ${stack_name}/${bench_name} ==="
  echo "Log: $log_file"

  # Important: do not let stale TunableOp results poison comparisons.
  run_in_container "$image" bash -lc \
    "unset PYTORCH_TUNABLEOP_ENABLED PYTORCH_TUNABLEOP_TUNING PYTORCH_TUNABLEOP_VERBOSE; \
     rm -f tunableop_results*.csv; \
     python3 sam_bench_amd.py \
       --checkpoint '$CHECKPOINT_CONTAINER' \
       --num-warmup '$NUM_WARMUP' \
       --num-runs '$NUM_RUNS' \
       $*" | tee "$log_file"
}

parse_logs() {
  python3 - "$RESULT_DIR" <<'PY'
import csv
import re
import sys
from pathlib import Path

result_dir = Path(sys.argv[1])
log_dir = result_dir / "logs"
rows = []

patterns = {
    "python": re.compile(r"^Python:\s*(.*)$"),
    "torch": re.compile(r"^PyTorch:\s*(.*)$"),
    "device": re.compile(r"^CUDA/HIP name:\s*(.*)$"),
    "hip": re.compile(r"^ROCm/HIP:\s*(.*)$"),
    "precision": re.compile(r"^precision:\s*(.*)$"),
    "points_per_batch": re.compile(r"^points_per_batch:\s*(.*)$"),
    "output_mode": re.compile(r"^output_mode:\s*(.*)$"),
    "encoder_section": re.compile(r"^=== Encoder-only diagnostic results ===$"),
    "amg_section": re.compile(r"^=== SAM1 AutomaticMaskGenerator results ===$"),
    "avg_latency": re.compile(r"^Average latency:\s*([0-9.]+) sec/image$"),
    "throughput": re.compile(r"^Throughput:\s*([0-9.]+) images/sec$"),
    "avg_masks": re.compile(r"^Average masks:\s*([0-9.]+)$"),
    "peak_mem": re.compile(r"^Peak GPU memory:\s*([0-9.]+) GB$"),
}

for path in sorted(log_dir.glob("*.log")):
    name = path.stem
    parts = name.split("_", 1)
    stack = parts[0]
    bench = parts[1] if len(parts) > 1 else name
    data = {
        "stack": stack,
        "bench": bench,
        "log_file": str(path.relative_to(result_dir)),
        "python": "",
        "torch": "",
        "hip_rocm": "",
        "device": "",
        "precision": "",
        "points_per_batch": "",
        "output_mode": "",
        "encoder_avg_s": "",
        "full_amg_avg_s": "",
        "throughput_img_s": "",
        "avg_masks": "",
        "peak_vram_gb": "",
    }
    section = None
    for line in path.read_text(errors="replace").splitlines():
        for key in ["python", "torch", "device", "hip", "precision", "points_per_batch", "output_mode"]:
            m = patterns[key].match(line)
            if m:
                out_key = "hip_rocm" if key == "hip" else key
                data[out_key] = m.group(1).strip()
        if patterns["encoder_section"].match(line):
            section = "encoder"
            continue
        if patterns["amg_section"].match(line):
            section = "amg"
            continue
        m = patterns["avg_latency"].match(line)
        if m:
            if section == "encoder":
                data["encoder_avg_s"] = m.group(1)
            elif section == "amg":
                data["full_amg_avg_s"] = m.group(1)
        m = patterns["throughput"].match(line)
        if m and section == "amg":
            data["throughput_img_s"] = m.group(1)
        m = patterns["avg_masks"].match(line)
        if m:
            data["avg_masks"] = m.group(1)
        m = patterns["peak_mem"].match(line)
        if m:
            data["peak_vram_gb"] = m.group(1)
    rows.append(data)

out_csv = result_dir / "benchmark_results.csv"
with out_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
    if rows:
        writer.writeheader()
        writer.writerows(rows)

latest = Path("results") / "latest.csv"
latest.parent.mkdir(exist_ok=True)
if rows:
    latest.write_text(out_csv.read_text())

print(f"Wrote {out_csv}")
if rows:
    print(f"Updated {latest}")
PY
}

main() {
  ensure_checkpoint

  if [[ "$ONLY" == "stable" || "$ONLY" == "all" ]]; then
    build_image_if_needed "$STABLE_IMAGE" "$STABLE_DOCKERFILE"
  fi
  if [[ "$ONLY" == "nightly" || "$ONLY" == "all" ]]; then
    build_image_if_needed "$NIGHTLY_IMAGE" "$NIGHTLY_DOCKERFILE"
  fi

  if [[ "$ONLY" == "stable" || "$ONLY" == "all" ]]; then
    run_bench stable "$STABLE_IMAGE" strict_fp32_binary \
      --points-per-batch 64 \
      --precision fp32 \
      --output-mode binary_mask \
      --profile-encoder

    run_bench stable "$STABLE_IMAGE" fast_amp_binary \
      --points-per-batch 128 \
      --precision amp-fp16 \
      --output-mode binary_mask \
      --profile-encoder

    run_bench stable "$STABLE_IMAGE" fast_amp_coco_rle \
      --points-per-batch 128 \
      --precision amp-fp16 \
      --output-mode coco_rle \
      --profile-encoder
  fi

  if [[ "$ONLY" == "nightly" || "$ONLY" == "all" ]]; then
    run_bench nightly "$NIGHTLY_IMAGE" strict_fp32_binary \
      --points-per-batch 64 \
      --precision fp32 \
      --output-mode binary_mask \
      --profile-encoder

    run_bench nightly "$NIGHTLY_IMAGE" fast_amp_binary \
      --points-per-batch 128 \
      --precision amp-fp16 \
      --output-mode binary_mask \
      --profile-encoder

    run_bench nightly "$NIGHTLY_IMAGE" fast_amp_coco_rle \
      --points-per-batch 128 \
      --precision amp-fp16 \
      --output-mode coco_rle \
      --profile-encoder
  fi

  parse_logs

  echo
  echo "Done. Results written to: $RESULT_DIR"
  echo "Latest CSV: results/latest.csv"
}

main "$@"
