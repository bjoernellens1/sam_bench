# SAM1 Benchmark on AMD ROCm

This repository benchmarks Meta SAM1 ViT-B `SamAutomaticMaskGenerator` on AMD ROCm systems, with a focus on the AMD Radeon AI PRO R9700 / RDNA4 platform.

The benchmark intentionally keeps the original SAM1 automatic-mask-generation workload comparable while adding ROCm-friendly options such as `torch.inference_mode()`, AMP fp16, configurable `points_per_batch`, encoder-only diagnostics, and Docker-based stable/nightly environment comparison.

## Quick results on AMD Radeon AI PRO R9700

Platform: AMD Radeon AI PRO R9700, `gfx1201`, 32 GB VRAM  
Image: PyTorch dog image, `1213 x 1546`  
Model: SAM1 ViT-B, `sam_vit_b_01ec64.pth`  
Workload: `SamAutomaticMaskGenerator(points_per_side=32)`

| Mode | Stack | Precision | points_per_batch | output_mode | Latency | Masks | Notes |
|---|---|---:|---:|---|---:|---:|---|
| Strict comparable | Docker nightly PyTorch 2.14.0.dev20260617 + ROCm 7.2 | fp32 | 64 | binary_mask | 2.632 s/image | 66 | Best strict/fp32 result |
| Fast practical | Docker nightly PyTorch 2.14.0.dev20260617 + ROCm 7.2 | AMP fp16 | 128 | coco_rle | 1.452 s/image | 63 | Best practical result |
| Host fast baseline | Host PyTorch 2.11.0 + ROCm 7.2 | AMP fp16 | 128 | binary_mask | 1.975 s/image | 63 | Previous best before nightly |
| Stable Docker regression | Docker PyTorch 2.10.0 + ROCm 7.2.4 | AMP fp16 | 128 | binary_mask | 5.24 s/image | 63 | Full AMG path regressed |

Main finding: the direct host result was faster than the first stable Docker result because the stable Docker image used a different and slower PyTorch/operator stack. The ROCm nightly PyTorch stack currently gives the best R9700 result.

See [docs/r9700_rocm_sam_findings.md](docs/r9700_rocm_sam_findings.md) for the full analysis.

## AMD platform findings

Detailed platform notes are documented in:

- [`docs/r9700_rocm_sam_findings.md`](docs/r9700_rocm_sam_findings.md)
- [`docs/gfx1151_strix_halo_sam_findings.md`](docs/gfx1151_strix_halo_sam_findings.md)

Current best known results:

| Platform | Stack | Mode | Latency |
|---|---|---|---:|
| Radeon AI PRO R9700 / gfx1201 | torch 2.14 nightly | fp32 strict | 2.632 s/image |
| Radeon AI PRO R9700 / gfx1201 | torch 2.14 nightly | AMP fp16 + coco_rle | 1.452 s/image |
| Radeon 8060S / gfx1151 | torch 2.14 nightly | fp32 strict | 6.331 s/image |
| Radeon 8060S / gfx1151 | torch 2.14 nightly | AMP fp16 + coco_rle | 4.012 s/image |

## Download SAM checkpoint

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
  --directory-prefix ~/Downloads/
```

The Docker run scripts mount `~/Downloads` into the container as `/downloads`.

## Run directly on the host with uv

```bash
uv sync
uv run python sam_bench_amd.py
```

Strict comparable mode:

```bash
uv run python sam_bench_amd.py \
  --checkpoint ~/Downloads/sam_vit_b_01ec64.pth \
  --points-per-batch 64 \
  --precision fp32 \
  --output-mode binary_mask
```

Fast host mode:

```bash
uv run python sam_bench_amd.py \
  --checkpoint ~/Downloads/sam_vit_b_01ec64.pth \
  --points-per-batch 128 \
  --precision amp-fp16 \
  --output-mode coco_rle \
  --profile-encoder
```

## Run interactively in Docker

Build the nightly image:

```bash
docker build -f Dockerfile.rocm-nightly -t sam-bench:rocm-nightly .
```

Start a ROCm-enabled container:

```bash
./run_rocm_container.sh sam-bench:rocm-nightly
```

Inside the container, verify the stack:

```bash
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("hip:", torch.version.hip)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
PY
```

Run strict comparable benchmark:

```bash
python3 sam_bench_amd.py \
  --checkpoint /downloads/sam_vit_b_01ec64.pth \
  --points-per-batch 64 \
  --precision fp32 \
  --output-mode binary_mask
```

Run fast practical benchmark:

```bash
python3 sam_bench_amd.py \
  --checkpoint /downloads/sam_vit_b_01ec64.pth \
  --points-per-batch 128 \
  --precision amp-fp16 \
  --output-mode coco_rle \
  --profile-encoder
```

## Run the full Docker benchmark matrix

The script below builds Docker images if missing and then runs stable + nightly comparisons:

```bash
scripts/run_docker_benchmark_matrix.sh
```

Useful options:

```bash
# Force rebuild both images
scripts/run_docker_benchmark_matrix.sh --rebuild

# Only benchmark nightly
scripts/run_docker_benchmark_matrix.sh --only nightly

# Only benchmark stable/release image
scripts/run_docker_benchmark_matrix.sh --only stable
```

Environment variables:

```bash
# Override checkpoint path on the host
CHECKPOINT_HOST=$HOME/Downloads/sam_vit_b_01ec64.pth \
  scripts/run_docker_benchmark_matrix.sh

# Shorter smoke test
NUM_WARMUP=1 NUM_RUNS=2 scripts/run_docker_benchmark_matrix.sh --only nightly
```

Results are written to:

```text
results/<timestamp>/benchmark_results.csv
results/<timestamp>/logs/*.log
results/latest.csv
```

## Standard rows to report

For fair fp32 comparison against other GPUs:

```text
strict comparable:
fp32, points_per_batch=64, output_mode=binary_mask
```

For practical AMD throughput:

```text
fast practical:
AMP fp16, points_per_batch=128, output_mode=coco_rle
```

The fast AMP mode should be reported separately because it changed the observed mask count from 66 to 63 on the R9700 benchmark image.

## TunableOp warning

Do not enable ROCm/PyTorch TunableOp for this benchmark by default. It caused GPU memory access faults during SAM AMG tuning on the tested R9700 nightly stack and can leave stale `tunableop_results*.csv` files behind.

Before benchmarking, the scripts remove stale TunableOp files and unset the relevant variables:

```bash
unset PYTORCH_TUNABLEOP_ENABLED
unset PYTORCH_TUNABLEOP_TUNING
unset PYTORCH_TUNABLEOP_VERBOSE
rm -f tunableop_results*.csv
```

If the GPU gets wedged after a memory access fault, this reset command worked on the test system:

```bash
sudo amd-smi reset -G
```

## Files

- `sam_bench.py` - original/simple benchmark variant.
- `sam_bench_amd.py` - ROCm-friendly configurable benchmark.
- `Dockerfile.rocm-stable` - stable/release ROCm PyTorch image for comparison.
- `Dockerfile.rocm-nightly` - nightly ROCm PyTorch image for best current performance.
- `run_rocm_container.sh` - interactive ROCm Docker launcher.
- `scripts/run_docker_benchmark_matrix.sh` - builds images if needed and runs the stable/nightly matrix.
- `docs/r9700_rocm_sam_findings.md` - analysis and interpretation for the R9700 platform.
- `results/r9700_sam_benchmark_results.csv` - manually documented current benchmark table.
