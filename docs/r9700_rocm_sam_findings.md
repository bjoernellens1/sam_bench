# AMD Radeon AI PRO R9700 SAM1 Benchmark Findings

Date: 2026-06-19  
Platform under test: AMD Radeon AI PRO R9700, `gfx1201`, 32 GB VRAM  
Benchmark: SAM1 ViT-B `SamAutomaticMaskGenerator` on the PyTorch dog image, `1213 x 1546`, `points_per_side=32`

## Executive summary

The AMD Radeon AI PRO R9700 result is strongly dependent on the PyTorch/ROCm stack. The original stable Docker image based on PyTorch 2.10 + ROCm 7.2.4 was much slower than the direct host environment, even though the ViT encoder itself was fast. Moving to a ROCm/PyTorch nightly stack improved the full SAM automatic mask generation path dramatically.

Best measured results on this platform:

| Result type | Stack | Precision | points_per_batch | output_mode | Latency | Masks | Notes |
|---|---|---:|---:|---|---:|---:|---|
| Strict comparable | Docker nightly PyTorch 2.14.0.dev20260617 + ROCm 7.2 | fp32 | 64 | binary_mask | 2.632 s/image | 66 | Best strict/fp32 result |
| Fast practical | Docker nightly PyTorch 2.14.0.dev20260617 + ROCm 7.2 | AMP fp16 | 128 | coco_rle | 1.452 s/image | 63 | Best practical result |
| Host fast baseline | Host PyTorch 2.11.0 + ROCm 7.2 | AMP fp16 | 128 | binary_mask | 1.975 s/image | 63 | Previous best before nightly |
| Bad Docker baseline | Docker PyTorch 2.10.0 + ROCm 7.2.4 | AMP fp16 | 128 | binary_mask | 5.240-5.265 s/image | 63 | Regressed AMG path |

Compared with the first stable Docker AMP result, nightly AMP + `coco_rle` is roughly `5.24 / 1.452 = 3.61x` faster. Compared with the host PyTorch 2.11 AMP result, nightly AMP + `coco_rle` is roughly `1.975 / 1.452 = 1.36x` faster.

## Benchmark workload

The benchmark is intentionally kept close to the original SAM1 automatic mask generation workload:

- Model: `sam_vit_b_01ec64.pth`
- Image: `https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg`
- Image resolution: `1213 x 1546`
- `points_per_side=32`
- `pred_iou_thresh=0.88`
- `stability_score_thresh=0.95`
- Default strict output: `binary_mask`

Important: `SamAutomaticMaskGenerator(points_per_side=32)` evaluates a grid of 1024 point prompts. With `points_per_batch=64`, this becomes 16 decoder batches. Therefore this benchmark is not just a single ViT encoder pass. It also includes repeated mask decoder execution, full-resolution mask upsampling/filtering, stability scoring, box extraction, NMS-like filtering, and mask serialization.

## Observed result table

| ID | Environment | PyTorch | HIP/ROCm | Precision | points_per_batch | output_mode | Encoder avg [s] | Full AMG avg [s] | Throughput [img/s] | Masks | Peak VRAM [GB] | Notes |
|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|
| host_fp32_b64_binary | Host uv | 2.11.0+rocm7.2 | 7.2.26015 | fp32 | 64 | binary_mask | n/a | 3.335 | 0.30 | 66 | 2.96 | Clean comparable host baseline |
| host_fp32_b128_binary | Host uv | 2.11.0+rocm7.2 | 7.2.26015 | fp32 | 128 | binary_mask | 0.457 | 3.362 | 0.30 | 66 | 5.49 | Batch size did not help |
| host_amp_b128_binary | Host uv | 2.11.0+rocm7.2 | 7.2.26015 | amp-fp16 | 128 | binary_mask | 0.129 | 1.975 | 0.51 | 63 | 5.39 | AMP is a major lever, but masks differ |
| host_amp_b256_binary | Host uv | 2.11.0+rocm7.2 | 7.2.26015 | amp-fp16 | 256 | binary_mask | 0.141 | 1.966 | 0.51 | 61 | 10.21 | No speed gain, more VRAM, fewer masks |
| docker210_fp32_b64_binary | Docker stable | 2.10.0+rocm7.2.4.git3d3aa833 | 7.2.53211 | fp32 | 64 | binary_mask | n/a | 6.431 | 0.16 | 66 | 2.96 | Bad regression in full AMG |
| docker210_amp_b128_binary | Docker stable | 2.10.0+rocm7.2.4.git3d3aa833 | 7.2.53211 | amp-fp16 | 128 | binary_mask | 0.075-0.080 | 5.240-5.265 | 0.19 | 63 | 5.57 | Encoder fast, full AMG slow |
| docker210_amp_b128_uncompressed_rle | Docker stable | 2.10.0+rocm7.2.4.git3d3aa833 | 7.2.53211 | amp-fp16 | 128 | uncompressed_rle | 0.075 | 5.168 | 0.19 | 63 | 5.57 | Output mode did not solve stable regression |
| docker210_amp_b128_coco_rle | Docker stable | 2.10.0+rocm7.2.4.git3d3aa833 | 7.2.53211 | amp-fp16 | 128 | coco_rle | 0.075 | 5.161 | 0.19 | 63 | 5.57 | Output mode did not solve stable regression |
| nightly_fp32_b64_binary | Docker nightly | 2.14.0.dev20260617+rocm7.2 | 7.2.53211 | fp32 | 64 | binary_mask | n/a | 2.632 | 0.38 | 66 | 2.96 | Best strict comparable result |
| nightly_amp_b128_binary | Docker nightly | 2.14.0.dev20260617+rocm7.2 | 7.2.53211 | amp-fp16 | 128 | binary_mask | 0.076 | 1.478 | 0.68 | 63 | 5.39 | Best comparable fast result |
| nightly_amp_b128_coco_rle | Docker nightly | 2.14.0.dev20260617+rocm7.2 | 7.2.53211 | amp-fp16 | 128 | coco_rle | 0.076 | 1.452 | 0.69 | 63 | 5.39 | Best practical result |

## Interpretation

### 1. PyTorch/ROCm nightly is the largest stable-stack lever

The Docker PyTorch 2.10 environment was much slower in the full AMG benchmark than the host PyTorch 2.11 environment. The nightly PyTorch 2.14 environment improved both fp32 and AMP full AMG performance:

- fp32 strict: `6.431 s -> 2.632 s` versus Docker PyTorch 2.10, about `2.44x` faster.
- AMP fast: `5.24 s -> 1.452 s` versus Docker PyTorch 2.10, about `3.61x` faster.
- AMP fast: `1.975 s -> 1.452 s` versus host PyTorch 2.11, about `1.36x` faster.

The important observation is that the Docker PyTorch 2.10 encoder-only AMP path was already very fast at about `0.075 s`, but the full AMG path was still around `5.24 s`. Therefore the regression was not general GPU access or container overhead; it was likely in the decoder/post-encoder tensor operator path.

### 2. AMP fp16 is the second major lever

On the host PyTorch 2.11 stack:

- fp32 batch 64: `3.335 s`, 66 masks.
- AMP fp16 batch 128: `1.975 s`, 63 masks.

On the nightly stack:

- fp32 batch 64: `2.632 s`, 66 masks.
- AMP fp16 batch 128 + `coco_rle`: `1.452 s`, 63 masks.

AMP is useful as a fast practical mode. It is not strictly identical to fp32 because the number of resulting masks changed from 66 to 63 in the measured run.

### 3. `points_per_batch` is not a major lever

Increasing `points_per_batch` from 64 to 128 did not improve fp32 throughput on the host (`3.335 s -> 3.362 s`) and increased memory use. Increasing to 256 in AMP did not improve runtime and increased peak VRAM from about 5.4 GB to about 10.2 GB while reducing mask count from 63 to 61.

Recommendation: use 64 for strict/fp32 comparison and 128 for the current fast AMP mode. Avoid 256 as a default.

### 4. `output_mode` is only a small lever on nightly

Nightly AMP with `binary_mask` was `1.478 s/image`; nightly AMP with `coco_rle` was `1.452 s/image`. This is only about a 1.8% improvement. Therefore final output serialization is not the main bottleneck.

On Docker PyTorch 2.10, changing output mode also barely helped (`5.240 s -> 5.161 s`), confirming that the bad stable Docker result is not mainly caused by final binary mask materialization.

### 5. The workload is probably hybrid CPU/GPU orchestration-bound, not purely CPU-bound

A live monitor screenshot showed high GPU utilization around 96% while the Python process used roughly one CPU core. This suggests the benchmark is not simply CPU-bound. The expensive path is still GPU-heavy, but SAM AMG has serial Python/control-flow orchestration and many smaller tensor operations after the encoder.

A better description is:

> The R9700 is not encoder-bound anymore. The remaining cost is dominated by automatic mask generation after the encoder: repeated prompt decoder calls, full-resolution mask operations, filtering, stability scoring, and NMS-like postprocessing. There is likely single-thread Python orchestration overhead, but the GPU is still actively used.

### 6. TunableOp should be avoided for this benchmark

`PYTORCH_TUNABLEOP_ENABLED=1` with tuning enabled caused GPU memory access faults on the nightly stack. A stale `tunableop_results0.csv` from PyTorch 2.14 also poisoned a later PyTorch 2.10 run and triggered validator mismatch warnings before another memory access fault.

Do not use TunableOp for this benchmark unless it is isolated on a simple matmul-only microbenchmark.

Recommended cleanup before benchmarking:

```bash
unset PYTORCH_TUNABLEOP_ENABLED
unset PYTORCH_TUNABLEOP_TUNING
unset PYTORCH_TUNABLEOP_VERBOSE
rm -f tunableop_results*.csv
```

If a memory access fault leaves the GPU in a bad state, the observed working reset command was:

```bash
sudo amd-smi reset -G
```

## Recommended standard benchmark rows

### Strict comparable row

```bash
python3 sam_bench_amd.py \
  --checkpoint /downloads/sam_vit_b_01ec64.pth \
  --points-per-batch 64 \
  --precision fp32 \
  --output-mode binary_mask
```

Report as:

```text
R9700 strict comparable:
Docker nightly PyTorch 2.14.0.dev20260617+rocm7.2
fp32, points_per_batch=64, binary_mask
2.632 s/image, 66 masks, 2.96 GB peak VRAM
```

### Fast practical row

```bash
python3 sam_bench_amd.py \
  --checkpoint /downloads/sam_vit_b_01ec64.pth \
  --points-per-batch 128 \
  --precision amp-fp16 \
  --output-mode coco_rle \
  --profile-encoder
```

Report as:

```text
R9700 fast practical:
Docker nightly PyTorch 2.14.0.dev20260617+rocm7.2
AMP fp16, points_per_batch=128, coco_rle
1.452 s/image, 63 masks, 5.39 GB peak VRAM
```

## Practical conclusions

- The original slow Docker result was caused by the software stack, not by Docker overhead itself.
- ROCm/PyTorch nightly is currently the best available stack for this SAM AMG workload on R9700.
- AMP fp16 is essential for practical performance, but should be reported separately because it changes mask count.
- `points_per_batch=128` is a reasonable fast-mode value; `256` is not worth it.
- `coco_rle` is slightly faster and more practical, but not a major lever.
- TunableOp is unstable for this workload and should not be part of the standard benchmark.
