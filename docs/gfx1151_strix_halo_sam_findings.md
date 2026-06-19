# SAM1 Benchmark on AMD Strix Halo / gfx1151 / Radeon 8060S

## Platform

Test platform:

- GPU: Radeon 8060S Graphics
- Architecture target: gfx1151 / RDNA3.5 APU
- ROCm/HIP runtime in Docker: 7.2.53211
- Reported GPU-accessible memory from PyTorch: 124.00 GB
- Model: SAM1 ViT-B
- Image: PyTorch dog image, 1213 x 1546 RGB
- Automatic mask generator settings:
  - `points_per_side=32`
  - `pred_iou_thresh=0.88`
  - `stability_score_thresh=0.95`
  - 2 warmup runs
  - 10 benchmark runs

Important interpretation: the reported 124 GB is not discrete VRAM. On RDNA3.5 APUs, ROCm maps system memory for GPU use through GPUVM/GTT. This gives excellent capacity, but it is not comparable to dedicated GDDR6/GDDR6X bandwidth on a discrete GPU.

## Result table

| Stack | Mode | Output | points_per_batch | Encoder avg | Full AMG avg | Throughput | Masks | Peak GPU mem |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stable Docker, torch 2.10.0+rocm7.2.4 | fp32 | binary_mask | 64 | 0.655 s | 12.763 s | 0.08 img/s | 66 | 2.99 GB |
| stable Docker, torch 2.10.0+rocm7.2.4 | amp-fp16 | binary_mask | 128 | 0.190 s | 10.853 s | 0.09 img/s | 62 | 5.57 GB |
| stable Docker, torch 2.10.0+rocm7.2.4 | amp-fp16 | coco_rle | 128 | 0.190 s | 10.844 s | 0.09 img/s | 62 | 5.57 GB |
| nightly Docker, torch 2.14.0.dev20260617+rocm7.2 | fp32 | binary_mask | 64 | 0.637 s | 6.331 s | 0.16 img/s | 66 | 2.92 GB |
| nightly Docker, torch 2.14.0.dev20260617+rocm7.2 | amp-fp16 | binary_mask | 128 | 0.248 s | 4.023 s | 0.25 img/s | 61 | 5.35 GB |
| nightly Docker, torch 2.14.0.dev20260617+rocm7.2 | amp-fp16 | coco_rle | 128 | 0.248 s | 4.012 s | 0.25 img/s | 61 | 5.35 GB |

## Main observations

### 1. ROCm/PyTorch nightly is essential on gfx1151

The stable PyTorch 2.10 stack is very slow:

- fp32: 12.763 s/image
- AMP fp16: 10.853 s/image

The nightly PyTorch 2.14 stack improves this strongly:

- fp32: 6.331 s/image
- AMP fp16: 4.012 s/image

Speedups:

- fp32 nightly vs stable: about 2.0x faster
- AMP nightly vs stable: about 2.7x faster

### 2. AMP helps, but less dramatically than on the R9700

On nightly:

- fp32: 6.331 s/image, 66 masks
- AMP fp16: 4.012 s/image, 61 masks

AMP gives about 1.58x speedup, but changes the final mask count.

### 3. `coco_rle` does not materially change runtime

On nightly AMP:

- binary_mask: 4.023 s/image
- coco_rle: 4.012 s/image

The difference is only about 0.3 percent. Output serialization is not the bottleneck.

### 4. Stable stack shows many hipBLASLt fallback warnings

The stable PyTorch 2.10 stack emits many warnings of the form:

```text
HIPBLAS_STATUS_NOT_SUPPORTED
Will attempt to recover by calling cublas instead.
