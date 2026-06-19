#!/usr/bin/env python3
"""SAM1 AutomaticMaskGenerator benchmark with ROCm/AMD-friendly options."""

from __future__ import annotations

import argparse
import contextlib
import platform
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry


IMAGE_URL = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"


def sync_if_gpu(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def get_autocast_context(device: str, precision: str):
    """Return an autocast context.

    PyTorch uses device_type='cuda' for both NVIDIA CUDA and ROCm/HIP builds.
    """
    if device != "cuda":
        return contextlib.nullcontext()

    if precision == "fp32":
        return contextlib.nullcontext()
    if precision == "amp-fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if precision == "amp-bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    raise ValueError(f"Unknown precision mode: {precision}")


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    k = (len(values) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def print_device_info(device: str) -> None:
    print("\n=== Device info ===")
    print(f"Python:        {platform.python_version()}")
    print(f"PyTorch:       {torch.__version__}")
    print(f"Device:        {device}")

    if device == "cuda":
        print(f"CUDA/HIP name: {torch.cuda.get_device_name(0)}")
        print(f"Device count:  {torch.cuda.device_count()}")

        hip_version = getattr(torch.version, "hip", None)
        cuda_version = getattr(torch.version, "cuda", None)

        if hip_version:
            print(f"ROCm/HIP:      {hip_version}")
        if cuda_version:
            print(f"CUDA runtime:  {cuda_version}")

        props = torch.cuda.get_device_properties(0)
        print(f"Total memory:  {props.total_memory / 1024**3:.2f} GB")


def load_image(image_path: Path) -> np.ndarray:
    if not image_path.exists():
        print(f"Downloading test image to {image_path}")
        r = requests.get(IMAGE_URL, timeout=30)
        r.raise_for_status()
        image_path.write_bytes(r.content)

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def benchmark_full_amg(
    *,
    mask_generator: SamAutomaticMaskGenerator,
    image_rgb: np.ndarray,
    device: str,
    precision: str,
    num_warmup: int,
    num_runs: int,
) -> tuple[list[float], list[int]]:
    print("\n=== Warmup: full automatic mask generation ===")
    for i in range(num_warmup):
        with torch.inference_mode(), get_autocast_context(device, precision):
            _ = mask_generator.generate(image_rgb)
        sync_if_gpu(device)
        print(f"Warmup {i + 1}/{num_warmup} done")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print("\n=== Benchmark: full automatic mask generation ===")
    times: list[float] = []
    mask_counts: list[int] = []

    for i in range(num_runs):
        sync_if_gpu(device)
        start = time.perf_counter()

        with torch.inference_mode(), get_autocast_context(device, precision):
            masks = mask_generator.generate(image_rgb)

        sync_if_gpu(device)
        elapsed = time.perf_counter() - start

        times.append(elapsed)
        mask_counts.append(len(masks))

        print(f"Run {i + 1:02d}: {elapsed:.3f}s, {len(masks)} masks")

    return times, mask_counts


def benchmark_encoder_only(
    *,
    sam,
    image_rgb: np.ndarray,
    device: str,
    precision: str,
    num_warmup: int,
    num_runs: int,
) -> list[float]:
    """Diagnostic only: measure SamPredictor.set_image().

    This includes preprocessing and the ViT image encoder. It does not replace
    the comparable full AutomaticMaskGenerator benchmark.
    """
    predictor = SamPredictor(sam)

    print("\n=== Warmup: encoder-only diagnostic ===")
    for i in range(num_warmup):
        with torch.inference_mode(), get_autocast_context(device, precision):
            predictor.set_image(image_rgb)
        sync_if_gpu(device)
        print(f"Encoder warmup {i + 1}/{num_warmup} done")

    print("\n=== Benchmark: encoder-only diagnostic ===")
    times: list[float] = []

    for i in range(num_runs):
        sync_if_gpu(device)
        start = time.perf_counter()

        with torch.inference_mode(), get_autocast_context(device, precision):
            predictor.set_image(image_rgb)

        sync_if_gpu(device)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        print(f"Encoder run {i + 1:02d}: {elapsed:.3f}s")

    return times


def summarize_times(times: list[float], label: str) -> None:
    avg = statistics.mean(times)
    med = statistics.median(times)
    p90 = percentile(times, 90)
    p95 = percentile(times, 95)
    throughput = 1.0 / avg

    print(f"\n=== {label} ===")
    print(f"Average latency: {avg:.3f} sec/image")
    print(f"Median latency:  {med:.3f} sec/image")
    print(f"P90 latency:     {p90:.3f} sec/image")
    print(f"P95 latency:     {p95:.3f} sec/image")
    print(f"Throughput:      {throughput:.2f} images/sec")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAM1 AutomaticMaskGenerator benchmark with AMD/ROCm-friendly options."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="~/Downloads/sam_vit_b_01ec64.pth",
        help="Path to SAM checkpoint.",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default="benchmark_image.jpg",
        help="Local image path. If missing, the dog image is downloaded.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="vit_b",
        choices=["vit_b", "vit_l", "vit_h"],
    )
    parser.add_argument("--num-warmup", type=int, default=2)
    parser.add_argument("--num-runs", type=int, default=10)

    # Keep original defaults for comparability.
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument(
        "--points-per-batch",
        type=int,
        default=64,
        help=(
            "Original SAM default is 64. Try 128 on AMD discrete GPUs. "
            "256 usually increases VRAM without improving this benchmark."
        ),
    )
    parser.add_argument("--pred-iou-thresh", type=float, default=0.88)
    parser.add_argument("--stability-score-thresh", type=float, default=0.95)
    parser.add_argument(
        "--output-mode",
        type=str,
        default="binary_mask",
        choices=["binary_mask", "uncompressed_rle", "coco_rle"],
        help=(
            "Keep binary_mask for strict comparability. coco_rle is a small "
            "practical optimization and requires pycocotools."
        ),
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "amp-fp16", "amp-bf16"],
        help=(
            "fp32 is most comparable. amp-fp16 is faster on R9700 but can change "
            "mask counts slightly."
        ),
    )
    parser.add_argument(
        "--model-half",
        action="store_true",
        help=(
            "Convert model weights to fp16. More aggressive than autocast. "
            "Use only with --precision amp-fp16 and validate results."
        ),
    )
    parser.add_argument(
        "--matmul-precision",
        type=str,
        default="high",
        choices=["highest", "high", "medium"],
        help="Controls torch.set_float32_matmul_precision().",
    )
    parser.add_argument(
        "--profile-encoder",
        action="store_true",
        help="Also benchmark SamPredictor.set_image() as encoder-only diagnostic.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision(args.matmul_precision)

    checkpoint_path = Path(args.checkpoint).expanduser()
    image_path = Path(args.image_path).expanduser()

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Download with:\n"
            "wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth "
            "--directory-prefix ~/Downloads/"
        )

    image_rgb = load_image(image_path)
    print(f"Image shape: {image_rgb.shape}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print_device_info(device)

    print("\n=== Benchmark config ===")
    print(f"Model:                 {args.model_type}")
    print(f"Checkpoint:            {checkpoint_path}")
    print(f"Warmup runs:           {args.num_warmup}")
    print(f"Benchmark runs:        {args.num_runs}")
    print(f"points_per_side:       {args.points_per_side}")
    print(f"points_per_batch:      {args.points_per_batch}")
    print(f"pred_iou_thresh:       {args.pred_iou_thresh}")
    print(f"stability_score:       {args.stability_score_thresh}")
    print(f"output_mode:           {args.output_mode}")
    print(f"precision:             {args.precision}")
    print(f"model_half:            {args.model_half}")
    print(f"matmul_precision:      {args.matmul_precision}")

    print("\n=== Loading SAM1 ===")
    sam = sam_model_registry[args.model_type](checkpoint=str(checkpoint_path))
    sam.to(device)
    sam.eval()

    if args.model_half:
        if args.precision != "amp-fp16":
            raise ValueError("--model-half should only be used together with --precision amp-fp16")
        if device != "cuda":
            raise ValueError("--model-half only makes sense on GPU")
        sam.half()

    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        output_mode=args.output_mode,
    )

    if args.profile_encoder:
        encoder_times = benchmark_encoder_only(
            sam=sam,
            image_rgb=image_rgb,
            device=device,
            precision=args.precision,
            num_warmup=args.num_warmup,
            num_runs=args.num_runs,
        )
        summarize_times(encoder_times, "Encoder-only diagnostic results")

    full_times, mask_counts = benchmark_full_amg(
        mask_generator=mask_generator,
        image_rgb=image_rgb,
        device=device,
        precision=args.precision,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
    )

    summarize_times(full_times, "SAM1 AutomaticMaskGenerator results")

    print(f"Average masks:   {statistics.mean(mask_counts):.1f}")
    print(f"Mask counts:     {mask_counts}")

    if device == "cuda":
        mem_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak GPU memory: {mem_gb:.2f} GB")


if __name__ == "__main__":
    main()
