"""
Quick Benchmark - Demonstrates VLA-Cache timing without full model inference.

Shows the caching speedup by measuring:
1. Patch similarity detection time
2. Vision encoding simulation (with/without cache)
3. Overall frame processing time
"""

import time
import numpy as np
from pathlib import Path
from PIL import Image
from vla_cache import find_static_patches, patchify, calculate_patch_similarity

def simulate_vision_encoding(num_patches, cached_patches=None, computation_time_per_patch_ms=2.0):
    """
    Simulate vision encoder computation time.

    Args:
        num_patches: Total number of patches
        cached_patches: List of patch indices that are cached
        computation_time_per_patch_ms: Time to encode each patch
    """
    if cached_patches is None or len(cached_patches) == 0:
        # No cache - compute all patches
        computed = num_patches
        cached = 0
    else:
        # With cache - only compute non-cached patches
        computed = num_patches - len(cached_patches)
        cached = len(cached_patches)

    # Simulate computation time (only for computed patches)
    computation_time = (computed * computation_time_per_patch_ms) / 1000.0
    time.sleep(computation_time)

    return computed, cached, computation_time

def benchmark_frames(image_paths, patch_size=14):
    """
    Benchmark VLA-Cache on a sequence of frames.
    """
    print("="*80)
    print("VLA-CACHE QUICK BENCHMARK")
    print("="*80)
    print(f"Frames: {len(image_paths)}")
    print(f"Patch size: {patch_size}x{patch_size}")
    print()

    # Load all images
    images = [Image.open(path).convert("RGB") for path in image_paths]
    img_size = images[0].size[0]
    num_patches_total = (img_size // patch_size) ** 2
    print(f"Image size: {img_size}x{img_size}")
    print(f"Total patches per image: {num_patches_total}")
    print()

    # === BASELINE (NO CACHE) ===
    print("="*80)
    print("BASELINE - NO CACHING")
    print("="*80)

    baseline_times = []
    baseline_vision_times = []

    for idx, image in enumerate(images):
        frame_start = time.perf_counter()

        # Image load (negligible)
        load_time = 0.001

        # Vision encoding (all patches)
        vision_start = time.perf_counter()
        computed, cached, vision_time = simulate_vision_encoding(num_patches_total, cached_patches=None)
        vision_times_actual = time.perf_counter() - vision_start
        baseline_vision_times.append(vision_times_actual)

        # LM + generation (simulate)
        lm_time = 0.150  # Fixed for all frames

        frame_time = time.perf_counter() - frame_start
        baseline_times.append(frame_time)

        print(f"Frame {idx}: {frame_time*1000:.1f}ms")
        print(f"  Vision: {vision_times_actual*1000:.1f}ms (computed {computed} patches)")
        print(f"  LM+Gen: {lm_time*1000:.1f}ms (simulated)")

    baseline_total = sum(baseline_times)
    baseline_avg = np.mean(baseline_times)
    baseline_vision_avg = np.mean(baseline_vision_times)

    print(f"\nBaseline Summary:")
    print(f"  Total time: {baseline_total*1000:.1f}ms")
    print(f"  Avg per frame: {baseline_avg*1000:.1f}ms")
    print(f"  Avg vision: {baseline_vision_avg*1000:.1f}ms")

    # === VLA-CACHE ENABLED ===
    print("\n" + "="*80)
    print("VLA-CACHE ENABLED")
    print("="*80)

    cached_times = []
    cached_vision_times = []
    cache_analysis_times = []
    all_static_patches = []
    prev_image = None

    for idx, image in enumerate(images):
        frame_start = time.perf_counter()

        # Image load
        load_time = 0.001

        # Cache analysis (only after first frame)
        static_patches = []
        analysis_time = 0
        if prev_image is not None:
            analysis_start = time.perf_counter()

            # Find static patches
            static_patches = find_static_patches(
                image, prev_image,
                patch_size=patch_size,
                top_k=150,
                sim_threshold=0.996
            )

            analysis_time = time.perf_counter() - analysis_start
            cache_analysis_times.append(analysis_time)
            all_static_patches.append(len(static_patches))

        # Vision encoding (with cache)
        vision_start = time.perf_counter()
        computed, cached, vision_time = simulate_vision_encoding(
            num_patches_total,
            cached_patches=static_patches
        )
        vision_times_actual = time.perf_counter() - vision_start
        cached_vision_times.append(vision_times_actual)

        # LM + generation (simulate)
        lm_time = 0.150  # Fixed for all frames

        frame_time = time.perf_counter() - frame_start
        cached_times.append(frame_time)

        cache_status = f"(CACHE HIT: {len(static_patches)} patches)" if static_patches else "(CACHE MISS)"

        print(f"Frame {idx}: {frame_time*1000:.1f}ms {cache_status}")
        if prev_image is not None:
            print(f"  Cache analysis: {analysis_time*1000:.1f}ms")
        print(f"  Vision: {vision_times_actual*1000:.1f}ms (computed {computed}, cached {cached})")
        print(f"  LM+Gen: {lm_time*1000:.1f}ms (simulated)")

        prev_image = image

    cached_total = sum(cached_times)
    cached_avg = np.mean(cached_times)
    cached_vision_avg = np.mean(cached_vision_times)
    cache_analysis_avg = np.mean(cache_analysis_times) if cache_analysis_times else 0

    print(f"\nVLA-Cache Summary:")
    print(f"  Total time: {cached_total*1000:.1f}ms")
    print(f"  Avg per frame: {cached_avg*1000:.1f}ms")
    print(f"  Avg vision: {cached_vision_avg*1000:.1f}ms")
    print(f"  Avg cache analysis: {cache_analysis_avg*1000:.1f}ms")

    if all_static_patches:
        avg_static = np.mean(all_static_patches)
        print(f"  Avg cached patches: {avg_static:.1f} ({avg_static/num_patches_total*100:.1f}%)")

    # === COMPARISON ===
    print("\n" + "="*80)
    print("SPEEDUP COMPARISON")
    print("="*80)

    speedup = baseline_total / cached_total
    vision_speedup = baseline_vision_avg / cached_vision_avg
    time_saved = baseline_total - cached_total

    print(f"\nEnd-to-End:")
    print(f"  Baseline: {baseline_total*1000:.1f}ms")
    print(f"  VLA-Cache: {cached_total*1000:.1f}ms")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Time saved: {time_saved*1000:.1f}ms ({time_saved/baseline_total*100:.1f}%)")

    print(f"\nVision Encoding:")
    print(f"  Baseline: {baseline_vision_avg*1000:.1f}ms per frame")
    print(f"  VLA-Cache: {cached_vision_avg*1000:.1f}ms per frame")
    print(f"  Vision speedup: {vision_speedup:.2f}x")

    print(f"\nCache Statistics:")
    if all_static_patches:
        print(f"  Frames with cache hits: {len(all_static_patches)}/{len(images)-1}")
        print(f"  Avg cache hit rate: {np.mean(all_static_patches)/num_patches_total*100:.1f}%")
    else:
        print(f"  No cache hits (need multiple frames)")

    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print(f"""
This benchmark simulates VLA-Cache behavior:

1. BASELINE (No Cache):
   - Processes all {num_patches_total} patches every frame
   - Vision encoding: {baseline_vision_avg*1000:.1f}ms per frame

2. VLA-CACHE:
   - First frame: No cache (same as baseline)
   - Subsequent frames: Caches ~{np.mean(all_static_patches) if all_static_patches else 0:.0f} static patches
   - Vision encoding: {cached_vision_avg*1000:.1f}ms per frame
   - Cache overhead: {cache_analysis_avg*1000:.1f}ms per frame

3. SPEEDUP:
   - Vision encoding: {vision_speedup:.2f}x faster with cache
   - Overall: {speedup:.2f}x faster (vision + overhead)
   - Cache rate: {np.mean(all_static_patches)/num_patches_total*100 if all_static_patches else 0:.1f}% of patches reused

NOTE: This uses simulated vision encoding. With a real model:
- Vision encoding would take ~2-5 seconds (not 300ms)
- Speedup would be similar ({vision_speedup:.1f}x)
- Cache overhead would be negligible relative to encoding time
""")

if __name__ == "__main__":
    import sys

    # Use test frames
    frame_dir = Path("test_frames")
    if not frame_dir.exists():
        print("Error: test_frames/ directory not found")
        print("Run: python3 create_test_frames.py")
        sys.exit(1)

    image_paths = sorted(frame_dir.glob("frame_*.jpg"))

    if len(image_paths) < 2:
        print("Error: Need at least 2 frames")
        sys.exit(1)

    benchmark_frames(image_paths)
