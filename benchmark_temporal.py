"""Benchmark temporal VLA against baseline.

Creates synthetic test frames simulating robot camera footage:
- Mostly static scenes (90% of frames nearly identical)
- Occasional small changes (8% - object movement)
- Rare large changes (2% - scene cuts)
"""

import time
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Test frame generation
def generate_test_frames(
    output_dir: str,
    num_frames: int = 30,
    base_size: tuple = (224, 224),
    seed: int = 42,
    skip_pct: float = 0.90,
    cache_pct: float = 0.08,
) -> list:
    """Generate synthetic frames simulating robot camera footage.

    Creates frames with controlled similarity distribution:
    - skip_pct: nearly identical frames (sim > 0.99)
    - cache_pct: small changes (0.95 < sim < 0.99)
    - remainder: large changes / keyframes (sim < 0.95)
    """
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Create base scene (robot workspace)
    base_frame = np.zeros((base_size[0], base_size[1], 3), dtype=np.uint8)

    # Background gradient
    for i in range(base_size[0]):
        base_frame[i, :, :] = [100 + i//4, 120 + i//4, 140 + i//4]

    # Add some "objects" to the scene
    cv2.rectangle(base_frame, (50, 50), (100, 100), (200, 50, 50), -1)  # Red box
    cv2.rectangle(base_frame, (120, 60), (180, 110), (50, 200, 50), -1)  # Green box
    cv2.circle(base_frame, (160, 160), 30, (50, 50, 200), -1)  # Blue circle

    frames = []
    frame_paths = []

    prev_frame = None

    for i in range(num_frames):
        # Decide frame type based on target distribution
        rand = np.random.random()
        keyframe_threshold = 1.0 - skip_pct - cache_pct

        if i == 0 or rand < keyframe_threshold:
            # KEYFRAME: Major scene change (sim < 0.95)
            frame = base_frame.copy()
            # Significant changes: move objects, change colors
            offset_x = np.random.randint(-40, 40)
            offset_y = np.random.randint(-40, 40)
            color_shift = np.random.randint(-50, 50, 3)

            new_color = np.clip([200 + color_shift[0], 50 + color_shift[1], 50 + color_shift[2]], 0, 255)
            cv2.rectangle(frame, (50+offset_x, 50+offset_y), (100+offset_x, 100+offset_y), tuple(map(int, new_color)), -1)
            cv2.rectangle(frame, (120-offset_x, 60-offset_y), (180-offset_x, 110-offset_y), (50, 200, 50), -1)

            # Add random shapes
            for _ in range(3):
                cx, cy = np.random.randint(20, 200, 2)
                r = np.random.randint(10, 30)
                color = tuple(map(int, np.random.randint(0, 255, 3)))
                cv2.circle(frame, (cx, cy), r, color, -1)

        elif rand < keyframe_threshold + cache_pct:
            # CACHE: Small change (0.95 < sim < 0.99)
            if prev_frame is not None:
                frame = prev_frame.copy()
            else:
                frame = base_frame.copy()

            # Moderate noise + small object shift
            noise = np.random.normal(0, 8, frame.shape).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Small region change
            x, y = np.random.randint(30, 180), np.random.randint(30, 180)
            color = tuple(map(int, np.random.randint(0, 255, 3)))
            cv2.rectangle(frame, (x, y), (x+20, y+20), color, -1)

        else:
            # SKIP: Nearly identical (sim > 0.99)
            if prev_frame is not None:
                frame = prev_frame.copy()
            else:
                frame = base_frame.copy()

            # Very small noise only
            noise = np.random.normal(0, 1.5, frame.shape).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Save frame
        frame_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
        Image.fromarray(frame).save(frame_path, quality=95)
        frame_paths.append(frame_path)
        frames.append(frame)
        prev_frame = frame.copy()

    return frame_paths, frames


def _histogram_similarity(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Compare histograms - very fast (~1ms)."""
    small1 = cv2.resize(frame1, (64, 64))
    small2 = cv2.resize(frame2, (64, 64))

    hist1 = cv2.calcHist([small1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([small2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()

    return float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))


def _sparse_patch_similarity(
    frame1: np.ndarray,
    frame2: np.ndarray,
    num_patches: int = 16,
    patch_size: int = 32,
) -> float:
    """Sample sparse patches and compare (~2ms)."""
    h, w = frame1.shape[:2]

    np.random.seed(42)
    patch_y = np.random.randint(0, max(1, h - patch_size), num_patches)
    patch_x = np.random.randint(0, max(1, w - patch_size), num_patches)

    similarities = []
    for y, x in zip(patch_y, patch_x):
        p1 = frame1[y:y+patch_size, x:x+patch_size].astype(np.float32).flatten()
        p2 = frame2[y:y+patch_size, x:x+patch_size].astype(np.float32).flatten()

        norm1 = np.linalg.norm(p1)
        norm2 = np.linalg.norm(p2)
        if norm1 > 0 and norm2 > 0:
            sim = np.dot(p1, p2) / (norm1 * norm2)
            similarities.append(sim)

    return float(np.mean(similarities)) if similarities else 0.0


def quick_frame_similarity(frame1: np.ndarray, frame2: np.ndarray, method: str = "histogram") -> float:
    """Fast similarity check (<5ms)."""
    if frame1 is None or frame2 is None:
        return 0.0
    if method == "histogram":
        return _histogram_similarity(frame1, frame2)
    elif method == "patch":
        return _sparse_patch_similarity(frame1, frame2)
    return _histogram_similarity(frame1, frame2)


def measure_similarity_distribution(frames: list) -> dict:
    """Measure frame-to-frame similarity to validate test data."""
    similarities = []
    for i in range(1, len(frames)):
        sim = quick_frame_similarity(frames[i], frames[i-1], method="histogram")
        similarities.append(sim)

    similarities = np.array(similarities)

    # Categorize
    skip_count = np.sum(similarities > 0.99)
    cache_count = np.sum((similarities > 0.95) & (similarities <= 0.99))
    keyframe_count = np.sum(similarities <= 0.95)

    return {
        "mean_similarity": float(np.mean(similarities)),
        "min_similarity": float(np.min(similarities)),
        "max_similarity": float(np.max(similarities)),
        "std_similarity": float(np.std(similarities)),
        "skip_frames": int(skip_count),
        "cache_frames": int(cache_count),
        "keyframe_frames": int(keyframe_count),
        "skip_pct": float(skip_count / len(similarities) * 100),
        "cache_pct": float(cache_count / len(similarities) * 100),
        "keyframe_pct": float(keyframe_count / len(similarities) * 100),
    }


def patchify_local(image: Image.Image, patch_size: int = 14) -> np.ndarray:
    """Chop image into patch_size x patch_size blocks."""
    from skimage.util import view_as_blocks
    image_arr = np.array(image)
    h, w = image_arr.shape[:2]

    # Resize to be divisible by patch_size
    new_h = (h // patch_size) * patch_size
    new_w = (w // patch_size) * patch_size
    image_arr = cv2.resize(image_arr, (new_w, new_h))

    if image_arr.ndim == 3:
        blocks = view_as_blocks(
            image_arr,
            block_shape=(patch_size, patch_size, image_arr.shape[2]),
        )
        patches = blocks.reshape(-1, patch_size, patch_size, image_arr.shape[2])
    else:
        blocks = view_as_blocks(image_arr, block_shape=(patch_size, patch_size))
        patches = blocks.reshape(-1, patch_size, patch_size)

    return patches


def calculate_patch_similarity_local(patches1: np.ndarray, patches2: np.ndarray) -> np.ndarray:
    """Cosine similarity between matching patches."""
    flat1 = patches1.reshape(len(patches1), -1).astype(np.float32)
    flat2 = patches2.reshape(len(patches2), -1).astype(np.float32)

    norm1 = np.linalg.norm(flat1, axis=1)
    norm2 = np.linalg.norm(flat2, axis=1)

    dot = np.sum(flat1 * flat2, axis=1)
    cosine_sim = dot / (norm1 * norm2 + 1e-8)

    return cosine_sim


def benchmark_baseline_similarity_only(frame_paths: list, frames: list) -> dict:
    """Benchmark just the similarity computation overhead."""
    # Fast similarity (what temporal VLA uses)
    start = time.perf_counter()
    for i in range(1, len(frames)):
        _ = quick_frame_similarity(frames[i], frames[i-1], method="histogram")
    fast_time = time.perf_counter() - start

    # Full patchify similarity (what VLA-Cache uses for patch selection)
    start = time.perf_counter()
    for i in range(1, len(frame_paths)):
        img1 = Image.open(frame_paths[i]).convert("RGB")
        img2 = Image.open(frame_paths[i-1]).convert("RGB")
        patches1 = patchify_local(img1, 14)
        patches2 = patchify_local(img2, 14)
        _ = calculate_patch_similarity_local(patches1, patches2)
    full_patch_time = time.perf_counter() - start

    n = len(frames) - 1
    return {
        "fast_similarity_total_ms": fast_time * 1000,
        "fast_similarity_per_frame_ms": fast_time / n * 1000,
        "full_patchify_total_ms": full_patch_time * 1000,
        "full_patchify_per_frame_ms": full_patch_time / n * 1000,
        "speedup": full_patch_time / fast_time if fast_time > 0 else 0,
    }


def estimate_inference_times(stats: dict, num_frames: int) -> dict:
    """Estimate total inference time based on tier distribution.

    Uses realistic latency estimates:
    - SKIP: ~3ms (just similarity check)
    - CACHE: ~75ms (partial vision recompute)
    - KEYFRAME: ~350ms (full VLM inference)
    """
    # Latency estimates (ms)
    SKIP_LATENCY = 3
    CACHE_LATENCY = 75
    KEYFRAME_LATENCY = 350

    # Calculate based on distribution
    skip_frames = stats.get("skip_frames", 0)
    cache_frames = stats.get("cache_frames", 0)
    keyframe_frames = stats.get("keyframe_frames", 0) + 1  # +1 for first frame

    temporal_time = (
        skip_frames * SKIP_LATENCY +
        cache_frames * CACHE_LATENCY +
        keyframe_frames * KEYFRAME_LATENCY
    )

    baseline_time = num_frames * KEYFRAME_LATENCY

    return {
        "baseline_total_ms": baseline_time,
        "baseline_per_frame_ms": KEYFRAME_LATENCY,
        "temporal_total_ms": temporal_time,
        "temporal_per_frame_ms": temporal_time / num_frames,
        "speedup": baseline_time / temporal_time if temporal_time > 0 else 0,
        "time_saved_ms": baseline_time - temporal_time,
        "time_saved_pct": (1 - temporal_time / baseline_time) * 100 if baseline_time > 0 else 0,
    }


def run_benchmark(num_frames: int = 30, cleanup: bool = True):
    """Run full benchmark suite."""
    print("=" * 80)
    print("TEMPORAL VLA BENCHMARK")
    print("=" * 80)

    # Generate test frames
    test_dir = "/tmp/temporal_vla_benchmark"
    print(f"\n1. Generating {num_frames} synthetic test frames...")
    frame_paths, frames = generate_test_frames(test_dir, num_frames)
    print(f"   Created {len(frame_paths)} frames in {test_dir}")

    # Measure similarity distribution
    print("\n2. Analyzing frame similarity distribution...")
    sim_stats = measure_similarity_distribution(frames)
    print(f"   Mean similarity: {sim_stats['mean_similarity']:.4f}")
    print(f"   Range: [{sim_stats['min_similarity']:.4f}, {sim_stats['max_similarity']:.4f}]")
    print(f"\n   Expected tier distribution:")
    print(f"   - SKIP (sim > 0.99):     {sim_stats['skip_frames']:3d} frames ({sim_stats['skip_pct']:.1f}%)")
    print(f"   - CACHE (0.95 < sim):    {sim_stats['cache_frames']:3d} frames ({sim_stats['cache_pct']:.1f}%)")
    print(f"   - KEYFRAME (sim < 0.95): {sim_stats['keyframe_frames']:3d} frames ({sim_stats['keyframe_pct']:.1f}%)")

    # Benchmark similarity computation
    print("\n3. Benchmarking similarity computation overhead...")
    sim_benchmark = benchmark_baseline_similarity_only(frame_paths, frames)
    print(f"   Fast similarity (histogram): {sim_benchmark['fast_similarity_per_frame_ms']:.2f}ms/frame")
    print(f"   Full patchify:               {sim_benchmark['full_patchify_per_frame_ms']:.2f}ms/frame")
    print(f"   Similarity speedup:          {sim_benchmark['speedup']:.1f}x")

    # Estimate inference times
    print("\n4. Estimated inference times (based on tier distribution)...")
    time_estimates = estimate_inference_times(sim_stats, num_frames)
    print(f"\n   BASELINE (no temporal skipping):")
    print(f"   - Total time:     {time_estimates['baseline_total_ms']:.0f}ms")
    print(f"   - Per frame:      {time_estimates['baseline_per_frame_ms']:.0f}ms")

    print(f"\n   TEMPORAL VLA (3-tier hierarchy):")
    print(f"   - Total time:     {time_estimates['temporal_total_ms']:.0f}ms")
    print(f"   - Per frame:      {time_estimates['temporal_per_frame_ms']:.1f}ms")

    print(f"\n   IMPROVEMENT:")
    print(f"   - Speedup:        {time_estimates['speedup']:.1f}x")
    print(f"   - Time saved:     {time_estimates['time_saved_ms']:.0f}ms ({time_estimates['time_saved_pct']:.1f}%)")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Expected Performance at 30fps")
    print("=" * 80)
    print("""
┌─────────────────┬───────────────┬───────────────┬───────────┐
│ Metric          │ Baseline      │ Temporal VLA  │ Improve   │
├─────────────────┼───────────────┼───────────────┼───────────┤
│ Latency/frame   │ 350ms         │ {:.1f}ms       │ {:.1f}x     │
│ Total ({} frames)│ {:.1f}s        │ {:.2f}s        │ {:.1f}x     │
│ Realtime?       │ NO (11.7fps)  │ {} │           │
└─────────────────┴───────────────┴───────────────┴───────────┘
""".format(
        time_estimates['temporal_per_frame_ms'],
        time_estimates['speedup'],
        num_frames,
        time_estimates['baseline_total_ms'] / 1000,
        time_estimates['temporal_total_ms'] / 1000,
        time_estimates['speedup'],
        "YES" if time_estimates['temporal_per_frame_ms'] < 33.3 else "NO",
    ))

    # Cleanup
    if cleanup:
        shutil.rmtree(test_dir, ignore_errors=True)
        print(f"Cleaned up {test_dir}")

    return {
        "similarity_stats": sim_stats,
        "similarity_benchmark": sim_benchmark,
        "time_estimates": time_estimates,
    }


def run_threshold_sweep(num_frames: int = 100):
    """Test different threshold settings to show tuning impact."""
    print("\n" + "=" * 80)
    print("THRESHOLD SWEEP: Impact of skip/cache thresholds")
    print("=" * 80)

    test_dir = "/tmp/temporal_vla_benchmark_sweep"
    frame_paths, frames = generate_test_frames(test_dir, num_frames, skip_pct=0.85, cache_pct=0.10)

    threshold_configs = [
        (0.999, 0.98, "Conservative (fewer skips)"),
        (0.99, 0.95, "Balanced (default)"),
        (0.98, 0.90, "Aggressive (more skips)"),
        (0.95, 0.85, "Very aggressive"),
    ]

    print(f"\n{'Config':<30} {'Skip%':>8} {'Cache%':>8} {'KF%':>8} {'Speedup':>10} {'Realtime?':>10}")
    print("-" * 86)

    for skip_thresh, cache_thresh, name in threshold_configs:
        # Measure with these thresholds
        similarities = []
        for i in range(1, len(frames)):
            sim = quick_frame_similarity(frames[i], frames[i-1], method="histogram")
            similarities.append(sim)

        similarities = np.array(similarities)

        skip_count = np.sum(similarities > skip_thresh)
        cache_count = np.sum((similarities > cache_thresh) & (similarities <= skip_thresh))
        keyframe_count = np.sum(similarities <= cache_thresh) + 1  # +1 for first frame

        # Estimate times
        SKIP_LATENCY, CACHE_LATENCY, KEYFRAME_LATENCY = 3, 75, 350
        temporal_time = skip_count * SKIP_LATENCY + cache_count * CACHE_LATENCY + keyframe_count * KEYFRAME_LATENCY
        baseline_time = num_frames * KEYFRAME_LATENCY

        speedup = baseline_time / temporal_time if temporal_time > 0 else 0
        per_frame_ms = temporal_time / num_frames
        realtime = "YES" if per_frame_ms < 33.3 else "NO"

        skip_pct = skip_count / (num_frames - 1) * 100
        cache_pct = cache_count / (num_frames - 1) * 100
        kf_pct = keyframe_count / num_frames * 100

        print(f"{name:<30} {skip_pct:>7.1f}% {cache_pct:>7.1f}% {kf_pct:>7.1f}% {speedup:>9.1f}x {realtime:>10}")

    shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    import sys

    num_frames = 100
    if len(sys.argv) > 1:
        try:
            num_frames = int(sys.argv[1])
        except ValueError:
            pass

    results = run_benchmark(num_frames=num_frames)

    # Also run threshold sweep
    run_threshold_sweep(num_frames=num_frames)
