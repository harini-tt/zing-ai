"""Efficient VLA: Combines temporal frame skipping + hierarchical token merging.

Two-stage optimization:
1. Temporal skipping - decide skip/cache/keyframe per frame (20x speedup)
2. Token merging - reduce token count within processed frames (30-100x reduction)

Combined effect: Near real-time VLA on long video streams.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
import time

import numpy as np
from PIL import Image

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None

try:
    from .token_merging import (
        HierarchicalTokenMerger,
        TokenMergingConfig,
    )
except ImportError:
    from token_merging import (
        HierarchicalTokenMerger,
        TokenMergingConfig,
    )


@dataclass
class EfficientVLAConfig:
    """Configuration for efficient VLA pipeline."""

    # Temporal skipping thresholds
    skip_threshold: float = 0.99
    cache_threshold: float = 0.95
    max_skip_streak: int = 10

    # Token merging settings
    spatial_merge_ratio: float = 0.5
    temporal_merge_ratio: float = 0.3
    saliency_top_k_ratio: float = 0.2
    temporal_segment_size: int = 5

    # Pipeline settings
    enable_temporal_skip: bool = True
    enable_token_merging: bool = True
    enable_saliency: bool = True


@dataclass
class PipelineStats:
    """Track combined pipeline statistics."""

    # Frame-level stats
    total_frames: int = 0
    skipped_frames: int = 0
    cached_frames: int = 0
    keyframes: int = 0

    # Token-level stats
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    spatial_merged: int = 0
    temporal_merged: int = 0
    saliency_pruned: int = 0

    # Timing
    total_time_ms: float = 0.0
    similarity_time_ms: float = 0.0
    inference_time_ms: float = 0.0
    merging_time_ms: float = 0.0

    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        return {
            # Frame stats
            "total_frames": self.total_frames,
            "skipped_frames": self.skipped_frames,
            "skip_rate": self.skipped_frames / max(1, self.total_frames),
            "cached_frames": self.cached_frames,
            "keyframes": self.keyframes,

            # Token stats
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "token_compression": self.total_input_tokens / max(1, self.total_output_tokens),
            "token_reduction": 1.0 - (self.total_output_tokens / max(1, self.total_input_tokens)),

            # Timing
            "total_time_ms": self.total_time_ms,
            "avg_time_per_frame_ms": self.total_time_ms / max(1, self.total_frames),
            "similarity_time_ms": self.similarity_time_ms,
            "inference_time_ms": self.inference_time_ms,
            "merging_time_ms": self.merging_time_ms,
        }


def quick_histogram_similarity(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Fast histogram-based frame similarity (~1ms)."""
    import cv2

    small1 = cv2.resize(frame1, (64, 64))
    small2 = cv2.resize(frame2, (64, 64))

    hist1 = cv2.calcHist([small1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([small2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()

    return float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))


class EfficientVLAPipeline:
    """Combined temporal skipping + token merging pipeline."""

    def __init__(self, config: Optional[EfficientVLAConfig] = None):
        self.config = config or EfficientVLAConfig()

        # Token merging
        self.token_merger = HierarchicalTokenMerger(TokenMergingConfig(
            spatial_merge_ratio=self.config.spatial_merge_ratio,
            temporal_merge_ratio=self.config.temporal_merge_ratio,
            saliency_top_k_ratio=self.config.saliency_top_k_ratio,
            temporal_segment_size=self.config.temporal_segment_size,
            use_saliency=self.config.enable_saliency,
        ))

        # State
        self.last_frame: Optional[np.ndarray] = None
        self.last_output: Optional[str] = None
        self.last_tokens: Optional[np.ndarray] = None
        self.skip_count: int = 0

        self.stats = PipelineStats()

    def decide_frame_action(self, frame: np.ndarray) -> str:
        """Decide whether to skip, cache, or fully process frame."""
        if not self.config.enable_temporal_skip:
            return "keyframe"

        if self.last_frame is None:
            return "keyframe"

        sim_start = time.perf_counter()
        similarity = quick_histogram_similarity(frame, self.last_frame)
        self.stats.similarity_time_ms += (time.perf_counter() - sim_start) * 1000

        # Check skip streak
        if self.skip_count >= self.config.max_skip_streak:
            return "keyframe"

        if similarity > self.config.skip_threshold:
            return "skip"
        elif similarity > self.config.cache_threshold:
            return "cache"
        else:
            return "keyframe"

    def process_tokens(
        self,
        tokens: np.ndarray,
        attention_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Process tokens through merging pipeline."""
        if not self.config.enable_token_merging:
            return tokens

        merge_start = time.perf_counter()
        merged_tokens, frame_stats = self.token_merger.process_frame(
            tokens, attention_weights
        )
        self.stats.merging_time_ms += (time.perf_counter() - merge_start) * 1000

        self.stats.spatial_merged += frame_stats.get("spatial_merged", 0)
        self.stats.saliency_pruned += frame_stats.get("saliency_pruned", 0)

        return merged_tokens

    def flush_token_buffer(self) -> Optional[np.ndarray]:
        """Flush token buffer for hierarchical pooling."""
        if self.token_merger.should_flush():
            pooled, _ = self.token_merger.flush_segment()
            if pooled is not None:
                self.stats.temporal_merged += (
                    sum(t.shape[0] for t in self.token_merger.frame_buffer) -
                    pooled.shape[0]
                )
            return pooled
        return None

    def process_frame(
        self,
        frame: np.ndarray,
        vision_tokens: Optional[np.ndarray] = None,
        attention_weights: Optional[np.ndarray] = None,
    ) -> Tuple[str, Optional[np.ndarray], Optional[str]]:
        """Process a single frame through the pipeline.

        Args:
            frame: RGB image as numpy array
            vision_tokens: Pre-computed vision tokens (if available)
            attention_weights: Attention weights for saliency

        Returns:
            (action, merged_tokens, output)
            - action: "skip" | "cache" | "keyframe"
            - merged_tokens: Reduced vision tokens (or None if skipped)
            - output: Model output string (reused if skipped)
        """
        frame_start = time.perf_counter()
        self.stats.total_frames += 1

        # Decide action
        action = self.decide_frame_action(frame)

        if action == "skip":
            self.skip_count += 1
            self.stats.skipped_frames += 1
            self.stats.total_time_ms += (time.perf_counter() - frame_start) * 1000
            return action, None, self.last_output

        # Reset skip count
        self.skip_count = 0

        if action == "cache":
            self.stats.cached_frames += 1
        else:
            self.stats.keyframes += 1

        # Process tokens if provided
        merged_tokens = None
        if vision_tokens is not None:
            self.stats.total_input_tokens += vision_tokens.shape[0]
            merged_tokens = self.process_tokens(vision_tokens, attention_weights)
            self.stats.total_output_tokens += merged_tokens.shape[0]

        # Update state
        self.last_frame = frame.copy()
        self.last_tokens = merged_tokens

        self.stats.total_time_ms += (time.perf_counter() - frame_start) * 1000

        return action, merged_tokens, None

    def reset(self):
        """Reset pipeline state."""
        self.last_frame = None
        self.last_output = None
        self.last_tokens = None
        self.skip_count = 0
        self.token_merger.reset()
        self.stats = PipelineStats()

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return self.stats.summary()


def estimate_combined_speedup(
    num_frames: int = 100,
    tokens_per_frame: int = 256,
    skip_rate: float = 0.90,
    cache_rate: float = 0.08,
    token_compression: float = 30.0,
) -> Dict[str, Any]:
    """Estimate combined speedup from both optimizations.

    Baseline costs (per frame):
    - Vision encoding: ~200ms
    - Token processing: ~100ms (scales with token count)
    - LLM generation: ~50ms

    With optimizations:
    - Skipped frames: ~3ms (similarity check only)
    - Cached frames: ~75ms (partial vision + reduced tokens)
    - Keyframes: ~350ms but with compressed tokens
    """
    # Baseline timing
    BASELINE_VISION_MS = 200
    BASELINE_TOKEN_MS = 100
    BASELINE_LLM_MS = 50
    BASELINE_TOTAL = BASELINE_VISION_MS + BASELINE_TOKEN_MS + BASELINE_LLM_MS

    # Optimized timing
    SKIP_MS = 3
    CACHE_VISION_MS = 50
    KEYFRAME_VISION_MS = 200

    # Token processing scales with compression
    token_scale = 1.0 / token_compression

    skip_frames = int(num_frames * skip_rate)
    cache_frames = int(num_frames * cache_rate)
    keyframes = num_frames - skip_frames - cache_frames

    # Calculate times
    baseline_time = num_frames * BASELINE_TOTAL

    skip_time = skip_frames * SKIP_MS
    cache_time = cache_frames * (CACHE_VISION_MS + BASELINE_TOKEN_MS * token_scale + BASELINE_LLM_MS)
    keyframe_time = keyframes * (KEYFRAME_VISION_MS + BASELINE_TOKEN_MS * token_scale + BASELINE_LLM_MS)

    optimized_time = skip_time + cache_time + keyframe_time

    return {
        "num_frames": num_frames,
        "baseline_time_ms": baseline_time,
        "baseline_per_frame_ms": BASELINE_TOTAL,
        "optimized_time_ms": optimized_time,
        "optimized_per_frame_ms": optimized_time / num_frames,
        "speedup": baseline_time / optimized_time,
        "time_saved_pct": (1 - optimized_time / baseline_time) * 100,
        "breakdown": {
            "skip_frames": skip_frames,
            "skip_time_ms": skip_time,
            "cache_frames": cache_frames,
            "cache_time_ms": cache_time,
            "keyframes": keyframes,
            "keyframe_time_ms": keyframe_time,
        },
        "token_stats": {
            "input_per_frame": tokens_per_frame,
            "output_per_frame": int(tokens_per_frame / token_compression),
            "compression": token_compression,
        },
    }


def run_combined_benchmark():
    """Benchmark the combined pipeline."""
    import cv2

    print("=" * 80)
    print("EFFICIENT VLA: Combined Temporal Skip + Token Merging")
    print("=" * 80)

    # Generate synthetic video
    num_frames = 100
    frame_size = (224, 224)
    tokens_per_frame = 256
    token_dim = 768

    print(f"\nGenerating {num_frames} synthetic frames...")
    np.random.seed(42)

    # Create frames with temporal coherence
    base_frame = np.random.randint(50, 200, (*frame_size, 3), dtype=np.uint8)
    frames = []
    video_tokens = []

    base_tokens = np.random.randn(tokens_per_frame, token_dim).astype(np.float32)

    for i in range(num_frames):
        # Frame generation (90% similar, 8% small change, 2% large change)
        rand = np.random.random()
        if rand < 0.02 or i == 0:
            # Large change
            base_frame = np.random.randint(50, 200, (*frame_size, 3), dtype=np.uint8)
            base_tokens = np.random.randn(tokens_per_frame, token_dim).astype(np.float32)
        elif rand < 0.10:
            # Small change
            noise = np.random.randint(-10, 10, base_frame.shape, dtype=np.int16)
            base_frame = np.clip(base_frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            base_tokens = base_tokens + np.random.randn(tokens_per_frame, token_dim).astype(np.float32) * 0.1
        else:
            # Nearly identical
            noise = np.random.randint(-2, 2, base_frame.shape, dtype=np.int16)
            base_frame = np.clip(base_frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            base_tokens = base_tokens + np.random.randn(tokens_per_frame, token_dim).astype(np.float32) * 0.02

        frames.append(base_frame.copy())
        video_tokens.append(base_tokens.copy())

    # Run pipeline
    print("\nRunning efficient VLA pipeline...")

    pipeline = EfficientVLAPipeline(EfficientVLAConfig(
        skip_threshold=0.99,
        cache_threshold=0.95,
        spatial_merge_ratio=0.5,
        temporal_merge_ratio=0.3,
        saliency_top_k_ratio=0.2,
    ))

    start = time.perf_counter()
    for i, (frame, tokens) in enumerate(zip(frames, video_tokens)):
        action, merged, output = pipeline.process_frame(frame, tokens)
    elapsed = time.perf_counter() - start

    stats = pipeline.get_stats()

    print("\n" + "-" * 80)
    print("RESULTS")
    print("-" * 80)

    print(f"\nFrame Processing:")
    print(f"  Total frames:    {stats['total_frames']}")
    print(f"  Skipped:         {stats['skipped_frames']} ({stats['skip_rate']*100:.1f}%)")
    print(f"  Cached:          {stats['cached_frames']}")
    print(f"  Keyframes:       {stats['keyframes']}")

    print(f"\nToken Reduction:")
    print(f"  Input tokens:    {stats['input_tokens']:,}")
    print(f"  Output tokens:   {stats['output_tokens']:,}")
    print(f"  Compression:     {stats['token_compression']:.1f}x")
    print(f"  Reduction:       {stats['token_reduction']*100:.1f}%")

    print(f"\nTiming (actual pipeline overhead):")
    print(f"  Total time:      {elapsed*1000:.1f}ms")
    print(f"  Per frame:       {elapsed/num_frames*1000:.2f}ms")
    print(f"  Throughput:      {num_frames/elapsed:.1f} fps")

    # Estimate full system speedup
    print("\n" + "-" * 80)
    print("ESTIMATED FULL SYSTEM SPEEDUP")
    print("-" * 80)

    estimates = estimate_combined_speedup(
        num_frames=num_frames,
        tokens_per_frame=tokens_per_frame,
        skip_rate=stats['skip_rate'],
        cache_rate=stats['cached_frames'] / stats['total_frames'],
        token_compression=stats['token_compression'],
    )

    print(f"\nBaseline (no optimization):")
    print(f"  Per frame:       {estimates['baseline_per_frame_ms']:.0f}ms")
    print(f"  Total:           {estimates['baseline_time_ms']/1000:.1f}s")

    print(f"\nWith Efficient VLA:")
    print(f"  Per frame:       {estimates['optimized_per_frame_ms']:.1f}ms")
    print(f"  Total:           {estimates['optimized_time_ms']/1000:.2f}s")

    print(f"\nImprovement:")
    print(f"  Speedup:         {estimates['speedup']:.1f}x")
    print(f"  Time saved:      {estimates['time_saved_pct']:.1f}%")
    print(f"  Realtime?        {'YES' if estimates['optimized_per_frame_ms'] < 33.3 else 'NO'} ({1000/estimates['optimized_per_frame_ms']:.0f} fps)")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Optimization Breakdown")
    print("=" * 80)
    print("""
┌────────────────────────┬────────────┬────────────┬────────────┐
│ Optimization           │ Baseline   │ Optimized  │ Savings    │
├────────────────────────┼────────────┼────────────┼────────────┤
│ Frame processing       │ 100%       │ {:.0f}%       │ {:.0f}% skipped │
│ Token count            │ {:,}      │ {:,}       │ {:.0f}x compress │
│ Latency/frame          │ {}ms      │ {:.0f}ms      │ {:.0f}x faster  │
│ Realtime capable       │ NO         │ {}        │            │
└────────────────────────┴────────────┴────────────┴────────────┘
""".format(
        (1 - stats['skip_rate']) * 100,
        stats['skip_rate'] * 100,
        stats['input_tokens'],
        stats['output_tokens'],
        stats['token_compression'],
        int(estimates['baseline_per_frame_ms']),
        estimates['optimized_per_frame_ms'],
        estimates['speedup'],
        "YES" if estimates['optimized_per_frame_ms'] < 33.3 else "NO",
    ))


if __name__ == "__main__":
    run_combined_benchmark()
