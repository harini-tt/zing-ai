"""Hierarchical token merging for efficient VLA inference.

Implements ideas from:
- LongVLM: Hierarchical merging within local segments before global pooling
- ViLaMP: Differential distillation for per-frame saliency scores
- ToMe: Bipartite soft matching for token merging

Key techniques:
1. Spatial merging - merge similar tokens within a frame
2. Temporal merging - merge tokens across consecutive frames
3. Saliency scoring - keep only high-importance tokens
4. Hierarchical pooling - local segments → global representation
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import time

import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None


@dataclass
class TokenMergingConfig:
    """Configuration for hierarchical token merging."""

    # Spatial merging (within frame)
    spatial_merge_ratio: float = 0.5  # Keep 50% of tokens per frame
    spatial_similarity_threshold: float = 0.9

    # Temporal merging (across frames)
    temporal_segment_size: int = 5  # Frames per local segment
    temporal_merge_ratio: float = 0.3  # Keep 30% of tokens per segment
    temporal_similarity_threshold: float = 0.85

    # Saliency-based selection
    use_saliency: bool = True
    saliency_top_k_ratio: float = 0.2  # Keep top 20% by saliency
    saliency_method: str = "attention"  # "attention" | "gradient" | "differential"

    # Hierarchical pooling
    num_hierarchy_levels: int = 2  # local → global
    pool_method: str = "attention_weighted"  # "mean" | "max" | "attention_weighted"


@dataclass
class MergingStats:
    """Track token reduction statistics."""

    input_tokens: int = 0
    output_tokens: int = 0
    spatial_merged: int = 0
    temporal_merged: int = 0
    saliency_pruned: int = 0

    @property
    def reduction_ratio(self) -> float:
        if self.input_tokens == 0:
            return 0.0
        return 1.0 - (self.output_tokens / self.input_tokens)

    @property
    def compression_factor(self) -> float:
        if self.output_tokens == 0:
            return float('inf')
        return self.input_tokens / self.output_tokens


class TokenMerger:
    """Bipartite soft matching for token merging (inspired by ToMe)."""

    def __init__(self, config: TokenMergingConfig):
        self.config = config

    def compute_similarity_matrix(
        self,
        tokens: np.ndarray,  # [N, D]
    ) -> np.ndarray:
        """Compute pairwise cosine similarity between tokens."""
        # Normalize
        norms = np.linalg.norm(tokens, axis=1, keepdims=True)
        normalized = tokens / (norms + 1e-8)

        # Cosine similarity
        similarity = normalized @ normalized.T
        return similarity

    def bipartite_soft_matching(
        self,
        tokens: np.ndarray,  # [N, D]
        num_keep: int,
    ) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
        """Merge tokens using bipartite matching.

        Splits tokens into two sets, finds best matches, merges similar pairs.

        Returns:
            merged_tokens: [num_keep, D]
            kept_indices: indices of tokens that were kept
            merge_groups: which original tokens merged into each output token
        """
        N, D = tokens.shape

        if num_keep >= N:
            return tokens, np.arange(N), [[i] for i in range(N)]

        # Split into two sets (alternating for spatial locality)
        set_a = tokens[0::2]  # Even indices
        set_b = tokens[1::2]  # Odd indices
        idx_a = np.arange(0, N, 2)
        idx_b = np.arange(1, N, 2)

        # Compute cross-set similarity
        norm_a = set_a / (np.linalg.norm(set_a, axis=1, keepdims=True) + 1e-8)
        norm_b = set_b / (np.linalg.norm(set_b, axis=1, keepdims=True) + 1e-8)
        cross_sim = norm_a @ norm_b.T  # [len_a, len_b]

        # Greedy matching: pair most similar tokens
        num_merge = N - num_keep
        merge_pairs = []
        used_a, used_b = set(), set()

        # Find top matches
        flat_idx = np.argsort(cross_sim.flatten())[::-1]
        for idx in flat_idx:
            if len(merge_pairs) >= num_merge:
                break
            i, j = divmod(idx, cross_sim.shape[1])
            if i not in used_a and j not in used_b:
                if cross_sim[i, j] >= self.config.spatial_similarity_threshold:
                    merge_pairs.append((i, j))
                    used_a.add(i)
                    used_b.add(j)

        # Build output
        merged_tokens = []
        kept_indices = []
        merge_groups = []

        merged_b = {j for _, j in merge_pairs}
        merge_map = {i: j for i, j in merge_pairs}

        for i in range(len(set_a)):
            if i in merge_map:
                # Merge with partner from set_b
                j = merge_map[i]
                merged = (set_a[i] + set_b[j]) / 2.0
                merged_tokens.append(merged)
                kept_indices.append(idx_a[i])
                merge_groups.append([idx_a[i], idx_b[j]])
            else:
                # Keep as-is
                merged_tokens.append(set_a[i])
                kept_indices.append(idx_a[i])
                merge_groups.append([idx_a[i]])

        for j in range(len(set_b)):
            if j not in merged_b:
                merged_tokens.append(set_b[j])
                kept_indices.append(idx_b[j])
                merge_groups.append([idx_b[j]])

        merged_tokens = np.array(merged_tokens[:num_keep])
        kept_indices = np.array(kept_indices[:num_keep])

        return merged_tokens, kept_indices, merge_groups[:num_keep]

    def merge_spatial(
        self,
        frame_tokens: np.ndarray,  # [N, D]
    ) -> Tuple[np.ndarray, List[List[int]]]:
        """Merge similar tokens within a single frame."""
        N = frame_tokens.shape[0]
        num_keep = max(1, int(N * self.config.spatial_merge_ratio))

        merged, _, groups = self.bipartite_soft_matching(frame_tokens, num_keep)
        return merged, groups


class SaliencyScorer:
    """Compute per-token importance scores."""

    def __init__(self, config: TokenMergingConfig):
        self.config = config

    def score_by_attention(
        self,
        tokens: np.ndarray,  # [N, D]
        attention_weights: Optional[np.ndarray] = None,  # [N, N] or [H, N, N]
    ) -> np.ndarray:
        """Score tokens by how much attention they receive."""
        N = tokens.shape[0]

        if attention_weights is None:
            # Fallback: use token norm as proxy for importance
            return np.linalg.norm(tokens, axis=1)

        # Average across heads if multi-head
        if attention_weights.ndim == 3:
            attention_weights = attention_weights.mean(axis=0)

        # Sum attention received by each token (column sum)
        incoming_attention = attention_weights.sum(axis=0)

        return incoming_attention

    def score_by_differential(
        self,
        curr_tokens: np.ndarray,  # [N, D]
        prev_tokens: Optional[np.ndarray],  # [N, D]
    ) -> np.ndarray:
        """Score tokens by how much they changed from previous frame."""
        if prev_tokens is None:
            # First frame: all tokens are important
            return np.ones(curr_tokens.shape[0])

        if curr_tokens.shape != prev_tokens.shape:
            return np.ones(curr_tokens.shape[0])

        # L2 distance between corresponding tokens
        diff = np.linalg.norm(curr_tokens - prev_tokens, axis=1)

        # Normalize to [0, 1]
        if diff.max() > 0:
            diff = diff / diff.max()

        return diff

    def score_by_gradient_magnitude(
        self,
        tokens: np.ndarray,  # [N, D]
    ) -> np.ndarray:
        """Estimate importance by local gradient (edge detection proxy)."""
        N = tokens.shape[0]

        # Assume tokens are in spatial order (grid)
        grid_size = int(np.sqrt(N))
        if grid_size * grid_size != N:
            # Not a perfect grid, use norm fallback
            return np.linalg.norm(tokens, axis=1)

        # Reshape to grid
        token_grid = tokens.reshape(grid_size, grid_size, -1)

        # Compute gradients
        grad_x = np.diff(token_grid, axis=1, prepend=token_grid[:, :1, :])
        grad_y = np.diff(token_grid, axis=0, prepend=token_grid[:1, :, :])

        # Gradient magnitude
        grad_mag = np.sqrt(
            np.sum(grad_x ** 2, axis=-1) + np.sum(grad_y ** 2, axis=-1)
        )

        return grad_mag.flatten()

    def compute_saliency(
        self,
        tokens: np.ndarray,
        prev_tokens: Optional[np.ndarray] = None,
        attention_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute saliency scores using configured method."""
        method = self.config.saliency_method

        if method == "attention":
            return self.score_by_attention(tokens, attention_weights)
        elif method == "differential":
            return self.score_by_differential(tokens, prev_tokens)
        elif method == "gradient":
            return self.score_by_gradient_magnitude(tokens)
        else:
            # Combined: weighted average of all methods
            attn_score = self.score_by_attention(tokens, attention_weights)
            diff_score = self.score_by_differential(tokens, prev_tokens)
            grad_score = self.score_by_gradient_magnitude(tokens)

            # Normalize each
            for score in [attn_score, diff_score, grad_score]:
                if score.max() > 0:
                    score /= score.max()

            return 0.4 * attn_score + 0.4 * diff_score + 0.2 * grad_score


class HierarchicalPooler:
    """Pool tokens hierarchically: local segments → global representation."""

    def __init__(self, config: TokenMergingConfig):
        self.config = config

    def pool_tokens(
        self,
        tokens: np.ndarray,  # [N, D]
        weights: Optional[np.ndarray] = None,  # [N]
        method: Optional[str] = None,
    ) -> np.ndarray:
        """Pool tokens into single representation."""
        method = method or self.config.pool_method

        if method == "mean":
            return tokens.mean(axis=0)
        elif method == "max":
            return tokens.max(axis=0)
        elif method == "attention_weighted" and weights is not None:
            # Normalize weights
            weights = weights / (weights.sum() + 1e-8)
            return (tokens * weights[:, np.newaxis]).sum(axis=0)
        else:
            return tokens.mean(axis=0)

    def pool_segment(
        self,
        segment_tokens: List[np.ndarray],  # List of [N_i, D] per frame
        saliency_scores: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """Pool tokens from a temporal segment."""
        # Concatenate all frame tokens
        all_tokens = np.concatenate(segment_tokens, axis=0)

        if saliency_scores is not None:
            all_scores = np.concatenate(saliency_scores, axis=0)
        else:
            all_scores = None

        # Keep top-k by saliency
        num_keep = max(1, int(len(all_tokens) * self.config.temporal_merge_ratio))

        if all_scores is not None:
            top_indices = np.argsort(all_scores)[-num_keep:]
            selected_tokens = all_tokens[top_indices]
            selected_scores = all_scores[top_indices]
        else:
            # Random selection if no saliency
            indices = np.random.choice(len(all_tokens), num_keep, replace=False)
            selected_tokens = all_tokens[indices]
            selected_scores = None

        return selected_tokens, selected_scores

    def hierarchical_pool(
        self,
        frame_tokens: List[np.ndarray],  # List of [N_i, D] per frame
        saliency_scores: Optional[List[np.ndarray]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply hierarchical pooling across all frames.

        Level 0: Individual frames (after spatial merging)
        Level 1: Local segments (temporal_segment_size frames)
        Level 2: Global representation
        """
        num_frames = len(frame_tokens)
        segment_size = self.config.temporal_segment_size

        stats = {
            "num_frames": num_frames,
            "num_segments": (num_frames + segment_size - 1) // segment_size,
            "tokens_per_level": [],
        }

        # Level 0: count input tokens
        level0_tokens = sum(t.shape[0] for t in frame_tokens)
        stats["tokens_per_level"].append(level0_tokens)

        # Level 1: Pool within segments
        segment_outputs = []
        segment_scores = []

        for seg_start in range(0, num_frames, segment_size):
            seg_end = min(seg_start + segment_size, num_frames)
            seg_frames = frame_tokens[seg_start:seg_end]

            if saliency_scores is not None:
                seg_saliency = saliency_scores[seg_start:seg_end]
            else:
                seg_saliency = None

            pooled, scores = self.pool_segment(seg_frames, seg_saliency)
            segment_outputs.append(pooled)
            if scores is not None:
                segment_scores.append(scores)

        level1_tokens = sum(s.shape[0] for s in segment_outputs)
        stats["tokens_per_level"].append(level1_tokens)

        # Level 2: Global pooling (if multiple segments)
        if len(segment_outputs) > 1:
            all_segment_tokens = np.concatenate(segment_outputs, axis=0)

            if segment_scores:
                all_segment_scores = np.concatenate(segment_scores, axis=0)
            else:
                all_segment_scores = None

            # Final reduction
            final_keep = max(1, int(len(all_segment_tokens) * self.config.temporal_merge_ratio))
            if all_segment_scores is not None:
                top_idx = np.argsort(all_segment_scores)[-final_keep:]
                global_tokens = all_segment_tokens[top_idx]
            else:
                global_tokens = all_segment_tokens[:final_keep]

            stats["tokens_per_level"].append(global_tokens.shape[0])
        else:
            global_tokens = segment_outputs[0] if segment_outputs else np.array([])

        return global_tokens, stats


class HierarchicalTokenMerger:
    """Complete hierarchical token merging pipeline."""

    def __init__(self, config: Optional[TokenMergingConfig] = None):
        self.config = config or TokenMergingConfig()
        self.token_merger = TokenMerger(self.config)
        self.saliency_scorer = SaliencyScorer(self.config)
        self.pooler = HierarchicalPooler(self.config)

        self.prev_frame_tokens: Optional[np.ndarray] = None
        self.frame_buffer: List[np.ndarray] = []
        self.saliency_buffer: List[np.ndarray] = []
        self.stats = MergingStats()

    def process_frame(
        self,
        frame_tokens: np.ndarray,  # [N, D] vision tokens for one frame
        attention_weights: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process a single frame's tokens through the pipeline."""
        frame_stats = {}
        original_count = frame_tokens.shape[0]
        self.stats.input_tokens += original_count

        # Step 1: Spatial merging within frame
        merged_tokens, merge_groups = self.token_merger.merge_spatial(frame_tokens)
        spatial_merged = original_count - merged_tokens.shape[0]
        self.stats.spatial_merged += spatial_merged
        frame_stats["spatial_merged"] = spatial_merged

        # Step 2: Compute saliency scores
        if self.config.use_saliency:
            saliency = self.saliency_scorer.compute_saliency(
                merged_tokens,
                prev_tokens=self.prev_frame_tokens,
                attention_weights=attention_weights,
            )

            # Keep top-k by saliency
            num_keep = max(1, int(len(merged_tokens) * self.config.saliency_top_k_ratio))
            top_indices = np.argsort(saliency)[-num_keep:]

            salient_tokens = merged_tokens[top_indices]
            salient_scores = saliency[top_indices]

            saliency_pruned = merged_tokens.shape[0] - salient_tokens.shape[0]
            self.stats.saliency_pruned += saliency_pruned
            frame_stats["saliency_pruned"] = saliency_pruned
        else:
            salient_tokens = merged_tokens
            salient_scores = np.ones(len(merged_tokens))

        # Buffer for temporal merging
        self.frame_buffer.append(salient_tokens)
        self.saliency_buffer.append(salient_scores)

        # Update state
        self.prev_frame_tokens = merged_tokens
        self.stats.output_tokens += salient_tokens.shape[0]

        frame_stats["input_tokens"] = original_count
        frame_stats["output_tokens"] = salient_tokens.shape[0]
        frame_stats["reduction"] = 1.0 - (salient_tokens.shape[0] / original_count)

        return salient_tokens, frame_stats

    def flush_segment(self) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Process buffered frames through hierarchical pooling."""
        if not self.frame_buffer:
            return None, {}

        # Apply hierarchical pooling
        pooled_tokens, pool_stats = self.pooler.hierarchical_pool(
            self.frame_buffer,
            self.saliency_buffer if self.config.use_saliency else None,
        )

        # Track temporal merging
        input_count = sum(t.shape[0] for t in self.frame_buffer)
        output_count = pooled_tokens.shape[0]
        self.stats.temporal_merged += input_count - output_count

        # Clear buffer
        self.frame_buffer = []
        self.saliency_buffer = []

        return pooled_tokens, pool_stats

    def should_flush(self) -> bool:
        """Check if we should flush the segment buffer."""
        return len(self.frame_buffer) >= self.config.temporal_segment_size

    def process_video_tokens(
        self,
        video_tokens: List[np.ndarray],  # List of [N_i, D] per frame
        attention_weights: Optional[List[np.ndarray]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process all frames from a video."""
        self.reset()

        all_outputs = []
        frame_stats_list = []

        for i, frame_tokens in enumerate(video_tokens):
            attn = attention_weights[i] if attention_weights else None
            output, stats = self.process_frame(frame_tokens, attn)
            frame_stats_list.append(stats)

            if self.should_flush():
                pooled, _ = self.flush_segment()
                if pooled is not None:
                    all_outputs.append(pooled)

        # Flush remaining
        if self.frame_buffer:
            pooled, _ = self.flush_segment()
            if pooled is not None:
                all_outputs.append(pooled)

        # Final concatenation
        if all_outputs:
            final_tokens = np.concatenate(all_outputs, axis=0)
        else:
            final_tokens = np.array([])

        overall_stats = {
            "total_input_tokens": self.stats.input_tokens,
            "total_output_tokens": final_tokens.shape[0] if len(final_tokens) > 0 else 0,
            "spatial_merged": self.stats.spatial_merged,
            "temporal_merged": self.stats.temporal_merged,
            "saliency_pruned": self.stats.saliency_pruned,
            "compression_factor": self.stats.input_tokens / max(1, final_tokens.shape[0] if len(final_tokens) > 0 else 1),
            "reduction_ratio": 1.0 - (final_tokens.shape[0] / max(1, self.stats.input_tokens)) if len(final_tokens) > 0 else 1.0,
        }

        return final_tokens, overall_stats

    def reset(self):
        """Reset all state."""
        self.prev_frame_tokens = None
        self.frame_buffer = []
        self.saliency_buffer = []
        self.stats = MergingStats()

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        return {
            "input_tokens": self.stats.input_tokens,
            "output_tokens": self.stats.output_tokens,
            "spatial_merged": self.stats.spatial_merged,
            "temporal_merged": self.stats.temporal_merged,
            "saliency_pruned": self.stats.saliency_pruned,
            "reduction_ratio": self.stats.reduction_ratio,
            "compression_factor": self.stats.compression_factor,
        }


def benchmark_token_merging(
    num_frames: int = 100,
    tokens_per_frame: int = 256,
    token_dim: int = 768,
):
    """Benchmark token merging performance."""
    print("=" * 80)
    print("HIERARCHICAL TOKEN MERGING BENCHMARK")
    print("=" * 80)

    # Generate synthetic video tokens
    print(f"\nGenerating {num_frames} frames x {tokens_per_frame} tokens x {token_dim}D...")
    np.random.seed(42)

    # Simulate video with temporal coherence
    base_tokens = np.random.randn(tokens_per_frame, token_dim).astype(np.float32)
    video_tokens = []

    for i in range(num_frames):
        # Add small noise to simulate frame changes
        noise_level = 0.1 if i % 10 != 0 else 0.5  # Larger change every 10 frames
        frame = base_tokens + np.random.randn(tokens_per_frame, token_dim).astype(np.float32) * noise_level
        video_tokens.append(frame)

        if i % 10 == 0:
            # Simulate scene change
            base_tokens = frame

    total_input = num_frames * tokens_per_frame
    print(f"Total input tokens: {total_input:,}")

    # Test different configurations
    configs = [
        ("Conservative", TokenMergingConfig(
            spatial_merge_ratio=0.7,
            temporal_merge_ratio=0.5,
            saliency_top_k_ratio=0.5,
        )),
        ("Balanced", TokenMergingConfig(
            spatial_merge_ratio=0.5,
            temporal_merge_ratio=0.3,
            saliency_top_k_ratio=0.2,
        )),
        ("Aggressive", TokenMergingConfig(
            spatial_merge_ratio=0.3,
            temporal_merge_ratio=0.2,
            saliency_top_k_ratio=0.1,
        )),
        ("ViLaMP-style (100x target)", TokenMergingConfig(
            spatial_merge_ratio=0.2,
            temporal_merge_ratio=0.1,
            saliency_top_k_ratio=0.05,
            temporal_segment_size=10,
        )),
    ]

    print(f"\n{'Config':<25} {'Output':>10} {'Compress':>10} {'Time':>10} {'Throughput':>15}")
    print("-" * 75)

    for name, config in configs:
        merger = HierarchicalTokenMerger(config)

        start = time.perf_counter()
        output_tokens, stats = merger.process_video_tokens(video_tokens)
        elapsed = time.perf_counter() - start

        output_count = output_tokens.shape[0] if len(output_tokens) > 0 else 0
        compression = total_input / max(1, output_count)
        throughput = num_frames / elapsed

        print(f"{name:<25} {output_count:>10,} {compression:>9.1f}x {elapsed*1000:>9.1f}ms {throughput:>12.1f} fps")

    # Detailed stats for balanced config
    print("\n" + "=" * 80)
    print("DETAILED BREAKDOWN (Balanced config)")
    print("=" * 80)

    merger = HierarchicalTokenMerger(TokenMergingConfig())
    output_tokens, stats = merger.process_video_tokens(video_tokens)

    print(f"\nToken flow:")
    print(f"  Input:           {stats['total_input_tokens']:>10,} tokens")
    print(f"  After spatial:   {stats['total_input_tokens'] - stats['spatial_merged']:>10,} tokens (-{stats['spatial_merged']:,} merged)")
    print(f"  After saliency:  {stats['total_input_tokens'] - stats['spatial_merged'] - stats['saliency_pruned']:>10,} tokens (-{stats['saliency_pruned']:,} pruned)")
    print(f"  After temporal:  {stats['total_output_tokens']:>10,} tokens (-{stats['temporal_merged']:,} merged)")
    print(f"\nCompression: {stats['compression_factor']:.1f}x ({stats['reduction_ratio']*100:.1f}% reduction)")

    # Memory savings estimate
    bytes_per_token = token_dim * 4  # float32
    input_memory = total_input * bytes_per_token
    output_memory = stats['total_output_tokens'] * bytes_per_token

    print(f"\nMemory estimate:")
    print(f"  Input:  {input_memory / 1e6:.1f} MB")
    print(f"  Output: {output_memory / 1e6:.1f} MB")
    print(f"  Saved:  {(input_memory - output_memory) / 1e6:.1f} MB ({(1 - output_memory/input_memory)*100:.1f}%)")


if __name__ == "__main__":
    import sys

    num_frames = 100
    if len(sys.argv) > 1:
        num_frames = int(sys.argv[1])

    benchmark_token_merging(num_frames=num_frames)
