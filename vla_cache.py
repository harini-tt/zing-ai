"""
VLA-Cache: Adaptive Token Caching for Vision-Language Models
Adapted from: https://github.com/siyuhsu/vla-cache

Implements frame-to-frame caching with:
1. Patch-level similarity detection
2. Attention-based task-relevant filtering
3. Layer-wise adaptive KV cache reuse
"""

import cv2
import numpy as np
import mlx.core as mx
from PIL import Image, ImageDraw
from skimage.util import view_as_blocks
from typing import List, Tuple, Optional, Dict, Any


# ============================================================================
# Patch-Level Similarity Detection
# ============================================================================

def patchify(image: Image.Image, patch_size: int = 14) -> np.ndarray:
    """
    Converts an image into non-overlapping patches.

    Args:
        image: PIL Image
        patch_size: Size of each square patch (default 14 for ViT)

    Returns:
        Array of patches with shape (num_patches, patch_size, patch_size, channels)
    """
    image_arr = np.array(image)
    h, w = image_arr.shape[:2]

    assert h % patch_size == 0 and w % patch_size == 0, \
        f"Image dimensions ({h}x{w}) must be divisible by patch_size ({patch_size})"

    if image_arr.ndim == 3:
        blocks = view_as_blocks(image_arr, block_shape=(patch_size, patch_size, image_arr.shape[2]))
        patches = blocks.reshape(-1, patch_size, patch_size, image_arr.shape[2])
    else:
        blocks = view_as_blocks(image_arr, block_shape=(patch_size, patch_size))
        patches = blocks.reshape(-1, patch_size, patch_size)

    return patches


def calculate_patch_similarity(patches1: np.ndarray, patches2: np.ndarray) -> np.ndarray:
    """
    Computes cosine similarity between two sets of patches.

    Args:
        patches1: First set of patches
        patches2: Second set of patches

    Returns:
        Array of cosine similarity scores per patch
    """
    # Flatten patches to vectors
    flat1 = patches1.reshape(len(patches1), -1).astype(np.float32)
    flat2 = patches2.reshape(len(patches2), -1).astype(np.float32)

    # Compute norms
    norm1 = np.linalg.norm(flat1, axis=1)
    norm2 = np.linalg.norm(flat2, axis=1)

    # Compute cosine similarity
    dot = np.sum(flat1 * flat2, axis=1)
    cosine_sim = dot / (norm1 * norm2 + 1e-8)

    return cosine_sim


def find_static_patches(
    img_curr: Image.Image,
    img_prev: Image.Image,
    patch_size: int = 14,
    top_k: int = 150,
    sim_threshold: float = 0.996
) -> List[int]:
    """
    Identifies patches with high similarity between consecutive frames.

    Args:
        img_curr: Current frame
        img_prev: Previous frame
        patch_size: Size of each patch
        top_k: Maximum number of static patches to return
        sim_threshold: Minimum similarity threshold

    Returns:
        List of patch indices that are static (sorted by similarity)
    """
    # Patchify both images
    patches_curr = patchify(img_curr, patch_size)
    patches_prev = patchify(img_prev, patch_size)

    # Calculate similarity
    similarity = calculate_patch_similarity(patches_curr, patches_prev)

    # Reshape to 2D grid
    img_size = np.array(img_curr).shape[0]  # Assume square image
    grid_size = img_size // patch_size
    similarity_2d = similarity.reshape(grid_size, grid_size)

    # Find patches above threshold
    patch_scores = [
        (i * grid_size + j, similarity_2d[i, j])
        for i in range(grid_size)
        for j in range(grid_size)
        if similarity_2d[i, j] >= sim_threshold
    ]

    # Sort by similarity (highest first) and take top-k
    patch_scores.sort(key=lambda x: x[1], reverse=True)
    top_patch_ids = [idx for idx, _ in patch_scores[:top_k]]

    return top_patch_ids


# ============================================================================
# Attention-Based Token Filtering
# ============================================================================

def get_layer_mask_schedule(
    attention_maps: List[mx.array],
    apply_weighted_growth: bool = True,
    growth_factor: float = 0.55
) -> mx.array:
    """
    Computes per-layer reuse proportions based on normalized attention entropy.

    Lower entropy = more focused attention = higher reuse potential

    Args:
        attention_maps: List of attention maps per layer (shape: [heads, tokens, tokens])
        apply_weighted_growth: Whether to smooth upward deltas
        growth_factor: Weight for smoothing

    Returns:
        Layer-wise reuse proportions, shape (num_layers - 1,)
    """
    entropies = []

    # Compute entropy for each layer (except last)
    for attn in attention_maps[:-1]:
        # Average across heads
        attn_mean = mx.mean(attn, axis=0)

        # Normalize to probabilities
        attn_sum = mx.sum(attn_mean, axis=-1, keepdims=True)
        attn_norm = attn_mean / (attn_sum + 1e-10)

        # Replace NaN with 0
        attn_norm = mx.where(mx.isnan(attn_norm), 0.0, attn_norm)

        # Compute entropy: -sum(p * log(p))
        log_attn = mx.log(attn_norm + 1e-10)
        token_entropy = -mx.sum(attn_norm * log_attn, axis=-1)
        entropies.append(mx.mean(token_entropy))

    entropies = mx.stack(entropies)

    # Normalize entropy to [0, 1]
    entropy_min = mx.min(entropies)
    entropy_max = mx.max(entropies)
    norm_entropy = (entropies - entropy_min) / (entropy_max - entropy_min + 1e-10)

    # Reuse = 1 - entropy (low entropy = high reuse)
    reuse = 1.0 - norm_entropy

    if apply_weighted_growth:
        # Smooth upward deltas
        reuse_list = reuse.tolist()
        for i in range(1, len(reuse_list)):
            delta = reuse_list[i] - reuse_list[i - 1]
            if delta > 0:
                reuse_list[i] = reuse_list[i - 1] + delta * growth_factor
        reuse = mx.array(reuse_list)

    return reuse


def token_attention_merge(
    attention_maps: List[mx.array],
    attention_positions: mx.array,
    layer_id: int = 15,
    v_token_start: int = 1,
    v_token_count: int = 256,
    t_token_start: int = 257,
    t_token_count: int = 35
) -> mx.array:
    """
    Computes mean attention from text tokens to vision tokens.

    Args:
        attention_maps: List of attention maps per layer
        attention_positions: Token position indices
        layer_id: Which layer to analyze
        v_token_start: Start index of vision tokens
        v_token_count: Number of vision tokens
        t_token_start: Start index of text tokens
        t_token_count: Number of text tokens

    Returns:
        Attention scores from text to vision tokens
    """
    # Get attention map for specified layer
    attn_map = attention_maps[layer_id].astype(mx.float32)

    # Average across heads
    if attn_map.ndim == 4:  # [batch, heads, tokens, tokens]
        attn_map = mx.mean(attn_map[0], axis=0)
    elif attn_map.ndim == 3:  # [heads, tokens, tokens]
        attn_map = mx.mean(attn_map, axis=0)

    # Define token ranges
    v_token_end = v_token_start + v_token_count
    t_token_end = t_token_start + t_token_count

    # Create text token mask
    text_mask = (attention_positions >= t_token_start) & (attention_positions < t_token_end)

    # Extract attention from text to vision tokens
    text_indices = mx.where(text_mask)[0]
    relation = attn_map[text_indices, v_token_start:v_token_end]

    # Average across text tokens
    return mx.mean(relation, axis=0)


def get_top_attention_patches(attn_scores: mx.array, top_k: int = 120) -> List[int]:
    """
    Selects top-k patch indices based on attention scores.

    Args:
        attn_scores: Attention scores for vision tokens
        top_k: Number of top patches to select

    Returns:
        List of top-k patch indices
    """
    # Convert to numpy for cv2
    attn_np = np.array(attn_scores)

    # Reshape to 2D grid (16x16 for 256 tokens)
    grid_size = int(np.sqrt(len(attn_np)))
    attn_2d = attn_np.reshape(grid_size, grid_size)

    # Resize to ensure consistent grid
    attn_resized = cv2.resize(attn_2d, (grid_size, grid_size))

    # Flatten and sort
    flat = [(i * grid_size + j, attn_resized[i, j])
            for i in range(grid_size)
            for j in range(grid_size)]
    flat.sort(key=lambda x: x[1], reverse=True)

    return [idx for idx, _ in flat[:top_k]]


def task_relevant_selection(
    attention_maps: List[mx.array],
    attention_positions: mx.array,
    image: Image.Image,
    static_patches: List[int],
    top_k: int = 120,
    layer_id: int = 15,
    v_token_start: int = 1
) -> Tuple[np.ndarray, List[int]]:
    """
    Filters static patches to keep only task-irrelevant ones for caching.

    Task-relevant patches (high attention) should be recomputed.
    Task-irrelevant patches (low attention but static) can be cached.

    Args:
        attention_maps: List of attention maps
        attention_positions: Token positions
        image: Current frame
        static_patches: Patches identified as visually static
        top_k: Number of top attention patches
        layer_id: Layer to analyze
        v_token_start: Vision token start index

    Returns:
        (visualization_image, cacheable_token_indices)
    """
    # Get attention scores
    attn_scores = token_attention_merge(
        attention_maps, attention_positions, layer_id,
        v_token_start=v_token_start
    )

    # Find top attention patches
    top_attn_patches = get_top_attention_patches(attn_scores, top_k)

    # Calculate set differences
    only_static = set(static_patches) - set(top_attn_patches)  # Cache these
    only_top = set(top_attn_patches) - set(static_patches)     # Recompute these
    overlap = set(static_patches) & set(top_attn_patches)      # Recompute these (important)

    # Visualize patch groups
    patch_groups = [
        (static_patches, (15, 67, 223)),      # Blue: static patches
        (top_attn_patches, (254, 55, 13)),    # Red: high attention
        (list(only_static), (40, 116, 166)),  # Teal: cacheable (static + low attn)
        (list(only_top), (241, 196, 15)),     # Yellow: recompute (dynamic + high attn)
        (list(overlap), (231, 76, 60)),       # Crimson: recompute (static + high attn)
    ]

    result_image = draw_patches_overlay(image, patch_groups, patch_size=14, alpha=0.4)

    # Convert patch IDs to token indices
    cacheable_tokens = sorted([pid + v_token_start for pid in only_static])

    return np.array(result_image), cacheable_tokens


# ============================================================================
# Visualization Utilities
# ============================================================================

def draw_patches_overlay(
    image: Image.Image,
    patch_groups: List[Tuple[List[int], Tuple[int, int, int]]],
    patch_size: int = 14,
    alpha: float = 0.4
) -> Image.Image:
    """
    Draws colored overlays on image for different patch groups.

    Args:
        image: PIL Image
        patch_groups: List of (patch_ids, RGB_color) tuples
        patch_size: Size of each patch
        alpha: Transparency of overlay

    Returns:
        Image with overlays
    """
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    width = image.size[0]
    num_patches = width // patch_size

    for patch_list, color in patch_groups:
        for pid in patch_list:
            i, j = divmod(pid, num_patches)
            top_left = (j * patch_size, i * patch_size)
            bottom_right = ((j + 1) * patch_size, (i + 1) * patch_size)
            draw.rectangle(
                [top_left, bottom_right],
                fill=color + (int(255 * alpha),)
            )

    return Image.alpha_composite(image, overlay).convert("RGB")


# ============================================================================
# VLA-Cache Manager
# ============================================================================

class VLACache:
    """
    Manages frame-to-frame vision token caching with adaptive reuse.
    """

    def __init__(
        self,
        patch_size: int = 14,
        sim_threshold: float = 0.996,
        max_static_patches: int = 150,
        max_attention_patches: int = 120,
        enable_visualization: bool = False
    ):
        """
        Args:
            patch_size: Size of image patches (14 for ViT)
            sim_threshold: Minimum similarity for static patches
            max_static_patches: Maximum static patches to track
            max_attention_patches: Top-k attention patches
            enable_visualization: Whether to generate visualizations
        """
        self.patch_size = patch_size
        self.sim_threshold = sim_threshold
        self.max_static_patches = max_static_patches
        self.max_attention_patches = max_attention_patches
        self.enable_visualization = enable_visualization

        # State
        self.prev_frame = None
        self.prev_vision_hidden = None
        self.prev_attention_maps = None
        self.prev_attention_positions = None

        # Statistics
        self.frame_count = 0
        self.cache_hits = 0
        self.total_tokens_cached = 0
        self.total_tokens_computed = 0

    def should_cache(self) -> bool:
        """Check if we have previous frame data to enable caching."""
        return self.prev_frame is not None

    def get_cacheable_tokens(
        self,
        curr_frame: Image.Image,
        curr_attention_maps: List[mx.array],
        curr_attention_positions: mx.array,
        layer_id: int = 15
    ) -> Tuple[Optional[List[int]], Optional[np.ndarray]]:
        """
        Identify which vision tokens can be cached from previous frame.

        Returns:
            (cacheable_token_indices, visualization_image)
        """
        if not self.should_cache():
            return None, None

        # Step 1: Find static patches
        static_patches = find_static_patches(
            curr_frame,
            self.prev_frame,
            patch_size=self.patch_size,
            top_k=self.max_static_patches,
            sim_threshold=self.sim_threshold
        )

        if len(static_patches) == 0:
            return None, None

        # Step 2: Filter by task relevance
        viz_image, cacheable_tokens = task_relevant_selection(
            curr_attention_maps,
            curr_attention_positions,
            curr_frame,
            static_patches,
            top_k=self.max_attention_patches,
            layer_id=layer_id
        )

        self.cache_hits += 1
        self.total_tokens_cached += len(cacheable_tokens)

        return cacheable_tokens, viz_image if self.enable_visualization else None

    def update(
        self,
        frame: Image.Image,
        vision_hidden: mx.array,
        attention_maps: List[mx.array],
        attention_positions: mx.array
    ):
        """Update cache with current frame data."""
        self.prev_frame = frame
        self.prev_vision_hidden = vision_hidden
        self.prev_attention_maps = attention_maps
        self.prev_attention_positions = attention_positions
        self.frame_count += 1

    def reset(self):
        """Reset cache state."""
        self.prev_frame = None
        self.prev_vision_hidden = None
        self.prev_attention_maps = None
        self.prev_attention_positions = None
        self.frame_count = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get caching statistics."""
        total_tokens = self.total_tokens_cached + self.total_tokens_computed
        cache_rate = self.total_tokens_cached / total_tokens if total_tokens > 0 else 0

        return {
            'frame_count': self.frame_count,
            'cache_hits': self.cache_hits,
            'tokens_cached': self.total_tokens_cached,
            'tokens_computed': self.total_tokens_computed,
            'cache_rate': cache_rate,
        }
