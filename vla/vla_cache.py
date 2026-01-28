"""VLA cache helpers for patch similarity, attention filtering, and cache state."""

import cv2
import numpy as np
import mlx.core as mx
from PIL import Image, ImageDraw
from skimage.util import view_as_blocks
from typing import List, Tuple, Optional, Dict, Any


# === Patch similarity ===

def patchify(image: Image.Image, patch_size: int = 14) -> np.ndarray:
    """Split an image into non-overlapping square patches."""
    image_arr = np.array(image)
    h, w = image_arr.shape[:2]

    assert h % patch_size == 0 and w % patch_size == 0, (
        f"Image dimensions ({h}x{w}) must be divisible by patch_size ({patch_size})"
    )

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


def calculate_patch_similarity(patches1: np.ndarray, patches2: np.ndarray) -> np.ndarray:
    """Return cosine similarity for corresponding patches."""
    flat1 = patches1.reshape(len(patches1), -1).astype(np.float32)
    flat2 = patches2.reshape(len(patches2), -1).astype(np.float32)

    norm1 = np.linalg.norm(flat1, axis=1)
    norm2 = np.linalg.norm(flat2, axis=1)

    dot = np.sum(flat1 * flat2, axis=1)
    cosine_sim = dot / (norm1 * norm2 + 1e-8)

    return cosine_sim


def find_static_patches(
    img_curr: Image.Image,
    img_prev: Image.Image,
    patch_size: int = 14,
    top_k: int = 150,
    sim_threshold: float = 0.996,
) -> List[int]:
    """Find patch indices that remain visually stable between frames."""
    patches_curr = patchify(img_curr, patch_size)
    patches_prev = patchify(img_prev, patch_size)

    similarity = calculate_patch_similarity(patches_curr, patches_prev)

    img_size = np.array(img_curr).shape[0]
    grid_size = img_size // patch_size
    similarity_2d = similarity.reshape(grid_size, grid_size)

    patch_scores = [
        (i * grid_size + j, similarity_2d[i, j])
        for i in range(grid_size)
        for j in range(grid_size)
        if similarity_2d[i, j] >= sim_threshold
    ]

    patch_scores.sort(key=lambda x: x[1], reverse=True)
    top_patch_ids = [idx for idx, _ in patch_scores[:top_k]]

    return top_patch_ids


# === Attention-guided filtering ===

def get_layer_mask_schedule(
    attention_maps: List[mx.array],
    apply_weighted_growth: bool = True,
    growth_factor: float = 0.55,
) -> mx.array:
    """Compute per-layer reuse ratios from attention entropy."""
    entropies = []

    for attn in attention_maps[:-1]:
        attn_mean = mx.mean(attn, axis=0)

        attn_sum = mx.sum(attn_mean, axis=-1, keepdims=True)
        attn_norm = attn_mean / (attn_sum + 1e-10)

        attn_norm = mx.where(mx.isnan(attn_norm), 0.0, attn_norm)

        log_attn = mx.log(attn_norm + 1e-10)
        token_entropy = -mx.sum(attn_norm * log_attn, axis=-1)
        entropies.append(mx.mean(token_entropy))

    entropies = mx.stack(entropies)

    entropy_min = mx.min(entropies)
    entropy_max = mx.max(entropies)
    norm_entropy = (entropies - entropy_min) / (entropy_max - entropy_min + 1e-10)

    reuse = 1.0 - norm_entropy

    if apply_weighted_growth:
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
    t_token_count: int = 35,
) -> mx.array:
    """Aggregate attention from text tokens to vision tokens."""
    attn_map = attention_maps[layer_id].astype(mx.float32)

    if attn_map.ndim == 4:
        attn_map = mx.mean(attn_map[0], axis=0)
    elif attn_map.ndim == 3:
        attn_map = mx.mean(attn_map, axis=0)

    v_token_end = v_token_start + v_token_count
    t_token_end = t_token_start + t_token_count

    text_mask = (attention_positions >= t_token_start) & (attention_positions < t_token_end)

    text_indices = mx.where(text_mask)[0]
    relation = attn_map[text_indices, v_token_start:v_token_end]

    return mx.mean(relation, axis=0)


def get_top_attention_patches(attn_scores: mx.array, top_k: int = 120) -> List[int]:
    """Pick the highest-attended vision patches."""
    attn_np = np.array(attn_scores)

    grid_size = int(np.sqrt(len(attn_np)))
    attn_2d = attn_np.reshape(grid_size, grid_size)

    attn_resized = cv2.resize(attn_2d, (grid_size, grid_size))

    flat = [
        (i * grid_size + j, attn_resized[i, j])
        for i in range(grid_size)
        for j in range(grid_size)
    ]
    flat.sort(key=lambda x: x[1], reverse=True)

    return [idx for idx, _ in flat[:top_k]]


def task_relevant_selection(
    attention_maps: List[mx.array],
    attention_positions: mx.array,
    image: Image.Image,
    static_patches: List[int],
    top_k: int = 120,
    layer_id: int = 15,
    v_token_start: int = 1,
) -> Tuple[np.ndarray, List[int]]:
    """Keep static patches that are low-attention and therefore cacheable."""
    attn_scores = token_attention_merge(
        attention_maps,
        attention_positions,
        layer_id,
        v_token_start=v_token_start,
    )

    top_attn_patches = get_top_attention_patches(attn_scores, top_k)

    only_static = set(static_patches) - set(top_attn_patches)
    only_top = set(top_attn_patches) - set(static_patches)
    overlap = set(static_patches) & set(top_attn_patches)

    patch_groups = [
        (static_patches, (15, 67, 223)),
        (top_attn_patches, (254, 55, 13)),
        (list(only_static), (40, 116, 166)),
        (list(only_top), (241, 196, 15)),
        (list(overlap), (231, 76, 60)),
    ]

    result_image = draw_patches_overlay(image, patch_groups, patch_size=14, alpha=0.4)

    cacheable_tokens = sorted([pid + v_token_start for pid in only_static])

    return np.array(result_image), cacheable_tokens


# === Visualization ===

def draw_patches_overlay(
    image: Image.Image,
    patch_groups: List[Tuple[List[int], Tuple[int, int, int]]],
    patch_size: int = 14,
    alpha: float = 0.4,
) -> Image.Image:
    """Overlay colored patch groups for debugging and demos."""
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
                fill=color + (int(255 * alpha),),
            )

    return Image.alpha_composite(image, overlay).convert("RGB")


# === Cache manager ===
class VLACache:
    """Track frame-to-frame state and cacheable token indices."""

    def __init__(
        self,
        patch_size: int = 14,
        sim_threshold: float = 0.996,
        max_static_patches: int = 150,
        max_attention_patches: int = 120,
        enable_visualization: bool = False,
    ):
        """Initialize cache settings and counters."""
        self.patch_size = patch_size
        self.sim_threshold = sim_threshold
        self.max_static_patches = max_static_patches
        self.max_attention_patches = max_attention_patches
        self.enable_visualization = enable_visualization

        self.prev_frame = None
        self.prev_vision_hidden = None
        self.prev_attention_maps = None
        self.prev_attention_positions = None

        self.frame_count = 0
        self.cache_hits = 0
        self.total_tokens_cached = 0
        self.total_tokens_computed = 0

    def should_cache(self) -> bool:
        """Return True if a previous frame is available."""
        return self.prev_frame is not None

    def get_cacheable_tokens(
        self,
        curr_frame: Image.Image,
        curr_attention_maps: List[mx.array],
        curr_attention_positions: mx.array,
        layer_id: int = 15,
    ) -> Tuple[Optional[List[int]], Optional[np.ndarray]]:
        """Compute cacheable token indices and optional visualization."""
        if not self.should_cache():
            return None, None

        static_patches = find_static_patches(
            curr_frame,
            self.prev_frame,
            patch_size=self.patch_size,
            top_k=self.max_static_patches,
            sim_threshold=self.sim_threshold,
        )

        if len(static_patches) == 0:
            return None, None

        viz_image, cacheable_tokens = task_relevant_selection(
            curr_attention_maps,
            curr_attention_positions,
            curr_frame,
            static_patches,
            top_k=self.max_attention_patches,
            layer_id=layer_id,
        )

        self.cache_hits += 1
        self.total_tokens_cached += len(cacheable_tokens)

        return cacheable_tokens, viz_image if self.enable_visualization else None

    def update(
        self,
        frame: Image.Image,
        vision_hidden: mx.array,
        attention_maps: List[mx.array],
        attention_positions: mx.array,
    ):
        """Store the current frame state for the next step."""
        self.prev_frame = frame
        self.prev_vision_hidden = vision_hidden
        self.prev_attention_maps = attention_maps
        self.prev_attention_positions = attention_positions
        self.frame_count += 1

    def reset(self):
        """Clear all cached state and counters."""
        self.prev_frame = None
        self.prev_vision_hidden = None
        self.prev_attention_maps = None
        self.prev_attention_positions = None
        self.frame_count = 0

    def get_stats(self) -> Dict[str, Any]:
        """Return cumulative cache statistics."""
        total_tokens = self.total_tokens_cached + self.total_tokens_computed
        cache_rate = self.total_tokens_cached / total_tokens if total_tokens > 0 else 0

        return {
            "frame_count": self.frame_count,
            "cache_hits": self.cache_hits,
            "tokens_cached": self.total_tokens_cached,
            "tokens_computed": self.total_tokens_computed,
            "cache_rate": cache_rate,
        }
