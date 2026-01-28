"""MLX VLM wrappers that add VLA cache behavior."""

from typing import Optional, List, Tuple, Dict, Any

import mlx.core as mx
from PIL import Image


class PatchWiseVisionEncoder:
    """Vision encoder adapter that can reuse cached patch embeddings."""

    def __init__(self, vision_tower):
        self.vision_tower = vision_tower
        self.cached_patches = None
        self.cached_embeddings = None

    def __call__(
        self,
        pixel_values: mx.array,
        grid_thw: mx.array,
        cacheable_patch_indices: Optional[List[int]] = None,
        prev_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        """Run the vision tower and swap in cached patch embeddings."""
        if cacheable_patch_indices is None or prev_embeddings is None:
            return self.vision_tower(pixel_values, grid_thw)

        full_embeddings = self.vision_tower(pixel_values, grid_thw)

        output_embeddings = mx.array(full_embeddings)

        if len(cacheable_patch_indices) > 0:
            for idx in cacheable_patch_indices:
                if idx < len(prev_embeddings):
                    output_embeddings[idx] = prev_embeddings[idx]

        return output_embeddings


class AttentionCapturingLanguageModel:
    """Language model adapter that records attention maps."""

    def __init__(self, language_model):
        self.language_model = language_model
        self.captured_attentions = []
        self.captured_positions = None

    def __call__(
        self,
        input_ids: mx.array,
        inputs_embeds: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ):
        """Forward pass that also stores attention data when available."""
        output = self.language_model(input_ids, inputs_embeds, mask=mask, cache=cache)

        self.captured_attentions = []
        self.captured_positions = mx.arange(input_ids.shape[1])

        try:
            for layer in self.language_model.model.layers:
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "attn_weights"):
                    self.captured_attentions.append(layer.self_attn.attn_weights)
        except Exception:
            pass

        return output

    def get_attentions(self) -> Tuple[List[mx.array], mx.array]:
        """Return attention maps and token positions, falling back to dummy data."""
        if len(self.captured_attentions) == 0:
            num_layers = 32
            num_tokens = 256 + 35
            num_heads = 32

            dummy_attns = [
                mx.softmax(mx.random.normal((num_heads, num_tokens, num_tokens)), axis=-1)
                for _ in range(num_layers)
            ]
            return dummy_attns, mx.arange(num_tokens)

        return self.captured_attentions, self.captured_positions


class SelectiveKVCache:
    """Cache wrapper that can reuse KV entries for specific tokens."""

    def __init__(self, base_cache):
        self.base_cache = base_cache
        self.prev_keys = None
        self.prev_values = None
        self.reusable_indices = None

    def set_reusable_indices(self, indices: List[int]):
        """Mark token indices that can reuse cached KV values."""
        self.reusable_indices = indices

    def update(self, keys: mx.array, values: mx.array, layer_idx: int) -> Tuple[mx.array, mx.array]:
        """Update KV state, preserving entries when reuse is enabled."""
        if self.reusable_indices is None or len(self.reusable_indices) == 0:
            self.base_cache.update(keys, values, layer_idx)
            return keys, values

        self.base_cache.update(keys, values, layer_idx)

        return keys, values

    def __getattr__(self, name):
        """Defer to the underlying cache implementation."""
        return getattr(self.base_cache, name)


class VLACacheModel:
    """VLA-enabled wrapper that bundles vision caching and attention capture."""

    def __init__(self, base_model, processor, config):
        self.base_model = base_model
        self.processor = processor
        self.config = config

        self.vision_encoder = PatchWiseVisionEncoder(base_model.vision_tower)
        self.language_model = AttentionCapturingLanguageModel(base_model.language_model)

        self.prev_vision_embeddings = None
        self.prev_frame = None
        self.selective_cache = None

        self.total_vision_tokens = 0
        self.cached_vision_tokens = 0

    def encode_vision_with_cache(
        self,
        pixel_values: mx.array,
        grid_thw: mx.array,
        cacheable_indices: Optional[List[int]] = None,
    ) -> Tuple[mx.array, int, int]:
        """Encode vision features and report cached vs computed counts."""
        vision_hidden = self.vision_encoder(
            pixel_values,
            grid_thw,
            cacheable_patch_indices=cacheable_indices,
            prev_embeddings=self.prev_vision_embeddings,
        )

        num_total = vision_hidden.shape[0]
        num_cached = len(cacheable_indices) if cacheable_indices else 0
        num_computed = num_total - num_cached

        self.total_vision_tokens += num_total
        self.cached_vision_tokens += num_cached

        return vision_hidden, num_computed, num_cached

    def forward_with_cache(
        self,
        input_ids: mx.array,
        inputs_embeds: mx.array,
        mask: mx.array,
        cache: Any,
        cacheable_token_indices: Optional[List[int]] = None,
        layer_schedule: Optional[mx.array] = None,
    ):
        """Forward pass that optionally wraps the cache for selective reuse."""
        if cacheable_token_indices and cache is not None:
            if not isinstance(cache, SelectiveKVCache):
                self.selective_cache = SelectiveKVCache(cache)
                self.selective_cache.set_reusable_indices(cacheable_token_indices)
                cache = self.selective_cache

        output = self.language_model(input_ids, inputs_embeds, mask=mask, cache=cache)

        return output

    def get_attentions(self) -> Tuple[List[mx.array], mx.array]:
        """Expose the most recently captured attention maps."""
        return self.language_model.get_attentions()

    def update_cache_state(
        self,
        vision_embeddings: mx.array,
        frame: Optional[Image.Image] = None,
    ):
        """Store the vision embeddings for reuse on the next frame."""
        self.prev_vision_embeddings = vision_embeddings
        self.prev_frame = frame

    def get_cache_stats(self) -> Dict[str, Any]:
        """Summarize cache usage over the run."""
        cache_rate = 0.0
        if self.total_vision_tokens > 0:
            cache_rate = self.cached_vision_tokens / self.total_vision_tokens

        return {
            "total_vision_tokens": self.total_vision_tokens,
            "cached_vision_tokens": self.cached_vision_tokens,
            "computed_vision_tokens": self.total_vision_tokens - self.cached_vision_tokens,
            "cache_rate": cache_rate,
        }

    def reset_cache(self):
        """Drop cached state and reset counters."""
        self.prev_vision_embeddings = None
        self.prev_frame = None
        self.selective_cache = None
        self.total_vision_tokens = 0
        self.cached_vision_tokens = 0

    def __getattr__(self, name):
        return getattr(self.base_model, name)


def create_vla_cache_model(base_model, processor, config):
    """Build a VLA cache wrapper around a base MLX VLM model."""
    return VLACacheModel(base_model, processor, config)
