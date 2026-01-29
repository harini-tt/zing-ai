"""Temporal frame skipping for low-latency VLA inference.

3-Tier hierarchy:
- SKIP (sim > 0.99): Reuse last output entirely (~2-5ms)
- CACHE (0.95 < sim < 0.99): Use VLA-Cache for partial recompute (~50-100ms)
- KEYFRAME (sim < 0.95 or forced): Full compute (~350ms)
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
import mlx.core as mx
from PIL import Image
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config, prepare_inputs
from mlx_vlm.models.cache import make_prompt_cache

from .vla_cache import VLACache, get_layer_mask_schedule, quick_frame_similarity
from .vla_model import create_vla_cache_model


class FrameDecision(Enum):
    SKIP = "skip"
    CACHE = "cache"
    KEYFRAME = "keyframe"


@dataclass
class TemporalConfig:
    """Tuning knobs for temporal frame skipping."""

    skip_threshold: float = 0.99
    cache_threshold: float = 0.95
    max_skip_streak: int = 10
    similarity_method: str = "histogram"
    confidence_decay_rate: float = 0.005
    min_skip_threshold: float = 0.95
    enable_motion_detection: bool = True
    motion_threshold: float = 0.1


@dataclass
class TemporalStats:
    """Track performance across frames."""

    skip_count: int = 0
    cache_count: int = 0
    keyframe_count: int = 0
    total_skip_time_ms: float = 0.0
    total_cache_time_ms: float = 0.0
    total_keyframe_time_ms: float = 0.0
    decisions: List[FrameDecision] = field(default_factory=list)

    @property
    def total_frames(self) -> int:
        return self.skip_count + self.cache_count + self.keyframe_count

    @property
    def total_time_ms(self) -> float:
        return self.total_skip_time_ms + self.total_cache_time_ms + self.total_keyframe_time_ms

    @property
    def avg_time_ms(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return self.total_time_ms / self.total_frames

    def summary(self) -> Dict[str, Any]:
        total = self.total_frames
        return {
            "total_frames": total,
            "skip_count": self.skip_count,
            "skip_pct": self.skip_count / total * 100 if total > 0 else 0,
            "cache_count": self.cache_count,
            "cache_pct": self.cache_count / total * 100 if total > 0 else 0,
            "keyframe_count": self.keyframe_count,
            "keyframe_pct": self.keyframe_count / total * 100 if total > 0 else 0,
            "total_time_ms": self.total_time_ms,
            "avg_time_ms": self.avg_time_ms,
            "avg_skip_time_ms": self.total_skip_time_ms / self.skip_count if self.skip_count > 0 else 0,
            "avg_cache_time_ms": self.total_cache_time_ms / self.cache_count if self.cache_count > 0 else 0,
            "avg_keyframe_time_ms": self.total_keyframe_time_ms / self.keyframe_count if self.keyframe_count > 0 else 0,
        }


class TemporalVLA:
    """VLA with temporal frame skipping for low-latency inference."""

    def __init__(
        self,
        model,
        processor,
        config,
        temporal_config: Optional[TemporalConfig] = None,
        prompt: str = "Describe what you see in this image.",
    ):
        self.base_model = model
        self.processor = processor
        self.model_config = config
        self.config = temporal_config or TemporalConfig()
        self.prompt = prompt

        self.vla_model = create_vla_cache_model(model, processor, config)
        self.vla_cache = VLACache(
            patch_size=14,
            sim_threshold=0.996,
            max_static_patches=150,
            max_attention_patches=120,
            enable_visualization=False,
        )

        self.last_output: Optional[str] = None
        self.last_frame: Optional[np.ndarray] = None
        self.last_frame_pil: Optional[Image.Image] = None
        self.skip_count: int = 0
        self.frame_index: int = 0

        self.current_skip_threshold = self.config.skip_threshold

        self.stats = TemporalStats()

        self.formatted_prompt = apply_chat_template(
            processor, config, prompt, num_images=1
        )

    def quick_similarity(self, frame: np.ndarray, prev_frame: Optional[np.ndarray]) -> float:
        """Fast similarity check (<5ms). NOT full patchify."""
        if prev_frame is None:
            return 0.0

        return quick_frame_similarity(
            frame, prev_frame, method=self.config.similarity_method
        )

    def detect_motion(self, frame: np.ndarray, prev_frame: Optional[np.ndarray]) -> float:
        """Detect rapid motion between frames."""
        if prev_frame is None:
            return 1.0

        gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5, levels=1, winsize=15,
            iterations=2, poly_n=5, poly_sigma=1.1, flags=0
        )

        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        return float(np.mean(magnitude))

    def is_keyframe(self, frame: np.ndarray, similarity: float) -> bool:
        """Detect if this should be a keyframe (full compute)."""
        if self.last_frame is None:
            return True

        if self.skip_count >= self.config.max_skip_streak:
            return True

        if similarity < self.config.cache_threshold:
            return True

        if self.config.enable_motion_detection:
            motion = self.detect_motion(frame, self.last_frame)
            if motion > self.config.motion_threshold:
                return True

        return False

    def decide_tier(self, frame: np.ndarray) -> Tuple[FrameDecision, float]:
        """Decide which tier to use for this frame."""
        sim_start = time.perf_counter()
        similarity = self.quick_similarity(frame, self.last_frame)
        sim_time = (time.perf_counter() - sim_start) * 1000

        if self.is_keyframe(frame, similarity):
            return FrameDecision.KEYFRAME, similarity

        effective_skip_threshold = max(
            self.current_skip_threshold,
            self.config.min_skip_threshold
        )

        if similarity > effective_skip_threshold and self.skip_count < self.config.max_skip_streak:
            return FrameDecision.SKIP, similarity

        if similarity > self.config.cache_threshold:
            return FrameDecision.CACHE, similarity

        return FrameDecision.KEYFRAME, similarity

    def run_skip(self) -> str:
        """Tier 1: SKIP - reuse last output entirely."""
        self.skip_count += 1
        self.current_skip_threshold -= self.config.confidence_decay_rate
        return self.last_output

    def run_with_cache(self, frame: np.ndarray, image_path: str) -> str:
        """Tier 2: CACHE - partial recompute using VLA-Cache."""
        self.skip_count = 0
        self.current_skip_threshold = self.config.skip_threshold

        image = Image.fromarray(frame)

        inputs = prepare_inputs(
            self.processor,
            images=[image_path],
            prompts=self.formatted_prompt
        )
        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]
        grid_thw = inputs["image_grid_thw"]
        mask = inputs["attention_mask"]

        cacheable_token_indices = None
        layer_schedule = None

        if self.vla_cache.should_cache():
            prev_attentions, prev_positions = self.vla_model.get_attentions()

            cacheable_token_indices, _ = self.vla_cache.get_cacheable_tokens(
                image,
                prev_attentions,
                prev_positions,
            )

            if cacheable_token_indices and len(cacheable_token_indices) > 0:
                layer_schedule = get_layer_mask_schedule(prev_attentions)

        dtype = self.base_model.vision_tower.patch_embed.proj.weight.dtype

        if cacheable_token_indices and len(cacheable_token_indices) > 0:
            vision_hidden, _, _ = self.vla_model.encode_vision_with_cache(
                pixel_values.astype(dtype),
                grid_thw,
                cacheable_indices=cacheable_token_indices,
            )
        else:
            vision_hidden = self.base_model.vision_tower(pixel_values.astype(dtype), grid_thw)
            mx.eval(vision_hidden)

        embeds = self.base_model.language_model.model.embed_tokens(input_ids)
        merged = self.base_model.merge_input_ids_with_image_features(
            self.base_model.config.image_token_id,
            self.base_model.config.video_token_id,
            vision_hidden,
            embeds,
            input_ids,
        )
        mx.eval(merged)

        lm_cache = make_prompt_cache(self.base_model.language_model)

        out = self.vla_model.forward_with_cache(
            input_ids,
            merged,
            mask=mask,
            cache=lm_cache,
            cacheable_token_indices=cacheable_token_indices,
            layer_schedule=layer_schedule,
        )
        mx.eval(out.logits)

        output = self._generate_tokens(out, lm_cache)

        self.vla_model.update_cache_state(vision_hidden, image)
        self.vla_cache.update(
            image,
            vision_hidden,
            *self.vla_model.get_attentions(),
        )

        return output

    def run_full(self, frame: np.ndarray, image_path: str) -> str:
        """Tier 3: KEYFRAME - full compute."""
        self.skip_count = 0
        self.current_skip_threshold = self.config.skip_threshold

        self.vla_cache.reset()
        self.vla_model.reset_cache()

        image = Image.fromarray(frame)

        inputs = prepare_inputs(
            self.processor,
            images=[image_path],
            prompts=self.formatted_prompt
        )
        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]
        grid_thw = inputs["image_grid_thw"]
        mask = inputs["attention_mask"]

        dtype = self.base_model.vision_tower.patch_embed.proj.weight.dtype
        vision_hidden = self.base_model.vision_tower(pixel_values.astype(dtype), grid_thw)
        mx.eval(vision_hidden)

        embeds = self.base_model.language_model.model.embed_tokens(input_ids)
        merged = self.base_model.merge_input_ids_with_image_features(
            self.base_model.config.image_token_id,
            self.base_model.config.video_token_id,
            vision_hidden,
            embeds,
            input_ids,
        )
        mx.eval(merged)

        lm_cache = make_prompt_cache(self.base_model.language_model)

        out = self.vla_model.forward_with_cache(
            input_ids,
            merged,
            mask=mask,
            cache=lm_cache,
        )
        mx.eval(out.logits)

        output = self._generate_tokens(out, lm_cache)

        self.vla_model.update_cache_state(vision_hidden, image)
        self.vla_cache.update(
            image,
            vision_hidden,
            *self.vla_model.get_attentions(),
        )

        return output

    def _generate_tokens(self, out, lm_cache, max_tokens: int = 100) -> str:
        """Generate output tokens from model."""
        tokens = []
        for _ in range(max_tokens):
            logits = out.logits[:, -1, :]
            next_tok = mx.argmax(logits, axis=-1)
            tokens.append(next_tok.item())

            if next_tok.item() == self.processor.tokenizer.eos_token_id:
                break

            next_embed = self.base_model.language_model.model.embed_tokens(
                next_tok.reshape(1, 1)
            )
            out = self.vla_model.forward_with_cache(
                next_tok.reshape(1, 1),
                next_embed,
                mask=None,
                cache=lm_cache,
            )
            mx.eval(out.logits)

        return self.processor.tokenizer.decode(tokens)

    def process_frame(self, frame: np.ndarray, image_path: str) -> Tuple[str, FrameDecision, float]:
        """Main entry point: decide skip/cache/keyframe and process accordingly."""
        frame_start = time.perf_counter()

        decision, similarity = self.decide_tier(frame)

        if decision == FrameDecision.SKIP:
            output = self.run_skip()
        elif decision == FrameDecision.CACHE:
            output = self.run_with_cache(frame, image_path)
        else:
            output = self.run_full(frame, image_path)

        frame_time_ms = (time.perf_counter() - frame_start) * 1000

        if decision == FrameDecision.SKIP:
            self.stats.skip_count += 1
            self.stats.total_skip_time_ms += frame_time_ms
        elif decision == FrameDecision.CACHE:
            self.stats.cache_count += 1
            self.stats.total_cache_time_ms += frame_time_ms
        else:
            self.stats.keyframe_count += 1
            self.stats.total_keyframe_time_ms += frame_time_ms

        self.stats.decisions.append(decision)

        self.last_output = output
        self.last_frame = frame.copy()
        self.last_frame_pil = Image.fromarray(frame)
        self.frame_index += 1

        return output, decision, similarity

    def reset(self):
        """Reset all state."""
        self.last_output = None
        self.last_frame = None
        self.last_frame_pil = None
        self.skip_count = 0
        self.frame_index = 0
        self.current_skip_threshold = self.config.skip_threshold
        self.vla_cache.reset()
        self.vla_model.reset_cache()
        self.stats = TemporalStats()

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.stats.summary()


def run_temporal_stream(
    image_paths: List[str],
    prompt: str = "Describe what you see in this image.",
    model_path: str = "mlx-community/Qwen2-VL-2B-Instruct-4bit",
    temporal_config: Optional[TemporalConfig] = None,
    verbose: bool = True,
):
    """Process a video/camera stream using temporal VLA."""
    if verbose:
        print(f"Model: {model_path}")
        print(f"Frames: {len(image_paths)}")
        print("=" * 80)

    load_start = time.perf_counter()
    model, processor = load(model_path)
    config = load_config(model_path)
    load_time = time.perf_counter() - load_start

    if verbose:
        print(f"Model load: {load_time*1000:.1f}ms")

    temporal_config = temporal_config or TemporalConfig()
    temporal_vla = TemporalVLA(model, processor, config, temporal_config, prompt)

    outputs = []

    for idx, image_path in enumerate(image_paths):
        image = Image.open(image_path).convert("RGB")
        frame = np.array(image)

        output, decision, similarity = temporal_vla.process_frame(frame, image_path)
        outputs.append(output)

        if verbose:
            frame_time = (
                temporal_vla.stats.total_skip_time_ms +
                temporal_vla.stats.total_cache_time_ms +
                temporal_vla.stats.total_keyframe_time_ms
            ) / temporal_vla.stats.total_frames if temporal_vla.stats.total_frames > 0 else 0

            current_frame_time = 0
            if decision == FrameDecision.SKIP and temporal_vla.stats.skip_count > 0:
                current_frame_time = temporal_vla.stats.total_skip_time_ms / temporal_vla.stats.skip_count
            elif decision == FrameDecision.CACHE and temporal_vla.stats.cache_count > 0:
                current_frame_time = temporal_vla.stats.total_cache_time_ms / temporal_vla.stats.cache_count
            elif decision == FrameDecision.KEYFRAME and temporal_vla.stats.keyframe_count > 0:
                current_frame_time = temporal_vla.stats.total_keyframe_time_ms / temporal_vla.stats.keyframe_count

            print(
                f"Frame {idx+1:3d}/{len(image_paths)} | "
                f"{decision.value:8s} | "
                f"sim={similarity:.4f} | "
                f"Output: {output[:50]}{'...' if len(output) > 50 else ''}"
            )

    if verbose:
        print("\n" + "=" * 80)
        print("TEMPORAL VLA SUMMARY")
        print("=" * 80)

        stats = temporal_vla.get_stats()
        print(f"Total frames: {stats['total_frames']}")
        print(f"  SKIP:     {stats['skip_count']:3d} ({stats['skip_pct']:.1f}%) - avg {stats['avg_skip_time_ms']:.1f}ms")
        print(f"  CACHE:    {stats['cache_count']:3d} ({stats['cache_pct']:.1f}%) - avg {stats['avg_cache_time_ms']:.1f}ms")
        print(f"  KEYFRAME: {stats['keyframe_count']:3d} ({stats['keyframe_pct']:.1f}%) - avg {stats['avg_keyframe_time_ms']:.1f}ms")
        print(f"\nTotal time: {stats['total_time_ms']:.1f}ms")
        print(f"Average time per frame: {stats['avg_time_ms']:.1f}ms")

        baseline_estimate = stats['total_frames'] * 350
        print(f"\nEstimated baseline (350ms/frame): {baseline_estimate:.1f}ms")
        print(f"Speedup: {baseline_estimate / stats['total_time_ms']:.1f}x" if stats['total_time_ms'] > 0 else "N/A")

        print("\nMemory:")
        print(f"  Active: {mx.get_active_memory() / 1e9:.2f} GB")
        print(f"  Peak:   {mx.get_peak_memory() / 1e9:.2f} GB")

    return outputs, temporal_vla.get_stats()


def run_comparison(
    image_paths: List[str],
    prompt: str = "Describe what you see in this image.",
    model_path: str = "mlx-community/Qwen2-VL-2B-Instruct-4bit",
):
    """Compare temporal VLA against baseline."""
    from .vla_inference import run_vla_cache_inference

    print("\n" + "=" * 80)
    print("BASELINE (NO TEMPORAL SKIPPING)")
    print("=" * 80)

    baseline_start = time.perf_counter()
    run_vla_cache_inference(image_paths, prompt, model_path, enable_cache=False)
    baseline_time = time.perf_counter() - baseline_start

    print("\n" + "=" * 80)
    print("TEMPORAL VLA (3-TIER HIERARCHY)")
    print("=" * 80)

    temporal_start = time.perf_counter()
    _, stats = run_temporal_stream(image_paths, prompt, model_path)
    temporal_time = time.perf_counter() - temporal_start

    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"Baseline time:  {baseline_time:.2f}s ({baseline_time/len(image_paths)*1000:.1f}ms/frame)")
    print(f"Temporal time:  {temporal_time:.2f}s ({temporal_time/len(image_paths)*1000:.1f}ms/frame)")
    print(f"Speedup: {baseline_time/temporal_time:.1f}x")
    print(f"Time saved: {baseline_time - temporal_time:.2f}s ({(1 - temporal_time/baseline_time)*100:.1f}%)")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m vla.temporal_vla <image1> [image2] [image3] ...")
        print("\nExamples:")
        print("  # Process frame sequence with temporal skipping")
        print("  python -m vla.temporal_vla frame1.jpg frame2.jpg frame3.jpg")
        print()
        print("  # Compare against baseline")
        print("  python -m vla.temporal_vla --compare frame1.jpg frame2.jpg frame3.jpg")
        print()
        print("  # Custom thresholds")
        print("  python -m vla.temporal_vla --skip-threshold 0.98 --cache-threshold 0.93 frame*.jpg")
        sys.exit(1)

    skip_threshold = 0.99
    cache_threshold = 0.95
    compare_mode = False
    image_paths = []

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--compare":
            compare_mode = True
        elif arg == "--skip-threshold":
            i += 1
            skip_threshold = float(sys.argv[i])
        elif arg == "--cache-threshold":
            i += 1
            cache_threshold = float(sys.argv[i])
        else:
            image_paths.append(arg)
        i += 1

    for path in image_paths:
        if not Path(path).exists():
            print(f"Error: Image not found: {path}")
            sys.exit(1)

    if len(image_paths) == 0:
        print("Error: No image paths provided")
        sys.exit(1)

    prompt = "Describe what you see in this image."
    temporal_config = TemporalConfig(
        skip_threshold=skip_threshold,
        cache_threshold=cache_threshold,
    )

    if compare_mode:
        run_comparison(image_paths, prompt)
    else:
        run_temporal_stream(image_paths, prompt, temporal_config=temporal_config)
