"""Run inference on image sequences with optional VLA caching."""

import time
from pathlib import Path
from typing import List

import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config, prepare_inputs
from mlx_vlm.models.cache import make_prompt_cache
from PIL import Image

from .vla_cache import VLACache, get_layer_mask_schedule
from .vla_model import create_vla_cache_model


def time_it(name):
    """Quick and dirty timing context manager."""
    class Timer:
        def __enter__(self):
            self.start = time.perf_counter()
            return self

        def __exit__(self, *args):
            self.elapsed = time.perf_counter() - self.start
            print(f"  {name}: {self.elapsed*1000:.1f}ms")
            return self

    return Timer()


def run_vla_cache_inference(
    image_paths: List[str],
    prompt: str = "Describe this image in detail.",
    model_path: str = "mlx-community/Qwen2-VL-2B-Instruct-4bit",
    enable_cache: bool = True,
    enable_visualization: bool = False,
):
    """Process a sequence of frames, caching vision tokens between them."""
    print(f"Model: {model_path}")
    print(f"Frames: {len(image_paths)}")
    print(f"VLA-Cache: {'ENABLED ✓' if enable_cache else 'DISABLED'}")
    print("=" * 80)

    with time_it("Model load"):
        base_model, processor = load(model_path)
        config = load_config(model_path)

    model = create_vla_cache_model(base_model, processor, config)

    cache_manager = VLACache(
        patch_size=14,
        sim_threshold=0.996,
        max_static_patches=150,
        max_attention_patches=120,
        enable_visualization=enable_visualization,
    )

    formatted_prompt = apply_chat_template(processor, config, prompt, num_images=1)

    total_time = 0
    total_vision_time = 0
    total_cached_time = 0
    outputs = []

    for frame_idx, image_path in enumerate(image_paths):
        print(f"\n{'='*80}")
        print(f"Frame {frame_idx + 1}/{len(image_paths)}: {Path(image_path).name}")
        print(f"{'='*80}")

        frame_start = time.perf_counter()

        with time_it("Image load"):
            image = Image.open(image_path).convert("RGB")

        with time_it("Prepare inputs"):
            inputs = prepare_inputs(processor, images=[image_path], prompts=formatted_prompt)
            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]
            grid_thw = inputs["image_grid_thw"]
            mask = inputs["attention_mask"]

        cacheable_token_indices = None
        viz_image = None
        layer_schedule = None

        if enable_cache and cache_manager.should_cache():
            with time_it("Cache analysis"):
                prev_attentions, prev_positions = model.get_attentions()

                cacheable_token_indices, viz_image = cache_manager.get_cacheable_tokens(
                    image,
                    prev_attentions,
                    prev_positions,
                )

                if cacheable_token_indices and len(cacheable_token_indices) > 0:
                    layer_schedule = get_layer_mask_schedule(prev_attentions)

                    print(f"  Cacheable tokens: {len(cacheable_token_indices)}")
                    print(
                        f"  Layer schedule: min={layer_schedule.min():.2f}, "
                        f"max={layer_schedule.max():.2f}, mean={layer_schedule.mean():.2f}"
                    )

        vision_start = time.perf_counter()

        dtype = base_model.vision_tower.patch_embed.proj.weight.dtype

        if enable_cache and cacheable_token_indices and len(cacheable_token_indices) > 0:
            vision_hidden, num_computed, num_cached = model.encode_vision_with_cache(
                pixel_values.astype(dtype),
                grid_thw,
                cacheable_indices=cacheable_token_indices,
            )

            vision_time = time.perf_counter() - vision_start
            total_cached_time += vision_time

            print(f"  Vision encoding (cached): {vision_time*1000:.1f}ms")
            print(f"    Computed: {num_computed} tokens")
            print(
                f"    Cached: {num_cached} tokens "
                f"({num_cached/(num_computed+num_cached)*100:.1f}%)"
            )
        else:
            vision_hidden = base_model.vision_tower(pixel_values.astype(dtype), grid_thw)
            mx.eval(vision_hidden)

            vision_time = time.perf_counter() - vision_start
            total_vision_time += vision_time

            print(f"  Vision encoding: {vision_time*1000:.1f}ms")
            print(f"    Computed: {vision_hidden.shape[0]} tokens")

        with time_it("Embed + merge"):
            embeds = base_model.language_model.model.embed_tokens(input_ids)
            merged = base_model.merge_input_ids_with_image_features(
                base_model.config.image_token_id,
                base_model.config.video_token_id,
                vision_hidden,
                embeds,
                input_ids,
            )
            mx.eval(merged)

        with time_it("LM prefill"):
            lm_cache = make_prompt_cache(base_model.language_model)

            out = model.forward_with_cache(
                input_ids,
                merged,
                mask=mask,
                cache=lm_cache,
                cacheable_token_indices=cacheable_token_indices,
                layer_schedule=layer_schedule,
            )
            mx.eval(out.logits)

        with time_it("Token generation"):
            max_tokens = 100
            tokens = []
            for _ in range(max_tokens):
                logits = out.logits[:, -1, :]
                next_tok = mx.argmax(logits, axis=-1)
                tokens.append(next_tok.item())

                if next_tok.item() == processor.tokenizer.eos_token_id:
                    break

                next_embed = base_model.language_model.model.embed_tokens(
                    next_tok.reshape(1, 1)
                )
                out = model.forward_with_cache(
                    next_tok.reshape(1, 1),
                    next_embed,
                    mask=None,
                    cache=lm_cache,
                )
                mx.eval(out.logits)

        output = processor.tokenizer.decode(tokens)
        outputs.append(output)

        frame_time = time.perf_counter() - frame_start
        total_time += frame_time

        print(f"\nOutput: {output[:100]}{'...' if len(output) > 100 else ''}")
        print(f"Frame time: {frame_time*1000:.1f}ms")

        if enable_cache:
            model.update_cache_state(vision_hidden, image)
            cache_manager.update(
                image,
                vision_hidden,
                *model.get_attentions(),
            )

        if enable_visualization and viz_image is not None:
            viz_path = f"vla_cache_viz_frame_{frame_idx:03d}.png"
            Image.fromarray(viz_image).save(viz_path)
            print(f"  Visualization saved: {viz_path}")

    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    print(f"Total frames: {len(image_paths)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per frame: {total_time/len(image_paths)*1000:.1f}ms")

    if enable_cache and len(image_paths) > 1:
        print("\nVision Encoding Breakdown:")
        print(f"  Uncached frames: {total_vision_time*1000:.1f}ms total")
        print(f"  Cached frames: {total_cached_time*1000:.1f}ms total")

        if total_cached_time > 0 and total_vision_time > 0:
            speedup = total_vision_time / (total_cached_time / (len(image_paths) - 1))
            print(f"  Vision speedup: {speedup:.2f}x")

        cache_stats = cache_manager.get_stats()
        print("\nCache Manager Stats:")
        print(f"  Frames processed: {cache_stats['frame_count']}")
        print(f"  Cache hits: {cache_stats['cache_hits']}")
        print(f"  Tokens cached: {cache_stats['tokens_cached']}")
        print(f"  Tokens computed: {cache_stats['tokens_computed']}")

        model_stats = model.get_cache_stats()
        print("\nModel Cache Stats:")
        print(f"  Total vision tokens: {model_stats['total_vision_tokens']}")
        print(f"  Cached vision tokens: {model_stats['cached_vision_tokens']}")
        print(f"  Cache rate: {model_stats['cache_rate']*100:.1f}%")

    print("\nMemory:")
    print(f"  Active: {mx.get_active_memory() / 1e9:.2f} GB")
    print(f"  Peak:   {mx.get_peak_memory() / 1e9:.2f} GB")

    return outputs


def run_comparison(image_paths: List[str], prompt: str):
    """Run with and without cache, print speedup."""
    print("\n" + "="*80)
    print("BASELINE (NO CACHE)")
    print("="*80)

    baseline_start = time.perf_counter()
    run_vla_cache_inference(image_paths, prompt, enable_cache=False)
    baseline_time = time.perf_counter() - baseline_start

    print("\n\n" + "="*80)
    print("VLA-CACHE ENABLED")
    print("="*80)

    cached_start = time.perf_counter()
    run_vla_cache_inference(image_paths, prompt, enable_cache=True, enable_visualization=True)
    cached_time = time.perf_counter() - cached_start

    print("\n\n" + "="*80)
    print("SPEEDUP COMPARISON")
    print("="*80)
    print(f"Baseline time: {baseline_time:.2f}s")
    print(f"Cached time: {cached_time:.2f}s")
    print(f"Speedup: {baseline_time/cached_time:.2f}x")
    print(
        f"Time saved: {(baseline_time - cached_time):.2f}s "
        f"({(1 - cached_time/baseline_time)*100:.1f}%)"
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m vla.vla_inference <image1> [image2] [image3] ...")
        print("\nExamples:")
        print("  # Single image (no caching benefit)")
        print("  python -m vla.vla_inference frame1.jpg")
        print()
        print("  # Image sequence (frame-to-frame caching)")
        print("  python -m vla.vla_inference frame1.jpg frame2.jpg frame3.jpg")
        print()
        print("  # Comparison mode")
        print("  python -m vla.vla_inference --compare frame1.jpg frame2.jpg frame3.jpg")
        sys.exit(1)

    if sys.argv[1] == "--compare":
        image_paths = sys.argv[2:]
        prompt = "Describe what you see in this image."
        run_comparison(image_paths, prompt)
    else:
        image_paths = sys.argv[1:]
        prompt = "Describe what you see in this image."

        for path in image_paths:
            if not Path(path).exists():
                print(f"Error: Image not found: {path}")
                sys.exit(1)

        if len(image_paths) == 1:
            print("Note: Single image - VLA-Cache most effective with sequences")
            run_vla_cache_inference(image_paths, prompt, enable_cache=False)
        else:
            run_vla_cache_inference(
                image_paths,
                prompt,
                enable_cache=True,
                enable_visualization=True,
            )
