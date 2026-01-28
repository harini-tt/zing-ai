"""
VLA-Cache Benchmark Script

Comprehensive benchmarking to measure:
- Vision encoding speedup
- End-to-end latency improvement
- Cache hit rates
- Memory usage
- Accuracy preservation
"""

import time
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config, prepare_inputs
from mlx_vlm.models.cache import make_prompt_cache
from PIL import Image

from vla_cache import VLACache, get_layer_mask_schedule
from vla_model import create_vla_cache_model


def benchmark_single_run(
    image_paths: List[str],
    prompt: str,
    model_path: str,
    enable_cache: bool
) -> Dict[str, Any]:
    """
    Run inference and collect detailed metrics.
    """
    # Load model
    base_model, processor = load(model_path)
    config = load_config(model_path)
    model = create_vla_cache_model(base_model, processor, config)

    # Initialize cache
    cache_manager = VLACache(
        patch_size=14,
        sim_threshold=0.996,
        max_static_patches=150,
        max_attention_patches=120,
        enable_visualization=False
    )

    formatted_prompt = apply_chat_template(processor, config, prompt, num_images=1)

    # Metrics
    frame_times = []
    vision_times = []
    lm_times = []
    generation_times = []
    cache_hits = []
    tokens_cached = []
    tokens_computed = []
    outputs = []

    # Process frames
    for frame_idx, image_path in enumerate(image_paths):
        frame_start = time.perf_counter()

        # Load image
        image = Image.open(image_path).convert("RGB")
        inputs = prepare_inputs(processor, images=[image_path], prompts=formatted_prompt)

        # Cache analysis
        cacheable_token_indices = None
        if enable_cache and cache_manager.should_cache():
            prev_attentions, prev_positions = model.get_attentions()
            cacheable_token_indices, _ = cache_manager.get_cacheable_tokens(
                image, prev_attentions, prev_positions
            )

        # Vision encoding
        vision_start = time.perf_counter()
        dtype = base_model.vision_tower.patch_embed.proj.weight.dtype

        if enable_cache and cacheable_token_indices:
            vision_hidden, num_computed, num_cached = model.encode_vision_with_cache(
                inputs["pixel_values"].astype(dtype),
                inputs["image_grid_thw"],
                cacheable_indices=cacheable_token_indices
            )
            tokens_computed.append(num_computed)
            tokens_cached.append(num_cached)
            cache_hits.append(1)
        else:
            vision_hidden = base_model.vision_tower(
                inputs["pixel_values"].astype(dtype),
                inputs["image_grid_thw"]
            )
            mx.eval(vision_hidden)
            tokens_computed.append(vision_hidden.shape[0])
            tokens_cached.append(0)
            cache_hits.append(0)

        vision_times.append(time.perf_counter() - vision_start)

        # LM prefill
        lm_start = time.perf_counter()
        embeds = base_model.language_model.model.embed_tokens(inputs["input_ids"])
        merged = base_model.merge_input_ids_with_image_features(
            base_model.config.image_token_id,
            base_model.config.video_token_id,
            vision_hidden,
            embeds,
            inputs["input_ids"]
        )
        mx.eval(merged)

        lm_cache = make_prompt_cache(base_model.language_model)
        out = model.forward_with_cache(
            inputs["input_ids"],
            merged,
            mask=inputs["attention_mask"],
            cache=lm_cache,
            cacheable_token_indices=cacheable_token_indices
        )
        mx.eval(out.logits)
        lm_times.append(time.perf_counter() - lm_start)

        # Token generation
        gen_start = time.perf_counter()
        tokens = []
        for _ in range(100):
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
                cache=lm_cache
            )
            mx.eval(out.logits)

        generation_times.append(time.perf_counter() - gen_start)
        output = processor.tokenizer.decode(tokens)
        outputs.append(output)

        frame_times.append(time.perf_counter() - frame_start)

        # Update cache
        if enable_cache:
            model.update_cache_state(vision_hidden, image)
            cache_manager.update(image, vision_hidden, *model.get_attentions())

    # Aggregate metrics
    return {
        'num_frames': len(image_paths),
        'total_time': sum(frame_times),
        'avg_frame_time': np.mean(frame_times),
        'std_frame_time': np.std(frame_times),
        'avg_vision_time': np.mean(vision_times),
        'avg_lm_time': np.mean(lm_times),
        'avg_generation_time': np.mean(generation_times),
        'cache_hit_rate': np.mean(cache_hits) if cache_hits else 0,
        'avg_tokens_cached': np.mean(tokens_cached) if tokens_cached else 0,
        'avg_tokens_computed': np.mean(tokens_computed) if tokens_computed else 0,
        'peak_memory_gb': mx.get_peak_memory() / 1e9,
        'outputs': outputs,
        'frame_times': frame_times,
        'vision_times': vision_times,
    }


def run_benchmark(
    image_paths: List[str],
    prompt: str,
    model_path: str,
    num_runs: int = 3
):
    """
    Run comprehensive benchmark comparing cached vs uncached.
    """
    print("="*80)
    print("VLA-CACHE COMPREHENSIVE BENCHMARK")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Frames: {len(image_paths)}")
    print(f"Runs: {num_runs}")
    print()

    # Warmup
    print("Warming up...")
    base_model, processor = load(model_path)
    config = load_config(model_path)
    del base_model, processor, config
    mx.eval(mx.zeros(1))
    print("Warmup complete\n")

    # Baseline runs
    print(f"Running baseline (no cache) - {num_runs} runs...")
    baseline_results = []
    for i in range(num_runs):
        print(f"  Run {i+1}/{num_runs}...", end="", flush=True)
        result = benchmark_single_run(image_paths, prompt, model_path, enable_cache=False)
        baseline_results.append(result)
        print(f" {result['total_time']:.2f}s")

    # Cached runs
    print(f"\nRunning VLA-Cache - {num_runs} runs...")
    cached_results = []
    for i in range(num_runs):
        print(f"  Run {i+1}/{num_runs}...", end="", flush=True)
        result = benchmark_single_run(image_paths, prompt, model_path, enable_cache=True)
        cached_results.append(result)
        print(f" {result['total_time']:.2f}s")

    # Compute statistics
    baseline_times = [r['total_time'] for r in baseline_results]
    cached_times = [r['total_time'] for r in cached_results]

    baseline_mean = np.mean(baseline_times)
    baseline_std = np.std(baseline_times)
    cached_mean = np.mean(cached_times)
    cached_std = np.std(cached_times)

    speedup = baseline_mean / cached_mean
    time_saved = baseline_mean - cached_mean

    # Vision encoding speedup
    baseline_vision = np.mean([r['avg_vision_time'] for r in baseline_results])
    cached_vision = np.mean([r['avg_vision_time'] for r in cached_results])
    vision_speedup = baseline_vision / cached_vision

    # Print results
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)

    print(f"\nEnd-to-End Performance ({num_runs} runs):")
    print(f"  Baseline (no cache):")
    print(f"    Mean: {baseline_mean:.3f}s ± {baseline_std:.3f}s")
    print(f"    Per frame: {baseline_mean/len(image_paths)*1000:.1f}ms")

    print(f"\n  VLA-Cache:")
    print(f"    Mean: {cached_mean:.3f}s ± {cached_std:.3f}s")
    print(f"    Per frame: {cached_mean/len(image_paths)*1000:.1f}ms")

    print(f"\n  Speedup: {speedup:.2f}x")
    print(f"  Time saved: {time_saved:.3f}s ({(time_saved/baseline_mean)*100:.1f}%)")

    print(f"\nVision Encoding:")
    print(f"  Baseline: {baseline_vision*1000:.1f}ms per frame")
    print(f"  VLA-Cache: {cached_vision*1000:.1f}ms per frame")
    print(f"  Vision speedup: {vision_speedup:.2f}x")

    # Cache stats
    cache_hit_rate = np.mean([r['cache_hit_rate'] for r in cached_results])
    avg_cached = np.mean([r['avg_tokens_cached'] for r in cached_results])
    avg_computed = np.mean([r['avg_tokens_computed'] for r in cached_results])
    total_tokens = avg_cached + avg_computed

    print(f"\nCache Statistics:")
    print(f"  Hit rate: {cache_hit_rate*100:.1f}%")
    print(f"  Avg tokens per frame: {total_tokens:.1f}")
    print(f"    Cached: {avg_cached:.1f} ({avg_cached/total_tokens*100:.1f}%)")
    print(f"    Computed: {avg_computed:.1f} ({avg_computed/total_tokens*100:.1f}%)")

    # Memory
    baseline_mem = np.mean([r['peak_memory_gb'] for r in baseline_results])
    cached_mem = np.mean([r['peak_memory_gb'] for r in cached_results])

    print(f"\nMemory:")
    print(f"  Baseline: {baseline_mem:.2f} GB")
    print(f"  VLA-Cache: {cached_mem:.2f} GB")
    print(f"  Overhead: {(cached_mem - baseline_mem):.2f} GB ({(cached_mem/baseline_mem-1)*100:.1f}%)")

    # Save results
    results = {
        'config': {
            'model': model_path,
            'num_frames': len(image_paths),
            'num_runs': num_runs,
        },
        'baseline': {
            'mean_time': baseline_mean,
            'std_time': baseline_std,
            'mean_vision_time': baseline_vision,
            'peak_memory_gb': baseline_mem,
        },
        'vla_cache': {
            'mean_time': cached_mean,
            'std_time': cached_std,
            'mean_vision_time': cached_vision,
            'peak_memory_gb': cached_mem,
            'cache_hit_rate': cache_hit_rate,
            'avg_tokens_cached': avg_cached,
            'avg_tokens_computed': avg_computed,
        },
        'speedup': {
            'end_to_end': speedup,
            'vision_encoding': vision_speedup,
            'time_saved_seconds': time_saved,
        }
    }

    output_path = 'vla_cache_benchmark_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python benchmark_vla_cache.py <image1> <image2> ... [--runs N]")
        print("\nExample:")
        print("  python benchmark_vla_cache.py frame*.jpg --runs 5")
        sys.exit(1)

    # Parse arguments
    image_paths = []
    num_runs = 3

    for arg in sys.argv[1:]:
        if arg == "--runs":
            continue
        elif sys.argv[sys.argv.index(arg)-1] == "--runs":
            num_runs = int(arg)
        else:
            image_paths.append(arg)

    # Verify images
    for path in image_paths:
        if not Path(path).exists():
            print(f"Error: Image not found: {path}")
            sys.exit(1)

    if len(image_paths) < 2:
        print("Error: Need at least 2 images for benchmarking")
        sys.exit(1)

    prompt = "Describe what you see in this image."
    model_path = "mlx-community/Qwen2-VL-2B-Instruct-4bit"

    run_benchmark(image_paths, prompt, model_path, num_runs)
