import time
import sys
from pathlib import Path

import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config, prepare_inputs
from mlx_vlm.models.cache import make_prompt_cache


def inspect_model(model):
    print("\n=== MODEL STRUCTURE ===")
    for name in dir(model):
        if not name.startswith("_"):
            attr = getattr(model, name, None)
            if hasattr(attr, "parameters"):
                params = sum(p.size for p in mx.utils.tree_flatten(attr.parameters()))
                print(f"  {name}: {params/1e6:.1f}M params")


def inspect_cache(cache):
    print(f"\n=== KV CACHE ({len(cache)} layers) ===")
    total_mb = 0
    for i, c in enumerate(cache[:3]):
        if hasattr(c, "keys") and c.keys is not None:
            k_mb = c.keys.nbytes / 1e6
            print(f"  Layer {i}: keys {c.keys.shape}, offset={c.offset} ({k_mb:.1f}MB)")
            total_mb += k_mb * 2
    print(f"  ... estimated total: {total_mb * len(cache) / 3:.0f}MB")


def main(image_path):
    model_path = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
    print(f"Loading {model_path}")
    model, processor = load(model_path)
    config = load_config(model_path)

    inspect_model(model)

    # prep inputs
    prompt = "What is this?"
    formatted = apply_chat_template(processor, config, prompt, num_images=1)
    inputs = prepare_inputs(processor, images=[image_path], prompts=formatted)

    input_ids = inputs["input_ids"]
    pixel_values = inputs.get("pixel_values")
    mask = inputs.get("attention_mask")

    print(f"\n=== INPUTS ===")
    print(f"  input_ids: {input_ids.shape} ({input_ids.size} tokens)")
    if "image_grid_thw" in inputs:
        t, h, w = inputs["image_grid_thw"][0]
        print(f"  image_grid: {t}x{h}x{w} = {int(t*h*w)} vision tokens")

    # run forward pass to populate cache
    cache = make_prompt_cache(model.language_model)
    print(f"\n=== RUNNING FORWARD PASS ===")
    t0 = time.perf_counter()
    out = model(input_ids, pixel_values, cache=cache, mask=mask)
    mx.eval(out.logits)
    print(f"  took {(time.perf_counter()-t0)*1000:.0f}ms")

    inspect_cache(cache)

    print(f"\n=== MEMORY ===")
    print(f"  peak: {mx.get_peak_memory()/1e9:.2f}GB")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python inspect_cache.py <image>")
        sys.exit(1)
    if not Path(sys.argv[1]).exists():
        print(f"not found: {sys.argv[1]}")
        sys.exit(1)
    main(sys.argv[1])
