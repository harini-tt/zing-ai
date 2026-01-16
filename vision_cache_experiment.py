import time
import sys
from pathlib import Path

import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config, prepare_inputs
from mlx_vlm.models.cache import make_prompt_cache


class SpatialVisionCache:
    def __init__(self, threshold=0.1):
        self.cache = {}
        self.threshold = threshold

    def _key(self, x, y, z):
        qx = round(x / self.threshold) * self.threshold
        qy = round(y / self.threshold) * self.threshold
        qz = round(z / self.threshold) * self.threshold
        return f"{qx:.2f},{qy:.2f},{qz:.2f}"

    def get(self, x, y, z):
        return self.cache.get(self._key(x, y, z))

    def put(self, x, y, z, hidden_states, grid_thw):
        key = self._key(x, y, z)
        self.cache[key] = {"hidden_states": hidden_states, "grid_thw": grid_thw}

    def stats(self):
        mem = sum(e["hidden_states"].nbytes for e in self.cache.values()) / 1e6
        return f"{len(self.cache)} entries, {mem:.1f}MB"


def run_with_cache(model, processor, image_path, prompt, cache, location):
    config = load_config("mlx-community/Qwen2-VL-2B-Instruct-4bit")
    formatted = apply_chat_template(processor, config, prompt, num_images=1)
    inputs = prepare_inputs(processor, images=[image_path], prompts=formatted)

    input_ids = inputs["input_ids"]
    pixel_values = inputs.get("pixel_values")
    mask = inputs.get("attention_mask")
    grid_thw = inputs.get("image_grid_thw")

    x, y, z = location
    cached = cache.get(x, y, z)

    # vision encoder
    if cached:
        print(f"  HIT @ {location}")
        hidden_states = cached["hidden_states"]
        t_vision = 0.0
    else:
        print(f"  MISS @ {location}")
        t0 = time.perf_counter()
        dtype = model.vision_tower.patch_embed.proj.weight.dtype
        hidden_states = model.vision_tower(pixel_values.astype(dtype), grid_thw)
        mx.eval(hidden_states)
        t_vision = time.perf_counter() - t0
        cache.put(x, y, z, hidden_states, grid_thw)

    # language model
    t0 = time.perf_counter()
    embeds = model.language_model.model.embed_tokens(input_ids)
    merged = model.merge_input_ids_with_image_features(
        model.config.image_token_id, model.config.video_token_id,
        hidden_states, embeds, input_ids
    )
    kv_cache = make_prompt_cache(model.language_model)
    out = model.language_model(input_ids, merged, mask=mask, cache=kv_cache)
    mx.eval(out.logits)
    t_lm = time.perf_counter() - t0

    return {"vision": t_vision, "lm": t_lm, "total": t_vision + t_lm, "hit": cached is not None}


def main(image_path):
    print("loading model...")
    model, processor = load("mlx-community/Qwen2-VL-2B-Instruct-4bit")

    cache = SpatialVisionCache(threshold=0.1)

    # simulate looking around
    locations = [
        (0.0, 0.0, 1.0),   # A: desk
        (1.0, 0.0, 1.0),   # B: window
        (0.0, 0.0, 1.0),   # A again (hit)
        (0.05, 0.0, 1.0),  # near A (hit)
        (2.0, 0.0, 1.0),   # C: door
        (0.0, 0.0, 1.0),   # A again (hit)
    ]

    results = []
    for i, loc in enumerate(locations):
        print(f"\n[{i+1}] looking at {loc}")
        r = run_with_cache(model, processor, image_path, "What do you see?", cache, loc)
        results.append(r)
        print(f"  vision: {r['vision']*1000:.0f}ms, lm: {r['lm']*1000:.0f}ms, total: {r['total']*1000:.0f}ms")

    # summary
    hits = [r for r in results if r["hit"]]
    misses = [r for r in results if not r["hit"]]

    print(f"\n=== SUMMARY ===")
    print(f"hits: {len(hits)}, misses: {len(misses)}")
    print(f"cache: {cache.stats()}")

    if hits and misses:
        avg_hit = sum(r["total"] for r in hits) / len(hits)
        avg_miss = sum(r["total"] for r in misses) / len(misses)
        print(f"avg hit:  {avg_hit*1000:.0f}ms")
        print(f"avg miss: {avg_miss*1000:.0f}ms")
        print(f"speedup:  {avg_miss/avg_hit:.1f}x")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python vision_cache_experiment.py <image>")
        sys.exit(1)
    if not Path(sys.argv[1]).exists():
        print(f"not found: {sys.argv[1]}")
        sys.exit(1)
    main(sys.argv[1])
