import time
import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config, prepare_inputs
from mlx_vlm.models.cache import make_prompt_cache

model_path = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
image_path = "experiments/example.jpg"

print("loading model...")
model, processor = load(model_path)
config = load_config(model_path)

prompt = apply_chat_template(processor, config, "Describe this image.", num_images=1)
inputs = prepare_inputs(processor, images=[image_path], prompts=prompt)

input_ids = inputs["input_ids"]
pixel_values = inputs["pixel_values"]
mask = inputs["attention_mask"]
grid_thw = inputs["image_grid_thw"]

# run vision encoder
dtype = model.vision_tower.patch_embed.proj.weight.dtype
t0 = time.perf_counter()
hidden = model.vision_tower(pixel_values.astype(dtype), grid_thw)
mx.eval(hidden)
print(f"vision encoder: {(time.perf_counter()-t0)*1000:.0f}ms, {hidden.shape[0]} tokens")

# prune by L2 norm - keep top K tokens
K = 200
norms = mx.linalg.norm(hidden, axis=1)
top_k_idx = mx.argpartition(-norms, K)[:K]  # indices of top K by norm
top_k_idx = mx.sort(top_k_idx)  # keep spatial order
pruned_hidden = hidden[top_k_idx]
print(f"pruned to {K} tokens")

# rebuild input_ids with correct number of image placeholders
image_token_id = model.config.image_token_id
orig_ids = input_ids[0].tolist()

# find where image tokens are and replace with K of them
new_ids = []
seen_image = False
for tok in orig_ids:
    if tok == image_token_id:
        if not seen_image:
            new_ids.extend([image_token_id] * K)
            seen_image = True
        # skip remaining image tokens
    else:
        new_ids.append(tok)

new_input_ids = mx.array([new_ids])
new_mask = mx.ones_like(new_input_ids)

print(f"input_ids: {input_ids.shape} -> {new_input_ids.shape}")

# embed + merge
t0 = time.perf_counter()
embeds = model.language_model.model.embed_tokens(new_input_ids)
merged = model.merge_input_ids_with_image_features(
    image_token_id, model.config.video_token_id,
    pruned_hidden, embeds, new_input_ids
)
mx.eval(merged)
print(f"embed + merge: {(time.perf_counter()-t0)*1000:.0f}ms")

# LM prefill
t0 = time.perf_counter()
cache = make_prompt_cache(model.language_model)
out = model.language_model(new_input_ids, merged, mask=new_mask, cache=cache)
mx.eval(out.logits)
print(f"LM prefill: {(time.perf_counter()-t0)*1000:.0f}ms")

# generate a few tokens to see if it works
print("\ngenerating...")
t0 = time.perf_counter()

max_tokens = 50
tokens = []
for i in range(max_tokens):
    logits = out.logits[:, -1, :]
    next_tok = mx.argmax(logits, axis=-1)
    tokens.append(next_tok.item())

    if next_tok.item() == processor.tokenizer.eos_token_id:
        break

    next_embed = model.language_model.model.embed_tokens(next_tok.reshape(1, 1))
    out = model.language_model(next_tok.reshape(1, 1), next_embed, cache=cache)
    mx.eval(out.logits)

print(f"generation: {(time.perf_counter()-t0)*1000:.0f}ms for {len(tokens)} tokens")
print(f"\noutput: {processor.tokenizer.decode(tokens)}")
