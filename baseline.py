import time
from pathlib import Path

import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from PIL import Image


def time_it(name):
    class Timer:
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        def __exit__(self, *args):
            self.elapsed = time.perf_counter() - self.start
            print(f"  {name}: {self.elapsed*1000:.1f}ms")
    return Timer()


def run_baseline(image_path: str, prompt: str = "Describe this image in detail."):
    model_path = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
    print(f"Model: {model_path}")
    print(f"Image: {image_path}")
    print("-" * 50)

    with time_it("Model load"):
        model, processor = load(model_path)
        config = load_config(model_path)

    with time_it("Image load"):
        image = Image.open(image_path)
        print(f"  Image size: {image.size}")

    with time_it("Prompt prep"):
        formatted_prompt = apply_chat_template(
            processor, config, prompt, num_images=1
        )

    print("\nInference:")
    with time_it("Total generation"):
        output = generate(
            model,
            processor,
            formatted_prompt,
            [image_path],
            max_tokens=256,
            verbose=False,
        )

    print("-" * 50)
    print(f"Output:\n{output}")
    print("-" * 50)

    print("\nMemory:")
    print(f"  Active: {mx.get_active_memory() / 1e9:.2f} GB")
    print(f"  Peak:   {mx.get_peak_memory() / 1e9:.2f} GB")

    return output


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python baseline.py <image_path> [prompt]")
        print("\nExample:")
        print("  python baseline.py photo.jpg")
        print("  python baseline.py photo.jpg 'What objects are in this image?'")
        sys.exit(1)

    image_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Describe this image in detail."

    if not Path(image_path).exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    run_baseline(image_path, prompt)
