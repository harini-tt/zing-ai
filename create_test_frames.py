"""
Create synthetic test frames for VLA-Cache benchmarking.
Generates frames with controlled amounts of change to simulate video sequences.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def create_test_frames(num_frames=5, image_size=224, output_dir="test_frames"):
    """
    Create synthetic test frames with progressive changes.

    Args:
        num_frames: Number of frames to generate
        image_size: Size of square images
        output_dir: Directory to save frames
    """
    os.makedirs(output_dir, exist_ok=True)

    # Base image with some structure
    base_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    # Add a gradient background
    for i in range(image_size):
        base_image[i, :, 0] = int(100 + (i / image_size) * 100)  # Red gradient
        base_image[:, i, 1] = int(80 + (i / image_size) * 120)   # Green gradient
        base_image[i, :, 2] = int(60 + (i / image_size) * 140)   # Blue gradient

    # Generate frames with progressive changes
    for frame_idx in range(num_frames):
        img_array = base_image.copy()
        img = Image.fromarray(img_array)
        draw = ImageDraw.Draw(img)

        # Static elements (should be cached)
        # Draw fixed rectangles
        draw.rectangle([10, 10, 50, 50], fill=(200, 50, 50))
        draw.rectangle([image_size-60, 10, image_size-20, 50], fill=(50, 200, 50))
        draw.rectangle([10, image_size-60, 50, image_size-20], fill=(50, 50, 200))

        # Static circle
        draw.ellipse([image_size//2 - 30, image_size//2 - 30,
                     image_size//2 + 30, image_size//2 + 30],
                     fill=(200, 200, 50))

        # Dynamic elements (should NOT be cached)
        # Moving rectangle (simulates motion)
        moving_x = 60 + frame_idx * 20
        moving_y = image_size // 2
        draw.rectangle([moving_x, moving_y, moving_x + 40, moving_y + 40],
                      fill=(255, 100, 100))

        # Frame number (changes every frame)
        try:
            draw.text((image_size - 80, image_size - 30),
                     f"Frame {frame_idx}",
                     fill=(255, 255, 255))
        except:
            # If font not available, draw a changing pattern instead
            draw.rectangle([image_size - 80, image_size - 30,
                          image_size - 30, image_size - 10],
                          fill=(255, 255, 255))

        # Add some noise to simulate camera noise (small amount)
        noise = np.random.randint(-5, 5, (image_size, image_size, 3), dtype=np.int16)
        img_array = np.array(img, dtype=np.int16)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

        # Save frame
        final_img = Image.fromarray(img_array)
        output_path = os.path.join(output_dir, f"frame_{frame_idx:03d}.jpg")
        final_img.save(output_path, quality=95)
        print(f"Created {output_path}")

    print(f"\nCreated {num_frames} test frames in {output_dir}/")
    print(f"Expected cache behavior:")
    print(f"  - Static elements (corners, center circle): HIGH similarity, CACHEABLE")
    print(f"  - Moving rectangle: LOW similarity, should NOT be cached")
    print(f"  - Frame label: Changes every frame, should NOT be cached")
    print(f"  - Expected cache hit rate: ~60-70% of patches")

if __name__ == "__main__":
    create_test_frames(num_frames=5, image_size=224)
