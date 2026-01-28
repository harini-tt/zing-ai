# VLA Cache for MLX VLM

Adaptive token caching for faster vision-language inference on Apple Silicon.

## Overview

VLA cache reuses vision tokens across adjacent frames by:

1. Detecting static image patches.
2. Filtering out task-relevant patches with attention scores.
3. Reusing cached patch embeddings during vision encoding.

## Installation

```bash
uv sync
```

```bash
pip install mlx mlx-vlm opencv-python scikit-image pillow
```

## Quick Start

```bash
python -m vla.vla_inference frame_001.jpg frame_002.jpg frame_003.jpg
```

```bash
python -m vla.vla_inference --compare frame1.jpg frame2.jpg frame3.jpg
```

## Usage Snippets

```python
from vla import VLACache, get_layer_mask_schedule

cache = VLACache(enable_visualization=True)
```

```python
from vla.vla_cache import find_static_patches, task_relevant_selection
```

## Package Layout

- `vla/vla_cache.py`: patch similarity, attention filtering, cache state
- `vla/vla_model.py`: MLX model wrappers for caching and attention capture
- `vla/vla_inference.py`: frame-by-frame inference harness
- `vla/__init__.py`: public API exports
