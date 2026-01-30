# Zing AI - On-Device ML Research Assistant

A vision-language research assistant that runs entirely on-device using Apple Silicon. Point your camera at ML papers, figures, and graphs to get instant analysis with context from your paper library.

## Features

- **On-Device Inference**: FastVLM models (0.5B/1.5B/7B) running locally via MLX - no cloud required
- **Research Context**: Automatically references your indexed papers when analyzing figures
- **Persistent Memory**: Save insights to local SQLite database, recall across sessions
- **Single-Shot Analysis**: Capture and analyze frames on demand with custom questions

## Demo

Coming soon!

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/harini-tt/zing-ai.git
cd zing-ai
uv sync
```

### 2. Download a model

```bash
cd fastvlm/app

# Choose one:
./get_pretrained_mlx_model.sh --model 0.5b --dest FastVLM/model  # Fast, lower quality
./get_pretrained_mlx_model.sh --model 1.5b --dest FastVLM/model  # Balanced (recommended)
./get_pretrained_mlx_model.sh --model 7b --dest FastVLM/model    # Best quality, needs 8GB+ RAM
```

### 3. Index your papers (optional)

```bash
# Add PDFs to papers/ folder, then:
cd research-assistant
uv run export_embeddings.py
# Copy output to app bundle or Documents folder
```

### 4. Build and run

Open `fastvlm/app/FastVLM.xcodeproj` in Xcode, select your device, and run.

## Usage

1. **Point camera** at a figure, graph, or table from an ML paper
2. **Edit question** (tap pencil icon) - or use default
3. **Tap "Capture Photo"** to analyze
4. **Read response** with specific values and insights
5. **Save to Memory** if you want to keep the insight

## Key Components

### FastVLM Model
Apple's efficient vision-language model optimized for on-device inference with MLX.

### Research Context
Keyword-based retrieval from indexed paper chunks. When you analyze a figure, relevant paper excerpts are included in the prompt.

### Insight Store
SQLite-based persistent storage for saving analysis results. Survives app restarts.

### Dynamic Token Optimization
Vision token pruning and ToMe (Token Merging) to reduce computation:
- **Pruning**: Removes low-importance vision tokens based on L2 norm scores
- **ToMe**: Merges similar tokens using bipartite soft matching ([Bolya et al.](https://arxiv.org/abs/2210.09461))

Configure via environment variables:
```bash
FASTVLM_KEEP_FRACTION=0.5      # Keep 50% of tokens after pruning
FASTVLM_TOME_ENABLED=true      # Enable token merging
FASTVLM_TOME_KEEP_RATIO=0.5    # Merge down to 50% of remaining tokens
```

### VLA Cache
Adaptive token caching for faster inference on video frames by reusing vision tokens across similar frames.

## Requirements

- macOS 15+ / iOS 18+
- Xcode 16+
- Apple Silicon (M1+ Mac or A14+ iPhone)
- 4GB+ free storage for models

## License

See LICENSE files in respective directories.
