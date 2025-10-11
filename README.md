# aw-llm-worker

AI-powered screenshot classification worker for ActivityWatch using Qwen2.5-VL vision language model.

## Features

- Automatically classifies screenshots from ActivityWatch watcher
- Uses Qwen2.5-VL-7B-Instruct (GGUF) with Metal GPU acceleration
- Intelligent project matching based on keywords
- Pushes structured classification events to ActivityWatch

## Installation

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.12
- [uv](https://github.com/astral-sh/uv) package manager
- [ActivityWatch running](https://activitywatch.net/)
- [ActivityWatch Screenshots Watcher running](https://github.com/Srakai/aw-watcher-screenshot/)

### Install with Metal GPU Support

```bash
# Clone the repository
git clone <your-repo-url>
cd aw-llm-worker

# Install with uv (includes Metal compilation flags)
uv sync

# Or manually ensure Metal support
CMAKE_ARGS="-DLLAMA_METAL=on" uv pip install --reinstall --no-cache llama-cpp-python
```

### Download Models

Download the Qwen2.5-VL model and mmproj files:

```bash
# Default paths used by the worker:
# Model: ~/.cache/lm-studio/models/unsloth/Qwen2.5-VL-7B-Instruct-GGUF/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf
# MMProj: ~/.cache/lm-studio/models/unsloth/Qwen2.5-VL-7B-Instruct-GGUF/mmproj-F16.gguf

# You can download from Hugging Face:
huggingface-cli download unsloth/Qwen2.5-VL-7B-Instruct-GGUF \
  --local-dir ~/.cache/lm-studio/models/unsloth/Qwen2.5-VL-7B-Instruct-GGUF
```

## Usage

### Basic Usage (llama-cpp-python)

```bash
uv run aw_llm_worker.py --spool-dir "/Users/filip/Library/Application Support/activitywatch/Screenshots"
```

### Faster execution (custom cli wrapper)

```bash
uv run aw_llm_worker.py \
  --spool-dir "/Users/filip/Library/Application Support/activitywatch/Screenshots" \
  --use-cli
```

### With Context File

```bash
uv run aw_llm_worker.py \
  --spool-dir "/Users/filip/Library/Application Support/activitywatch/Screenshots" \
  --context context.yaml \
  --use-cli
```

## Performance

With Metal GPU acceleration on M2 Apple Silicon:

- **Image encoding**: ~10-15 seconds
- **Classification**: ~18 tokens/second
- **Total per screenshot**: ~20-30 seconds

If you see slow performance (>60 seconds for encoding), ensure Metal support is properly compiled:

```bash
CMAKE_ARGS="-DLLAMA_METAL=on" uv pip install --reinstall --no-cache llama-cpp-python
```

## Output Schema

Events are pushed to ActivityWatch with this structure:

```json
{
  "src": {
    "path": "/path/to/screenshot.png",
    "sha256": "...",
    "app": "VSCode",
    "title": "...",
    "pid": 12345,
    "win_id": 6291463,
    "bbox": [0, 0, 1920, 1080]
  },
  "label": {
    "coarse_activity": "coding",
    "app_guess": "Visual Studio Code",
    "summary": "Working on Python script for screenshot classification",
    "tags": ["python", "vscode", "coding"],
    "project": {
      "name": "ActivityWatch Development",
      "confidence": 0.85,
      "reason": "keywords: activitywatch, python"
    },
    "confidence": 0.9
  },
  "llm": {
    "model": "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
    "mmproj": "mmproj-F16.gguf",
    "temperature": 0.2,
    "prompt_rev": "qwen2.5vl-screenshot-labeler/v1"
  },
  "context": {
    "context_id": "abc123...",
    "matched_keywords": "keywords: activitywatch, python"
  }
}
```
