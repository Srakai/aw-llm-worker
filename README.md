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
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- ActivityWatch running

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

### Basic Usage (Python Library - Slower)

```bash
uv run aw_llm_worker.py --spool-dir "/Users/filip/Library/Application Support/activitywatch/Screenshots"
```

### CLI Mode (3x Faster - Recommended!)

```bash
uv run aw_llm_worker.py \
  --spool-dir "/Users/filip/Library/Application Support/activitywatch/Screenshots" \
  --use-cli
```

The `--use-cli` flag uses `llama-mtmd-cli` directly instead of the Python library, providing:

- **~12 seconds** image encoding (vs ~38 seconds with Python library)
- **3x faster** overall performance
- Same Metal GPU acceleration
- No additional dependencies (uses system llama.cpp installation)

### With Context File

```bash
uv run aw_llm_worker.py \
  --spool-dir "/Users/filip/Library/Application Support/activitywatch/Screenshots" \
  --context context.yaml \
  --use-cli
```

### Custom Model Paths

```bash
uv run aw_llm_worker.py \
  --spool-dir "/path/to/spool" \
  --model /path/to/model.gguf \
  --mmproj /path/to/mmproj.gguf \
  --use-cli
```

### Specify CLI Path

If `llama-mtmd-cli` is not in your PATH:

```bash
uv run aw_llm_worker.py \
  --spool-dir "/path/to/spool" \
  --use-cli \
  --cli-path /usr/local/bin/llama-mtmd-cli
```

### All Options

```bash
uv run aw_llm_worker.py --help
```

## Context File Format

Create a `context.yaml` file to provide project hints:

```yaml
role: "Software Engineer"
org: "Your Company"

projects:
  - name: "ActivityWatch Development"
    keywords:
      - "activitywatch"
      - "aw-"
      - "aw_"
      - "python"
      - "watcher"

  - name: "Web Project"
    keywords:
      - "react"
      - "nextjs"
      - "tailwind"

routing:
  prefer_exact_match: false # true = always use keyword match over LLM guess
```

## Performance

With Metal GPU acceleration on Apple Silicon:

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

## Troubleshooting

### Metal GPU Not Working

Check if llama-cpp-python has Metal support:

```bash
python -c "from llama_cpp import Llama; print(Llama.has_metal())"
```

If it returns `False`, reinstall with Metal flags:

```bash
CMAKE_ARGS="-DLLAMA_METAL=on" uv pip install --reinstall --no-cache llama-cpp-python
```

### Slow Performance

- Ensure Metal support is enabled (see above)
- Check GPU layers: `--n-gpu-layers -1` (default) offloads all layers
- Monitor GPU usage with `sudo powermetrics --samplers gpu_power -i 1000`

### Missing Screenshots

Ensure the screenshot watcher is running and writing to the correct spool directory.

## License

[Add your license here]
