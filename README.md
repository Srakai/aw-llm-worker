# aw-llm-worker

ActivityWatch worker with dual functionality:

1. **Time Summarization** (main feature) - Analyze ActivityWatch events and classify time blocks using text convolution
2. **Screenshot Classification** (optional) - LLM-based screenshot labeling

## Features

- Automatically classifies screenshots from ActivityWatch watcher
- Uses Qwen2.5-VL-7B-Instruct (GGUF) with Metal GPU acceleration
- Intelligent project matching based on keywords
- Pushes structured classification events to ActivityWatch
- **Custom visualization** - Interactive timeline view of time blocks and screenshots

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

### Summarization Only (Main Feature)

```bash
python aw_llm_worker.py \
  --mode summarization \
  --summarization-interval 6.0 \
  --lookback-hours 8 \
  --context topics.yaml
```

### Screenshots Only

```bash
python aw_llm_worker.py \
  --mode screenshots \
  --spool-dir /path/to/screenshot/spool \
  --screenshot-poll 0.8
```

### Both Modes (Dual Operation)

```bash
python aw_llm_worker.py \
  --mode both \
  --spool-dir /path/to/spool \
  --screenshot-poll 0.8 \
  --summarization-interval 6.0 \
  --source-buckets aw-watcher-window_hostname \
  --source-buckets aw-watcher-afk_hostname
```

### With Context File

```bash
uv run aw_llm_worker.py \
  --spool-dir "/Users/filip/Library/Application Support/activitywatch/Screenshots" \
  --context context.yaml \
  --use-cli
```

## Configuration

### Topics Configuration (YAML)

```yaml
topics:
  Coding:
    phrases: ["coding", "python", "vscode", "git", "terminal"]
    kernel_width: 15 # ~4 minutes at 15s resolution
    threshold: 0.3 # Minimum match score
    min_duration_s: 120 # 2 min minimum block
    merge_gap_s: 60 # Merge blocks within 60s

  Writing:
    phrases: ["writing", "docs", "obsidian", "notes"]
    kernel_width: 10
    threshold: 0.4
    min_duration_s: 180
    merge_gap_s: 60

projects:
  - name: "MyProject"
    keywords: ["myproject", "special-keyword"]
```

## Visualization

The project includes a custom web-based visualization that displays both LLM time blocks and screenshot events on an interactive timeline.

### Setup Custom Visualization

**Step 1: Configure ActivityWatch Server**

Add the visualization path to your `aw-server.toml` config file:

**macOS/Linux:**
```bash
# One-liner to add visualization config (modify path as needed)
echo -e '\n[server.custom_static]\naw-llm-worker = "'$(pwd)'/visualization"' >> ~/Library/"Application Support"/activitywatch/aw-server/aw-server.toml
```

Or manually add to `~/Library/Application Support/activitywatch/aw-server/aw-server.toml`:
```toml
[server.custom_static]
aw-llm-worker = "/Users/YOUR_USERNAME/code/aw-llm-worker/visualization"
```

**Windows:**

Manually add to `%LOCALAPPDATA%\activitywatch\aw-server\aw-server.toml`:
```toml
[server.custom_static]
aw-llm-worker = '''C:\Users\YOUR_USERNAME\code\aw-llm-worker\visualization'''
```

**Linux:**
```bash
# One-liner to add visualization config
echo -e '\n[server.custom_static]\naw-llm-worker = "'$(pwd)'/visualization"' >> ~/.config/activitywatch/aw-server/aw-server.toml
```

**⚠️ Important:** Make sure you edit `aw-server.toml`, **NOT** `aw-qt.toml`!

**Step 2: Restart ActivityWatch**

Completely quit and restart ActivityWatch for the configuration to take effect.

**Step 3: Add Custom Visualization**

1. Open ActivityWatch web interface (usually http://localhost:5600)
2. Go to **Activity** → **Edit View**
3. Click **Add Visualization** → **Custom Visualization**
4. Enter `aw-llm-worker` as the visualization name
5. The timeline visualization will now appear in your activity view

### Visualization Features

- **📊 Activity Time Blocks** - Color-coded horizontal bars showing classified time periods (Coding, Writing, etc.)
- **📸 Screenshot Timeline** - Interactive scatter plot of screenshot events
- **🎨 Dark Theme** - Easy on the eyes with modern styling
- **⏱️ Time Controls** - Adjust time range and hostname
- **💡 Interactive Tooltips** - Hover over data points for detailed information
- **🎯 Auto-Detection** - Automatically detects hostname from existing buckets

### URL Parameters

You can also open the visualization directly with custom parameters:

```
http://localhost:5600/static/aw-llm-worker/?start=2025-10-12T00:00&end=2025-10-12T23:59&hostname=YOUR_HOSTNAME
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

## Key Features

### 1. **Dual-Mode Operation**

- Separate poll intervals for screenshots vs. summarization
- Screenshots: Fast poll (0.8s default)
- Summarization: Periodic runs (6h default)

### 2. **Model Lifecycle Management**

- Vision model loaded on-demand for screenshot batches
- Automatically unloaded when no work or before summarization
- Explicit garbage collection to free VRAM

### 3. **Overlap Handling (Conv-Style)**

- Events contribute fractionally to time frames based on overlap
- Similar to 1D convolution: smooth aggregation across boundaries
- L2-normalized frame vectors for consistent magnitude

### 4. **Performance Optimizations**

- State saves decoupled from marking (batch writes)
- Pre-computed text vector cache per discretization
- Efficient stride tricks for sliding windows
- Batched event emission to ActivityWatch (500/batch)

### 5. **Robust Error Handling**

- Per-bucket fetch with graceful degradation
- Malformed event skipping
- Main loop exception recovery with backoff

## Output

### Summarization Bucket

- **Bucket**: `aw-llm-blocks_{hostname}`
- **Event Type**: `classified_time`
- **Data Schema**:
  ```json
  {
    "label": "Coding",
    "confidence": 0.75
  }
  ```

### Screenshot Bucket

- **Bucket**: `aw-watcher-screenshot-llm_{hostname}`
- **Event Type**: `app.screenshot.label`
- **Data Schema**: (existing format with label, project, tags, etc.)

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

## Development Notes

### Why Hash-Based Text Encoding?

- No vocabulary required (works with any text)
- Deterministic (reproducible across runs)
- Fast (no model loading)
- Collision-resistant (SHA1-based)

### Conv1D Analogy

1. **Events** → Discretized time-frame matrix `M[t, d]`
2. **Topic kernels** → Triangular-weighted phrase vectors `K[w, d]`
3. **Convolution** → Slide kernel over time, compute similarity scores
4. **Thresholding** → Extract high-confidence segments
5. **Merging** → Combine nearby segments (gap < 60s)

## Future Enhancements

- [ ] Topic discovery (unsupervised clustering)
- [ ] Multi-scale analysis (different time resolutions)
- [ ] Combined timeline view (blocks + screenshots on same axis)
- [ ] Export visualization data to JSON/CSV
- [ ] Screenshot thumbnail support in tooltips
