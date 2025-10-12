#!/usr/bin/env python3
"""Main CLI entry point for aw-llm-worker."""

import os
import json
import time
import glob
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List

import click
from aw_client import ActivityWatchClient
from aw_core import Event

from awllm.utils.helpers import load_yaml_or_json, sha1, read_json, iso_to_dt
from awllm.utils.state import State
from awllm.models import QwenVLPython, QwenVLCLI
from awllm.prompt import PROMPT_REV, route_project_keywords

# Analysis imports
from awllm.analysis.textconv import HashSketch, conv1d_text, make_kernel_from_phrases
from awllm.analysis.aggregator import discretize
from awllm.analysis.classifier import find_segments, segments_to_blocks
from awllm.analysis.io import (
    fetch_aw_events,
    load_events_for_discretization,
    emit_blocks_to_aw,
)

LOG = logging.getLogger("aw-llm-worker")


class AWSink:
    """ActivityWatch event sink."""

    def __init__(self, testing: bool, bucket_suffix: str):
        self.client = ActivityWatchClient("aw-watcher-screenshot-llm", testing=testing)
        self.client.wait_for_start()
        self.client.connect()
        self.bucket = f"{self.client.client_name}_{self.client.client_hostname}"
        if bucket_suffix:
            self.bucket = (
                f"aw-watcher-screenshot-{bucket_suffix}_{self.client.client_hostname}"
            )
        self.eventtype = "app.screenshot.label"
        self.client.create_bucket(self.bucket, self.eventtype, queued=False)
        LOG.info("Created bucket: %s", self.bucket)

    def push(self, ts: datetime, payload: Dict[str, Any]):
        """Push event to ActivityWatch."""
        ev = Event(timestamp=ts, duration=timedelta(seconds=0), data=payload)
        self.client.insert_event(self.bucket, ev)


def process_screenshots(
    spool_dir: str,
    state: State,
    sink: AWSink,
    model: Any,
    ctx: Dict[str, Any],
    ctx_id: str,
    model_path: str,
    mmproj_path: str,
    temp: float,
    max_per_iter: int,
) -> int:
    """Process screenshot spool files. Returns number of screenshots processed."""
    files = sorted(glob.glob(os.path.join(spool_dir, "*.json")), key=os.path.getmtime)
    processed = 0

    for fp in files:
        try:
            rec = read_json(fp)
        except Exception as e:
            LOG.debug("Skip unreadable %s: %r", fp, e)
            continue

        if state.seen(rec):
            continue

        # Basic validation
        path = rec.get("path")
        if not path or not os.path.exists(path):
            state.mark(rec)  # mark to avoid busy-looping dead entries
            continue

        ts = rec.get("ts") or datetime.now(tz=timezone.utc).isoformat()
        ts_dt = iso_to_dt(ts)
        meta = {
            "title": rec.get("title"),
            "app": rec.get("app"),
            "pid": rec.get("pid"),
            "win_id": rec.get("win_id"),
            "bbox": rec.get("bbox"),
            "path": path,
        }

        # LLM classify
        try:
            label = model.classify(path, meta, ctx)
            LOG.debug("LLM output: %s", label)
        except Exception as e:
            LOG.error("LLM classify failed for %s: %r", path, e)
            state.mark(rec)
            continue

        # Optional keyword routing override/boost
        label = route_project_keywords(meta, ctx, label)

        # Build AW payload
        payload = {
            "src": {
                "path": os.path.abspath(path),
                "sha256": rec.get("sha256"),
                "app": rec.get("app"),
                "title": rec.get("title"),
                "pid": rec.get("pid"),
                "win_id": rec.get("win_id"),
                "bbox": rec.get("bbox"),
                "ts": rec.get("ts"),
            },
            "label": label,
            "llm": {
                "model": os.path.basename(model_path),
                "mmproj": os.path.basename(mmproj_path),
                "temperature": temp,
                "prompt_rev": PROMPT_REV,
            },
            "context": {
                "context_id": ctx_id,
                "matched_keywords": label.get("project", {}).get("reason", ""),
            },
        }

        # Push to ActivityWatch
        try:
            sink.push(ts_dt, payload)
        except Exception as e:
            LOG.error("Failed to push AW event: %r", e)
            continue

        # Mark done
        state.mark(rec)
        processed += 1
        LOG.info(
            "Labeled: %s | act=%s proj=%s conf=%.2f",
            os.path.basename(path),
            label.get("coarse_activity"),
            (label.get("project") or {}).get("name"),
            float(label.get("confidence", 0.0)),
        )

        if processed >= max_per_iter:
            break

    return processed


def run_summarization(
    aw_host: str,
    source_buckets: List[str],
    lookback_hours: float,
    topics: Dict[str, Dict],
    sketch_dim: int,
    dt_seconds: float,
    hostname: str,
):
    """Run time-block summarization and classification."""
    LOG.info("Starting summarization run...")
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=lookback_hours)

    # Fetch and discretize data
    raw_events = fetch_aw_events(source_buckets, start, end, host=aw_host)
    events = load_events_for_discretization(raw_events)

    if not events:
        LOG.info("No events to summarize.")
        return

    sketch = HashSketch(dim=sketch_dim, ngram=(1, 2))
    M, frame_starts = discretize(events, start, end, dt_seconds, sketch)

    if M.shape[0] == 0:
        LOG.info("No data matrix generated.")
        return

    all_blocks = []

    # Run classification for each topic
    for label, config in topics.items():
        LOG.info(f"  Classifying topic: {label}")
        kernel = make_kernel_from_phrases(
            config["phrases"], sketch, config["kernel_width"]
        )

        # Convolve to get scores
        scores = conv1d_text(M, kernel, stride=1, padding="same")

        # Find segments above threshold
        segments = find_segments(
            scores,
            threshold=config["threshold"],
            min_duration_s=config["min_duration_s"],
            dt=dt_seconds,
            merge_gap_s=config.get("merge_gap_s", 60.0),
        )

        # Convert to AW-compatible blocks
        blocks = segments_to_blocks(segments, frame_starts, label, scores)
        all_blocks.extend(blocks)
        LOG.info(f"    Found {len(blocks)} blocks for '{label}'.")

    # Emit all found blocks to ActivityWatch
    if all_blocks:
        bucket_id = f"aw-llm-blocks_{hostname}"
        emit_blocks_to_aw(
            all_blocks, host=aw_host, bucket_id=bucket_id, client_name="aw-llm-worker"
        )
        LOG.info(f"Emitted {len(blocks)} blocks total.")
    else:
        LOG.info("No blocks found to emit.")


@click.command()
@click.option(
    "--mode",
    type=click.Choice(["screenshots", "summarization", "both"], case_sensitive=False),
    default="both",
    show_default=True,
    help="Operating mode: screenshots only, summarization only, or both.",
)
@click.option(
    "--spool-dir",
    type=click.Path(file_okay=False),
    help="Directory containing spool JSON from the screenshot watcher (required for screenshot mode).",
)
@click.option(
    "--model",
    "model_path",
    type=click.Path(exists=True),
    default="/Users/filip/.cache/lm-studio/models/unsloth/Qwen2.5-VL-7B-Instruct-GGUF/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
    show_default=True,
    help="Path to model (for screenshot mode).",
)
@click.option(
    "--mmproj",
    "mmproj_path",
    type=click.Path(exists=True),
    default="/Users/filip/.cache/lm-studio/models/unsloth/Qwen2.5-VL-7B-Instruct-GGUF/mmproj-F16.gguf",
    show_default=True,
    help="Path to mmproj (for screenshot mode).",
)
@click.option(
    "--context",
    "context_file",
    type=click.Path(exists=True),
    help="YAML/JSON file with user/project context.",
)
@click.option(
    "--screenshot-poll",
    type=float,
    default=0.8,
    show_default=True,
    help="Polling cadence for screenshots (s).",
)
@click.option(
    "--summarization-interval",
    type=float,
    default=6.0,
    show_default=True,
    help="Hours between summarization runs.",
)
@click.option(
    "--max-per-iter",
    type=int,
    default=2,
    show_default=True,
    help="Process up to N screenshots per tick.",
)
@click.option("--n-ctx", type=int, default=4096, show_default=True)
@click.option(
    "--n-gpu-layers",
    type=int,
    default=-1,
    show_default=True,
    help="-1 = all layers on GPU (Metal).",
)
@click.option("--threads", type=int, default=0, show_default=True, help="0 = auto")
@click.option("--temp", type=float, default=0.2, show_default=True)
@click.option("--max-new", type=int, default=256, show_default=True)
@click.option(
    "--bucket-suffix",
    default="llm",
    show_default=True,
    help="Suffix for AW bucket name.",
)
@click.option(
    "--testing", is_flag=True, default=False, help="ActivityWatch client testing mode."
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARN", "ERROR"], case_sensitive=False),
)
@click.option(
    "--use-cli",
    is_flag=True,
    default=False,
    help="Use llama-mtmd-cli instead of Python library (for screenshots).",
)
@click.option(
    "--cli-path",
    type=click.Path(exists=True),
    help="Path to llama-mtmd-cli binary (auto-detected if not specified).",
)
@click.option(
    "--aw-host",
    default="http://127.0.0.1:5600",
    show_default=True,
    help="ActivityWatch server URL.",
)
@click.option(
    "--source-buckets",
    multiple=True,
    help="AW buckets to fetch for summarization (can specify multiple times).",
)
@click.option(
    "--lookback-hours",
    type=float,
    default=8.0,
    show_default=True,
    help="How far back to analyze for summarization.",
)
@click.option(
    "--sketch-dim",
    type=int,
    default=256,
    show_default=True,
    help="Dimensionality of text vectors for summarization.",
)
@click.option(
    "--dt-seconds",
    type=float,
    default=15.0,
    show_default=True,
    help="Time resolution for summarization (seconds).",
)
def main(
    mode,
    spool_dir,
    model_path,
    mmproj_path,
    context_file,
    screenshot_poll,
    summarization_interval,
    max_per_iter,
    n_ctx,
    n_gpu_layers,
    threads,
    temp,
    max_new,
    bucket_suffix,
    testing,
    log_level,
    use_cli,
    cli_path,
    aw_host,
    source_buckets,
    lookback_hours,
    sketch_dim,
    dt_seconds,
):
    """ActivityWatch LLM Worker - Dual-mode: Screenshot classification + Time summarization."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    mode = mode.lower()
    do_screenshots = mode in ["screenshots", "both"]
    do_summarization = mode in ["summarization", "both"]

    # Validate requirements
    if do_screenshots and not spool_dir:
        raise click.UsageError(
            "--spool-dir is required when screenshots mode is enabled"
        )

    LOG.info(f"Starting aw-llm-worker | mode={mode}")

    # Load context
    ctx = load_yaml_or_json(context_file)
    ctx_id = sha1(json.dumps(ctx, sort_keys=True))
    LOG.info("Loaded context (id=%s)", ctx_id[:8])

    # Initialize screenshot components if needed
    state = None
    sink = None
    model = None

    if do_screenshots:
        LOG.info(f"Screenshot mode enabled | spool={spool_dir}")
        state = State(os.path.join(spool_dir, ".llm_state.json"))
        sink = AWSink(testing=testing, bucket_suffix=bucket_suffix)

    # Get hostname for bucket naming
    from aw_client import ActivityWatchClient

    temp_client = ActivityWatchClient("temp", testing=testing)
    hostname = temp_client.client_hostname

    # Setup summarization topics from context
    topics = {}
    if do_summarization:
        LOG.info("Summarization mode enabled")
        # Default topics if not in context
        topics = ctx.get(
            "topics",
            {
                "Coding": {
                    "phrases": [
                        "coding",
                        "python",
                        "vscode",
                        "terminal",
                        "git",
                        "code",
                        "develop",
                    ],
                    "kernel_width": 15,
                    "threshold": 0.3,
                    "min_duration_s": 120,
                    "merge_gap_s": 60,
                },
                "Writing": {
                    "phrases": [
                        "writing",
                        "docs",
                        "obsidian",
                        "notes",
                        "readme",
                        "document",
                    ],
                    "kernel_width": 10,
                    "threshold": 0.4,
                    "min_duration_s": 180,
                    "merge_gap_s": 60,
                },
            },
        )

        # Default source buckets if not specified
        if not source_buckets:
            source_buckets = [
                f"aw-watcher-window_{hostname}",
                f"aw-watcher-afk_{hostname}",
            ]
        LOG.info(f"Summarization buckets: {list(source_buckets)}")

    # Timers
    last_summarization = datetime.now() - timedelta(
        hours=summarization_interval
    )  # Run immediately

    # Main loop
    while True:
        try:
            # --- Screenshot processing ---
            if do_screenshots:
                # Load model only when needed
                if model is None:
                    LOG.info("Loading vision model for screenshot batch...")
                    if use_cli:
                        LOG.info("Using CLI backend (llama-mtmd-cli)")
                        model = QwenVLCLI(
                            model_path=model_path,
                            mmproj_path=mmproj_path,
                            n_ctx=n_ctx,
                            n_gpu_layers=n_gpu_layers,
                            temp=temp,
                            max_tokens=max_new,
                            threads=threads,
                            verbose=False,
                            cli_path=cli_path,
                        )
                    else:
                        LOG.info("Using Python library backend")
                        model = QwenVLPython(
                            model_path=model_path,
                            mmproj_path=mmproj_path,
                            n_ctx=n_ctx,
                            n_gpu_layers=n_gpu_layers,
                            temp=temp,
                            max_tokens=max_new,
                            threads=threads,
                            verbose=log_level.upper() == "DEBUG",
                        )

                processed = process_screenshots(
                    spool_dir,
                    state,
                    sink,
                    model,
                    ctx,
                    ctx_id,
                    model_path,
                    mmproj_path,
                    temp,
                    max_per_iter,
                )

                if processed > 0:
                    state.save()
                    LOG.debug("Saved screenshot state to disk.")

                # Unload model if no more work
                if processed == 0:
                    if model is not None:
                        LOG.info("No screenshots to process, unloading model...")
                        del model
                        model = None
                        import gc

                        gc.collect()

                time.sleep(screenshot_poll)

            # --- Summarization ---
            if do_summarization:
                now = datetime.now()
                elapsed_hours = (now - last_summarization).total_seconds() / 3600

                if elapsed_hours >= summarization_interval:
                    # Unload screenshot model if loaded
                    if model is not None:
                        LOG.info("Unloading screenshot model for summarization...")
                        del model
                        model = None
                        import gc

                        gc.collect()

                    run_summarization(
                        aw_host=aw_host,
                        source_buckets=list(source_buckets),
                        lookback_hours=lookback_hours,
                        topics=topics,
                        sketch_dim=sketch_dim,
                        dt_seconds=dt_seconds,
                        hostname=hostname,
                    )
                    last_summarization = now

            # If only summarization mode and not time yet, sleep
            if not do_screenshots and do_summarization:
                time.sleep(60)  # Check every minute

        except KeyboardInterrupt:
            LOG.info("Exiting.")
            break
        except Exception as e:
            LOG.error(f"Error in main loop: {e}", exc_info=True)
            time.sleep(5)


if __name__ == "__main__":
    main()
