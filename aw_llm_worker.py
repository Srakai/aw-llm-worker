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
from awllm.analysis.textconv import HashSketch
from awllm.analysis.aggregator import discretize
from awllm.analysis.classifier import (
    create_time_windows,
    classify_windows_with_llm,
    merge_classified_windows,
)
from awllm.analysis.io import (
    fetch_aw_events,
    load_events_for_discretization,
    emit_blocks_to_aw,
    emit_raw_windows_to_aw,
    emit_project_blocks_to_aw,
    separate_events_by_source,
    find_screenshots_for_window,
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
    model_path: str,
    mmproj_path: str,
    n_ctx: int,
    n_gpu_layers: int,
    temp: float,
    max_new: int,
    threads: int,
    use_cli: bool,
    cli_path: Optional[str],
    log_level: str,
    ctx: Dict[str, Any],
    spool_dir: Optional[str] = None,
    force_screenshot_enrichment: bool = False,
):
    """Run time-block summarization and classification using an LLM."""
    LOG.info("Starting summarization run...")
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=lookback_hours)

    # 1. Fetch events from all buckets including vscode and screenshots
    # Automatically add vscode and screenshot buckets if not specified
    all_buckets = list(source_buckets)

    # Add VS Code bucket
    vscode_bucket = f"aw-watcher-vscode_{hostname}"
    if vscode_bucket not in all_buckets:
        all_buckets.append(vscode_bucket)

    # Add screenshot bucket
    screenshot_bucket = f"aw-watcher-screenshot-llm_{hostname}"
    if screenshot_bucket not in all_buckets:
        all_buckets.append(screenshot_bucket)

    LOG.info(f"Fetching from buckets: {all_buckets}")
    raw_events = fetch_aw_events(all_buckets, start, end, host=aw_host)

    # Separate events by source
    events, vscode_events, screenshot_summaries = separate_events_by_source(raw_events)

    if not events:
        LOG.info("No events to summarize.")
        return

    LOG.info(f"Loaded {len(events)} regular events")
    if vscode_events:
        LOG.info(f"Found {len(vscode_events)} VS Code events")
    if screenshot_summaries:
        LOG.info(f"Found {len(screenshot_summaries)} screenshot summaries")

    sketch = HashSketch(dim=sketch_dim, ngram=(1, 2))
    M, frame_starts, frame_texts = discretize(events, start, end, dt_seconds, sketch)

    if len(frame_starts) == 0:
        LOG.info("No data to process after discretization.")
        return

    # 2. Create sliding windows of text with enriched context
    # Get windowing parameters from topics config or use defaults
    window_duration_m = topics.get("_config", {}).get("window_duration_m", 30)
    window_step_m = topics.get("_config", {}).get("window_step_m", 5)

    window_size_steps = int((window_duration_m * 60) / dt_seconds)
    step_size_steps = int((window_step_m * 60) / dt_seconds)

    LOG.info(
        f"Creating windows: {window_duration_m}min window, {window_step_m}min step "
        f"({window_size_steps} steps x {step_size_steps} step)"
    )

    windows = create_time_windows(
        frame_texts,
        frame_starts,
        window_size_steps,
        step_size_steps,
        vscode_events=vscode_events,
        screenshot_summaries=screenshot_summaries,
    )

    if not windows:
        LOG.info("No windows created from discretized data.")
        return

    LOG.info(f"Created {len(windows)} time windows to classify.")
    if windows:
        # Log enrichment info
        sample_window = windows[0]
        if sample_window.get("num_vscode"):
            LOG.info(f"Windows enriched with VS Code events")
        if sample_window.get("num_screenshots"):
            LOG.info(f"Windows enriched with screenshot summaries")

    # 3. Load LLM model for text classification
    LOG.info("Loading LLM model for text classification...")
    try:
        if use_cli:
            LOG.info("Using CLI backend for summarization")
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
            LOG.info("Using Python library backend for summarization")
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
    except Exception as e:
        LOG.error(f"Failed to load model for summarization: {e}")
        return

    # 4. Classify windows with LLM
    # Extract topic names (exclude _config if present)
    topic_names = [k for k in topics.keys() if not k.startswith("_")]
    LOG.info(f"Classifying windows into topics: {topic_names}")

    # Extract projects from context for LLM
    projects = ctx.get("projects", [])
    if projects:
        LOG.info(f"Using {len(projects)} project definitions for classification")

    try:
        classified_windows = classify_windows_with_llm(
            windows, model, topic_names, projects
        )

        # Apply keyword-based project routing to boost/override LLM results
        for window in classified_windows:
            # Build a "meta" dict from window content for keyword matching
            content = window.get("content", "")
            meta = {
                "title": content[:200],  # Use first part of content
                "app": "",  # We don't have individual app info at window level
            }

            # Apply keyword routing similar to screenshots
            # Create a temporary label dict to pass through route_project_keywords
            temp_label = {
                "project": {
                    "name": window.get("project"),
                    "confidence": 0.5 if window.get("project") else 0.0,
                    "reason": "LLM classification",
                }
            }

            # Route and update
            routed_label = route_project_keywords(meta, ctx, temp_label)

            # Update window with routed project info
            if routed_label.get("project"):
                window["project"] = routed_label["project"].get("name")
                # Optionally log if keywords boosted the project
                if routed_label["project"].get("confidence", 0) > 0.6:
                    LOG.debug(f"Keyword boost: {routed_label['project'].get('reason')}")

        # NEW: Process screenshot refinement requests
        # Use forced mode or LLM decision
        if force_screenshot_enrichment:
            screenshot_requests = classified_windows
            LOG.info(
                f"Force screenshot enrichment enabled - processing all {len(screenshot_requests)} windows"
            )
        else:
            screenshot_requests = [
                w for w in classified_windows if w.get("request_screenshot_analysis")
            ]

        if screenshot_requests:
            if not force_screenshot_enrichment:
                LOG.info(
                    f"Found {len(screenshot_requests)} windows requesting screenshot analysis"
                )

            # Use the raw screenshot bucket (aw-watcher-screenshot, not -llm)
            screenshot_bucket = f"aw-watcher-screenshot_{hostname}"

            for window in screenshot_requests:
                try:
                    # Find screenshots for this window from ActivityWatch bucket
                    screenshots = find_screenshots_for_window(
                        host=aw_host,
                        bucket_id=screenshot_bucket,
                        window_start=window["start"],
                        window_end=window["end"],
                        max_screenshots=3,  # Default: 1 screenshot for stability
                    )

                    if screenshots:
                        LOG.info(
                            f"Refining window {window['start'].strftime('%H:%M')}-{window['end'].strftime('%H:%M')} "
                            f"with {len(screenshots)} screenshot(s)"
                        )

                        # Build initial classification from window
                        initial_classification = {
                            "label": window.get("label", "misc"),
                            "confidence": window.get("confidence", 0.0),
                            "project": window.get("project"),
                            "activity_description": window.get(
                                "activity_description", ""
                            ),
                        }

                        # Refine with screenshots
                        refined = model.refine_with_screenshots(
                            initial_classification,
                            window.get("content", ""),
                            screenshots,
                            topic_names,
                            projects,
                        )

                        # Update window with refined data
                        window["label"] = refined.get("label", window["label"])
                        window["confidence"] = refined.get(
                            "confidence", window["confidence"]
                        )
                        window["project"] = refined.get("project", window["project"])
                        window["activity_description"] = refined.get(
                            "activity_description", window["activity_description"]
                        )
                        window["enriched_with_screenshots"] = True
                        window["num_screenshots_analyzed"] = refined.get(
                            "num_screenshots_analyzed", len(screenshots)
                        )
                        window["screenshot_insights"] = refined.get(
                            "screenshot_insights", []
                        )

                        LOG.info(
                            f"Refinement complete: confidence {initial_classification['confidence']:.2f} -> {refined.get('confidence', 0.0):.2f}"
                        )
                    else:
                        LOG.debug(
                            f"No screenshots found for window {window['start'].strftime('%H:%M')}-{window['end'].strftime('%H:%M')}"
                        )

                except Exception as e:
                    LOG.error(f"Failed to refine window with screenshots: {e}")
                    continue

    except Exception as e:
        LOG.error(f"Failed to classify windows: {e}")
        return

    # Build model info for metadata
    model_info = {
        "model": os.path.basename(model_path),
        "backend": "cli" if use_cli else "python",
        "temperature": temp,
        "max_tokens": max_new,
        "method": "text_classification",
    }

    # NEW: Emit raw classified windows to separate bucket
    if classified_windows:
        raw_window_bucket = f"aw-llm-windows_{hostname}"
        LOG.info(
            f"Emitting {len(classified_windows)} raw windows to {raw_window_bucket}"
        )

        emit_raw_windows_to_aw(
            classified_windows,
            host=aw_host,
            bucket_id=raw_window_bucket,
            client_name="aw-llm-worker",
            model_info=model_info,
        )

    # Clean up model before merging (free memory)
    del model
    import gc

    gc.collect()

    # 5. Merge classified windows into final blocks
    merge_gap_s = topics.get("_config", {}).get("merge_gap_s", 300.0)
    min_confidence = topics.get("_config", {}).get("min_confidence", 0.3)

    LOG.info(
        f"Merging windows with merge_gap={merge_gap_s}s, min_confidence={min_confidence}"
    )

    final_blocks = merge_classified_windows(
        classified_windows, merge_gap_s=merge_gap_s, min_confidence=min_confidence
    )

    # 6. Emit blocks to ActivityWatch
    if final_blocks:
        # Build analysis metadata
        analysis_metadata = {
            "window_duration_m": window_duration_m,
            "window_step_m": window_step_m,
            "merge_gap_s": merge_gap_s,
            "min_confidence": min_confidence,
            "lookback_hours": lookback_hours,
            "sketch_dim": sketch_dim,
            "dt_seconds": dt_seconds,
            "num_source_buckets": len(source_buckets),
            "topics": topic_names,
        }

        # Emit to main blocks bucket (by category)
        bucket_id = f"aw-llm-blocks_{hostname}"
        emit_blocks_to_aw(
            final_blocks,
            host=aw_host,
            bucket_id=bucket_id,
            client_name="aw-llm-worker",
            model_info=model_info,
            analysis_metadata=analysis_metadata,
        )
        LOG.info(f"Emitted {len(final_blocks)} blocks to {bucket_id}")

        # NEW: Emit to per-project buckets
        project_bucket_prefix = f"aw-llm-project_{hostname}"
        emit_project_blocks_to_aw(
            final_blocks,
            host=aw_host,
            bucket_prefix=project_bucket_prefix,
            client_name="aw-llm-worker",
            model_info=model_info,
            analysis_metadata=analysis_metadata,
        )

        # Log summary
        enriched_count = sum(
            1 for b in final_blocks if b.get("enriched_with_screenshots")
        )
        if enriched_count:
            LOG.info(
                f"{enriched_count}/{len(final_blocks)} blocks were enriched with screenshots"
            )
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
@click.option(
    "--debug-format",
    is_flag=True,
    default=False,
    help="Debug mode: print formatted event text and exit (no LLM).",
)
@click.option(
    "--force-screenshot-enrichment",
    is_flag=True,
    default=False,
    help="Always enrich all windows with screenshots, bypassing LLM decision.",
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
    debug_format,
    force_screenshot_enrichment,
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

    # DEBUG MODE: Just format and print
    if debug_format:
        LOG.info("=== DEBUG FORMAT MODE ===")
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=lookback_hours)

        # Get hostname
        from aw_client import ActivityWatchClient

        temp_client = ActivityWatchClient("temp", testing=testing)
        hostname = temp_client.client_hostname

        # Default source buckets if not specified
        if not source_buckets:
            source_buckets = [
                f"aw-watcher-window_{hostname}",
                f"aw-watcher-afk_{hostname}",
            ]

        LOG.info(f"Fetching from buckets: {list(source_buckets)}")
        LOG.info(
            f"Time range: {start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%Y-%m-%d %H:%M')}"
        )

        # Fetch and process
        raw_events = fetch_aw_events(source_buckets, start, end, host=aw_host)
        events = load_events_for_discretization(raw_events)
        LOG.info(f"Loaded {len(events)} raw events")

        if not events:
            LOG.info("No events found.")
            return

        sketch = HashSketch(dim=sketch_dim, ngram=(1, 2))
        M, frame_starts, frame_texts = discretize(
            events, start, end, dt_seconds, sketch
        )

        LOG.info(f"Created {len(frame_texts)} frames")

        # Print sample frames
        print("\n" + "=" * 80)
        print("SAMPLE FORMATTED FRAMES (first 10 non-empty):")
        print("=" * 80)

        count = 0
        for i, (frame_start, text) in enumerate(zip(frame_starts, frame_texts)):
            if text.strip() and count < 10:
                print(f"\nFrame {i} [{frame_start.strftime('%Y-%m-%d %H:%M:%S')}]:")
                print(f"  Length: {len(text)} chars")
                print(f"  Content: {text[:500]}{'...' if len(text) > 500 else ''}")
                count += 1

        print("\n" + "=" * 80)
        print("FRAME STATISTICS:")
        print("=" * 80)
        non_empty = [t for t in frame_texts if t.strip()]
        if non_empty:
            lengths = [len(t) for t in non_empty]
            print(f"Total frames: {len(frame_texts)}")
            print(f"Non-empty frames: {len(non_empty)}")
            print(f"Avg length: {sum(lengths)/len(lengths):.1f} chars")
            print(f"Min length: {min(lengths)} chars")
            print(f"Max length: {max(lengths)} chars")
        else:
            print("No non-empty frames!")

        print("\n" + "=" * 80)
        print("TESTING WINDOW CREATION:")
        print("=" * 80)

        # Test window creation with deduplication
        topics = ctx.get("topics", {})
        window_duration_m = topics.get("_config", {}).get("window_duration_m", 30)
        window_step_m = topics.get("_config", {}).get("window_step_m", 5)

        window_size_steps = int((window_duration_m * 60) / dt_seconds)
        step_size_steps = int((window_step_m * 60) / dt_seconds)

        print(f"Window size: {window_duration_m}min ({window_size_steps} steps)")
        print(f"Step size: {window_step_m}min ({step_size_steps} steps)")

        windows = create_time_windows(
            frame_texts, frame_starts, window_size_steps, step_size_steps
        )

        print(f"Created {len(windows)} windows")

        # Show first 5 windows with their deduplicated content
        print("\n" + "=" * 80)
        print("SAMPLE WINDOWS (first 5):")
        print("=" * 80)

        for i, window in enumerate(windows[:5]):
            print(f"\nWindow {i+1}:")
            print(
                f"  Time: {window['start'].strftime('%H:%M')} - {window['end'].strftime('%H:%M')}"
            )
            print(f"  Content length: {len(window['content'])} chars")
            print(f"  Content preview:")
            # Show first 400 chars with better formatting
            preview = window["content"][:400]
            if len(window["content"]) > 400:
                preview += "..."
            # Split into multiple lines for readability
            for line in [preview[i : i + 80] for i in range(0, len(preview), 80)]:
                print(f"    {line}")

        print("\n" + "=" * 80)
        return

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
                        model_path=model_path,
                        mmproj_path=mmproj_path,
                        n_ctx=n_ctx,
                        n_gpu_layers=n_gpu_layers,
                        temp=temp,
                        max_new=max_new,
                        threads=threads,
                        use_cli=use_cli,
                        cli_path=cli_path,
                        log_level=log_level,
                        ctx=ctx,
                        spool_dir=spool_dir,
                        force_screenshot_enrichment=force_screenshot_enrichment,
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
