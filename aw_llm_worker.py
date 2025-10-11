#!/usr/bin/env python3
"""Main CLI entry point for aw-llm-worker."""

import os
import json
import time
import glob
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any

import click
from aw_client import ActivityWatchClient
from aw_core import Event

from awllm.utils.helpers import load_yaml_or_json, sha1, read_json, iso_to_dt
from awllm.utils.state import State
from awllm.models import QwenVLPython, QwenVLCLI
from awllm.prompt import PROMPT_REV, route_project_keywords

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


@click.command()
@click.option(
    "--spool-dir",
    type=click.Path(file_okay=False),
    required=True,
    help="Directory containing spool JSON from the screenshot watcher.",
)
@click.option(
    "--model",
    "model_path",
    type=click.Path(exists=True),
    required=False,
    default="/Users/filip/.cache/lm-studio/models/unsloth/Qwen2.5-VL-7B-Instruct-GGUF/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
    show_default=True,
)
@click.option(
    "--mmproj",
    "mmproj_path",
    type=click.Path(exists=True),
    required=False,
    default="/Users/filip/.cache/lm-studio/models/unsloth/Qwen2.5-VL-7B-Instruct-GGUF/mmproj-F16.gguf",
    show_default=True,
)
@click.option(
    "--context",
    "context_file",
    type=click.Path(exists=True),
    help="YAML/JSON file with user/project context.",
)
@click.option(
    "--poll", type=float, default=0.8, show_default=True, help="Polling cadence (s)."
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
    help="Use llama-mtmd-cli instead of Python library (3x faster!).",
)
@click.option(
    "--cli-path",
    type=click.Path(exists=True),
    help="Path to llama-mtmd-cli binary (auto-detected if not specified).",
)
def main(
    spool_dir,
    model_path,
    mmproj_path,
    context_file,
    poll,
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
):
    """ActivityWatch LLM Worker - Screenshot classification daemon."""
    logging.basicConfig(level=getattr(logging, log_level.upper()))
    LOG.info("Starting aw-llm-worker | spool=%s", spool_dir)

    ctx = load_yaml_or_json(context_file)
    ctx_id = sha1(json.dumps(ctx, sort_keys=True))
    LOG.info("Loaded context (id=%s)", ctx_id[:8])

    state = State(os.path.join(spool_dir, ".llm_state.json"))
    sink = AWSink(testing=testing, bucket_suffix=bucket_suffix)

    # Choose inference backend
    if use_cli:
        LOG.info("Using CLI backend (llama-mtmd-cli) for 3x faster performance")
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
        LOG.info("Using Python library backend (llama-cpp-python)")
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

    while True:
        # Find candidate spool records
        files = sorted(
            glob.glob(os.path.join(spool_dir, "*.json")), key=os.path.getmtime
        )
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

        # Heartbeat
        time.sleep(poll)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOG.info("Exiting.")
