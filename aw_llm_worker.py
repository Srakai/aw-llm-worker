#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aw-llm-worker:
- Polls a screenshot spool dir (JSON records written by your watcher).
- For each new screenshot, runs Qwen2.5-VL-7B-Instruct (GGUF) via llama-cpp (Metal).
- Pushes a classification event to ActivityWatch.

Bucket:
    aw-watcher-screenshot-llm_{hostname}
Event type:
    app.screenshot.label
Event schema (data):
{
  "src": { "path", "sha256", "app", "title", "pid", "win_id", "bbox" },
  "label": {
    "coarse_activity",    # enum-ish
    "app_guess",
    "summary",
    "tags",               # list[str]
    "project": {"name", "confidence", "reason"},
    "confidence"          # 0..1
  },
  "llm": { "model", "mmproj", "temperature", "prompt_rev" },
  "context": {"matched_keywords": ["..."], "context_id": "sha1(...)"}
}

Assumes spool record format produced by the earlier watcher:
{
  "ts": "2025-10-09T12-34-56.789Z",
  "title": "...",
  "app": "...",
  "pid": 123,
  "win_id": 6291463,
  "bbox": [L,T,R,B] | null,
  "path": "/abs/file.png",
  "sha256": "..."
}
"""
import os
import re
import io
import sys
import json
import time
import glob
import uuid
import yaml
import math
import base64
import queue
import hashlib
import socket
import logging
from typing import Any, Dict, Optional, List
from datetime import datetime, timezone, timedelta

import click
from aw_client import ActivityWatchClient
from aw_core import Event
from llama_cpp import Llama

LOG = logging.getLogger("aw-llm-worker")


# --------- utils
def iso_to_dt(s: str) -> datetime:
    # tolerant ISO8601 parser; supports trailing 'Z' and hyphen-separated time
    s = s.strip().replace("Z", "+00:00")

    # Handle format like "2025-10-10T22-34-15.007+00-00"
    # Convert hyphens in time portion to colons and in timezone to colon
    if "T" in s:
        date_part, time_part = s.split("T", 1)
        # Replace first two hyphens in time part with colons (HH-MM-SS -> HH:MM:SS)
        # But preserve the timezone offset
        if "+" in time_part:
            time_only, tz_part = time_part.rsplit("+", 1)
            time_only = time_only.replace("-", ":", 2)
            tz_part = tz_part.replace("-", ":", 1)
            s = f"{date_part}T{time_only}+{tz_part}"
        elif time_part.count("-") > 0 and "-" in time_part.split(".")[0]:
            # No timezone, just fix time portion
            time_only = time_part.replace("-", ":", 2)
            s = f"{date_part}T{time_only}"

    return datetime.fromisoformat(s)


def to_file_url(path: str) -> str:
    # llama.cpp accepts file:// or file:
    p = os.path.abspath(path)
    return "file://" + p


def load_yaml_or_json(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    try:
        if path.endswith(".json"):
            return json.loads(txt)
        return yaml.safe_load(txt) or {}
    except Exception as e:
        LOG.error("Failed to parse context file %s: %r", path, e)
        return {}


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: str, obj: Any) -> None:
    tmp = f"{path}.tmp.{uuid.uuid4().hex}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def ensure_dir(d: str) -> str:
    os.makedirs(d, exist_ok=True)
    return d


# --------- state (dedupe)
class State:
    def __init__(self, path: str):
        self.path = path
        self.data = {"seen": {}}  # key -> ts
        if os.path.exists(path):
            try:
                self.data = read_json(path)
            except Exception:
                pass

    def key(self, rec: Dict[str, Any]) -> str:
        return rec.get("sha256") or ("path:" + rec.get("path", ""))

    def seen(self, rec: Dict[str, Any]) -> bool:
        k = self.key(rec)
        return k in self.data["seen"]

    def mark(self, rec: Dict[str, Any]):
        k = self.key(rec)
        self.data["seen"][k] = rec.get("ts")
        write_json_atomic(self.path, self.data)


# --------- prompt
PROMPT_REV = "qwen2.5vl-screenshot-labeler/v1"

COARSE_ENUM = [
    "coding",
    "terminal",
    "writing",
    "reading",
    "browsing",
    "chat",
    "email",
    "meeting",
    "notes",
    "spreadsheet",
    "slides",
    "design",
    "file-manager",
    "media",
    "settings",
    "misc",
]

SYSTEM_PROMPT = f"""
You are a strict JSON classifier for screenshots. No preamble, no markdown.
Return a single JSON object. If uncertain, choose "misc" with lower confidence.

In tags place any visible keywords, project names etc.
JSON schema:
{{
  "what_do_i_see_here": string,    # <= 20 words explaining your choice
  "what_user_might_be_doing": string, # <= 20 words explaining your choice
  "what_user_wants_me_to_write_about_this": string, # <= 20 words explaining your choice
  "coarse_activity": one of {COARSE_ENUM},
  "app_guess": string,             # application name (guess if needed)
  "summary": string,               # <= 40 words describing the task
  "tags": [string, ...],           # 1..6 short tokens
  "project": {{
    "name": string|null,           # map to user projects if likely, else null
    "confidence": number,          # 0..1
    "reason": string               # <= 20 words
  }},
  "confidence": number             # 0..1 for the overall label
}}
Respond with JSON only.
""".strip()


def build_user_prompt(
    meta: Dict[str, Any], ctx: Dict[str, Any]
) -> List[Dict[str, Any]]:
    # Compose OpenAI-style content array with image + text
    projects = ctx.get("projects", [])
    routing = ctx.get("routing", {})
    role = ctx.get("role", "")
    org = ctx.get("org", "")

    # Minimal context string
    proj_lines = []
    for p in projects:
        ks = ", ".join(p.get("keywords", [])[:12])
        proj_lines.append(f"- {p.get('name')}: {ks}")
    proj_block = "\n".join(proj_lines) if proj_lines else "(none)"

    meta_text = (
        # f"Window title: {meta.get('title')}\n"
        # f"Source app: {meta.get('app')}\n"
        # f"PID: {meta.get('pid')}, WinID: {meta.get('win_id')}\n"
        # f"User role: {role}\n"
        # f"Org: {org}\n"
        # f"Project hints:\n{proj_block}\n"
        # f"Routing: prefer_exact_match={routing.get('prefer_exact_match', False)}\n"
        "Classify what the user is doing now. If matches any project by keywords, set project.name accordingly."
    )

    return [
        {"type": "image_url", "image_url": {"url": to_file_url(meta["path"])}},
        {"type": "text", "text": meta_text},
    ]


def extract_json(txt: str) -> Dict[str, Any]:
    # take first {...} block
    m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in LLM output")
    s = m.group(0)
    return json.loads(s)


# --------- simple keyword router for projects (pre/post LLM)
def route_project_keywords(
    meta: Dict[str, Any], ctx: Dict[str, Any], label: Dict[str, Any]
) -> Dict[str, Any]:
    projects = ctx.get("projects", [])
    prefer_exact = ctx.get("routing", {}).get("prefer_exact_match", False)

    hay = " ".join([str(meta.get("title") or ""), str(meta.get("app") or "")]).lower()
    best = None
    for p in projects:
        name = p.get("name")
        kws = [str(k).lower() for k in p.get("keywords", [])]
        if not kws:
            continue
        hits = [k for k in kws if k in hay]
        if hits and (best is None or len(hits) > len(best["hits"])):
            best = {"name": name, "hits": hits}

    if best:
        # boost / override if LLM was unsure or routing demands exact match
        prev = label.get("project", {}) or {}
        prev_name = prev.get("name")
        prev_conf = float(prev.get("confidence", 0.0) or 0.0)
        new_conf = max(prev_conf, 0.75 if prefer_exact else 0.6)
        if prefer_exact or (prev_name in [None, "", "null"] or prev_conf < 0.5):
            label["project"] = {
                "name": best["name"],
                "confidence": new_conf,
                "reason": f"keywords: {', '.join(best['hits'][:4])}",
            }
    return label


# --------- inference core
class QwenVL:
    def __init__(
        self,
        model_path: str,
        mmproj_path: str,
        n_ctx: int,
        n_gpu_layers: int,
        temp: float,
        max_tokens: int,
        threads: int,
        verbose: bool,
    ):
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.temp = temp
        self.max_tokens = max_tokens
        self.llm = Llama(
            model_path=model_path,
            mmproj_path=mmproj_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=threads,
            chat_format="qwen",  # Use chatml format which is compatible
            logits_all=False,
            verbose=verbose,
        )

    def classify(
        self, img_path: str, meta: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Dict[str, Any]:
        sys_msg = {"role": "system", "content": SYSTEM_PROMPT}
        user_msg = {"role": "user", "content": build_user_prompt(meta, ctx)}
        out = self.llm.create_chat_completion(
            messages=[sys_msg, user_msg],
            temperature=self.temp,
            max_tokens=self.max_tokens,
        )
        txt = out["choices"][0]["message"]["content"]
        obj = extract_json(txt)
        print(obj)
        # basic normalization
        obj["coarse_activity"] = str(obj.get("coarse_activity", "misc")).lower()
        if obj["coarse_activity"] not in COARSE_ENUM:
            obj["coarse_activity"] = "misc"
        # clamp confidences
        try:
            obj["confidence"] = float(max(0.0, min(1.0, obj.get("confidence", 0.0))))
        except Exception:
            obj["confidence"] = 0.0
        try:
            pj = obj.get("project") or {}
            pj["confidence"] = float(max(0.0, min(1.0, pj.get("confidence", 0.0))))
            obj["project"] = pj
        except Exception:
            obj["project"] = {"name": None, "confidence": 0.0, "reason": ""}
        return obj


# --------- AW sink
class AWSink:
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
        ev = Event(timestamp=ts, duration=timedelta(seconds=0), data=payload)
        self.client.insert_event(self.bucket, ev)


# --------- main loop
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
):
    logging.basicConfig(level=getattr(logging, log_level.upper()))
    LOG.info("Starting aw-llm-worker | spool=%s", spool_dir)

    ctx = load_yaml_or_json(context_file)
    ctx_id = sha1(json.dumps(ctx, sort_keys=True))
    LOG.info("Loaded context (id=%s)", ctx_id[:8])

    state = State(os.path.join(spool_dir, ".llm_state.json"))
    sink = AWSink(testing=testing, bucket_suffix=bucket_suffix)
    model = QwenVL(
        model_path=model_path,
        mmproj_path=mmproj_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        temp=temp,
        max_tokens=max_new,
        threads=threads,
        verbose=False,
    )

    while True:
        # find candidate spool records
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

            # basic validation
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
                # don't mark -> retry next tick
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

        # heartbeat (optional)
        time.sleep(poll)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOG.info("Exiting.")
