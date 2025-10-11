"""Prompt templates and builders for screenshot classification."""

import re
import json
from typing import Any, Dict, List
from awllm.utils.helpers import to_file_url


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

JSON schema:
{{
  "what_user_might_be_doing": string, # <= 20 words explaining your choice
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
    """Compose OpenAI-style content array with image + text."""
    meta_text = "Classify what the user is doing now. If matches any project by keywords, set project.name accordingly."

    return [
        {"type": "image_url", "image_url": {"url": to_file_url(meta["path"])}},
        {"type": "text", "text": meta_text},
    ]


def extract_json(txt: str) -> Dict[str, Any]:
    """Extract first JSON object from LLM output."""
    m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in LLM output")
    s = m.group(0)
    return json.loads(s)


def route_project_keywords(
    meta: Dict[str, Any], ctx: Dict[str, Any], label: Dict[str, Any]
) -> Dict[str, Any]:
    """Apply keyword-based project routing (pre/post LLM)."""
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
