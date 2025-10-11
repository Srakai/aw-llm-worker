"""Utility functions for aw-llm-worker."""

import os
import json
import yaml
import hashlib
from typing import Any, Dict, Optional
from datetime import datetime


def iso_to_dt(s: str) -> datetime:
    """Tolerant ISO8601 parser; supports trailing 'Z' and hyphen-separated time."""
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
    """Convert file path to file:// URL format."""
    p = os.path.abspath(path)
    return "file://" + p


def load_yaml_or_json(path: Optional[str]) -> Dict[str, Any]:
    """Load YAML or JSON file."""
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    try:
        if path.endswith(".json"):
            return json.loads(txt)
        return yaml.safe_load(txt) or {}
    except Exception as e:
        raise ValueError(f"Failed to parse context file {path}: {e}")


def sha1(s: str) -> str:
    """Calculate SHA1 hash of string."""
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def read_json(path: str) -> Dict[str, Any]:
    """Read JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(d: str) -> str:
    """Ensure directory exists."""
    os.makedirs(d, exist_ok=True)
    return d
