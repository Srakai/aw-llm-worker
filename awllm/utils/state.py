"""State management for deduplication."""

import os
import uuid
import json
from typing import Any, Dict
from .helpers import read_json


def write_json_atomic(path: str, obj: Any) -> None:
    """Write JSON atomically to avoid corruption."""
    tmp = f"{path}.tmp.{uuid.uuid4().hex}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class State:
    """Tracks processed screenshots to avoid duplicates."""

    def __init__(self, path: str):
        self.path = path
        self.data = {"seen": {}}  # key -> ts
        if os.path.exists(path):
            try:
                self.data = read_json(path)
            except Exception:
                pass

    def key(self, rec: Dict[str, Any]) -> str:
        """Generate unique key for a record."""
        return rec.get("sha256") or ("path:" + rec.get("path", ""))

    def seen(self, rec: Dict[str, Any]) -> bool:
        """Check if record has been processed."""
        k = self.key(rec)
        return k in self.data["seen"]

    def mark(self, rec: Dict[str, Any]):
        """Mark record as processed."""
        k = self.key(rec)
        self.data["seen"][k] = rec.get("ts")
        write_json_atomic(self.path, self.data)
