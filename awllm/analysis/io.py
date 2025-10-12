"""ActivityWatch I/O operations for event analysis."""

import requests
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List
from .aggregator import Event

LOG = logging.getLogger("aw-llm-worker")


def _iso(dt: datetime) -> str:
    """Convert datetime to ISO format for ActivityWatch."""
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def fetch_aw_events(
    buckets: List[str], start: datetime, end: datetime, host: str
) -> Dict[str, List[Dict]]:
    """Fetch events from multiple AW buckets with error handling."""
    out = {}
    for bucket_id in buckets:
        url = f"{host}/api/0/buckets/{bucket_id}/events"
        params = {"start": _iso(start), "end": _iso(end)}
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            out[bucket_id] = r.json()
        except requests.RequestException as e:
            LOG.warning(f"Error fetching events from bucket {bucket_id}: {e}")
            out[bucket_id] = []
    return out


def aw_event_to_text(bucket_id: str, ev: Dict) -> str:
    """Extract a descriptive text string from an AW event."""
    d = ev.get("data", {})
    if not d:
        return ""

    if "aw-watcher-window" in bucket_id:
        return f'{d.get("app","")} {d.get("title","")}'.strip()
    if "aw-watcher-vscode" in bucket_id:
        return f'vscode {d.get("project","")} {d.get("file","")}'.strip()
    if "aw-watcher-screenshot" in bucket_id or "aw-llm-worker" in bucket_id:
        lbl = d.get("label", {})
        if not lbl:
            return ""
        return " ".join(
            filter(
                None,
                [
                    lbl.get("coarse_activity"),
                    lbl.get("app_guess"),
                    lbl.get("summary"),
                    " ".join(lbl.get("tags", [])),
                    (lbl.get("project") or {}).get("name"),
                ],
            )
        ).strip()
    return " ".join(str(v) for v in d.values() if isinstance(v, str)).strip()


def load_events_for_discretization(
    aw_events_by_bucket: Dict[str, List[Dict]],
) -> List[Event]:
    """Convert raw AW events into the aggregator's Event format."""
    out: List[Event] = []
    for bid, event_list in aw_events_by_bucket.items():
        for ev in event_list:
            ts_str = ev.get("timestamp")
            if not ts_str:
                continue

            try:
                st = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                dur = ev.get("duration", 0.0)
                en = st + timedelta(seconds=dur)
                text = aw_event_to_text(bid, ev)
                if text:
                    out.append(Event(st, en, text, 1.0))
            except (ValueError, TypeError):
                continue  # Skip malformed events
    return out


def emit_blocks_to_aw(
    segments: List[Dict], host: str, bucket_id: str, client_name: str
):
    """Create a bucket and send classified time blocks to AW."""
    if not segments:
        return

    try:
        # Create bucket using proper ActivityWatch API format
        bucket_data = {
            "client": client_name,
            "type": "app.time.block",
            "hostname": bucket_id.split("_")[-1] if "_" in bucket_id else "unknown",
        }

        resp = requests.post(
            f"{host}/api/0/buckets/{bucket_id}",
            json=bucket_data,
            timeout=10,
        )
        # 304 = already exists, which is fine
        if resp.status_code not in (200, 304):
            LOG.warning(
                f"Bucket creation returned status {resp.status_code}: {resp.text}"
            )

        payload = []
        for seg in segments:
            st = datetime.fromisoformat(seg["start"])
            en = datetime.fromisoformat(seg["end"])
            payload.append(
                {
                    "timestamp": seg["start"],
                    "duration": (en - st).total_seconds(),
                    "data": {"label": seg["label"], "confidence": seg["confidence"]},
                }
            )

        # Send events in batches
        for i in range(0, len(payload), 500):
            batch = payload[i : i + 500]
            r = requests.post(
                f"{host}/api/0/buckets/{bucket_id}/events", json=batch, timeout=30
            )
            r.raise_for_status()
        LOG.info(f"Emitted {len(payload)} blocks to bucket {bucket_id}")
    except requests.RequestException as e:
        LOG.error(f"Error emitting blocks to ActivityWatch: {e}")
