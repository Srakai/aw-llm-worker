"""ActivityWatch I/O operations for event analysis."""

import requests
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple
from .aggregator import Event

LOG = logging.getLogger("aw-llm-worker")


def _iso(dt: datetime) -> str:
    """Convert datetime to ISO format for ActivityWatch."""
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def extract_vscode_summary(events: List[Dict]) -> str:
    """
    Extract a summary of VS Code activity from aw-watcher-vscode events.
    
    Args:
        events: List of VS Code watcher events
        
    Returns:
        Summary string like "VSCode: project1 (3 files), project2 (2 files)"
    """
    if not events:
        return ""
    
    # Aggregate by project
    project_files = {}
    
    for ev in events:
        d = ev.get("data", {})
        project = d.get("project", "unknown")
        file_path = d.get("file", "")
        
        if project not in project_files:
            project_files[project] = set()
        
        if file_path:
            project_files[project].add(file_path)
    
    # Format summary
    parts = []
    for project, files in sorted(project_files.items(), key=lambda x: len(x[1]), reverse=True):
        if project and project != "unknown":
            parts.append(f"{project} ({len(files)} files)")
    
    if parts:
        return "VSCode: " + ", ".join(parts[:3])  # Limit to top 3 projects
    return ""


def extract_screenshot_summaries(events: List[Dict]) -> List[str]:
    """
    Extract all screenshot summaries from screenshot-llm events.
    
    Args:
        events: List of screenshot watcher events
        
    Returns:
        List of summary strings with timestamps
    """
    summaries = []
    
    for ev in events:
        d = ev.get("data", {})
        label = d.get("label", {})
        
        if not label:
            continue
        
        summary = label.get("summary", "")
        activity = label.get("coarse_activity", "")
        app = label.get("app_guess", "")
        timestamp = ev.get("timestamp", "")
        
        if summary:
            # Parse timestamp for display
            try:
                ts_dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                time_str = ts_dt.strftime("%H:%M")
            except:
                time_str = ""
            
            # Format: "HH:MM [app] summary"
            formatted = f"{time_str} [{app or activity}] {summary}" if time_str else f"[{app or activity}] {summary}"
            summaries.append(formatted)
    
    return summaries


def separate_events_by_source(
    aw_events_by_bucket: Dict[str, List[Dict]]
) -> Tuple[List[Event], str, List[str]]:
    """
    Separate events into regular events, VS Code summary, and screenshot summaries.
    
    Args:
        aw_events_by_bucket: Raw events grouped by bucket
        
    Returns:
        Tuple of (regular_events, vscode_summary, screenshot_summaries)
    """
    regular_events = []
    vscode_summary = ""
    screenshot_summaries = []
    
    for bid, event_list in aw_events_by_bucket.items():
        # Handle VS Code events separately
        if "aw-watcher-vscode" in bid:
            vscode_summary = extract_vscode_summary(event_list)
            continue
        
        # Handle screenshot events separately
        if "aw-watcher-screenshot-llm" in bid or "aw-llm-blocks" in bid:
            screenshot_summaries.extend(extract_screenshot_summaries(event_list))
            continue
        
        # Process regular events (window, afk, etc.)
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
                    regular_events.append(Event(st, en, text, 1.0))
            except (ValueError, TypeError):
                continue
    
    return regular_events, vscode_summary, screenshot_summaries


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
    segments: List[Dict],
    host: str,
    bucket_id: str,
    client_name: str,
    model_info: Dict = None,
    analysis_metadata: Dict = None,
):
    """Create a bucket and send classified time blocks to AW with rich structure."""
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

            # Build rich structured data similar to screenshot format
            event_data = {
                "classification": {
                    "label": seg["label"],
                    "confidence": seg["confidence"],
                    "min_confidence": seg.get("min_confidence", seg["confidence"]),
                    "max_confidence": seg.get("max_confidence", seg["confidence"]),
                    "project": seg.get("project"),
                    "activity_description": seg.get("activity_description", ""),
                },
                "analysis": {
                    "duration_minutes": seg.get("duration_minutes", 0),
                    "num_windows": seg.get("num_windows", 1),
                    "content_sample": seg.get("content_sample", ""),
                    "method": "llm_text_classification",
                    "window_size_minutes": (
                        analysis_metadata.get("window_duration_m")
                        if analysis_metadata
                        else None
                    ),
                    "window_step_minutes": (
                        analysis_metadata.get("window_step_m")
                        if analysis_metadata
                        else None
                    ),
                },
            }

            # Add model info if provided
            if model_info:
                event_data["llm"] = model_info

            # Add analysis metadata if provided
            if analysis_metadata:
                event_data["config"] = {
                    "merge_gap_s": analysis_metadata.get("merge_gap_s"),
                    "min_confidence": analysis_metadata.get("min_confidence"),
                    "lookback_hours": analysis_metadata.get("lookback_hours"),
                }

            payload.append(
                {
                    "timestamp": seg["start"],
                    "duration": (en - st).total_seconds(),
                    "data": event_data,
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
