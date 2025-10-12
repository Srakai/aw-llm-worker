"""ActivityWatch I/O operations for event analysis."""

import requests
import logging
import os
import glob
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional
from .aggregator import Event

LOG = logging.getLogger("aw-llm-worker")


def _iso(dt: datetime) -> str:
    """Convert datetime to ISO format for ActivityWatch."""
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def extract_vscode_events(events: List[Dict]) -> List[Dict]:
    """
    Extract VS Code events with timestamps for window-based filtering.

    Args:
        events: List of VS Code watcher events

    Returns:
        List of dicts with timestamp, project, and file information
    """
    if not events:
        return []

    vscode_events = []
    for ev in events:
        d = ev.get("data", {})
        timestamp = ev.get("timestamp", "")

        if not timestamp:
            continue

        project = d.get("project", "")
        file_path = d.get("file", "")

        if project or file_path:
            try:
                ts_dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                time_str = ts_dt.strftime("%H:%M")

                # Format: "HH:MM VSCode: project/file"
                parts = []
                if project:
                    parts.append(project)
                if file_path:
                    # Just get filename, not full path
                    filename = (
                        file_path.split("/")[-1] if "/" in file_path else file_path
                    )
                    parts.append(filename)

                content = "VSCode: " + "/".join(parts) if parts else ""

                if content:
                    vscode_events.append(
                        {
                            "timestamp": ts_dt,
                            "time_str": time_str,
                            "content": content,
                            "formatted": f"{time_str} {content}",
                        }
                    )
            except:
                continue

    return vscode_events


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
            formatted = (
                f"{time_str} [{app or activity}] {summary}"
                if time_str
                else f"[{app or activity}] {summary}"
            )
            summaries.append(formatted)

    return summaries


def separate_events_by_source(
    aw_events_by_bucket: Dict[str, List[Dict]],
) -> Tuple[List[Event], List[Dict], List[str]]:
    """
    Separate events into regular events, VS Code events, and screenshot summaries.

    Args:
        aw_events_by_bucket: Raw events grouped by bucket

    Returns:
        Tuple of (regular_events, vscode_events, screenshot_summaries)
    """
    regular_events = []
    vscode_events = []
    screenshot_summaries = []

    for bid, event_list in aw_events_by_bucket.items():
        # Handle VS Code events separately
        if "aw-watcher-vscode" in bid:
            vscode_events = extract_vscode_events(event_list)
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

    return regular_events, vscode_events, screenshot_summaries


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

            # Add screenshot enrichment info if present
            if seg.get("enriched_with_screenshots"):
                event_data["screenshot_enrichment"] = {
                    "enriched": True,
                    "num_screenshots": seg.get("num_screenshots_analyzed", 0),
                    "insights": seg.get("screenshot_insights", []),
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


def emit_raw_windows_to_aw(
    windows: List[Dict],
    host: str,
    bucket_id: str,
    client_name: str,
    model_info: Dict = None,
):
    """Emit raw classified windows to AW for granular time tracking.

    This creates a separate bucket for storing individual time windows (e.g., 30-min intervals)
    before they are merged into larger blocks.
    """
    if not windows:
        return

    try:
        # Create bucket for raw windows
        bucket_data = {
            "client": client_name,
            "type": "app.time.window",
            "hostname": bucket_id.split("_")[-1] if "_" in bucket_id else "unknown",
        }

        resp = requests.post(
            f"{host}/api/0/buckets/{bucket_id}",
            json=bucket_data,
            timeout=10,
        )

        if resp.status_code not in (200, 304):
            LOG.warning(
                f"Window bucket creation returned status {resp.status_code}: {resp.text}"
            )

        payload = []
        for window in windows:
            st = window["start"]
            en = window["end"]

            # Build window event data
            event_data = {
                "classification": {
                    "label": window.get("label", "misc"),
                    "confidence": window.get("confidence", 0.0),
                    "project": window.get("project"),
                    "activity_description": window.get("activity_description", ""),
                },
                "window": {
                    "start_idx": window.get("start_idx"),
                    "end_idx": window.get("end_idx"),
                    "content_preview": window.get("content", "")[:500],
                    "num_vscode": window.get("num_vscode", 0),
                    "num_screenshots": window.get("num_screenshots", 0),
                },
            }

            if model_info:
                event_data["llm"] = model_info

            # Add screenshot request flag if present
            if window.get("request_screenshot_analysis"):
                event_data["screenshot_request"] = {
                    "requested": True,
                    "reason": window.get("screenshot_analysis_reason", ""),
                }

            payload.append(
                {
                    "timestamp": st.isoformat(),
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

        LOG.info(f"Emitted {len(payload)} raw windows to bucket {bucket_id}")

    except requests.RequestException as e:
        LOG.error(f"Error emitting raw windows to ActivityWatch: {e}")


def emit_project_blocks_to_aw(
    segments: List[Dict],
    host: str,
    bucket_prefix: str,
    client_name: str,
    model_info: Dict = None,
    analysis_metadata: Dict = None,
):
    """Emit blocks grouped by project to separate per-project buckets.

    This creates one bucket per project (e.g., aw-llm-project-myproject_hostname)
    and emits all blocks for that project to its dedicated bucket.
    """
    if not segments:
        return

    # Group segments by project
    by_project = {}
    for seg in segments:
        project = seg.get("project") or "unassigned"
        if project not in by_project:
            by_project[project] = []
        by_project[project].append(seg)

    LOG.info(f"Emitting blocks to {len(by_project)} project-specific buckets")

    # Emit to each project's bucket
    for project, project_segments in by_project.items():
        # Sanitize project name for bucket ID
        safe_project = project.lower().replace(" ", "-").replace("_", "-")
        bucket_id = f"{bucket_prefix}-{safe_project}"

        try:
            # Create project-specific bucket
            bucket_data = {
                "client": client_name,
                "type": "app.project.block",
                "hostname": bucket_id.split("_")[-1] if "_" in bucket_id else "unknown",
            }

            resp = requests.post(
                f"{host}/api/0/buckets/{bucket_id}",
                json=bucket_data,
                timeout=10,
            )

            if resp.status_code not in (200, 304):
                LOG.warning(
                    f"Project bucket creation returned status {resp.status_code}: {resp.text}"
                )

            payload = []
            for seg in project_segments:
                st = datetime.fromisoformat(seg["start"])
                en = datetime.fromisoformat(seg["end"])

                # Build rich structured data
                event_data = {
                    "project": project,
                    "classification": {
                        "label": seg["label"],
                        "confidence": seg["confidence"],
                        "min_confidence": seg.get("min_confidence", seg["confidence"]),
                        "max_confidence": seg.get("max_confidence", seg["confidence"]),
                        "activity_description": seg.get("activity_description", ""),
                    },
                    "analysis": {
                        "duration_minutes": seg.get("duration_minutes", 0),
                        "num_windows": seg.get("num_windows", 1),
                        "content_sample": seg.get("content_sample", ""),
                        "method": "llm_text_classification",
                    },
                }

                if model_info:
                    event_data["llm"] = model_info

                if analysis_metadata:
                    event_data["config"] = {
                        "merge_gap_s": analysis_metadata.get("merge_gap_s"),
                        "min_confidence": analysis_metadata.get("min_confidence"),
                    }

                # Add screenshot enrichment info if present
                if seg.get("enriched_with_screenshots"):
                    event_data["screenshot_enrichment"] = {
                        "enriched": True,
                        "num_screenshots": seg.get("num_screenshots_analyzed", 0),
                        "insights": seg.get("screenshot_insights", []),
                    }

                payload.append(
                    {
                        "timestamp": seg["start"],
                        "duration": (en - st).total_seconds(),
                        "data": event_data,
                    }
                )

            # Send events
            for i in range(0, len(payload), 500):
                batch = payload[i : i + 500]
                r = requests.post(
                    f"{host}/api/0/buckets/{bucket_id}/events", json=batch, timeout=30
                )
                r.raise_for_status()

            LOG.info(f"Emitted {len(payload)} blocks to project bucket {bucket_id}")

        except requests.RequestException as e:
            LOG.error(f"Error emitting to project bucket {bucket_id}: {e}")


def find_screenshots_for_window(
    spool_dir: str,
    window_start: datetime,
    window_end: datetime,
    max_screenshots: int = 1,
) -> List[str]:
    """Find screenshot files that fall within a time window.

    Args:
        spool_dir: Directory containing screenshot spool files
        window_start: Start time of the window
        window_end: End time of the window
        max_screenshots: Maximum number of screenshots to return

    Returns:
        List of screenshot file paths that fall within the window
    """
    if not spool_dir or not os.path.exists(spool_dir):
        return []

    # Find all JSON spool files
    spool_files = sorted(
        glob.glob(os.path.join(spool_dir, "*.json")),
        key=os.path.getmtime,
        reverse=True,  # Most recent first
    )

    matching_screenshots = []

    for spool_file in spool_files:
        try:
            import json

            with open(spool_file, "r") as f:
                rec = json.load(f)

            # Get timestamp and image path
            ts_str = rec.get("ts")
            img_path = rec.get("path")

            if not ts_str or not img_path or not os.path.exists(img_path):
                continue

            # Parse timestamp - handle both standard ISO format and screenshot watcher format
            # Screenshot watcher may use format like: "2025-10-10T22-37-01.006+00-00"
            # We need to convert it to: "2025-10-10T22:37:01.006+00:00"
            ts_normalized = ts_str.replace("Z", "+00:00")

            # Fix the time portion if it uses hyphens instead of colons
            # Match pattern: T[hour]-[min]-[sec]
            import re

            ts_normalized = re.sub(
                r"T(\d{2})-(\d{2})-(\d{2})", r"T\1:\2:\3", ts_normalized
            )

            # Fix the timezone portion if it uses hyphens
            # Match pattern: +00-00 or -00-00 at the end
            ts_normalized = re.sub(r"([+-]\d{2})-(\d{2})$", r"\1:\2", ts_normalized)

            try:
                ts_dt = datetime.fromisoformat(ts_normalized)
            except ValueError as e:
                LOG.debug(
                    f"Could not parse timestamp '{ts_str}' (normalized: '{ts_normalized}'): {e}"
                )
                continue

            # Check if screenshot falls within window (with small buffer)
            buffer_minutes = 2
            if ts_dt >= window_start - timedelta(
                minutes=buffer_minutes
            ) and ts_dt <= window_end + timedelta(minutes=buffer_minutes):
                matching_screenshots.append(img_path)

                if len(matching_screenshots) >= max_screenshots:
                    break

        except Exception as e:
            LOG.debug(f"Error reading spool file {spool_file}: {e}")
            continue

    return matching_screenshots
