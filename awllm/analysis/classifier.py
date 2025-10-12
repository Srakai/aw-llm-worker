"""Classify aggregated time events into larger blocks using an LLM."""

import logging
import numpy as np
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any

LOG = logging.getLogger("aw-llm-worker")


def create_time_windows(
    frame_texts: List[str],
    frame_starts: List[datetime],
    window_size: int,
    step_size: int,
    vscode_summary: str = "",
    screenshot_summaries: List[str] = None,
) -> List[Dict[str, Any]]:
    """
    Create overlapping windows of text data from time frames.

    Args:
        frame_texts: List of text content for each time step.
        frame_starts: List of start times for each time step.
        window_size: The number of time steps in each window.
        step_size: The number of time steps to advance for the next window.
        vscode_summary: Summary of VS Code activity (shared across all windows).
        screenshot_summaries: List of screenshot summaries to append.

    Returns:
        A list of windows, where each window is a dictionary containing
        the start time, end time, window index range, and concatenated text.
    """
    if screenshot_summaries is None:
        screenshot_summaries = []
    
    windows = []
    for i in range(0, len(frame_texts) - window_size + 1, step_size):
        start_idx = i
        end_idx = i + window_size - 1

        # Calculate time range
        dt = (
            (frame_starts[1] - frame_starts[0]).total_seconds()
            if len(frame_starts) > 1
            else 0
        )
        start_time = frame_starts[start_idx]
        end_time = frame_starts[end_idx] + timedelta(seconds=dt)

        # Extract unique events from the window
        window_text = _summarize_window_events(frame_texts[start_idx : end_idx + 1])

        if not window_text:
            continue

        # Build enriched content with additional context
        enriched_content = window_text
        
        # Add VS Code summary if available (once per window)
        if vscode_summary:
            enriched_content = f"{vscode_summary}\n\n{enriched_content}"
        
        # Add screenshot summaries that fall within this window's time range
        relevant_screenshots = []
        for screenshot_summary in screenshot_summaries:
            # Screenshot summaries already have timestamps, just include all for now
            # In future, could filter by window time range
            relevant_screenshots.append(screenshot_summary)
        
        if relevant_screenshots:
            screenshots_text = "\n".join(relevant_screenshots)
            enriched_content = f"{enriched_content}\n\nScreenshots:\n{screenshots_text}"

        windows.append(
            {
                "start": start_time,
                "end": end_time,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "content": enriched_content,
                "vscode_summary": vscode_summary,
                "num_screenshots": len(relevant_screenshots),
            }
        )
    return windows


def _summarize_window_events(frame_texts: List[str]) -> str:
    """
    Summarize a window by extracting unique events from all frames.
    Sort by time spent and filter out irrelevant/short activities.

    Args:
        frame_texts: List of formatted frame texts

    Returns:
        Deduplicated summary of events in the window, sorted by time spent
    """
    # Parse all events from frame texts and track time spent
    # Format is "HH:MM-HH:MM: event text; HH:MM: event text; ..."
    event_time_map = {}  # event_text -> total_seconds

    for frame_text in frame_texts:
        if not frame_text.strip():
            continue

        # Split by semicolon to get individual events
        event_parts = frame_text.split(";")

        for part in event_parts:
            part = part.strip()
            if not part:
                continue

            # Parse time range and extract event text
            # Formats: "HH:MM-HH:MM: text" or "HH:MM: text"
            if ":" in part:
                # Find the time portion and text portion
                match = re.match(r"(\d{2}:\d{2})(?:-(\d{2}:\d{2}))?:\s*(.+)", part)
                if match:
                    start_time = match.group(1)
                    end_time = match.group(2) or start_time
                    event_text = match.group(3).strip()

                    if not event_text:
                        continue

                    # Calculate duration (rough estimate)
                    try:
                        start_h, start_m = map(int, start_time.split(":"))
                        end_h, end_m = map(int, end_time.split(":"))
                        duration_minutes = (end_h * 60 + end_m) - (
                            start_h * 60 + start_m
                        )
                        if duration_minutes < 0:
                            duration_minutes += 24 * 60  # Handle day wrap
                        duration_seconds = duration_minutes * 60
                    except:
                        duration_seconds = 60  # Default to 1 minute

                    # Accumulate time for this event
                    if event_text not in event_time_map:
                        event_time_map[event_text] = 0
                    event_time_map[event_text] += duration_seconds

    if not event_time_map:
        return ""

    # Sort by time spent (descending) and filter
    sorted_events = sorted(event_time_map.items(), key=lambda x: x[1], reverse=True)

    # Filter out very short activities (less than 2 minutes total)
    significant_events = [
        (text, seconds) for text, seconds in sorted_events if seconds >= 120
    ]

    # If nothing significant, take top 5 anyway
    if not significant_events:
        significant_events = sorted_events[:5]
    else:
        # Limit to top 10 most time-consuming activities
        significant_events = significant_events[:10]

    # Format as simple list (no timestamps, just activities)
    event_texts = [text for text, _ in significant_events]
    result = "; ".join(event_texts)

    # Limit total length
    if len(result) > 1000:
        result = result[:1000] + "..."

    return result


def classify_windows_with_llm(
    windows: List[Dict[str, Any]], llm_model: Any, topics: List[str]
) -> List[Dict[str, Any]]:
    """
    Use an LLM to classify the content of each time window.

    Args:
        windows: List of window dictionaries with 'content', 'start', 'end' keys.
        llm_model: The LLM model instance with a classify_text method.
        topics: List of valid topic labels for classification.

    Returns:
        List of classified windows with label and confidence added.
    """
    classified = []

    for i, window in enumerate(windows):
        try:
            # Call the LLM to classify this window's text
            result = llm_model.classify_text(window["content"], topics)

            classified_window = {
                **window,
                "label": result.get("label", "misc"),
                "confidence": result.get("confidence", 0.0),
            }
            classified.append(classified_window)

            LOG.debug(
                f"Window {i+1}/{len(windows)}: {window['start'].strftime('%H:%M')}-{window['end'].strftime('%H:%M')} "
                f"-> {result.get('label')} (conf={result.get('confidence', 0.0):.2f})"
            )

        except Exception as e:
            LOG.error(f"Failed to classify window {i}: {e}")
            # Create a fallback classification
            classified.append(
                {
                    **window,
                    "label": "misc",
                    "confidence": 0.0,
                }
            )

    return classified


def merge_classified_windows(
    classified_windows: List[Dict[str, Any]],
    merge_gap_s: float = 300.0,
    min_confidence: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Merge consecutive windows with the same classification label into larger blocks.

    Args:
        classified_windows: List of classified window dictionaries.
        merge_gap_s: Maximum gap in seconds to merge blocks with the same label.
        min_confidence: Minimum confidence threshold to include a window.

    Returns:
        List of merged time blocks with rich structure similar to screenshot events.
    """
    if not classified_windows:
        return []

    # Filter by confidence threshold
    valid_windows = [
        w for w in classified_windows if w.get("confidence", 0.0) >= min_confidence
    ]

    if not valid_windows:
        LOG.info("No windows above confidence threshold.")
        return []

    # Sort by start time
    valid_windows.sort(key=lambda w: w["start"])

    blocks = []
    current_block = {
        "start": valid_windows[0]["start"],
        "end": valid_windows[0]["end"],
        "label": valid_windows[0]["label"],
        "confidences": [valid_windows[0]["confidence"]],
        "content_samples": [valid_windows[0].get("content", "")[:200]],
        "num_windows": 1,
        "project": valid_windows[0].get("project"),
        "activity_description": valid_windows[0].get("activity_description", ""),
    }

    for window in valid_windows[1:]:
        gap_seconds = (window["start"] - current_block["end"]).total_seconds()
        same_label = window["label"] == current_block["label"]

        # Merge if same label and gap is small enough
        if same_label and gap_seconds <= merge_gap_s:
            current_block["end"] = window["end"]
            current_block["confidences"].append(window["confidence"])
            current_block["num_windows"] += 1
            # Keep up to 3 content samples for reference
            if len(current_block["content_samples"]) < 3:
                current_block["content_samples"].append(window.get("content", "")[:200])
            # Update project if not set or if new window has higher confidence
            if not current_block["project"] and window.get("project"):
                current_block["project"] = window.get("project")
            # Merge activity descriptions (prefer longer, more detailed ones)
            new_desc = window.get("activity_description", "")
            if len(new_desc) > len(current_block["activity_description"]):
                current_block["activity_description"] = new_desc
        else:
            # Finalize current block
            duration_s = (current_block["end"] - current_block["start"]).total_seconds()
            blocks.append(
                {
                    "start": current_block["start"].isoformat(),
                    "end": current_block["end"].isoformat(),
                    "label": current_block["label"],
                    "confidence": float(np.mean(current_block["confidences"])),
                    "duration_minutes": round(duration_s / 60, 1),
                    "num_windows": current_block["num_windows"],
                    "content_sample": (
                        current_block["content_samples"][0]
                        if current_block["content_samples"]
                        else ""
                    ),
                    "min_confidence": float(min(current_block["confidences"])),
                    "max_confidence": float(max(current_block["confidences"])),
                    "project": current_block["project"],
                    "activity_description": current_block["activity_description"],
                }
            )

            # Start new block
            current_block = {
                "start": window["start"],
                "end": window["end"],
                "label": window["label"],
                "confidences": [window["confidence"]],
                "content_samples": [window.get("content", "")[:200]],
                "num_windows": 1,
                "project": window.get("project"),
                "activity_description": window.get("activity_description", ""),
            }

    # Don't forget the last block
    duration_s = (current_block["end"] - current_block["start"]).total_seconds()
    blocks.append(
        {
            "start": current_block["start"].isoformat(),
            "end": current_block["end"].isoformat(),
            "label": current_block["label"],
            "confidence": float(np.mean(current_block["confidences"])),
            "duration_minutes": round(duration_s / 60, 1),
            "num_windows": current_block["num_windows"],
            "content_sample": (
                current_block["content_samples"][0]
                if current_block["content_samples"]
                else ""
            ),
            "min_confidence": float(min(current_block["confidences"])),
            "max_confidence": float(max(current_block["confidences"])),
            "project": current_block["project"],
            "activity_description": current_block["activity_description"],
        }
    )

    return blocks
