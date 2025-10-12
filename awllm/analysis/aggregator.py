"""Discretize and aggregate time-series events into a matrix."""

import math
import re
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Set

from .textconv import HashSketch


@dataclass
class Event:
    start: datetime  # tz-aware
    end: datetime  # tz-aware
    text: str
    weight: float = 1.0


def clean_event_text(text: str) -> str:
    """
    Aggressively clean event text to remove redundant patterns and noise.

    Args:
        text: Raw event text

    Returns:
        Cleaned, deduplicated text
    """
    if not text:
        return ""

    # Remove "Google Chrome" duplicates
    text = re.sub(r"Google Chrome\s+", "", text, count=1)  # Remove first occurrence
    text = re.sub(r"\s+Google Chrome\s*$", "", text)  # Remove trailing
    text = re.sub(
        r"\s+—\s+Google Chrome.*$", "", text
    )  # Remove "— Google Chrome" suffix

    # Simplify "Code - Insiders" to "VSCode"
    text = re.sub(r"^Code\s*-?\s*Insiders\s+", "VSCode: ", text)

    # Remove noise tags
    text = re.sub(r"\s*\(Incognito\)\s*", " ", text)
    text = re.sub(r"\s*–\s*Audio playing\s*", " ", text)
    text = re.sub(r"\s*\(\d+\s+new\s+items?\)\s*", " ", text)

    # Remove excessive dashes and separators
    text = re.sub(r"\s*—+\s*", " ", text)
    text = re.sub(r"\s*–+\s*", " ", text)
    text = re.sub(r"\s+-\s+", " ", text)

    # Remove trailing usernames (Filip, etc.)
    text = re.sub(r"\s+\w+\s*$", "", text)

    # Remove "BEST" workspace tags
    text = re.sub(r"\s+BEST\s*$", "", text)
    text = re.sub(r"\s+—\s+BEST\s*$", "", text)

    # Shorten common app names
    text = re.sub(r"\s+Google Chrome\s*", " Chrome ", text)
    text = re.sub(r"^Chrome\s+", "", text)  # Remove Chrome prefix

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def deduplicate_and_merge_events(events: List[Event]) -> List[Event]:
    """
    Deduplicate and merge events to create clean, non-overlapping blocks.

    Args:
        events: Raw list of events from various sources

    Returns:
        Merged, deduplicated events sorted by time
    """
    if not events:
        return []

    # First pass: clean and normalize all events
    cleaned_events = []

    for event in events:
        # Normalize the text
        text = event.text.strip()
        if not text:
            continue

        # Filter out junk
        text_lower = text.lower()
        if any(skip in text_lower for skip in ["afk", "loginwindow", "screensaver"]):
            continue

        # Apply aggressive cleaning
        clean_text = clean_event_text(text)

        if not clean_text or len(clean_text) < 3:
            continue

        # Truncate overly long texts
        if len(clean_text) > 80:
            clean_text = clean_text[:80] + "..."

        cleaned_events.append(
            Event(
                start=event.start, end=event.end, text=clean_text, weight=event.weight
            )
        )

    if not cleaned_events:
        return []

    # Second pass: group by time buckets and merge identical texts
    # Use 1-minute buckets
    time_buckets = {}

    for event in cleaned_events:
        # Round to nearest minute
        bucket_time = event.start.replace(second=0, microsecond=0)

        if bucket_time not in time_buckets:
            time_buckets[bucket_time] = {}

        # Group by text within this time bucket
        if event.text not in time_buckets[bucket_time]:
            time_buckets[bucket_time][event.text] = {
                "start": event.start,
                "end": event.end,
                "text": event.text,
            }
        else:
            # Extend the time range
            existing = time_buckets[bucket_time][event.text]
            existing["start"] = min(existing["start"], event.start)
            existing["end"] = max(existing["end"], event.end)

    # Flatten back to list
    merged = []
    for bucket_time in sorted(time_buckets.keys()):
        for text, data in time_buckets[bucket_time].items():
            merged.append(
                Event(
                    start=data["start"], end=data["end"], text=data["text"], weight=1.0
                )
            )

    # Third pass: merge consecutive events with the same text
    if not merged:
        return []

    merged.sort(key=lambda e: e.start)
    final_merged = []
    current = merged[0]

    for event in merged[1:]:
        gap = (event.start - current.end).total_seconds()

        # Merge if same text and within 2 minutes
        if event.text == current.text and gap <= 120:
            current.end = max(current.end, event.end)
        else:
            final_merged.append(current)
            current = event

    final_merged.append(current)

    return final_merged


def format_events_for_llm(events: List[Event], max_length: int = 2000) -> str:
    """
    Format a list of events into a clean, readable summary for the LLM.

    Args:
        events: List of deduplicated, merged events
        max_length: Maximum character length for output

    Returns:
        Formatted string like "09:00-09:15: VSCode file.py; 09:15-09:30: Chrome docs"
    """
    if not events:
        return ""

    parts = []
    for event in events:
        # Format time range
        start_str = event.start.strftime("%H:%M")
        end_str = event.end.strftime("%H:%M")

        # Only show range if different
        if start_str != end_str:
            time_str = f"{start_str}-{end_str}"
        else:
            time_str = start_str

        parts.append(f"{time_str}: {event.text}")

    result = "; ".join(parts)

    # Truncate if too long
    if len(result) > max_length:
        result = result[:max_length] + "..."

    return result


def discretize(
    events: List[Event], start: datetime, end: datetime, dt: float, sketch: HashSketch
) -> Tuple[np.ndarray, List[datetime], List[str]]:
    """
    Return (M, frame_starts, frame_texts):
      M: (T,D) frame vectors (L2-normalized).
      frame_starts: List of datetimes for the start of each frame.
      frame_texts: Aggregated text content for each frame with time ranges.
    """
    if not (start.tzinfo and end.tzinfo):
        raise ValueError("Start and end datetimes must be timezone-aware.")

    D = sketch.dim
    total_seconds = (end - start).total_seconds()
    if total_seconds <= 0:
        return np.zeros((0, D), dtype=np.float32), [], []

    T = int(math.ceil(total_seconds / dt))
    M = np.zeros((T, D), dtype=np.float32)
    frame_starts = [start + timedelta(seconds=i * dt) for i in range(T)]

    # First, deduplicate and merge all events globally
    merged_events = deduplicate_and_merge_events(events)

    # Pre-compute event vectors (use merged events)
    unique_texts = {ev.text for ev in merged_events}
    cache: Dict[str, np.ndarray] = {text: sketch.encode(text) for text in unique_texts}

    # Build frame vectors using merged events
    for ev in merged_events:
        st, en = max(start, ev.start), min(end, ev.end)
        if en <= st:
            continue

        v = cache[ev.text] * ev.weight

        i0 = int((st - start).total_seconds() / dt)
        i1 = int(math.ceil((en - start).total_seconds() / dt))

        for i in range(max(0, i0), min(T, i1)):
            frame_start = frame_starts[i]
            frame_end = frame_start + timedelta(seconds=dt)
            overlap = (min(en, frame_end) - max(st, frame_start)).total_seconds()
            if overlap > 0:
                M[i] += (overlap / dt) * v

    # Normalize frames
    norm = np.linalg.norm(M, axis=1, keepdims=True)
    non_zero_rows = norm[:, 0] > 1e-9
    M[non_zero_rows] /= norm[non_zero_rows]

    # Format frame texts: for each frame, find events that overlap
    frame_texts = []
    for i in range(T):
        frame_start = frame_starts[i]
        frame_end = frame_start + timedelta(seconds=dt)

        # Find events that overlap this frame
        overlapping = []
        for ev in merged_events:
            if ev.end > frame_start and ev.start < frame_end:
                overlapping.append(ev)

        # Format the overlapping events
        text = format_events_for_llm(overlapping, max_length=500)
        frame_texts.append(text)

    return M, frame_starts, frame_texts
