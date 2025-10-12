"""Classify aggregated time events into larger blocks."""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple


def find_segments(
    scores: np.ndarray,
    threshold: float,
    min_duration_s: float,
    dt: float,
    merge_gap_s: float = 60.0,
) -> List[Tuple[int, int]]:
    """
    Find contiguous segments where score is above a threshold.
    Merges segments that are close to each other.
    """
    if scores.size == 0:
        return []

    above_threshold = np.where(scores > threshold)[0]
    if not len(above_threshold):
        return []

    # Find contiguous segments
    segments = []
    start_idx = above_threshold[0]
    for i in range(1, len(above_threshold)):
        if above_threshold[i] > above_threshold[i - 1] + 1:
            segments.append((start_idx, above_threshold[i - 1]))
            start_idx = above_threshold[i]
    segments.append((start_idx, above_threshold[-1]))

    # Merge close segments
    merge_gap_steps = int(merge_gap_s / dt)
    if not segments:
        return []

    merged = [segments[0]]
    for current_start, current_end in segments[1:]:
        last_start, last_end = merged[-1]
        if current_start - last_end <= merge_gap_steps:
            merged[-1] = (last_start, current_end)
        else:
            merged.append((current_start, current_end))

    # Filter for minimum duration
    min_len_steps = int(min_duration_s / dt)
    final_segments = [(s, e) for s, e in merged if (e - s + 1) >= min_len_steps]

    return final_segments


def segments_to_blocks(
    segments: List[Tuple[int, int]],
    frame_starts: List[datetime],
    label: str,
    scores: np.ndarray,
) -> List[Dict]:
    """Convert index-based segments to timestamped blocks with labels."""
    blocks = []
    if not frame_starts:
        return blocks

    dt_seconds = (
        (frame_starts[1] - frame_starts[0]).total_seconds()
        if len(frame_starts) > 1
        else 0
    )

    for start_idx, end_idx in segments:
        start_time = frame_starts[start_idx]
        end_time = frame_starts[end_idx] + timedelta(seconds=dt_seconds)

        segment_scores = scores[start_idx : end_idx + 1]
        confidence = float(np.mean(segment_scores)) if segment_scores.size > 0 else 0.0

        blocks.append(
            {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "label": label,
                "confidence": confidence,
            }
        )
    return blocks
