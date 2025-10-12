"""Discretize and aggregate time-series events into a matrix."""

import math
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from .textconv import HashSketch


@dataclass
class Event:
    start: datetime  # tz-aware
    end: datetime  # tz-aware
    text: str
    weight: float = 1.0


def discretize(
    events: List[Event], start: datetime, end: datetime, dt: float, sketch: HashSketch
) -> Tuple[np.ndarray, List[datetime]]:
    """
    Return (M, frame_starts):
      M: (T,D) frame vectors (L2-normalized), each is overlap-weighted sum of event vectors.
      Overlap handling == conv-style: each frame receives fractional coverage from any event.
    """
    if not (start.tzinfo and end.tzinfo):
        raise ValueError("Start and end datetimes must be timezone-aware.")

    D = sketch.dim
    total_seconds = (end - start).total_seconds()
    if total_seconds <= 0:
        return np.zeros((0, D), dtype=np.float32), []

    T = int(math.ceil(total_seconds / dt))
    M = np.zeros((T, D), dtype=np.float32)
    frame_starts = [start + timedelta(seconds=i * dt) for i in range(T)]

    # Pre-compute event vectors
    unique_texts = {ev.text for ev in events}
    cache: Dict[str, np.ndarray] = {text: sketch.encode(text) for text in unique_texts}

    for ev in events:
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

    return M, frame_starts
