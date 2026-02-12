"""Utilities to map SED detection windows to ASR word timestamps.

Provides functions to convert window/sample indexes to seconds, compute
overlap, and label words as stuttered when overlapping with SED windows.
"""
from typing import List, Dict, Tuple


def samples_to_seconds(idx: int, sr: int) -> float:
    return float(idx) / float(sr)


def seconds_to_samples(sec: float, sr: int) -> int:
    return int(round(sec * sr))


def overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Return overlap length in seconds between intervals a and b."""
    s = max(a_start, b_start)
    e = min(a_end, b_end)
    return max(0.0, e - s)


def map_windows_to_words(windows: List[Tuple[int, int]], words: List[Dict], sr: int, min_overlap_ratio: float = 0.2) -> List[Dict]:
    """Given SED windows in samples and ASR words with second timestamps,
    label words as stuttered if they overlap by at least min_overlap_ratio
    of the word duration.

    Inputs:
      windows: list of (start_sample, end_sample)
      words: list of {"word": str, "start": sec, "end": sec}
      sr: sample rate
      min_overlap_ratio: fraction of word duration that must overlap

    Returns: list of words enriched with `stutter` bool and overlap seconds.
    """
    # Convert windows to seconds
    win_secs = [(w[0] / sr, w[1] / sr) for w in windows]
    out = []
    for w in words:
        wstart = float(w.get("start", 0.0))
        wend = float(w.get("end", wstart))
        wdur = max(1e-6, wend - wstart)
        overlap = 0.0
        for ws, we in win_secs:
            overlap += overlap_seconds(wstart, wend, ws, we)
        stutter = (overlap / wdur) >= min_overlap_ratio
        rec = dict(w)
        rec.update({"stutter": bool(stutter), "overlap_s": float(overlap)})
        out.append(rec)
    return out
