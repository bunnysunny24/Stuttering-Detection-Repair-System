"""Simple Whisper wrapper for transcription with approximate word-level timestamps.

This module uses the `whisper` package (OpenAI) to transcribe an audio file
and converts segment-level timestamps into approximate per-word timestamps by
splitting segment text and allocating time proportionally. This is a CPU-
friendly prototype; timestamps are approximate but good enough to map SED
windows to words for a first-pass pipeline.

Notes:
- Requires `pip install -U openai-whisper` (and system `ffmpeg`). On Windows
  you may need to install FFmpeg separately if not already present.
"""
from typing import List, Dict, Optional
import math
import os
import numpy as np
import soundfile as sf

# Try to prefer faster-whisper (GPU) if available, otherwise fall back to OpenAI's whisper
try:
    from faster_whisper import WhisperModel as FasterWhisperModel  # type: ignore
    _HAS_FASTER = True
except Exception:
    FasterWhisperModel = None  # type: ignore
    _HAS_FASTER = False
try:
    import whisper as _openai_whisper  # type: ignore
    _HAS_OPENAI = True
except Exception:
    _openai_whisper = None
    _HAS_OPENAI = False


def load_whisper_model(model_size: str = "small", prefer_gpu: bool = True):
    """Load a Whisper model.

    Tries to load `faster-whisper` (GPU) if available and prefer_gpu is True.
    Falls back to OpenAI `whisper` package.
    """
    device = None
    # decide device string for faster-whisper
    if prefer_gpu:
        try:
            import torch
            if getattr(torch, 'cuda', None) and torch.cuda.is_available():
                device = 'cuda'
        except Exception:
            device = None

    if _HAS_FASTER:
        # compute_type float16 on GPU for speed, float32 on CPU
        compute_type = 'float16' if device == 'cuda' else 'float32'
        model = FasterWhisperModel(model_size, device=(device or 'cpu'), compute_type=compute_type)
        # mark implementation
        model._impl = 'faster_whisper'
        return model

    if _HAS_OPENAI:
        # use OpenAI whisper as fallback
        dev = 'cpu' if device is None else device
        model = _openai_whisper.load_model(model_size, device=dev)
        model._impl = 'openai_whisper'
        return model

    raise RuntimeError('No whisper backend available: install faster-whisper or openai-whisper')


def transcribe_with_segments(model, audio_path: str, language: Optional[str] = None, task: str = "transcribe") -> Dict:
    """Transcribe audio and return segments in a consistent dict format.

    Supports both faster-whisper and OpenAI's whisper model objects.
    Returns dict with keys 'text' and 'segments' where segments is a list of
    dicts containing 'start','end','text'.
    """
    # faster-whisper returns (segments, info)
    impl = getattr(model, '_impl', None)
    if impl == 'faster_whisper':
        segments, info = model.transcribe(audio_path, beam_size=5, language=language)
        segs = []
        text_parts = []
        for s in segments:
            segs.append({'start': float(s.start), 'end': float(s.end), 'text': s.text, 'audio_path': audio_path})
            text_parts.append(s.text)
        return {'text': ' '.join(text_parts).strip(), 'segments': segs}

    # fallback: OpenAI whisper
    if impl == 'openai_whisper' or impl is None:
        # whisper handles file reading via ffmpeg; audio_path can be file path
        res = model.transcribe(audio_path, language=language, task=task)
        return res

    raise RuntimeError('Unknown whisper model implementation')


def segments_to_word_timestamps(segments: List[Dict]) -> List[Dict]:
    """Convert whisper segments to approximate per-word timestamps.

    Each segment has 'start', 'end', 'text'. We split the text into words and
    assign each word a start/end by dividing the segment duration proportionally
    by character length (preserves approximate spacing for multi-word tokens).
    """
    words = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        dur = max(1e-6, end - start)
        toks = text.split()
        if not toks:
            continue

        # Try an energy-weighted split: load the audio for the segment and
        # allocate word boundaries by cumulative energy proportional to token
        # character lengths. This is a cheap heuristic that often improves
        # alignment vs uniform split and is much faster than full CTC.
        audio_path = seg.get("audio_path") or seg.get("file")
        try:
            if audio_path and os.path.exists(audio_path):
                info = sf.info(audio_path)
                sr = info.samplerate
                s_frame = int(max(0, round(start * sr)))
                e_frame = int(min(round(end * sr), info.frames))
                if e_frame > s_frame:
                    samples, _ = sf.read(audio_path, start=s_frame, stop=e_frame, dtype='float32')
                    samples = np.asarray(samples)
                    if samples.ndim > 1:
                        samples = np.mean(samples, axis=1)
                    energy = np.abs(samples)
                    total_energy = float(np.sum(energy))
                else:
                    samples = None
                    total_energy = 0.0
            else:
                samples = None
                total_energy = 0.0
        except Exception:
            samples = None
            total_energy = 0.0

        toks_lens = [max(1, len(t)) for t in toks]
        total_chars = sum(toks_lens)

        if samples is None or total_energy <= 1e-6:
            # fallback: proportional by character length (original behavior)
            if total_chars == 0:
                per = dur / len(toks)
                cursor = start
                for t in toks:
                    wstart = cursor
                    wend = cursor + per
                    words.append({"word": t, "start": wstart, "end": wend})
                    cursor = wend
            else:
                cursor = start
                for t, l in zip(toks, toks_lens):
                    frac = l / total_chars
                    wdur = frac * dur
                    wstart = cursor
                    wend = cursor + max(1e-6, wdur)
                    words.append({"word": t, "start": wstart, "end": wend})
                    cursor = wend
        else:
            # cumulative energy allocation
            cumsum = np.cumsum(np.abs(samples))
            tot = cumsum[-1]
            # build target energy thresholds for word boundaries
            targets = []
            acc = 0
            for l in toks_lens:
                acc += l
                targets.append(acc / total_chars * tot)

            # find sample indices where cumulative energy crosses targets
            boundaries = [0]
            ti = 0
            for target in targets:
                while ti < len(cumsum) and cumsum[ti] < target:
                    ti += 1
                boundaries.append(min(ti, len(cumsum)))

            # convert boundaries to times and assign
            prev_idx = 0
            for i, t in enumerate(toks):
                b0 = boundaries[i]
                b1 = boundaries[i + 1]
                # convert to seconds relative to segment start
                wstart = start + (b0 / len(samples)) * dur
                wend = start + (b1 / len(samples)) * dur
                if wend <= wstart:
                    wend = wstart + 1e-4
                words.append({"word": t, "start": float(wstart), "end": float(wend)})
                prev_idx = b1
    return words


def transcribe_to_words(audio_path: str, model_size: str = "small", language: Optional[str] = None) -> Dict:
    """High-level helper: load model, transcribe, and return words with timestamps.

    Returns dict: {"text": full_text, "segments": segments, "words": [ {word,start,end}, ... ]}
    """
    model = load_whisper_model(model_size)
    res = transcribe_with_segments(model, audio_path, language=language)
    segs = res.get("segments", [])
    words = segments_to_word_timestamps(segs)
    return {"text": res.get("text", ""), "segments": segs, "words": words}
