"""End-to-end CPU prototype: ASR (Whisper) + SED mapping + repair.

This script:
  - Runs the sliding-window SED detector (existing CNN) to find stutter windows.
  - Runs Whisper transcription and derives approximate per-word timestamps.
  - Maps SED windows to words and labels words as stuttered.
  - Applies a surgical repair (remove / silence / attenuate) using word boundaries.

Notes:
  - Uses CPU-only Whisper (model sizes: tiny, small, medium). This will download
    model weights on first run and can be slow on CPU.
  - Word timestamps are approximate (segment->word proportional split). For
    production, replace with a forced-aligner after fine-tuning.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import soundfile as sf

try:
    import torch
except Exception:
    torch = None

from Models.model_cnn import SimpleCNN
from Models.repair import load_audio, window_iter, predict_window
from Models.asr_whisper import transcribe_to_words
from Models.map_sed_words import map_windows_to_words, seconds_to_samples


def apply_word_level_repair(wave: np.ndarray, sr: int, word_records: list, mode: str = "attenuate", attenuate_db: float = 10.0, crossfade_ms: float = 10.0, preserve_silence_ms: float = 0.0):
    """Apply repair using word-level boundaries. word_records is list of dicts with 'start'/'end' in seconds and 'stutter' bool.

    crossfade_ms: millisecond crossfade applied at edited boundaries to avoid clicks.
    """
    repaired = wave.copy()
    crossfade_samples = max(1, int(round((crossfade_ms / 1000.0) * sr)))

    def apply_fade(seg: np.ndarray, fade_in: int = 0, fade_out: int = 0):
        # linear fade in/out
        out = seg.copy()
        n = len(out)
        if fade_in > 0:
            ramp = np.linspace(0.0, 1.0, min(fade_in, n))
            out[:len(ramp)] = out[:len(ramp)] * ramp
        if fade_out > 0:
            ramp = np.linspace(1.0, 0.0, min(fade_out, n))
            out[-len(ramp):] = out[-len(ramp):] * ramp
        return out

    if mode == 'silence':
        for w in word_records:
            if w.get('stutter'):
                s = seconds_to_samples(w['start'], sr)
                e = seconds_to_samples(w['end'], sr)
                s0 = max(0, s - crossfade_samples)
                e0 = min(len(repaired), e + crossfade_samples)
                # silence with crossfade
                repaired[s0:s] = apply_fade(repaired[s0:s], fade_out=min(crossfade_samples, s - s0))
                repaired[s:e] = 0.0
                repaired[e:e0] = apply_fade(repaired[e:e0], fade_in=min(crossfade_samples, e0 - e))
    elif mode == 'attenuate':
        factor = 10 ** (-abs(attenuate_db) / 20.0)
        for w in word_records:
            if w.get('stutter'):
                s = seconds_to_samples(w['start'], sr)
                e = seconds_to_samples(w['end'], sr)
                s0 = max(0, s - crossfade_samples)
                e0 = min(len(repaired), e + crossfade_samples)
                # fade out, attenuate, fade in
                repaired[s0:s] = apply_fade(repaired[s0:s], fade_out=min(crossfade_samples, s - s0))
                repaired[s:e] = repaired[s:e] * factor
                repaired[e:e0] = apply_fade(repaired[e:e0], fade_in=min(crossfade_samples, e0 - e))
    else:  # remove
        parts = []
        cursor = 0
        preserve_samples = int(round((preserve_silence_ms / 1000.0) * sr)) if preserve_silence_ms and preserve_silence_ms > 0.0 else 0
        for w in word_records:
            s = seconds_to_samples(w['start'], sr)
            e = seconds_to_samples(w['end'], sr)
            if s > cursor:
                parts.append(repaired[cursor:s])
            if w.get('stutter'):
                # either skip (remove) or insert short silence if requested
                if preserve_samples > 0:
                    parts.append(np.zeros(preserve_samples, dtype=repaired.dtype))
                # advance cursor past the stuttered section
                cursor = max(cursor, e)
            else:
                cursor = max(cursor, e)
        if cursor < len(repaired):
            parts.append(repaired[cursor:])

        # now concatenate parts with crossfades between adjacent parts to avoid clicks
        if not parts:
            repaired = np.zeros(0, dtype=repaired.dtype)
        else:
            out_parts = [parts[0]]
            for seg in parts[1:]:
                prev = out_parts.pop()
                if crossfade_samples > 0 and len(prev) >= crossfade_samples and len(seg) >= crossfade_samples:
                    tail = prev[-crossfade_samples:]
                    head = seg[:crossfade_samples]
                    ramp = np.linspace(0.0, 1.0, crossfade_samples)
                    mix = tail * (1.0 - ramp) + head * ramp
                    out_parts.append(np.concatenate([prev[:-crossfade_samples], mix, seg[crossfade_samples:]]))
                else:
                    out_parts.append(np.concatenate([prev, seg]))
            repaired = np.concatenate(out_parts)
    return repaired


def run_pipeline(args):
    inp = Path(args.input_file)
    if not inp.exists():
        raise SystemExit(f"Input not found: {inp}")

    wave, sr = load_audio(inp)

    # determine device: default CPU, but allow overriding via args.device
    device = torch.device('cpu')
    try:
        if hasattr(args, 'device') and args.device:
            dev_req = str(args.device).lower()
            if dev_req.startswith('cuda') and torch is not None and torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
    except Exception:
        device = torch.device('cpu')

    # load SED model
    sed_model = SimpleCNN(n_mels=args.n_mels, n_classes=len(SimpleCNN().fc.bias))
    state = __import__('torch').load(args.model_path, map_location=device)
    sed_model.load_state_dict(state)
    sed_model.to(device)
    sed_model.eval()

    # sliding-window detection (samples)
    stutter_windows = []
    window_infos = []
    # prepare thresholds: either global or per-class
    per_class_thresh = None
    if args.per_class_thresholds:
        try:
            with open(args.per_class_thresholds, 'r', encoding='utf-8') as f:
                j = json.load(f)
                per_class_thresh = j.get('thresholds') if isinstance(j, dict) else j
        except Exception:
            per_class_thresh = None
    if per_class_thresh is None:
        per_class_thresh = [args.threshold] * len(SimpleCNN().fc.bias)

    provisional_flags = []
    window_bounds = []
    for start, end, seg in window_iter(wave, sr, args.win_s, args.hop_s):
        probs = predict_window(sed_model, seg, sr, device, n_mels=args.n_mels)
        # per-class booleans
        class_bools = [bool(p >= t) for p, t in zip(probs, per_class_thresh)]
        is_stutter = any(class_bools)
        window_infos.append({"start_sample": int(start), "end_sample": int(min(end, len(wave))), "probs": probs.tolist(), "class_bools": class_bools, "stutter": is_stutter})
        provisional_flags.append(bool(is_stutter))
        window_bounds.append((int(start), int(min(end, len(wave)))))

    # apply consecutive-window smoothing if requested
    if args.consec_windows and args.consec_windows > 1:
        flags = np.array(provisional_flags, dtype=int)
        k = int(args.consec_windows)
        conv = np.convolve(flags, np.ones(k, dtype=int), mode='same')
        smoothed = conv >= k
    else:
        smoothed = np.array(provisional_flags, dtype=bool)

    for (samp_bounds, info), sm in zip(zip(window_bounds, window_infos), smoothed):
        info['stutter'] = bool(sm)
        if sm:
            stutter_windows.append((samp_bounds[0], samp_bounds[1]))

    # ASR transcription -> words with timestamps (seconds)
    print("Running Whisper ASR (this will download model weights if needed)...")
    asr = transcribe_to_words(str(inp), model_size=args.whisper_model, language=args.language)
    words = asr.get('words', [])
    # optionally try CTC-based refinement for better word boundaries
    if args.use_ctc_align:
        try:
            from Models.ctc_align import ctc_align
            transcript = asr.get('text', '')
            print("Attempting CTC-based alignment refinement (requires extra packages)...")
            refined = ctc_align(str(inp), transcript, wav2vec_model=args.ctc_model)
            if refined:
                print("CTC alignment succeeded; using refined word timestamps")
                words = refined
            else:
                print("CTC alignment not available or failed; keeping Whisper timestamps")
        except Exception as e:
            print("CTC alignment import/exec failed:", e)
    # map SED windows (samples) to words
    word_records = map_windows_to_words(stutter_windows, words, sr, min_overlap_ratio=args.min_overlap_ratio)

    # apply repair using word-level boundaries (with smoothing)
    repaired = apply_word_level_repair(wave, sr, word_records, mode=args.mode, attenuate_db=args.attenuate_db, crossfade_ms=args.crossfade_ms, preserve_silence_ms=args.preserve_silence_ms)

    # save repaired audio
    outp = Path(args.output_file)
    outp.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(outp), repaired, sr)

    # save report
    report = {
        "input": str(inp),
        "output": str(outp),
        "sr": sr,
        "sed_windows": window_infos,
        "asr_text": asr.get('text', ''),
        "word_records": word_records,
    }
    rep_path = outp.with_suffix('.asr_repair.json')
    with open(rep_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f"Wrote repaired audio: {outp}")
    print(f"Wrote report: {rep_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True, help='SED model checkpoint (pth)')
    p.add_argument('--input_file', required=True)
    p.add_argument('--output_file', required=True)
    p.add_argument('--win_s', type=float, default=1.0)
    p.add_argument('--hop_s', type=float, default=0.5)
    p.add_argument('--threshold', type=float, default=0.4)
    p.add_argument('--n_mels', type=int, default=80)
    p.add_argument('--whisper_model', default='small', help='whisper model size: tiny, small, medium')
    p.add_argument('--language', default=None, help='language code for whisper (optional)')
    p.add_argument('--mode', choices=['remove', 'silence', 'attenuate'], default='attenuate')
    p.add_argument('--attenuate_db', type=float, default=10.0)
    p.add_argument('--min_overlap_ratio', type=float, default=0.2)
    p.add_argument('--use_ctc_align', action='store_true', help='try CTC-based alignment refinement (requires extra deps)')
    p.add_argument('--ctc_model', default='facebook/wav2vec2-large-960h', help='Wav2Vec2 model for CTC alignment')
    p.add_argument('--crossfade_ms', type=float, default=10.0, help='millisecond crossfade used when removing/attenuating to avoid clicks')
    p.add_argument('--per_class_thresholds', default=None, help='JSON file with per-class thresholds, list of floats length 5')
    p.add_argument('--consec_windows', type=int, default=1, help='Require N consecutive positive windows to mark a window as stutter (smoothing).')
    p.add_argument('--preserve_silence_ms', type=float, default=0.0, help='When removing stuttered words, replace them with this many milliseconds of silence instead of deleting completely.')
    p.add_argument('--device', default=None, help="device to run SED model on: 'cpu' or 'cuda' (if available)")
    args = p.parse_args()
    run_pipeline(args)


if __name__ == '__main__':
    main()
