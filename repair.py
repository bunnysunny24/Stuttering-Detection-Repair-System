"""Simple stutter repair pipeline (naive) using the SEP-28k clip classifier.

This script performs a sliding-window pass over an input audio file, uses the
trained `SimpleCNN` to predict whether each window contains a stutter event,
and then applies a simple repair operation to windows flagged as stutter.

WARNING: This is a heuristic proof-of-concept. Automatic "fixing" of stuttered
speech is a complex task; the script below implements simple removal/attenuation
or silence insertion. For production-quality editing you'd use a dedicated
speech-inpainting or TTS-based regeneration pipeline.

Usage example:
  . .venv_models\Scripts\Activate.ps1
  python -m Models.repair --model_path Models/tmp/cnn_best.pth --input_file datasets/clips/stuttering-clips/clips/HeStutters_11_108.wav \
    --output_file Models/tmp/HeStutters_11_108_repaired.wav --win_s 1.0 --hop_s 0.5 --threshold 0.4 --mode remove

Modes:
  remove    - excise detected stutter windows and stitch audio back together (with tiny crossfade)
  silence   - replace stutter windows with silence of same duration
  attenuate - reduce volume (dB) of stutter windows (safer)

Note: the classifier was trained at clip-level; sliding-window detection is approximate.
"""
from pathlib import Path
import argparse
import numpy as np
import soundfile as sf
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:
    import torch
except Exception:
    torch = None

from model_cnn import SimpleCNN
from features import waveform_to_logmel


def load_audio(path: Path):
    data, sr = sf.read(str(path), dtype="float32")
    arr = np.asarray(data)
    if arr.ndim == 1:
        arr = np.expand_dims(arr, 0)
    else:
        arr = arr.T
    # convert to mono (mean)
    mono = np.mean(arr, axis=0)
    return mono.astype(np.float32), sr


def window_iter(wave: np.ndarray, sr: int, win_s: float, hop_s: float):
    win_n = int(round(win_s * sr))
    hop_n = int(round(hop_s * sr))
    if win_n <= 0:
        raise ValueError("win_s must be > 0")
    for start in range(0, max(1, len(wave) - 1), hop_n):
        end = start + win_n
        seg = wave[start:end]
        yield start, end, seg


def predict_window(model, seg: np.ndarray, sr: int, device, n_mels=80):
    # seg: 1D numpy (samples,) -> make (1, n_samples)
    wav = np.expand_dims(seg, 0)
    spec = waveform_to_logmel(wav, sr=sr, n_mels=n_mels)
    # spec shape (n_mels, T)
    # Ensure minimum time frames so the CNN's pooling doesn't reduce tensors to zero
    min_frames = 16
    t = spec.shape[1]
    if t < min_frames:
        pad = np.zeros((spec.shape[0], min_frames - t), dtype=spec.dtype)
        spec = np.concatenate([spec, pad], axis=1)
    X = np.zeros((1, 1, spec.shape[0], spec.shape[1]), dtype=np.float32)
    X[0, 0, :, : spec.shape[1]] = spec
    X_t = torch.from_numpy(X).to(device)
    with torch.no_grad():
        logits = model(X_t)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    return probs


def _cosine_fade(n):
    # cosine window from 1 -> 0
    t = np.linspace(0, np.pi, n)
    return 0.5 * (1 + np.cos(t))


def crossfade_concat(parts, sr, fade_ms=10, preserve_ms=20):
    """
    Concatenate parts with a short preserved silence between clips and
    cosine crossfades to avoid clicks.

    parts: list of 1D numpy arrays
    fade_ms: length of crossfade in milliseconds
    preserve_ms: length of short silence to insert between edited joins (ms)
    """
    if not parts:
        return np.zeros(0, dtype=np.float32)
    fade = int(sr * fade_ms / 1000)
    preserve = int(sr * preserve_ms / 1000)
    out = parts[0].astype(np.float32)
    for p in parts[1:]:
        # insert short silence to preserve natural pacing
        if preserve > 0:
            sil = np.zeros(preserve, dtype=np.float32)
            # crossfade between out tail and silence, then silence->p
            # first join out + silence
            if fade > 0 and len(out) >= fade and len(sil) >= fade:
                win = _cosine_fade(fade)
                left_tail = out[-fade:]
                right_head = sil[:fade]
                mix = left_tail * win + right_head * (1 - win)
                out = np.concatenate([out[:-fade], mix, sil[fade:]])
            else:
                out = np.concatenate([out, sil])
            # then crossfade silence -> p
            if fade > 0 and len(out) >= fade and len(p) >= fade:
                left_tail = out[-fade:]
                right_head = p[:fade]
                win = _cosine_fade(fade)
                mix = left_tail * win + right_head * (1 - win)
                out = np.concatenate([out[:-fade], mix, p[fade:]])
            else:
                out = np.concatenate([out, p])
        else:
            # direct crossfade between out and p
            if fade > 0 and len(out) >= fade and len(p) >= fade:
                left_tail = out[-fade:]
                right_head = p[:fade]
                win = _cosine_fade(fade)
                mix = left_tail * win + right_head * (1 - win)
                out = np.concatenate([out[:-fade], mix, p[fade:]])
            else:
                out = np.concatenate([out, p])
    return out


def repair_audio(args):
    inp = Path(args.input_file)
    outp = Path(args.output_file)
    wave, sr = load_audio(inp)
    device = torch.device('cpu')

    # load model
    model = SimpleCNN(n_mels=args.n_mels, n_classes=len(SimpleCNN().fc.bias))
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    stutter_windows = []
    window_infos = []
    for start, end, seg in window_iter(wave, sr, args.win_s, args.hop_s):
        probs = predict_window(model, seg, sr, device, n_mels=args.n_mels)
        is_stutter = bool((probs >= args.threshold).any())
        window_infos.append({"start": int(start), "end": int(min(end, len(wave))), "probs": probs.tolist(), "stutter": is_stutter})
        if is_stutter:
            stutter_windows.append((start, min(end, len(wave))))

    # Build repaired audio
    if args.mode == 'silence':
        repaired = wave.copy()
        for s, e in stutter_windows:
            repaired[s:e] = 0.0
    elif args.mode == 'attenuate':
        factor = 10 ** (-abs(args.attenuate_db) / 20.0)
        repaired = wave.copy()
        for s, e in stutter_windows:
            repaired[s:e] = repaired[s:e] * factor
    else:  # remove
        parts = []
        cursor = 0
        for s, e in stutter_windows:
            if s > cursor:
                parts.append(wave[cursor:s])
            cursor = max(cursor, e)
        if cursor < len(wave):
            parts.append(wave[cursor:])
        repaired = crossfade_concat(parts, sr, fade_ms=10, preserve_ms=args.preserve_silence_ms)

    outp.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(outp), repaired, sr)

    # save diagnostics
    diag = Path(args.output_file).with_suffix('.repair.json')
    diag_obj = {"input": str(inp), "output": str(outp), "sr": sr, "win_s": args.win_s, "hop_s": args.hop_s, "threshold": args.threshold, "mode": args.mode, "windows": window_infos}

    # generate spectrogram diagnostic PNG overlaying detection windows
    try:
        spec = waveform_to_logmel(np.expand_dims(wave, 0), sr=sr, n_mels=args.n_mels)
        # hop length used in features: default 160 samples
        hop_length = 160
        times = np.arange(spec.shape[1]) * (hop_length / sr)
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(spec, origin='lower', aspect='auto', cmap='magma', extent=[times[0], times[-1] if len(times)>0 else 0, 0, args.n_mels])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Mel band')
        # overlay windows
        for w in window_infos:
            s = w['start'] / sr
            e = w['end'] / sr
            rect = Rectangle((s, 0), e - s, args.n_mels, facecolor='none', edgecolor='cyan', linewidth=1.0, alpha=0.6)
            ax.add_patch(rect)
        fig.colorbar(im, ax=ax, format='%+2.0f dB')
        pngp = Path(args.output_file).with_suffix('.repair.png')
        fig.tight_layout()
        fig.savefig(str(pngp), dpi=150)
        plt.close(fig)
        diag_obj['spectrogram_png'] = str(pngp)
    except Exception:
        diag_obj['spectrogram_png'] = None

    with open(diag, 'w', encoding='utf-8') as f:
        json.dump(diag_obj, f, indent=2)

    print(f"Wrote repaired audio to {outp}. Diagnostic JSON: {diag}")
    if diag_obj.get('spectrogram_png'):
        print(f"Spectrogram diagnostic: {diag_obj['spectrogram_png']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--win_s', type=float, default=1.0)
    parser.add_argument('--hop_s', type=float, default=0.5)
    parser.add_argument('--threshold', type=float, default=0.4)
    parser.add_argument('--mode', choices=['remove', 'silence', 'attenuate'], default='remove')
    parser.add_argument('--attenuate_db', type=float, default=10.0, help='dB to reduce when mode=attenuate')
    parser.add_argument('--n_mels', type=int, default=80)
    parser.add_argument('--preserve_silence_ms', type=int, default=20, help='ms of short silence to preserve between removed segments')
    args = parser.parse_args()
    repair_audio(args)


if __name__ == '__main__':
    main()
