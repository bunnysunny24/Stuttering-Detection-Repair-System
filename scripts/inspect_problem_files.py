"""Inspect problematic audio samples: generate waveform + spectrogram PNGs.

Usage:
  # after collecting problematic WAVs into datasets/problematic_samples
  conda activate agni
  python scripts/inspect_problem_files.py --input-dir datasets/problematic_samples --out-dir output/problem_inspect --max 50

Creates PNGs named <stem>_wave.png and <stem>_spec.png in out-dir.
"""
import argparse
import os
from pathlib import Path
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

try:
    import librosa
    import librosa.display
    USE_LIBROSA = True
except Exception:
    USE_LIBROSA = False


def plot_waveform(y, sr, out_path):
    plt.figure(figsize=(8,2))
    times = np.arange(len(y)) / float(sr)
    plt.fill_between(times, y, color='gray')
    plt.xlim(0, times[-1])
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_spectrogram(y, sr, out_path, n_mels=80, n_fft=512, hop_length=160):
    plt.figure(figsize=(8,3))
    if USE_LIBROSA:
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        S_db = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    else:
        # simple STFT magnitude fallback
        from scipy import signal
        f, t, Zxx = signal.stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft-hop_length)
        mag = np.abs(Zxx)
        plt.pcolormesh(t, f, 20 * np.log10(np.maximum(mag, 1e-10)), shading='gouraud')
        plt.ylabel('Freq (Hz)')
        plt.xlabel('Time (s)')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--max', type=int, default=100)
    args = p.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted([p for p in in_dir.glob('*.wav')])[:args.max]
    if not wavs:
        print('No WAV files found in', in_dir)
        return

    for w in wavs:
        try:
            y, sr = sf.read(str(w))
            if y.ndim > 1:
                y = np.mean(y, axis=1)
        except Exception as e:
            print('Failed to read', w, e)
            continue

        stem = w.stem
        wave_out = out_dir / f"{stem}_wave.png"
        spec_out = out_dir / f"{stem}_spec.png"

        try:
            plot_waveform(y, sr, wave_out)
            plot_spectrogram(y, sr, spec_out)
            print('Saved', wave_out, spec_out)
        except Exception as e:
            print('Failed plotting', w, e)

if __name__ == '__main__':
    main()
