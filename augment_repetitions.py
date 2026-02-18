"""
Create synthetic repetition augmentations for rare 'Word Repetition' class.

This script finds short clips in `datasets/clips` or features and creates
augmented NPZs by duplicating short segments to simulate repeated words.

Usage:
  python Models/augment_repetitions.py --src datasets/clips/stuttering-clips/clips --out datasets/features/train_aug --n 200
"""
import argparse
from pathlib import Path
import soundfile as sf
import numpy as np
import os

def augment_file(wav_path, out_path, sr=16000, repeat_ms=150):
    audio, r = sf.read(str(wav_path))
    if r != sr:
        # simple resample using numpy (fast fallback)
        import math
        from scipy.signal import resample
        num = int(len(audio) * sr / r)
        audio = resample(audio, num)

    if len(audio) < sr // 2:
        return False

    # pick random short segment
    seg_len = int(sr * (repeat_ms / 1000.0))
    start = np.random.randint(0, max(1, len(audio) - seg_len))
    segment = audio[start:start+seg_len]

    # insert duplicate after it
    new_audio = np.concatenate([audio[:start+seg_len], segment, audio[start+seg_len:]])

    # write out
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), new_audio, sr)
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--n', type=int, default=200)
    args = parser.parse_args()

    src = Path(args.src)
    outdir = Path(args.out)
    wavs = list(src.glob('*.wav'))
    if len(wavs) == 0:
        print('No wav files found in', src)
        return

    idx = 0
    for i in range(args.n):
        src_f = np.random.choice(wavs)
        out_f = outdir / f"aug_rep_{i}_{src_f.stem}.wav"
        ok = augment_file(src_f, out_f)
        if ok:
            idx += 1

    print(f"Wrote {idx} augmented files to {outdir}")

if __name__ == '__main__':
    main()
