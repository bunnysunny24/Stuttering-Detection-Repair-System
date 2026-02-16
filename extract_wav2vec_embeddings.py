"""Extract wav2vec2 embeddings for existing feature files.

Usage:
  python Models/extract_wav2vec_embeddings.py --workers 4

This script looks for .npz files under `datasets/features/{train,val}` and
for each file tries to find the corresponding audio in
`datasets/clips/stuttering-clips/clips/{name}.wav|.flac`. If found, it
computes a fixed-length embedding using HuggingFace Wav2Vec2 and writes
an `embedding` array back into the .npz (overwriting file).
"""
import os
import argparse
import glob
import numpy as np
import soundfile as sf
import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model


def find_audio_path(stem):
    base = os.path.join('datasets', 'clips', 'stuttering-clips', 'clips')
    for ext in ('.wav', '.flac'):
        p = os.path.join(base, f"{stem}{ext}")
        if os.path.exists(p):
            return p
    return None


def process_file(npz_path, processor, model, device='cpu'):
    stem = os.path.splitext(os.path.basename(npz_path))[0]
    audio_path = find_audio_path(stem)
    if audio_path is None:
        return False, f"audio not found for {stem}"

    try:
        wav, sr = librosa.load(audio_path, sr=16000)
        inputs = processor(wav, sampling_rate=16000, return_tensors='pt', padding=True)
        input_values = inputs.input_values.to(device)
        with torch.no_grad():
            outputs = model(input_values)
        # Mean pooling across time to get fixed-size vector
        emb = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()

        # Load existing npz and write back with embedding
        data = dict(np.load(npz_path))
        data['embedding'] = emb
        np.savez_compressed(npz_path, **data)
        return True, f"embedded {stem}"
    except Exception as e:
        return False, str(e)


def main(worker_count=1):
    print("Loading wav2vec2 model (this may download weights)...")
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
    model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    files = glob.glob(os.path.join('datasets', 'features', '**', '*.npz'), recursive=True)
    print(f"Found {len(files)} feature files")

    success = 0
    for i, npz_path in enumerate(files, 1):
        ok, msg = process_file(npz_path, processor, model, device=device)
        if ok:
            success += 1
        if i % 500 == 0:
            print(f"Processed {i}/{len(files)} - successes: {success}")

    print(f"Done. Embedded {success}/{len(files)} files.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    main(worker_count=args.workers)
