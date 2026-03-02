"""
Extract TEMPORAL frame-level wav2vec2-base features from audio clips.

Instead of collapsing each clip to a single 1536-dim vector (mean+std),
this saves the FULL temporal sequence (768, T) per clip, preserving the
frame-by-frame speech representations that are critical for detecting
temporal stuttering patterns (repetitions, prolongations, blocks).

KEY UPGRADE: Multi-layer weighted average
  By default, extracts a weighted average of the top 6 encoder layers
  (layers 7-12) rather than just the last hidden state. Different layers
  capture different information:
    - Lower layers: acoustic/phonetic features (important for prolongation)
    - Middle layers: phoneme patterns (important for sound repetition)
    - Upper layers: linguistic features (important for word repetition)
  The weighted average captures ALL these levels.

Output: datasets/features_w2v_temporal/{train,val}/*.npz
  Each NPZ contains:
    'temporal_embedding': np.float32 array of shape (768, T)
    'labels': np.float32 array of shape (5,)

Usage:
    python Models/extract_wav2vec2_temporal.py
    python Models/extract_wav2vec2_temporal.py --max-frames 200 --batch-size 8
    python Models/extract_wav2vec2_temporal.py --layer-mode last  # last layer only
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import scipy.io.wavfile as wavfile
from scipy.signal import resample_poly
from pathlib import Path
from tqdm import tqdm

# ---- CPU Optimization: use ALL cores for PyTorch ops ----
N_CORES = os.cpu_count() or 4
torch.set_num_threads(N_CORES)
torch.set_num_interop_threads(N_CORES)
# Set env vars for all math backends
os.environ.setdefault('OMP_NUM_THREADS', str(N_CORES))
os.environ.setdefault('MKL_NUM_THREADS', str(N_CORES))

sys.path.insert(0, str(Path(__file__).parent))

from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor


def load_label_map(features_dir: str):
    """Build filename -> labels mapping from existing NPZ files (cached).

    First call scans all NPZ files and saves a compact cache file.
    Subsequent calls load from cache in <1 second.
    """
    cache_path = Path(features_dir) / '_label_cache.pkl'

    # Try loading from cache first
    if cache_path.exists():
        import pickle
        with open(cache_path, 'rb') as fh:
            label_map = pickle.load(fh)
        print(f"  Loaded {len(label_map)} labels from cache")
        return label_map

    # Build from individual NPZ files (slow first run)
    label_map = {}
    for split in ('train', 'val'):
        split_dir = Path(features_dir) / split
        if not split_dir.exists():
            continue
        files = sorted(split_dir.glob('*.npz'))
        print(f"  Scanning {len(files)} {split} files for labels...")
        for i, f in enumerate(files):
            try:
                data = np.load(f)
                if 'labels' in data.files:
                    label_map[f.stem] = data['labels'].astype(np.float32)
                data.close()
            except Exception:
                continue
            if (i + 1) % 5000 == 0:
                print(f"    ... {i+1}/{len(files)}")

    # Save cache for next time
    import pickle
    with open(cache_path, 'wb') as fh:
        pickle.dump(label_map, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Cached {len(label_map)} labels to {cache_path}")

    return label_map


def get_split_files(features_dir: str):
    """Get train/val split assignments from existing feature directory."""
    splits = {}
    for split in ('train', 'val'):
        split_dir = Path(features_dir) / split
        if split_dir.exists():
            stems = {f.stem for f in split_dir.glob('*.npz')}
            for stem in stems:
                splits[stem] = split
    return splits


def find_audio_files(clips_dir: str):
    """Find all WAV files in clips directory (recursive)."""
    clips_path = Path(clips_dir)
    audio_files = {}
    for ext in ('*.wav', '*.WAV', '*.mp3', '*.flac'):
        for f in clips_path.rglob(ext):
            audio_files[f.stem] = f
    return audio_files


# ---- Multi-layer feature combination ----
# Optimal weights for stuttering detection based on SSL probing literature.
# Upper layers (10-12) get higher weight for semantic/linguistic features,
# middle layers (7-9) for phoneme-level patterns.
LAYER_WEIGHTS_DEFAULT = {
    7: 0.10,   # acoustic patterns
    8: 0.12,   # phoneme transitions
    9: 0.15,   # syllable patterns
    10: 0.18,  # word-level features
    11: 0.20,  # contextual features
    12: 0.25,  # highest-level representations (last layer)
}
# Normalise so they sum to 1.0
_ws = sum(LAYER_WEIGHTS_DEFAULT.values())
LAYER_WEIGHTS_DEFAULT = {k: v / _ws for k, v in LAYER_WEIGHTS_DEFAULT.items()}


def compute_multilayer_hidden(outputs, layer_weights):
    """Compute weighted average of selected hidden layers.

    Args:
        outputs: model output with output_hidden_states=True
            outputs.hidden_states is a tuple of (num_layers+1,) tensors,
            each of shape (B, T, 768).
            Index 0 = CNN feature extractor output
            Indices 1-12 = transformer encoder layers 1-12
        layer_weights: dict mapping layer_index -> weight

    Returns:
        weighted_hidden: (B, T, 768) weighted average
    """
    hidden_states = outputs.hidden_states  # tuple of 13 tensors
    result = None
    for layer_idx, weight in layer_weights.items():
        h = hidden_states[layer_idx]  # (B, T, 768)
        if result is None:
            result = weight * h
        else:
            result = result + weight * h
    return result


def extract_temporal_embeddings(args):
    """Main extraction loop — saves frame-level temporal features."""
    layer_mode = args.layer_mode

    print(f"=== wav2vec2 TEMPORAL Feature Extraction ===")
    print(f"Model: {args.model}")
    print(f"Layer mode: {layer_mode}")
    print(f"Max frames: {args.max_frames}")
    print(f"Clips dir: {args.clips_dir}")
    print(f"Output dir: {args.output_dir}")

    # Load pretrained model
    print("Loading pretrained model...")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.model)
    model = Wav2Vec2Model.from_pretrained(args.model)
    model.eval()

    # For multi-layer mode, we need output_hidden_states=True
    use_multilayer = (layer_mode == 'weighted-avg')
    if use_multilayer:
        model.config.output_hidden_states = True
        layer_weights = LAYER_WEIGHTS_DEFAULT
        print(f"Using weighted average of layers: {list(layer_weights.keys())}")
        print(f"  Weights: {', '.join(f'L{k}={v:.2f}' for k, v in layer_weights.items())}")
    else:
        layer_weights = None
        print("Using last hidden layer only")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Load label map from existing features
    print("Loading labels from existing features...")
    label_map = load_label_map(args.features_dir)
    print(f"Found labels for {len(label_map)} files")

    # Get train/val split assignments
    splits = get_split_files(args.features_dir)
    print(f"Split assignments: {sum(1 for v in splits.values() if v=='train')} train, "
          f"{sum(1 for v in splits.values() if v=='val')} val")

    # Find audio files
    print("Scanning for audio files...")
    audio_files = find_audio_files(args.clips_dir)
    print(f"Found {len(audio_files)} audio files")

    # Match: only process files that have both labels AND split assignment
    to_process = []
    for stem in label_map:
        if stem in audio_files and stem in splits:
            to_process.append((stem, audio_files[stem], label_map[stem], splits[stem]))

    print(f"Files to process: {len(to_process)}")
    if len(to_process) == 0:
        print("ERROR: No matching files found! Check paths.")
        return

    # Create output directories
    output_path = Path(args.output_dir)
    (output_path / 'train').mkdir(parents=True, exist_ok=True)
    (output_path / 'val').mkdir(parents=True, exist_ok=True)

    # Skip already extracted
    existing_train = {f.stem for f in (output_path / 'train').glob('*.npz')}
    existing_val = {f.stem for f in (output_path / 'val').glob('*.npz')}
    existing = existing_train | existing_val

    to_process_filtered = [(s, a, l, sp) for s, a, l, sp in to_process if s not in existing]
    print(f"Already extracted: {len(existing)}, remaining: {len(to_process_filtered)}")

    if len(to_process_filtered) == 0:
        print("All files already extracted!")
        return

    target_sr = 16000
    max_frames = args.max_frames
    errors = 0
    t_start = time.time()
    batch_size = args.batch_size
    frame_lengths = []

    print(f"CPU threads: {torch.get_num_threads()} (cores: {N_CORES})")
    print(f"Batch size: {batch_size}")

    total = len(to_process_filtered)
    pbar = tqdm(total=total, desc="Extracting temporal wav2vec2 features")

    for batch_start in range(0, total, batch_size):
        batch_items = to_process_filtered[batch_start:batch_start + batch_size]

        # Load and preprocess audio
        batch_waveforms = []
        batch_meta = []

        for stem, audio_path, labels, split in batch_items:
            try:
                sr, data = wavfile.read(str(audio_path))
                # Convert to float32 in [-1, 1]
                if data.dtype == np.int16:
                    data = data.astype(np.float32) / 32768.0
                elif data.dtype == np.int32:
                    data = data.astype(np.float32) / 2147483648.0
                elif data.dtype == np.float64:
                    data = data.astype(np.float32)
                # Mono
                if data.ndim > 1:
                    data = data.mean(axis=1)
                # Resample if needed
                if sr != target_sr:
                    from math import gcd
                    g = gcd(target_sr, sr)
                    data = resample_poly(data, target_sr // g, sr // g).astype(np.float32)
                batch_waveforms.append(data)
                batch_meta.append((stem, labels, split))
            except Exception as e:
                errors += 1
                if errors <= 10:
                    print(f"\nError loading {stem}: {e}")

        if not batch_waveforms:
            pbar.update(len(batch_items))
            continue

        try:
            inputs = processor(
                batch_waveforms,
                sampling_rate=target_sr,
                return_tensors='pt',
                padding=True
            )

            with torch.no_grad():
                outputs = model(**inputs)

            # Choose how to get hidden states
            if use_multilayer:
                hidden = compute_multilayer_hidden(outputs, layer_weights)
            else:
                hidden = outputs.last_hidden_state  # (B, T, 768)

            # Compute real (unpadded) frame lengths
            if 'attention_mask' in inputs:
                attn_mask = inputs['attention_mask']
                T_out = hidden.shape[1]
                T_in = attn_mask.shape[1]
                real_lengths = []
                for b in range(attn_mask.shape[0]):
                    real_input_len = attn_mask[b].sum().item()
                    approx_out_len = max(1, int(real_input_len / (T_in / T_out + 1e-8)))
                    approx_out_len = min(approx_out_len, T_out)
                    real_lengths.append(approx_out_len)
            else:
                real_lengths = [hidden.shape[1]] * hidden.shape[0]

            hidden_np = hidden.numpy()  # (B, T, 768)

            # Save each sample
            for i, (stem, labels, split) in enumerate(batch_meta):
                real_T = real_lengths[i]
                frames = hidden_np[i, :real_T, :]  # (real_T, 768)

                if frames.shape[0] > max_frames:
                    frames = frames[:max_frames, :]

                temporal_embedding = frames.T.astype(np.float32)  # (768, T)
                frame_lengths.append(temporal_embedding.shape[1])

                out_path = output_path / split / f"{stem}.npz"
                np.savez_compressed(
                    str(out_path),
                    temporal_embedding=temporal_embedding,
                    labels=labels
                )

        except Exception as e:
            # Fallback: process one-by-one
            for i, waveform_np in enumerate(batch_waveforms):
                try:
                    inp = processor(
                        waveform_np, sampling_rate=target_sr,
                        return_tensors='pt', padding=True
                    )
                    with torch.no_grad():
                        out = model(**inp)

                    if use_multilayer:
                        h = compute_multilayer_hidden(out, layer_weights).squeeze(0).numpy()
                    else:
                        h = out.last_hidden_state.squeeze(0).numpy()

                    if h.shape[0] > max_frames:
                        h = h[:max_frames, :]
                    temporal_emb = h.T.astype(np.float32)
                    frame_lengths.append(temporal_emb.shape[1])

                    stem, labels, split = batch_meta[i]
                    np.savez_compressed(
                        str(output_path / split / f"{stem}.npz"),
                        temporal_embedding=temporal_emb,
                        labels=labels
                    )
                except Exception as e2:
                    errors += 1
                    if errors <= 10:
                        print(f"\nError processing {batch_meta[i][0]}: {e2}")

        pbar.update(len(batch_items))

        elapsed = time.time() - t_start
        done = pbar.n
        if done > 0:
            eta_seconds = (elapsed / done) * (total - done)
            pbar.set_postfix({
                'ETA': f'{eta_seconds/3600:.1f}h',
                'errors': errors
            })

    pbar.close()

    elapsed_total = time.time() - t_start
    print(f"\n=== Temporal Extraction Complete ===")
    print(f"Time: {elapsed_total/3600:.2f} hours ({elapsed_total:.0f}s)")
    print(f"Errors: {errors}")
    print(f"Layer mode: {layer_mode}")

    if frame_lengths:
        fl = np.array(frame_lengths)
        print(f"Frame lengths: min={fl.min()}, max={fl.max()}, "
              f"mean={fl.mean():.1f}, median={np.median(fl):.1f}")

    n_train = len(list((output_path / 'train').glob('*.npz')))
    n_val = len(list((output_path / 'val').glob('*.npz')))
    print(f"Output: {n_train} train, {n_val} val in {args.output_dir}")

    total_bytes = sum(
        f.stat().st_size
        for split in ('train', 'val')
        for f in (output_path / split).glob('*.npz')
    )
    print(f"Total storage: {total_bytes / (1024**3):.2f} GB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract temporal wav2vec2 features')
    parser.add_argument('--model', type=str, default='facebook/wav2vec2-base',
                        help='HuggingFace model name')
    parser.add_argument('--clips-dir', type=str,
                        default='datasets/clips/stuttering-clips/clips',
                        help='Directory containing audio WAV files')
    parser.add_argument('--features-dir', type=str, default='datasets/features',
                        help='Existing features dir (for labels and split assignments)')
    parser.add_argument('--output-dir', type=str,
                        default='datasets/features_w2v_temporal',
                        help='Output directory for temporal features')
    parser.add_argument('--max-frames', type=int, default=200,
                        help='Max temporal frames per clip (200 ~ 4s at 50fps)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size (lower for CPU)')
    parser.add_argument('--layer-mode', type=str, default='weighted-avg',
                        choices=['last', 'weighted-avg'],
                        help='How to combine layers: last=last hidden state only, '
                             'weighted-avg=weighted average of layers 7-12 (recommended)')

    args = parser.parse_args()
    extract_temporal_embeddings(args)
