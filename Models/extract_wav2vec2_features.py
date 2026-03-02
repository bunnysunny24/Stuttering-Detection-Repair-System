"""
Extract wav2vec2-base embeddings from audio clips for stuttering detection.

This replaces mel-spectrogram features with 768-dim pretrained speech 
representations from wav2vec2-base (trained on 960h LibriSpeech).
These embeddings capture deep speech patterns that hand-crafted features cannot.

Usage:
    python Models/extract_wav2vec2_features.py
    python Models/extract_wav2vec2_features.py --model facebook/hubert-base-ls960
    python Models/extract_wav2vec2_features.py --batch-size 16 --num-workers 4

Output: datasets/features_w2v/{train,val}/*.npz  (each with 'embedding' + 'labels')
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm

# ---- CPU Optimization: use ALL cores for PyTorch ops ----
N_CORES = os.cpu_count() or 4
torch.set_num_threads(N_CORES)
torch.set_num_interop_threads(min(N_CORES, 4))

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor


def load_label_map(features_dir: str):
    """Build filename -> labels mapping from existing NPZ files."""
    label_map = {}
    for split in ('train', 'val'):
        split_dir = Path(features_dir) / split
        if not split_dir.exists():
            continue
        for f in split_dir.glob('*.npz'):
            try:
                data = np.load(f)
                labels = data.get('labels')
                if labels is not None:
                    # Key = stem (e.g., "FluencyBank_010_0")
                    label_map[f.stem] = np.array(labels, dtype=np.float32)
            except Exception:
                continue
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


def extract_embeddings(args):
    """Main extraction loop."""
    print(f"=== wav2vec2 Feature Extraction ===")
    print(f"Model: {args.model}")
    print(f"Clips dir: {args.clips_dir}")
    print(f"Output dir: {args.output_dir}")
    
    # Load pretrained model
    print("Loading pretrained model...")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.model)
    model = Wav2Vec2Model.from_pretrained(args.model)
    model.eval()
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Load label map from existing features
    print("Loading labels from existing features...")
    label_map = load_label_map(args.features_dir)
    print(f"Found labels for {len(label_map)} files")
    
    # Get train/val split assignments
    splits = get_split_files(args.features_dir)
    print(f"Split assignments: {sum(1 for v in splits.values() if v=='train')} train, {sum(1 for v in splits.values() if v=='val')} val")
    
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
    
    # Extract with batched inference
    target_sr = 16000
    errors = 0
    t_start = time.time()
    batch_size = args.batch_size
    
    print(f"CPU threads: {torch.get_num_threads()} (cores: {N_CORES})")
    print(f"Batch size: {batch_size}")
    
    # Process in batches for better CPU utilization
    total = len(to_process_filtered)
    pbar = tqdm(total=total, desc="Extracting wav2vec2 embeddings")
    
    for batch_start in range(0, total, batch_size):
        batch_items = to_process_filtered[batch_start:batch_start + batch_size]
        
        # Load and preprocess all audio in this batch
        batch_waveforms = []
        batch_meta = []  # (stem, labels, split) for each successful load
        
        for stem, audio_path, labels, split in batch_items:
            try:
                waveform, sr = torchaudio.load(str(audio_path))
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                if sr != target_sr:
                    resampler = torchaudio.transforms.Resample(sr, target_sr)
                    waveform = resampler(waveform)
                batch_waveforms.append(waveform.squeeze().numpy())
                batch_meta.append((stem, labels, split))
            except Exception as e:
                errors += 1
                if errors <= 10:
                    print(f"\nError loading {stem}: {e}")
        
        if not batch_waveforms:
            pbar.update(len(batch_items))
            continue
        
        try:
            # Batch process through wav2vec2
            inputs = processor(
                batch_waveforms,
                sampling_rate=target_sr,
                return_tensors='pt',
                padding=True
            )
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Pool each sample in the batch
            hidden = outputs.last_hidden_state  # (B, T, 768)
            
            # Handle attention mask for proper pooling (ignore padding)
            if 'attention_mask' in inputs:
                mask = inputs['attention_mask'].unsqueeze(-1).float()  # (B, T, 1)
                lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1, 1)
                mean_pool = (hidden * mask).sum(dim=1) / lengths.squeeze(1)  # (B, 768)
                var = ((hidden - mean_pool.unsqueeze(1)) ** 2 * mask).sum(dim=1) / lengths.squeeze(1)
                std_pool = var.clamp(min=1e-8).sqrt()  # (B, 768)
            else:
                mean_pool = hidden.mean(dim=1)  # (B, 768)
                std_pool = hidden.std(dim=1)    # (B, 768)
            
            mean_np = mean_pool.numpy()  # (B, 768)
            std_np = std_pool.numpy()    # (B, 768)
            
            # Save each sample
            for i, (stem, labels, split) in enumerate(batch_meta):
                embedding = np.concatenate([mean_np[i], std_np[i]]).astype(np.float32)
                out_path = output_path / split / f"{stem}.npz"
                np.savez_compressed(str(out_path), embedding=embedding, labels=labels)
        
        except Exception as e:
            # Fallback: process one-by-one if batch fails
            for i, waveform_np in enumerate(batch_waveforms):
                try:
                    inp = processor(waveform_np, sampling_rate=target_sr, return_tensors='pt', padding=True)
                    with torch.no_grad():
                        out = model(**inp)
                    h = out.last_hidden_state
                    m = h.mean(dim=1).squeeze().numpy()
                    s = h.std(dim=1).squeeze().numpy()
                    emb = np.concatenate([m, s]).astype(np.float32)
                    stem, labels, split = batch_meta[i]
                    np.savez_compressed(str(output_path / split / f"{stem}.npz"), embedding=emb, labels=labels)
                except Exception as e2:
                    errors += 1
                    if errors <= 10:
                        print(f"\nError processing {batch_meta[i][0]}: {e2}")
        
        pbar.update(len(batch_items))
        
        # Update ETA
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
    print(f"\n=== Extraction Complete ===")
    print(f"Time: {elapsed_total/3600:.1f} hours")
    print(f"Errors: {errors}")
    
    # Count outputs
    n_train = len(list((output_path / 'train').glob('*.npz')))
    n_val = len(list((output_path / 'val').glob('*.npz')))
    print(f"Output: {n_train} train, {n_val} val in {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract wav2vec2 embeddings')
    parser.add_argument('--model', type=str, default='facebook/wav2vec2-base',
                        help='HuggingFace model name (default: facebook/wav2vec2-base)')
    parser.add_argument('--clips-dir', type=str, default='datasets/clips/stuttering-clips/clips',
                        help='Directory containing audio WAV files')
    parser.add_argument('--features-dir', type=str, default='datasets/features',
                        help='Existing features dir (for labels and split assignments)')
    parser.add_argument('--output-dir', type=str, default='datasets/features_w2v',
                        help='Output directory for wav2vec2 features')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for inference (16 is good for 32GB+ RAM)')
    
    args = parser.parse_args()
    extract_embeddings(args)
