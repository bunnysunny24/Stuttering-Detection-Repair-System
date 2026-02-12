#!/usr/bin/env python3
"""
Preprocess audio data: Extract features and create train/val splits.
Outputs .npz files for training.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import librosa

# Add Models to path
sys.path.insert(0, str(Path(__file__).parent))

from features import waveform_to_logmel

# ============================================================================
# Configuration
# ============================================================================

CLIPS_DIR = Path(__file__).parent.parent / 'datasets' / 'clips' / 'stuttering-clips' / 'clips'
LABELS_DIR = Path(__file__).parent.parent / 'datasets'
OUTPUT_DIR = Path(__file__).parent.parent / 'datasets' / 'features'
SR = 16000
N_MELS = 80
MAX_FRAMES = 256  # ~4 seconds at 16kHz with hop_length=160

# Label columns (5 stutter classes)
LABEL_COLS = ['Prolongation', 'Block', 'SoundRep', 'WordRep', 'Interjection']

# ============================================================================
# Functions
# ============================================================================

def load_labels():
    """Load and merge labels from both datasets."""
    labels_list = []
    
    # Sep-28k labels
    sep_csv = LABELS_DIR / 'SEP-28k_labels.csv'
    if sep_csv.exists():
        df_sep = pd.read_csv(sep_csv)
        print(f"Loaded {len(df_sep)} labels from SEP-28k")
        labels_list.append(df_sep)
    
    # FluencyBank labels
    fb_csv = LABELS_DIR / 'fluencybank_labels.csv'
    if fb_csv.exists():
        df_fb = pd.read_csv(fb_csv)
        print(f"Loaded {len(df_fb)} labels from FluencyBank")
        labels_list.append(df_fb)
    
    if not labels_list:
        raise FileNotFoundError("No label CSVs found!")
    
    df = pd.concat(labels_list, ignore_index=True)
    print(f"Total labels: {len(df)}")
    return df

def get_label_vector(row, label_cols):
    """Extract multi-hot label vector from row."""
    labels = np.zeros(len(label_cols), dtype=np.float32)
    for i, col in enumerate(label_cols):
        if col in row and row[col] == 1:
            labels[i] = 1.0
    return labels

def load_audio(audio_path, sr=16000, duration=4.0):
    """Load audio file and pad/trim to fixed length."""
    try:
        waveform, sr_loaded = librosa.load(str(audio_path), sr=sr, mono=True)
        
        # Pad or trim to fixed duration
        target_samples = int(sr * duration)
        if len(waveform) < target_samples:
            waveform = np.pad(waveform, (0, target_samples - len(waveform)))
        else:
            waveform = waveform[:target_samples]
        
        return waveform
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None

def extract_features(waveform, sr=16000, n_mels=80):
    """Extract log-Mel spectrogram."""
    spectrogram = waveform_to_logmel(waveform, sr=sr, n_mels=n_mels)
    
    # Pad/trim to MAX_FRAMES
    if spectrogram.shape[1] < MAX_FRAMES:
        spectrogram = np.pad(spectrogram, ((0, 0), (0, MAX_FRAMES - spectrogram.shape[1])))
    else:
        spectrogram = spectrogram[:, :MAX_FRAMES]
    
    return spectrogram.astype(np.float32)

def get_audio_filename(row):
    """Construct audio filename from Show, EpId, ClipId columns."""
    try:
        show = str(row.get('Show', '')).strip()
        ep_id = int(row.get('EpId', -1))
        clip_id = int(row.get('ClipId', -1))
        
        if show and ep_id >= 0 and clip_id >= 0:
            # Format: Show_EpId_ClipId (e.g., HeStutters_0_0.wav or FluencyBank_010_0.wav)
            # Note: Some shows use padding (FluencyBank), others don't (HeStutters)
            audio_id = f"{show}_{ep_id}_{clip_id}"
            audio_path = CLIPS_DIR / f"{audio_id}.wav"
            
            if audio_path.exists():
                return audio_path
        
        return None
    except Exception:
        return None

def preprocess_data(train_ratio=0.8):
    """Main preprocessing pipeline."""
    
    # Create output directories
    train_dir = OUTPUT_DIR / 'train'
    val_dir = OUTPUT_DIR / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Load labels
    df = load_labels()
    
    # Shuffle and split
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * train_ratio)
    df_train = df[:split_idx]
    df_val = df[split_idx:]
    
    print(f"\nTrain: {len(df_train)} | Val: {len(df_val)}")
    
    # Process training data
    print("\n" + "="*60)
    print("Processing TRAINING data...")
    print("="*60)
    process_split(df_train, train_dir, "train")
    
    # Process validation data
    print("\n" + "="*60)
    print("Processing VALIDATION data...")
    print("="*60)
    process_split(df_val, val_dir, "val")
    
    print("\n✅ Preprocessing complete!")

def process_split(df, output_dir, split_name):
    """Process a data split (train or val)."""
    successful = 0
    failed = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{split_name.upper()}"):
        # Get audio path
        audio_path = get_audio_filename(row)
        if audio_path is None:
            failed += 1
            continue
        
        # Load audio
        waveform = load_audio(audio_path, sr=SR)
        if waveform is None:
            failed += 1
            continue
        
        # Extract features
        spectrogram = extract_features(waveform, sr=SR, n_mels=N_MELS)
        
        # Get labels
        labels = get_label_vector(row, LABEL_COLS)
        
        # Save as .npz
        audio_id = audio_path.stem
        output_path = output_dir / f"{audio_id}.npz"
        np.savez(output_path, spectrogram=spectrogram, labels=labels)
        
        successful += 1
    
    print(f"✅ {split_name.upper()}: {successful} processed, {failed} failed")

# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("Starting data preprocessing...")
    print(f"Clips directory: {CLIPS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Sample rate: {SR} Hz")
    print(f"Mel bands: {N_MELS}")
    print(f"Max frames: {MAX_FRAMES}")
    
    preprocess_data(train_ratio=0.8)
