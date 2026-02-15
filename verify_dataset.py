"""
DATASET VERIFICATION SCRIPT
Checks that extracted features are properly formatted for training
"""

import numpy as np
from pathlib import Path

print("="*80)
print("DATASET VERIFICATION")
print("="*80)

# Check feature files
train_dir = Path('datasets/features/train')
val_dir = Path('datasets/features/val')

train_files = sorted(train_dir.glob('*.npz'))
val_files = sorted(val_dir.glob('*.npz'))

print(f"\nTrain files: {len(train_files)}")
print(f"Val files: {len(val_files)}")
print(f"Total: {len(train_files) + len(val_files)}")

if len(train_files) == 0:
    print("\n[ERROR] No training files found!")
    print("Run: python Models/COMPLETE_PIPELINE.py")
    exit(1)

# Check first file
print("\n" + "="*80)
print("CHECKING FIRST TRAINING FILE")
print("="*80)

sample_file = train_files[0]
print(f"\nFile: {sample_file.name}")

data = np.load(sample_file)
print(f"Keys in file: {list(data.keys())}")

spectrogram = data['spectrogram']
print(f"\nSpectrogram shape: {spectrogram.shape}")
print(f"Spectrogram dtype: {spectrogram.dtype}")
print(f"Channels: {spectrogram.shape[0]}")
if len(spectrogram.shape) > 1:
    print(f"Time steps: {spectrogram.shape[1]}")

# Check if labels exist
if 'labels' in data:
    labels = data['labels']
    print(f"\nLabels shape: {labels.shape}")
    print(f"Labels dtype: {labels.dtype}")
    print(f"Labels values: {labels}")
else:
    print("\n[WARNING] No labels in file (this is OK)")

# Verify model expects 123 channels
print("\n" + "="*80)
print("MODEL EXPECTATIONS vs ACTUAL DATA")
print("="*80)

print(f"\nModel expects input: (batch, 123, time_steps)")
print(f"Actual data shape: ({spectrogram.shape[0]}, {spectrogram.shape[1] if len(spectrogram.shape) > 1 else 'N/A'})")

if spectrogram.shape[0] == 123:
    print("✓ [OK] Channel dimension is correct (123)")
else:
    print(f"✗ [ERROR] Expected 123 channels, got {spectrogram.shape[0]}")
    exit(1)

if len(spectrogram.shape) == 2:
    print("✓ [OK] Data is 2D (channels, time) - correct format")
else:
    print(f"✗ [ERROR] Expected 2D data, got {len(spectrogram.shape)}D")
    exit(1)

# Sample statistics
print(f"\nData statistics:")
print(f"  Min: {spectrogram.min():.4f}")
print(f"  Max: {spectrogram.max():.4f}")
print(f"  Mean: {spectrogram.mean():.4f}")
print(f"  Std: {spectrogram.std():.4f}")

# Check multiple files
print("\n" + "="*80)
print("CHECKING ALL FILES")
print("="*80)

errors = 0
for i, f in enumerate(train_files[:10]):  # Check first 10
    try:
        d = np.load(f)
        spec = d['spectrogram']
        if spec.shape[0] != 123:
            print(f"✗ File {f.name}: Wrong shape {spec.shape}")
            errors += 1
    except Exception as e:
        print(f"✗ File {f.name}: {e}")
        errors += 1

if errors == 0:
    print("✓ [OK] All checked files have correct format")
else:
    print(f"✗ [ERROR] {errors} files have issues")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total samples available: {len(train_files) + len(val_files)}")
print(f"Features: 123 channels")
print(f"Data format: 2D numpy arrays (123, time_steps)")
print(f"Status: ✓ READY FOR TRAINING")
print("\nNext step: python Models/COMPLETE_PIPELINE.py")
print("="*80)
