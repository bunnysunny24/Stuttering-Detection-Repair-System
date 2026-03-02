"""
SELF-TRAINING LABEL REFINEMENT
-------------------------------
Uses a trained model's confident predictions to clean noisy labels,
then saves refined labels for a second training pass.

This addresses the #1 bottleneck for SEP-28k: noisy annotator labels.
Typical improvement: +3-8% F1 on noisy datasets.

Usage:
    python Models/self_train_refine.py \
        --checkpoint Models/checkpoints/temporal_bilstm_best.pth \
        --data-dir datasets/features_w2v_temporal \
        --output-dir datasets/features_w2v_temporal_refined \
        --confidence-high 0.90 \
        --confidence-low 0.10
"""

import os
import sys
import argparse
import shutil
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))


def load_model_from_checkpoint(checkpoint_path, device='cpu'):
    """Auto-detect and load model from checkpoint."""
    ckpt = torch.load(str(checkpoint_path), map_location=device)

    if isinstance(ckpt, dict):
        sd = ckpt.get('model_state', ckpt.get('state_dict', ckpt))
    else:
        sd = ckpt

    keys = set(sd.keys())

    # Detect architecture from state dict keys
    if any('lstm_norm.' in k for k in keys):
        # TemporalBiLSTMClassifier
        from model_temporal_bilstm import TemporalBiLSTMClassifier
        # Detect input_dim from projection weight
        input_dim = 768
        for k in keys:
            if 'proj.0.weight' in k:
                input_dim = sd[k].shape[1]
                break
        model = TemporalBiLSTMClassifier(
            input_dim=input_dim, n_classes=5, hidden_dim=256,
            lstm_hidden=128, lstm_layers=2, dropout=0.0
        )
    elif any('proj.' in k for k in keys) and any('temporal_blocks.' in k for k in keys):
        # TemporalStutterClassifier
        from model_temporal_w2v import TemporalStutterClassifier
        input_dim = 768
        for k in keys:
            if 'proj.0.weight' in k:
                input_dim = sd[k].shape[1]
                break
        model = TemporalStutterClassifier(input_dim=input_dim, n_classes=5, hidden_dim=256, dropout=0.0)
    else:
        raise ValueError("Cannot detect model architecture from checkpoint keys")

    model.load_state_dict(sd)
    model.eval()
    model.to(device)
    return model


def refine_labels(args):
    device = torch.device('cpu')
    print(f"=== Self-Training Label Refinement ===")
    print(f"Checkpoint:      {args.checkpoint}")
    print(f"Data dir:        {args.data_dir}")
    print(f"Output dir:      {args.output_dir}")
    print(f"Confidence high: {args.confidence_high}")
    print(f"Confidence low:  {args.confidence_low}")

    # Load model
    print("Loading model...")
    model = load_model_from_checkpoint(args.checkpoint, device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # Process both splits
    total_refined = 0
    total_files = 0
    class_names = ['Prolongation', 'Block', 'SoundRep', 'WordRep', 'Interjection']
    flip_counts = np.zeros(5, dtype=int)
    flip_0to1 = np.zeros(5, dtype=int)
    flip_1to0 = np.zeros(5, dtype=int)

    for split in ('train', 'val'):
        src_dir = Path(args.data_dir) / split
        dst_dir = Path(args.output_dir) / split
        dst_dir.mkdir(parents=True, exist_ok=True)

        if not src_dir.exists():
            print(f"  Skipping {split}: not found")
            continue

        files = sorted(src_dir.glob('*.npz'))
        print(f"\n  Processing {split}: {len(files)} files")

        for f in tqdm(files, desc=f"  Refining {split}"):
            data = np.load(f)
            total_files += 1

            if 'temporal_embedding' in data:
                feat = torch.from_numpy(data['temporal_embedding']).float()
                # z-score normalize (same as training)
                mean = feat.mean(dim=-1, keepdim=True)
                std = feat.std(dim=-1, keepdim=True).clamp(min=1e-6)
                feat = (feat - mean) / std
            elif 'embedding' in data:
                feat = torch.from_numpy(data['embedding']).float()
            else:
                # Copy as-is
                shutil.copy2(str(f), str(dst_dir / f.name))
                continue

            labels = data['labels'].astype(np.float32) if 'labels' in data else np.zeros(5, dtype=np.float32)
            orig_labels = (labels > 0).astype(np.float32)

            # Run inference
            with torch.no_grad():
                x = feat.unsqueeze(0).to(device)
                logits = model(x)
                probs = torch.sigmoid(logits).squeeze(0).numpy()

            # Refine labels based on confidence
            refined = orig_labels.copy()
            refined_this = False

            for c in range(5):
                if probs[c] > args.confidence_high and orig_labels[c] == 0:
                    # Model very confident positive, but label says negative -> flip to positive
                    refined[c] = 1.0
                    flip_counts[c] += 1
                    flip_0to1[c] += 1
                    refined_this = True
                elif probs[c] < args.confidence_low and orig_labels[c] == 1:
                    # Model very confident negative, but label says positive -> flip to negative
                    refined[c] = 0.0
                    flip_counts[c] += 1
                    flip_1to0[c] += 1
                    refined_this = True

            if refined_this:
                total_refined += 1

            # Save to output dir (copy features, use refined labels)
            save_dict = {}
            for key in data.files:
                if key == 'labels':
                    save_dict['labels'] = refined
                else:
                    save_dict[key] = data[key]

            np.savez_compressed(str(dst_dir / f.name), **save_dict)
            data.close()

    # Report
    print(f"\n=== Label Refinement Results ===")
    print(f"Total files: {total_files}")
    print(f"Files with label changes: {total_refined} ({100*total_refined/max(1,total_files):.1f}%)")
    print(f"\nPer-class flips:")
    for c in range(5):
        print(f"  {class_names[c]:15s}: {flip_counts[c]:5d} flips "
              f"(0->1: {flip_0to1[c]}, 1->0: {flip_1to0[c]})")
    print(f"\nRefined data saved to: {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Self-training label refinement')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--data-dir', type=str, required=True, help='Input features directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for refined features')
    parser.add_argument('--confidence-high', type=float, default=0.90, help='Threshold for flipping 0->1')
    parser.add_argument('--confidence-low', type=float, default=0.10, help='Threshold for flipping 1->0')
    args = parser.parse_args()
    refine_labels(args)
