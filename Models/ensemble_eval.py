"""
ENSEMBLE EVALUATION WITH TEST-TIME AUGMENTATION (TTA)
------------------------------------------------------
Loads multiple model checkpoints, runs inference with TTA,
and averages predictions for maximum accuracy.

Techniques:
1. Multi-model ensemble: Average predictions from N models trained with different seeds
2. Test-time augmentation: Multiple inference passes with time shifts
3. Optimal per-class thresholds via calibration on ensemble predictions

Usage:
    python Models/ensemble_eval.py \
        --checkpoints ckpt1.pth ckpt2.pth ckpt3.pth \
        --data-dir datasets/features_w2v_temporal \
        --tta-shifts 5
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, classification_report
)

sys.path.insert(0, str(Path(__file__).parent))

CLASS_NAMES = ['Prolongation', 'Block', 'SoundRep', 'WordRep', 'Interjection']


def load_model_from_checkpoint(checkpoint_path, device='cpu'):
    """Auto-detect and load model from checkpoint."""
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(ckpt, dict):
        sd = ckpt.get('model_state', ckpt.get('state_dict', ckpt))
    else:
        sd = ckpt

    keys = set(sd.keys())
    if any('lstm_norm.' in k for k in keys):
        from model_temporal_bilstm import TemporalBiLSTMClassifier
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
        from model_temporal_w2v import TemporalStutterClassifier
        input_dim = 768
        for k in keys:
            if 'proj.0.weight' in k:
                input_dim = sd[k].shape[1]
                break
        model = TemporalStutterClassifier(input_dim=input_dim, n_classes=5, hidden_dim=256, dropout=0.0)
    else:
        raise ValueError(f"Cannot detect architecture from {checkpoint_path}")

    model.load_state_dict(sd)
    model.eval()
    model.to(device)
    return model


def prepare_input(data):
    """Load and normalize a single NPZ file."""
    if 'temporal_embedding' in data:
        feat = torch.from_numpy(data['temporal_embedding']).float()
        mean = feat.mean(dim=-1, keepdim=True)
        std = feat.std(dim=-1, keepdim=True).clamp(min=1e-6)
        feat = (feat - mean) / std
        return feat
    elif 'embedding' in data:
        return torch.from_numpy(data['embedding']).float()
    return None


def tta_predict(model, feat, n_shifts=5, device='cpu'):
    """Test-time augmentation with time shifts + original.

    Returns averaged probability vector (5,).
    """
    all_probs = []

    # Original
    with torch.no_grad():
        x = feat.unsqueeze(0).to(device)
        logits = model(x)
        all_probs.append(torch.sigmoid(logits).squeeze(0).cpu().numpy())

    # Time-shifted versions (circular shift)
    if feat.dim() == 2:  # temporal: (C, T)
        T = feat.shape[1]
        if T > 10:
            for i in range(1, n_shifts):
                shift = int(T * i / n_shifts)
                shifted = torch.roll(feat, shifts=shift, dims=1)
                with torch.no_grad():
                    x = shifted.unsqueeze(0).to(device)
                    logits = model(x)
                    all_probs.append(torch.sigmoid(logits).squeeze(0).cpu().numpy())

    # Average all TTA predictions
    return np.mean(all_probs, axis=0)


def optimize_thresholds(all_probs, all_labels, n_classes=5):
    """Find optimal per-class thresholds by grid search on F1."""
    thresholds = []
    for c in range(n_classes):
        best_t = 0.5
        best_f1 = 0.0
        for t in np.arange(0.1, 0.9, 0.01):
            preds = (all_probs[:, c] > t).astype(int)
            f1 = f1_score(all_labels[:, c], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        thresholds.append(float(best_t))
    return thresholds


def ensemble_evaluate(args):
    device = torch.device('cpu')

    print(f"\n{'='*60}")
    print(f"  ENSEMBLE EVALUATION WITH TEST-TIME AUGMENTATION")
    print(f"{'='*60}")
    print(f"  Models:     {len(args.checkpoints)} checkpoints")
    print(f"  TTA shifts: {args.tta_shifts}")
    print(f"  Data:       {args.data_dir}")
    print()

    # Load all models
    models = []
    for cp in args.checkpoints:
        print(f"  Loading: {cp}")
        m = load_model_from_checkpoint(cp, device)
        models.append(m)
    print(f"  Loaded {len(models)} models\n")

    # Load validation data
    val_dir = Path(args.data_dir) / 'val'
    files = sorted(val_dir.glob('*.npz'))
    print(f"  Validation files: {len(files)}")

    all_probs = []
    all_labels = []

    for f in tqdm(files, desc="  Ensemble inference"):
        data = np.load(f)
        feat = prepare_input(data)
        if feat is None:
            data.close()
            continue

        labels = data['labels'].astype(np.float32) if 'labels' in data else np.zeros(5, dtype=np.float32)
        labels_bin = (labels > 0).astype(np.float32)

        # Get predictions from each model (with TTA) and average
        model_probs = []
        for model in models:
            probs = tta_predict(model, feat, n_shifts=args.tta_shifts, device=device)
            model_probs.append(probs)

        # Ensemble average across models
        ensemble_prob = np.mean(model_probs, axis=0)

        all_probs.append(ensemble_prob)
        all_labels.append(labels_bin)
        data.close()

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    print(f"\n  Total samples: {len(all_probs)}")

    # Optimize thresholds
    print(f"\n  Optimizing per-class thresholds...")
    thresholds = optimize_thresholds(all_probs, all_labels)

    # Final predictions with optimized thresholds
    preds = np.zeros_like(all_probs)
    for c in range(5):
        preds[:, c] = (all_probs[:, c] > thresholds[c]).astype(float)

    # Metrics
    print(f"\n{'='*60}")
    print(f"  ENSEMBLE RESULTS ({len(models)} models x {args.tta_shifts} TTA)")
    print(f"{'='*60}\n")

    for c in range(5):
        f1 = f1_score(all_labels[:, c], preds[:, c], zero_division=0)
        prec = precision_score(all_labels[:, c], preds[:, c], zero_division=0)
        rec = recall_score(all_labels[:, c], preds[:, c], zero_division=0)
        acc = accuracy_score(all_labels[:, c], preds[:, c])
        try:
            auc = roc_auc_score(all_labels[:, c], all_probs[:, c])
        except Exception:
            auc = 0.0
        print(f"  {CLASS_NAMES[c]:15s}  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  Acc={acc:.4f}  AUC={auc:.4f}  Thresh={thresholds[c]:.2f}")

    # Macro averages
    macro_f1 = f1_score(all_labels, preds, average='macro', zero_division=0)
    macro_prec = precision_score(all_labels, preds, average='macro', zero_division=0)
    macro_rec = recall_score(all_labels, preds, average='macro', zero_division=0)

    # Per-class accuracy average
    per_class_acc = []
    for c in range(5):
        per_class_acc.append(accuracy_score(all_labels[:, c], preds[:, c]))
    mean_acc = np.mean(per_class_acc)

    # Sample-level accuracy (all 5 classes correct)
    sample_acc = np.mean(np.all(preds == all_labels, axis=1))

    print(f"\n  {'MACRO AVERAGE':15s}  F1={macro_f1:.4f}  Prec={macro_prec:.4f}  Rec={macro_rec:.4f}")
    print(f"  {'MEAN CLASS ACC':15s}  {mean_acc:.4f} ({mean_acc*100:.1f}%)")
    print(f"  {'SAMPLE ACC':15s}     {sample_acc:.4f} ({sample_acc*100:.1f}%)")

    # Save results
    out_dir = Path(args.output_dir) if args.output_dir else Path('output/ensemble_eval')
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'n_models': len(models),
        'tta_shifts': args.tta_shifts,
        'thresholds': {CLASS_NAMES[c]: thresholds[c] for c in range(5)},
        'per_class': {
            CLASS_NAMES[c]: {
                'f1': float(f1_score(all_labels[:, c], preds[:, c], zero_division=0)),
                'precision': float(precision_score(all_labels[:, c], preds[:, c], zero_division=0)),
                'recall': float(recall_score(all_labels[:, c], preds[:, c], zero_division=0)),
                'accuracy': float(accuracy_score(all_labels[:, c], preds[:, c])),
            }
            for c in range(5)
        },
        'macro_f1': float(macro_f1),
        'macro_precision': float(macro_prec),
        'macro_recall': float(macro_rec),
        'mean_class_accuracy': float(mean_acc),
        'sample_accuracy': float(sample_acc),
        'checkpoints': [str(cp) for cp in args.checkpoints],
    }

    results_path = out_dir / 'ensemble_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    # Save optimized thresholds
    thresh_path = out_dir / 'ensemble_thresholds.json'
    thresh_dict = {CLASS_NAMES[c]: thresholds[c] for c in range(5)}
    with open(thresh_path, 'w') as f:
        json.dump(thresh_dict, f, indent=2)
    print(f"  Thresholds saved to: {thresh_path}")

    # Also save as output/thresholds.json for the main pipeline
    main_thresh = Path('output/thresholds.json')
    with open(main_thresh, 'w') as f:
        json.dump(thresh_dict, f, indent=2)
    print(f"  Also saved to: {main_thresh}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ensemble evaluation with TTA')
    parser.add_argument('--checkpoints', nargs='+', required=True, help='Model checkpoint paths')
    parser.add_argument('--data-dir', type=str, required=True, help='Features directory')
    parser.add_argument('--tta-shifts', type=int, default=5, help='Number of TTA time shifts')
    parser.add_argument('--output-dir', type=str, default='output/ensemble_eval', help='Output directory')
    args = parser.parse_args()
    ensemble_evaluate(args)
