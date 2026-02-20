#!/usr/bin/env python3
"""Ensemble checkpoints: average predicted probabilities across checkpoints and report macro ROC AUC and F1.

Usage:
  python tools/ensemble_checkpoints.py --checkpoints Models/checkpoints/training_20260220_005807/improved_90plus_best.pth ...
"""
import argparse
import json
import numpy as np
from pathlib import Path
import torch
from Models.model_improved_90plus import ImprovedStutteringCNN

NUM_CLASSES = 5


def load_model(checkpoint_path, device):
    model = ImprovedStutteringCNN(n_channels=123, n_classes=NUM_CLASSES, dropout=0.2)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def compute_metrics(y_true, y_probs, thresholds=None):
    # y_true, y_probs: numpy arrays (N, C)
    if thresholds is None:
        thresholds = np.full(y_probs.shape[1], 0.5)
    y_pred = (y_probs > thresholds).astype(int)

    # compute per-class roc auc via rank method
    roc_auc_list = []
    f1_list = []
    for i in range(y_true.shape[1]):
        y_true_i = y_true[:, i]
        y_prob_i = y_probs[:, i]
        # ROC-AUC by Mann-Whitney
        if len(np.unique(y_true_i)) > 1:
            n_pos = int(np.sum(y_true_i == 1))
            n_neg = int(len(y_true_i) - n_pos)
            if n_pos > 0 and n_neg > 0:
                ranks = np.argsort(np.argsort(y_prob_i)) + 1
                sum_ranks_pos = np.sum(ranks[y_true_i == 1])
                auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
            else:
                auc = 0.5
        else:
            auc = 0.5
        roc_auc_list.append(float(auc))

        tp = int(np.sum((y_true_i == 1) & (y_pred[:, i] == 1)))
        fp = int(np.sum((y_true_i == 0) & (y_pred[:, i] == 1)))
        fn = int(np.sum((y_true_i == 1) & (y_pred[:, i] == 0)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)

    return {'roc_auc_macro': float(np.mean(roc_auc_list)), 'f1_macro': float(np.mean(f1_list)), 'per_class_auc': roc_auc_list}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints', nargs='+', required=True)
    p.add_argument('--data-dir', default='datasets/features')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device=='cuda' else 'cpu')

    # load val dataset
    val_dir = Path(args.data_dir) / 'val'
    files = sorted(val_dir.glob('**/*.npz'))
    if not files:
        print('No val files found in', val_dir)
        return 1

    # load checkpoints
    models = [load_model(Path(p), device) for p in args.checkpoints]

    # accumulate probs
    all_probs = []
    all_truth = []
    batch_size = args.batch_size
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i+batch_size]
        batch_specs = []
        batch_labels = []
        for f in batch_files:
            data = np.load(f)
            if 'spectrogram' in data:
                spec = torch.from_numpy(data['spectrogram']).float()
            elif 'embedding' in data:
                spec = torch.from_numpy(data['embedding']).float()
            else:
                continue
            if spec.dim() == 1:
                # embedding
                X = spec.unsqueeze(0).to(device)
            else:
                if spec.dim() == 3:
                    spec = spec.squeeze(0)
                # keep as (1, C, T)
                X = spec.unsqueeze(0).to(device)
            batch_specs.append(X)
            lbl = data.get('labels')
            if lbl is None:
                lbl = np.zeros((NUM_CLASSES,), dtype=np.int32)
            lbl = np.asarray(lbl)
            lbl_bin = (lbl > 0).astype(int)
            batch_labels.append(lbl_bin)

        if not batch_specs:
            continue
        X_batch = torch.cat(batch_specs, dim=0)

        # get probs from each model
        probs_accum = None
        with torch.no_grad():
            for m in models:
                logits = m(X_batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                if probs_accum is None:
                    probs_accum = probs
                else:
                    probs_accum += probs
        probs_avg = probs_accum / float(len(models))
        all_probs.append(probs_avg)
        all_truth.append(np.vstack(batch_labels))

    y_probs = np.vstack(all_probs)
    y_true = np.vstack(all_truth)

    results = compute_metrics(y_true, y_probs)
    out = Path('output')
    out.mkdir(exist_ok=True)
    outp = out / 'ensemble_results.json'
    outp.write_text(json.dumps(results, indent=2))
    print('Wrote ensemble results to', outp)
    print('Results:', results)


if __name__ == '__main__':
    main()
import argparse
from pathlib import Path
import torch
import numpy as np
import json
from tqdm import tqdm

from Models.diagnose_best_checkpoint import load_thresholds
from Models.model_cnn_bilstm import CNNBiLSTM
from Models.eval_validation import load_npz, pad_to_length, TOTAL_CHANNELS, NUM_CLASSES


def load_model(checkpoint_path):
    state = torch.load(str(checkpoint_path), map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    state = {k.replace('module.', ''): v for k, v in state.items()} if isinstance(state, dict) else state
    model = CNNBiLSTM(in_channels=TOTAL_CHANNELS, n_classes=NUM_CLASSES)
    model.load_state_dict(state)
    model.eval()
    return model


def main(checkpoints, data_dir, out_dir, thresholds=None):
    ckpts = [Path(p) for p in checkpoints]
    val_dir = Path(data_dir) / 'val'
    files = sorted(val_dir.glob('**/*.npz'))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [load_model(p) for p in ckpts]

    all_probs = []
    max_len = 0
    for p in files:
        spec, labels = load_npz(p)
        if spec is None:
            continue
        max_len = max(max_len, spec.shape[1])

    for p in tqdm(files, desc='Ensemble eval'):
        spec, labels = load_npz(p)
        if spec is None:
            continue
        if spec.shape[0] != TOTAL_CHANNELS:
            if spec.shape[0] < TOTAL_CHANNELS:
                spec = np.pad(spec, ((0, TOTAL_CHANNELS - spec.shape[0]), (0, 0)), mode='constant')
            else:
                spec = spec[:TOTAL_CHANNELS, :]
        spec = pad_to_length(spec, max_len)
        x = torch.from_numpy(spec).unsqueeze(0)
        probs = []
        for m in models:
            with torch.no_grad():
                logits = m(x)
                probs.append(torch.sigmoid(logits).cpu().numpy()[0])
        probs = np.vstack(probs)
        avg = np.mean(probs, axis=0)
        all_probs.append({'file': str(p), 'probs': avg.tolist(), 'labels': labels.tolist()})

    # compute simple metrics using thresholds or 0.5
    if thresholds:
        th = np.array(json.load(open(thresholds)).get('thresholds', [0.5]*NUM_CLASSES))
    else:
        th = np.array([0.5]*NUM_CLASSES)

    all_probs_arr = np.vstack([r['probs'] for r in all_probs])
    all_labels_arr = np.vstack([r['labels'] for r in all_probs])
    preds = (all_probs_arr >= th[np.newaxis, :]).astype(int)
    truths = (all_labels_arr > 0).astype(int)

    per_class = {}
    for i in range(NUM_CLASSES):
        y_true = truths[:, i]
        y_pred = preds[:, i]
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[i] = {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall, 'f1': f1}

    out = {'per_class': per_class, 'n_files': len(all_probs), 'thresholds_used': th.tolist()}
    json.dump(out, open(out_dir / 'ensemble_metrics.json', 'w'), indent=2)
    # save averaged probs for inspection
    json.dump(all_probs, open(out_dir / 'ensemble_probs.json', 'w'), indent=2)
    print('Wrote ensemble results to', out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--data-dir', default='datasets/features')
    parser.add_argument('--out', default='output/ensemble_results')
    parser.add_argument('--thresholds', default=None)
    args = parser.parse_args()
    main(args.checkpoints, args.data_dir, args.out, thresholds=args.thresholds)
