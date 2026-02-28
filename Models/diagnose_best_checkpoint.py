import json
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from eval_validation import load_npz, pad_to_length, TOTAL_CHANNELS, NUM_CLASSES
from model_cnn_bilstm import CNNBiLSTM


def load_thresholds(path: Path):
    if not path.exists():
        return np.array([0.5] * NUM_CLASSES)
    j = json.load(open(path))
    return np.array(j.get('thresholds', [0.5] * NUM_CLASSES))


def main(ckpt_path, data_dir, thresholds_path=None, out_dir=None, top_k: int = 20):
    ckpt = Path(ckpt_path)
    val_dir = Path(data_dir) / 'val'
    out_dir = Path(out_dir) if out_dir is not None else (Path('output') / f'diag_{ckpt.stem}')
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cpu')
    state = torch.load(str(ckpt), map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    state = {k.replace('module.', ''): v for k, v in state.items()} if isinstance(state, dict) else state

    model = CNNBiLSTM(in_channels=TOTAL_CHANNELS, n_classes=NUM_CLASSES)
    model.load_state_dict(state)
    model.to(device).eval()

    thresh = load_thresholds(Path(thresholds_path) if thresholds_path else Path('output/thresholds.json'))

    files = sorted(val_dir.glob('**/*.npz'))
    rows = []
    all_probs = []
    all_labels = []
    max_len = 0
    for p in files:
        spec, labels = load_npz(p)
        if spec is None:
            continue
        max_len = max(max_len, spec.shape[1])

    with torch.no_grad():
        for p in tqdm(files, desc='Diag'):
            spec, labels = load_npz(p)
            if spec is None:
                continue
            if spec.shape[0] != TOTAL_CHANNELS:
                if spec.shape[0] < TOTAL_CHANNELS:
                    spec = np.pad(spec, ((0, TOTAL_CHANNELS - spec.shape[0]), (0, 0)), mode='constant')
                else:
                    spec = spec[:TOTAL_CHANNELS, :]
            spec = pad_to_length(spec, max_len)
            x = torch.from_numpy(spec).unsqueeze(0).to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            all_probs.append(probs)
            all_labels.append(labels)
            rows.append({'file': str(p), 'probs': probs.tolist(), 'labels': labels.tolist()})

    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    preds = (all_probs >= thresh[np.newaxis, :]).astype(int)
    truths = (all_labels > 0).astype(int)

    # per-class confusion
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

    # find top FPs and FNs per class
    top_errors = {}
    for i in range(NUM_CLASSES):
        entries = []
        for r, prob_vec, label_vec in zip(rows, all_probs, all_labels):
            pred = int(prob_vec[i] >= thresh[i])
            true = int(label_vec[i] > 0)
            if pred == 1 and true == 0:
                entries.append((prob_vec[i], r['file']))
        entries = sorted(entries, key=lambda x: -x[0])[:top_k]
        top_errors[f'class_{i}_FP'] = [{'prob': float(p), 'file': f} for p, f in entries]

        entries = []
        for r, prob_vec, label_vec in zip(rows, all_probs, all_labels):
            pred = int(prob_vec[i] >= thresh[i])
            true = int(label_vec[i] > 0)
            if pred == 0 and true == 1:
                entries.append((prob_vec[i], r['file']))
        entries = sorted(entries, key=lambda x: x[0])[:top_k]
        top_errors[f'class_{i}_FN'] = [{'prob': float(p), 'file': f} for p, f in entries]

    out = {'per_class': per_class, 'top_errors': top_errors, 'thresholds': thresh.tolist()}
    json.dump(out, open(out_dir / 'diagnostics.json', 'w'), indent=2)
    print('Wrote diagnostics to', out_dir / 'diagnostics.json')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', help='Checkpoint path (legacy)', default='Models/checkpoints/training_20260219_015326/cnn_bilstm_best.pth')
    p.add_argument('--checkpoint', help='Checkpoint path (alias)', dest='ckpt_alias')
    p.add_argument('--data-dir', default='datasets/features')
    p.add_argument('--thresholds', default='output/thresholds.json')
    p.add_argument('--out', help='Output diagnostics directory (overrides default)', default=None)
    p.add_argument('--top', type=int, help='Number of top errors (FP/FN) per class to record', default=20)
    # compatibility args (accepted but not used for now)
    p.add_argument('--batch-size', type=int, default=96, help='Ignored: present for CLI compatibility')
    p.add_argument('--num-workers', type=int, default=4, help='Ignored: present for CLI compatibility')

    args = p.parse_args()
    ckpt = args.ckpt_alias if args.ckpt_alias else args.ckpt
    main(ckpt, args.data_dir, args.thresholds, out_dir=args.out, top_k=args.top)
