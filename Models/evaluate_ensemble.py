import argparse
import json
import numpy as np
import torch
from pathlib import Path

def load_checkpoint_state(path):
    state = torch.load(str(path), map_location='cpu')
    if isinstance(state, dict):
        if 'model_state' in state:
            return state['model_state']
        if 'state_dict' in state:
            return state['state_dict']
        if 'model' in state and isinstance(state['model'], dict):
            return state['model']
    return state

def compute_metrics(y_true, y_probs, thresholds=None):
    # y_true: (N, C) binary
    # y_probs: (N, C) floats
    if thresholds is None:
        thresholds = np.full(y_probs.shape[1], 0.5)
    y_pred = (y_probs > thresholds).astype(float)
    per_class_f1 = []
    per_class_precision = []
    per_class_recall = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class_precision.append(prec)
        per_class_recall.append(rec)
        per_class_f1.append(f1)
    return {
        'f1_macro': float(np.mean(per_class_f1)),
        'precision_macro': float(np.mean(per_class_precision)),
        'recall_macro': float(np.mean(per_class_recall)),
        'per_class_f1': per_class_f1
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', nargs='+', required=False, help='List of checkpoint files for ensemble')
    parser.add_argument('--data-dir', type=str, default='datasets/features', help='Data dir')
    parser.add_argument('--arch', type=str, default='improved_90plus_large')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--out', type=str, default='output/ensemble_results.json')
    args = parser.parse_args()

    # Lazy import of dataset & model to avoid duplication
    try:
        from train_90plus_final import AudioDataset, NUM_CLASSES
    except Exception:
        from Models.train_90plus_final import AudioDataset, NUM_CLASSES

    # Select model class
    if args.arch == 'improved_90plus_large':
        try:
            from model_improved_90plus_large import ImprovedStutteringCNNLarge as ModelClass
        except Exception:
            from model_improved_90plus import ImprovedStutteringCNN as ModelClass
    else:
        from model_improved_90plus import ImprovedStutteringCNN as ModelClass

    # Find checkpoints if not provided
    cks = args.checkpoints
    if not cks:
        # search Models/checkpoints for recent *_best.pth or recent .pth files
        p = Path('Models/checkpoints')
        if p.exists():
            bests = list(p.rglob('*_best.pth'))
            if bests:
                bests = sorted(bests, key=lambda x: x.stat().st_mtime, reverse=True)
                cks = [str(bests[0])]
            else:
                anypth = list(p.rglob('*.pth'))
                anypth = sorted(anypth, key=lambda x: x.stat().st_mtime, reverse=True)
                cks = [str(x) for x in anypth[:3]]
    if not cks:
        print('No checkpoints found for ensemble'); return

    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')

    # Build val loader
    val_ds = AudioDataset(args.data_dir, split='val', augment=False)
    # reuse collate from train script if present
    try:
        from train_90plus_final import collate_variable_length as collate_fn
    except Exception:
        from Models.train_90plus_final import collate_variable_length as collate_fn
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    all_probs = []
    for ck in cks:
        print('Loading checkpoint', ck)
        model = ModelClass(n_channels=123, n_classes=NUM_CLASSES)
        sd = load_checkpoint_state(ck)
        try:
            model.load_state_dict(sd)
        except Exception:
            try:
                model.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()})
            except Exception as e:
                print('Failed to load state for', ck, e)
                continue
        model.to(device)
        model.eval()
        probs_list = []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                logits = model(X)
                probs = torch.sigmoid(logits).cpu().numpy()
                probs_list.append(probs)
        probs_all = np.vstack(probs_list)
        all_probs.append(probs_all)

    # Average probabilities
    avg_probs = np.mean(np.stack(all_probs, axis=0), axis=0)

    # Collect y_true from val set (once)
    y_true_list = []
    for _, y in val_loader:
        y_true_list.append(y.numpy())
    y_true = np.vstack(y_true_list)

    metrics = compute_metrics(y_true, avg_probs)
    print('Ensemble metrics:', metrics)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({'checkpoints': cks, 'metrics': metrics}, indent=2))
    print('Saved ensemble results to', args.out)

if __name__ == '__main__':
    main()
