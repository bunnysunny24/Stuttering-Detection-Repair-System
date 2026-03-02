"""
Calibrate per-class thresholds on validation set and write output/thresholds.json

Usage:
  python Models/calibrate_thresholds.py --checkpoint <path> --data-dir datasets/features --out output/thresholds.json
"""
import argparse
import json
import numpy as np
import torch
from pathlib import Path

def load_model(checkpoint_path, device):
    # Flexible loader: inspect state_dict keys to select the right model class.
    state = torch.load(str(checkpoint_path), map_location=device)

    # If common encapsulation was used (e.g., {'state_dict': ...}), unwrap it
    if isinstance(state, dict) and ('state_dict' in state or 'model_state_dict' in state or 'model_state' in state):
        state = state.get('state_dict', state.get('model_state_dict', state.get('model_state')))

    # Strip common DataParallel/module. prefix if present
    def _strip_module_prefix(d):
        if not isinstance(d, dict):
            return d
        if any(k.startswith('module.') for k in d.keys()):
            return {k.replace('module.', ''): v for k, v in d.items()}
        return d

    state = _strip_module_prefix(state)

    # Heuristic detection based on keys
    keys = list(state.keys()) if isinstance(state, dict) else []
    # Detect model architecture from state dict keys
    use_large = any(k.startswith('ms_conv1.') for k in keys)  # ms_conv1 is unique to ImprovedStutteringCNNLarge
    use_improved = any(k.startswith('block1.') for k in keys)
    use_cnn_bilstm = any(k.startswith('conv1') or k.startswith('lstm') or k.startswith('classifier') for k in keys)
    use_embedding_mlp = any(k.startswith('input_bn.') or k.startswith('backbone.') for k in keys)
    use_temporal_w2v = any(k.startswith('proj.') and any(k2.startswith('temporal_blocks.') for k2 in keys) for k in keys)
    use_temporal_bilstm = any(k.startswith('lstm_norm.') and any(k2.startswith('temporal_blocks.') for k2 in keys) for k in keys)

    # Try all model classes in order of likelihood
    candidates = []
    if use_temporal_bilstm:
        _input_dim = 768
        for k, v in state.items():
            if k == 'proj.0.weight':
                _input_dim = v.shape[1]
                break
        candidates.append(('model_temporal_bilstm', 'TemporalBiLSTMClassifier', {'input_dim': _input_dim, 'n_classes': 5, 'hidden_dim': 256, 'lstm_hidden': 128, 'lstm_layers': 2, 'dropout': 0.3}))
    if use_temporal_w2v and not use_temporal_bilstm:
        # Detect input_dim from projection layer
        _input_dim = 768
        for k, v in state.items():
            if k == 'proj.0.weight':  # Conv1d weight shape: (out, in, kernel)
                _input_dim = v.shape[1]
                break
        candidates.append(('model_temporal_w2v', 'TemporalStutterClassifier', {'input_dim': _input_dim, 'n_classes': 5, 'hidden_dim': 256, 'dropout': 0.3}))
    if use_embedding_mlp:
        # Detect input_dim from the first linear layer weight shape
        _emb_dim = 1536
        for k, v in state.items():
            if k == 'backbone.0.fc.weight':
                _emb_dim = v.shape[1]
                break
            elif k == 'input_bn.weight':
                _emb_dim = v.shape[0]
                break
        candidates.append(('model_embedding_mlp', 'EmbeddingMLPClassifier', {'input_dim': _emb_dim, 'n_classes': 5, 'dropout': 0.3}))
    if use_large:
        candidates.append(('model_improved_90plus_large', 'ImprovedStutteringCNNLarge', {'n_channels': 123, 'n_classes': 5, 'dropout': 0.35}))
    if use_improved:
        candidates.append(('model_improved_90plus', 'ImprovedStutteringCNN', {'n_channels': 123, 'n_classes': 5, 'dropout': 0.4}))
    if use_cnn_bilstm:
        candidates.append(('model_cnn_bilstm', 'CNNBiLSTM', {'in_channels': 123, 'n_classes': 5}))
    # Always add all as fallbacks
    candidates.extend([
        ('model_improved_90plus_large', 'ImprovedStutteringCNNLarge', {'n_channels': 123, 'n_classes': 5, 'dropout': 0.35}),
        ('model_improved_90plus', 'ImprovedStutteringCNN', {'n_channels': 123, 'n_classes': 5, 'dropout': 0.4}),
        ('model_cnn_bilstm', 'CNNBiLSTM', {'in_channels': 123, 'n_classes': 5}),
    ])

    model = None
    for mod_name, cls_name, kwargs in candidates:
        try:
            mod = __import__(mod_name)
            ModelClass = getattr(mod, cls_name)
            m = ModelClass(**kwargs)
            m.load_state_dict(state)
            model = m
            print(f'Loaded checkpoint as {cls_name}')
            break
        except Exception:
            try:
                stripped = {k.replace('module.', ''): v for k, v in state.items()}
                m.load_state_dict(stripped)
                model = m
                print(f'Loaded checkpoint as {cls_name} (stripped keys)')
                break
            except Exception:
                continue

    if model is None:
        raise RuntimeError('Could not load checkpoint with any known model class')

    model.to(device).eval()
    return model

def gather_probs_and_labels(model, data_dir, device, batch_size=32, num_workers=2):
    from train_90plus_final import AudioDataset, collate_variable_length
    from torch.utils.data import DataLoader

    val_ds = AudioDataset(data_dir, split='val', augment=False)
    dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_variable_length)

    all_probs = []
    all_labels = []
    with torch.no_grad():
        for X, y in dl:
            X = X.to(device)
            logits = model(X)
            all_probs.append(logits.cpu().numpy())
            all_labels.append(y.numpy())

    all_logits = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    return all_logits, all_labels


def optimize_temperature(logits_np, labels_np, device, lr=0.01, steps=300):
    """Optimize a single temperature scalar to minimize BCE NLL on validation logits."""
    import torch
    logits = torch.from_numpy(logits_np).float().to(device)
    labels = torch.from_numpy(labels_np).float().to(device)

    # initialize temperature > 0
    temp = torch.nn.Parameter(torch.ones(1, device=device, dtype=torch.float32))
    optimizer = torch.optim.Adam([temp], lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for _ in range(steps):
        optimizer.zero_grad()
        scaled = logits / temp.clamp(min=1e-2)
        loss = loss_fn(scaled, labels)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            # keep temperature in a reasonable range
            temp.clamp_(0.05, 10.0)

    return float(temp.item())

def calibrate_thresholds(probs, labels):
    num_classes = probs.shape[1]
    thresholds = np.zeros(num_classes)
    best_metrics = {}
    for i in range(num_classes):
        best_f1 = -1.0
        best_t = 0.5
        y_true = labels[:, i].astype(int)
        if y_true.sum() == 0:
            thresholds[i] = 0.5
            best_metrics[i] = {'f1': 0.0}
            continue
        for t in np.linspace(0.01, 0.99, 99):
            y_pred = (probs[:, i] > t).astype(int)
            tp = int(((y_true == 1) & (y_pred == 1)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            fn = int(((y_true == 1) & (y_pred == 0)).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        thresholds[i] = float(best_t)
        best_metrics[i] = {'f1': float(best_f1)}
    return thresholds, best_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-dir', default='datasets/features')
    parser.add_argument('--out', default='output/thresholds.json')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=2)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.checkpoint, device)

    logits, labels = gather_probs_and_labels(model, args.data_dir, device, batch_size=args.batch_size, num_workers=args.num_workers)

    # Optimize temperature on validation logits
    try:
        temperature = optimize_temperature(logits, labels, device=device, lr=0.02, steps=200)
    except Exception:
        temperature = 1.0

    # Apply temperature and compute probabilities for threshold search
    probs = 1.0 / (1.0 + np.exp(-logits / float(temperature)))

    thresholds, metrics = calibrate_thresholds(probs, labels)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'thresholds': thresholds.tolist(),
        'per_class_metrics': metrics,
        'temperature': float(temperature)
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote thresholds to {out_path}")
    # Write a simple lock file indicating thresholds are set for deployment
    try:
        lock_path = out_path.parent / 'thresholds.lock'
        lock_payload = {
            'created': str(out_path.resolve()),
            'timestamp': __import__('datetime').datetime.now().isoformat()
        }
        lock_path.write_text(json.dumps(lock_payload))
    except Exception:
        pass

if __name__ == '__main__':
    main()
