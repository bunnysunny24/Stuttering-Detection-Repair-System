import argparse
from pathlib import Path
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import seaborn as sns

from calibrate_thresholds import load_model, gather_probs_and_labels, optimize_temperature, calibrate_thresholds


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def plot_roc_pr(probs, labels, out_dir: Path, class_names):
    out_dir.mkdir(parents=True, exist_ok=True)
    n_classes = labels.shape[1]
    roc_data = {}
    pr_data = {}
    for i in range(n_classes):
        y_true = labels[:, i]
        y_score = probs[:, i]
        try:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
        except Exception:
            fpr, tpr, roc_auc = np.array([0, 1]), np.array([0, 1]), float('nan')
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            ap = average_precision_score(y_true, y_score)
        except Exception:
            precision, recall, ap = np.array([0, 1]), np.array([1, 0]), float('nan')

        # ROC
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, label=f'AUC={roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title(f'ROC — {class_names[i]}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f'roc_class_{i}.png')
        plt.close()

        # PR
        plt.figure(figsize=(5, 5))
        plt.plot(recall, precision, label=f'AP={ap:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall — {class_names[i]}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f'pr_class_{i}.png')
        plt.close()

        roc_data[class_names[i]] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': float(roc_auc)}
        pr_data[class_names[i]] = {'precision': precision.tolist(), 'recall': recall.tolist(), 'ap': float(ap)}

    # Save summary
    json.dump({'roc': roc_data, 'pr': pr_data}, open(out_dir / 'roc_pr_data.json', 'w'), indent=2)


def plot_confusion_and_metrics(probs, labels, thresholds, out_dir: Path, class_names):
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = (probs >= thresholds[np.newaxis, :]).astype(int)
    truths = (labels > 0).astype(int)
    per_class = {}
    for i, cname in enumerate(class_names):
        y_t = truths[:, i]
        y_p = preds[:, i]
        tp = int(((y_t == 1) & (y_p == 1)).sum())
        fp = int(((y_t == 0) & (y_p == 1)).sum())
        fn = int(((y_t == 1) & (y_p == 0)).sum())
        tn = int(((y_t == 0) & (y_p == 0)).sum())
        per_class[cname] = {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}

        # confusion matrix heatmap
        cm = np.array([[tn, fp], [fn, tp]])
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Pred')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix — {cname}')
        plt.tight_layout()
        plt.savefig(out_dir / f'confusion_{i}_{cname.replace(" ","_")}.png')
        plt.close()

    json.dump({'per_class_confusion': per_class, 'thresholds': thresholds.tolist()}, open(out_dir / 'confusion_summary.json', 'w'), indent=2)


def plot_loss_vs_epoch(metrics_json_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_json_path, 'r') as f:
        metrics = json.load(f)

    train = metrics.get('train', [])
    val = metrics.get('val', [])
    epochs = max(len(train), len(val))
    train_losses = [e.get('loss', None) for e in train]
    val_losses = [e.get('loss', None) for e in val]

    plt.figure(figsize=(6, 4))
    if any(x is not None for x in train_losses):
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    if any(x is not None for x in val_losses):
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training / Validation Loss vs Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'loss_vs_epoch.png')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-dir', default='datasets/features')
    parser.add_argument('--metrics-json', default='output/training_20260217_013705/improved_90plus_metrics.json')
    parser.add_argument('--out-dir', default='output/eval_from_checkpoint')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=2)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model (auto-detect) using calibrate_thresholds helper
    model = load_model(args.checkpoint, device)

    # Gather logits and labels
    logits, labels = gather_probs_and_labels(model, args.data_dir, device, batch_size=args.batch_size, num_workers=args.num_workers)

    # Optimize temperature (safe guard)
    try:
        temperature = optimize_temperature(logits, labels, device=device, lr=0.02, steps=200)
    except Exception:
        temperature = 1.0

    probs = 1.0 / (1.0 + np.exp(-logits / float(temperature)))

    # Calibrate thresholds (maximize F1)
    thresholds, metrics = calibrate_thresholds(probs, labels)

    class_names = ['Prolongation', 'Block', 'Sound Repetition', 'Word Repetition', 'Interjection']

    plot_roc_pr(probs, labels, out_dir=out_dir, class_names=class_names)
    plot_confusion_and_metrics(probs, labels, np.array(thresholds), out_dir=out_dir, class_names=class_names)
    plot_loss_vs_epoch(Path(args.metrics_json), out_dir=out_dir)

    # Save thresholds and per-class metrics
    json.dump({'thresholds': thresholds.tolist(), 'per_class_metrics': metrics}, open(out_dir / 'thresholds_and_metrics.json', 'w'), indent=2)

    print('Saved ROC/PR, confusion matrices, loss plot, and thresholds to', out_dir)


if __name__ == '__main__':
    main()
