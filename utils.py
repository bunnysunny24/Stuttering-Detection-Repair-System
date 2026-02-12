"""Utility functions: metrics and helpers.

Provides multi-label F1/precision/recall computation using sklearn where
available. These are minimal helpers used by the training script.
"""
from typing import Tuple
import numpy as np

try:
    from sklearn.metrics import precision_recall_fscore_support
except Exception:
    precision_recall_fscore_support = None

import os
import json
import torch
import numpy as _np


class FocalLoss(torch.nn.Module):
    """Multi-label focal loss (for logits).

    Implements a numerically-stable focal loss built on BCEWithLogitsLoss.
    alpha: weighting factor for positive class (float or list per-class)
    gamma: focusing parameter
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: (N, C), targets: {0,1} same shape
        prob = torch.sigmoid(logits)
        targets = targets.type_as(prob)
        ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        if isinstance(self.alpha, (float, int)):
            loss = loss * self.alpha
        else:
            # per-class alpha
            alpha_t = torch.as_tensor(self.alpha, dtype=loss.dtype, device=loss.device)
            loss = loss * alpha_t.unsqueeze(0)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def save_epoch_diagnostics(out_dir: str, epoch: int, train_metrics: dict, val_metrics: dict, logits_stats: dict):
    """Persist diagnostics for an epoch into a human-readable text file and a JSON summary.

    - writes Models/diagnostics/epoch_{epoch:02d}.txt with readable layout
    - appends a summary line to Models/diagnostics/diagnostics_summary.jsonl
    """
    os.makedirs(out_dir, exist_ok=True)
    txt_path = os.path.join(out_dir, f"epoch_{epoch:02d}.txt")
    jsonl_path = os.path.join(out_dir, "diagnostics_summary.jsonl")

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Epoch {epoch}\n")
        f.write("\n--- TRAIN METRICS ---\n")
        for k, v in (train_metrics or {}).items():
            f.write(f"{k}: {v}\n")
        f.write("\n--- VAL METRICS ---\n")
        for k, v in (val_metrics or {}).items():
            f.write(f"{k}: {v}\n")
        f.write("\n--- LOGITS STATS ---\n")
        for k, v in (logits_stats or {}).items():
            f.write(f"{k}: {v}\n")

    # write one-line json record for quick parsing
    record = {
        'epoch': epoch,
        'train': train_metrics,
        'val': val_metrics,
        'logits': logits_stats,
    }
    with open(jsonl_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record) + "\n")


def multilabel_f1(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute precision/recall/f1 for multi-label binary predictions.

    y_true, y_pred: shape (N, C) arrays
    """
    if y_pred.dtype != np.float32:
        y_pred = y_pred.astype(np.float32)
    pred_labels = (y_pred >= threshold).astype(int)
    if precision_recall_fscore_support is not None:
        p, r, f1, _ = precision_recall_fscore_support(y_true, pred_labels, average=None, zero_division=0)
        micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(y_true, pred_labels, average="micro", zero_division=0)
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, pred_labels, average="macro", zero_division=0)
        return {
            "per_class_p": p.tolist(),
            "per_class_r": r.tolist(),
            "per_class_f1": f1.tolist(),
            "micro_p": float(micro_p),
            "micro_r": float(micro_r),
            "micro_f1": float(micro_f1),
            "macro_f1": float(macro_f1),
        }
    else:
        # naive fallback: compute per-class precision/recall/f1
        eps = 1e-8
        C = y_true.shape[1]
        per = {"per_class_p": [], "per_class_r": [], "per_class_f1": []}
        for c in range(C):
            y_t = y_true[:, c]
            y_p = pred_labels[:, c]
            tp = int(((y_t == 1) & (y_p == 1)).sum())
            fp = int(((y_t == 0) & (y_p == 1)).sum())
            fn = int(((y_t == 1) & (y_p == 0)).sum())
            p = tp / (tp + fp + eps)
            r = tp / (tp + fn + eps)
            f1 = 2 * p * r / (p + r + eps)
            per["per_class_p"].append(float(p))
            per["per_class_r"].append(float(r))
            per["per_class_f1"].append(float(f1))
        per["micro_f1"] = float(np.mean(per["per_class_f1"]))
        per["macro_f1"] = float(np.mean(per["per_class_f1"]))
        return per
