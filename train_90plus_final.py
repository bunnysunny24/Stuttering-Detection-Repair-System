"""
TRAINING SCRIPT FOR 90+ ACCURACY
Replaces: Models/improved_train_enhanced.py

Improvements:
1. Uses 8-layer model (model_improved_90plus.py) 
2. Accepts 123 channels (enhanced features)
3. Data augmentation (pitch shift, time stretch, masking)
4. Stronger regularization (dropout 0.4-0.5)
5. Better class weighting for imbalance
6. Optimized thresholds (0.70, 0.75, 0.70, 0.75, 0.70)
7. Scheduled learning rate with warmup
8. Advanced early stopping
"""

import os
import sys
import json
import argparse
import logging
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from datetime import datetime
import pickle
import shutil
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Suppress noisy decorative separator lines printed by imported modules or
# legacy scripts. This replaces sys.stdout/sys.stderr with a light filter that
# drops writes that consist only of repeated '=' characters (and optional
# whitespace/newline) so the training progress output remains readable.
import sys as _sys
_original_stdout = _sys.stdout
_original_stderr = _sys.stderr

class _SeparatorFilter:
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def write(self, s):
        try:
            # Only consider string-like writes; ignore non-str
            if not isinstance(s, str):
                return self._wrapped.write(s)

            stripped = s.strip('\r\n')
            # If the stripped content is only '=' characters and long, drop it
            if len(stripped) >= 10 and set(stripped) == {'='}:
                return
            return self._wrapped.write(s)
        except Exception:
            # On any error, fallback to original writer
            return self._wrapped.write(s)

    def flush(self):
        try:
            return self._wrapped.flush()
        except Exception:
            return None

_sys.stdout = _SeparatorFilter(_original_stdout)
_sys.stderr = _SeparatorFilter(_original_stderr)

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import models and utilities
from model_improved_90plus import ImprovedStutteringCNN
from constants import TOTAL_CHANNELS, NUM_CLASSES, SCHEDULER_PATIENCE, THRESH_SEARCH_START, THRESH_SEARCH_END, THRESH_SEARCH_STEP, THRESHOLD_OPT_EPOCHS, AUG_TIME_MASK_P, AUG_FREQ_MASK_P, AUG_NOISE_P, AUG_STRETCH_P, AUG_PITCH_P, AUG_SNR_P
from utils import FocalLoss as UtilsFocalLoss

# GPU optimization
# GPU optimization - MAXIMUM PERFORMANCE
torch.backends.cudnn.benchmark = True
# Note: avoid forcing deterministic/non-deterministic modes and thread counts here
# to allow the environment or user to control reproducibility and parallelism.
os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')


# Use centralized FocalLoss implementation from utils
# `UtilsFocalLoss` supports `pos_weight` and matches usage below.


class LabelSmoothingBCELoss(nn.Module):
    """Binary cross entropy with label smoothing for extreme imbalance."""
    def __init__(self, smoothing=0.1, pos_weight=None):
        super().__init__()
        self.smoothing = float(smoothing)
        self.pos_weight = pos_weight

    def forward(self, predictions, targets):
        # Smooth targets toward 0.5
        smooth_t = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return torch.nn.functional.binary_cross_entropy_with_logits(predictions, smooth_t, pos_weight=self.pos_weight)


class ExponentialMovingAverage:
    """Simple EMA implementation for model parameters with context manager.

    Usage:
        ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
        ...
        ema.update()
        with ema.average_parameters():
            run_validation()
    """
    def __init__(self, parameters, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.collected = []
        for p in parameters:
            if p.requires_grad:
                self.shadow[p] = p.data.clone().detach()
                self.collected.append(p)

    def update(self):
        for p in self.collected:
            assert p in self.shadow
            new_avg = (1.0 - self.decay) * p.data + self.decay * self.shadow[p]
            self.shadow[p].copy_(new_avg)

    from contextlib import contextmanager

    @contextmanager
    def average_parameters(self):
        # Swap model parameters with EMA shadow values
        backup = {}
        for p in self.collected:
            backup[p] = p.data.clone()
            p.data.copy_(self.shadow[p])
        try:
            yield
        finally:
            for p in self.collected:
                p.data.copy_(backup[p])


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

class AudioAugmentation:
    """
    Apply augmentation to audio features for better generalization.
    Works with mel-spectrogram and enhanced features.
    """
    
    def __init__(self, augment_prob=0.5, time_mask_p=AUG_TIME_MASK_P, freq_mask_p=AUG_FREQ_MASK_P, noise_p=AUG_NOISE_P, stretch_p=AUG_STRETCH_P, pitch_p=AUG_PITCH_P, snr_p=AUG_SNR_P):
        self.augment_prob = augment_prob
        self.time_mask_p = float(time_mask_p)
        self.freq_mask_p = float(freq_mask_p)
        self.noise_p = float(noise_p)
        self.stretch_p = float(stretch_p)
        self.pitch_p = float(pitch_p)
        self.snr_p = float(snr_p)
    
    def __call__(self, spectrogram):
        """Apply random augmentations."""
        if np.random.random() < self.augment_prob:
            spectrogram = self._apply_augmentations(spectrogram)
        return spectrogram
    
    def _apply_augmentations(self, spec):
        """Apply one or more augmentations."""
        choices = ['time_mask', 'freq_mask', 'noise', 'stretch', 'pitch', 'snr_noise']
        probs = np.array([self.time_mask_p, self.freq_mask_p, self.noise_p, self.stretch_p, self.pitch_p, self.snr_p], dtype=float)
        # Normalize probabilities to sum to 1.0 (robust to edits of constants)
        total = probs.sum()
        if total <= 0:
            probs = np.ones_like(probs) / float(len(probs))
        else:
            probs = probs / float(total)
        augmentation_type = np.random.choice(choices, p=probs)
        
        if augmentation_type == 'time_mask':
            # Mask random time segments (simulate speech gaps)
            spec = self._time_masking(spec)
        elif augmentation_type == 'freq_mask':
            # Mask random frequency bands
            spec = self._freq_masking(spec)
        elif augmentation_type == 'noise':
            # Add gaussian noise
            spec = self._add_noise(spec)
        elif augmentation_type == 'snr_noise':
            spec = self._add_noise_snr(spec)
        elif augmentation_type == 'pitch':
            spec = self._pitch_shift(spec)
        else:  # stretch
            # Time stretching (simulate faster/slower speech)
            spec = self._time_stretch(spec)
        
        return spec
    
    def _time_masking(self, spec):
        """Mask random time region."""
        time_length = spec.shape[1]
        # Determine mask length robustly for short clips
        mask_max = max(1, int(time_length * 0.3))
        mask_min = min(5, mask_max)
        if mask_min >= mask_max:
            mask_len = mask_max
        else:
            mask_len = np.random.randint(mask_min, mask_max + 1)

        # Choose start position safely
        start_max = max(0, time_length - mask_len)
        if start_max <= 0:
            mask_start = 0
        else:
            mask_start = np.random.randint(0, start_max + 1)

        spec = spec.copy()
        end = mask_start + mask_len
        spec[:, mask_start:end] = np.mean(spec)
        return spec
    
    def _freq_masking(self, spec):
        """Mask random frequency band."""
        freq_length = spec.shape[0]
        # Determine mask length robustly for narrow frequency axes
        mask_max = max(1, int(freq_length * 0.2))
        mask_min = min(5, mask_max)
        if mask_min >= mask_max:
            mask_len = mask_max
        else:
            mask_len = np.random.randint(mask_min, mask_max + 1)

        # Choose start position safely
        start_max = max(0, freq_length - mask_len)
        if start_max <= 0:
            mask_start = 0
        else:
            mask_start = np.random.randint(0, start_max + 1)

        spec = spec.copy()
        end = mask_start + mask_len
        spec[mask_start:end, :] = np.mean(spec)
        return spec
    
    def _add_noise(self, spec):
        """Add gaussian noise."""
        noise = np.random.normal(0, 0.01, spec.shape)
        return spec + noise

    def _add_noise_snr(self, spec):
        """Add gaussian noise scaled to achieve a random SNR between 10 and 30 dB."""
        # compute signal power
        power = np.mean(spec ** 2)
        snr_db = np.random.uniform(10.0, 30.0)
        snr = 10 ** (snr_db / 10.0)
        noise_power = power / max(snr, 1e-9)
        noise = np.random.normal(0, np.sqrt(noise_power), spec.shape)
        return spec + noise

    def _pitch_shift(self, spec):
        """Simple pitch shift by rolling frequency axis.

        This is a crude approximation applied on spectrogram bins: shifting
        up/down by a small number of bins and filling the edge with mean.
        """
        max_shift = max(1, int(spec.shape[0] * 0.05))
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift == 0:
            return spec
        shifted = np.roll(spec, shift, axis=0)
        if shift > 0:
            # filled top rows with mean of original
            shifted[:shift, :] = np.mean(spec, axis=1)[:shift, None]
        else:
            shifted[shift:, :] = np.mean(spec, axis=1)[shift:, None]
        return shifted
    
    def _time_stretch(self, spec):
        """Time stretching (simple implementation)."""
        stretch_factor = np.random.uniform(0.95, 1.05)
        orig_length = spec.shape[1]
        new_length = max(1, int(orig_length * stretch_factor))

        # If shorter, crop then pad back to original length
        if new_length < orig_length:
            cropped = spec[:, :new_length]
            pad_width = orig_length - new_length
            spec = np.pad(cropped, ((0, 0), (0, pad_width)), mode='edge')
        else:
            # If longer or equal, pad by repeating edge columns then trim to original length
            if new_length > orig_length:
                padded = np.pad(spec, ((0, 0), (0, new_length - orig_length)), mode='edge')
                spec = padded[:, :orig_length]
            else:
                # unchanged
                spec = spec[:, :orig_length]

        return spec


# ============================================================================
# COLLATE FUNCTION FOR VARIABLE LENGTH SEQUENCES
# ============================================================================

def collate_variable_length(batch):
    """FIXED: Custom collate function to handle variable length sequences."""
    specs = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Handle 1D embeddings
    if specs[0].dim() == 1:
        specs_batch = torch.stack(specs, dim=0)
        labels_batch = torch.stack(labels, dim=0)
        return specs_batch, labels_batch

    # FIXED: Ensure all specs are 2D (channels, time)
    processed_specs = []
    for spec in specs:
        if spec.dim() == 3:
            spec = spec.squeeze(0)  # Remove batch dimension
        elif spec.dim() == 1:
            spec = spec.unsqueeze(0)
        processed_specs.append(spec)
    
    # Find max time length
    max_time = max(spec.shape[-1] for spec in processed_specs)

    # Pad to max length
    padded_specs = []
    for spec in processed_specs:
        if spec.shape[-1] < max_time:
            pad_amount = max_time - spec.shape[-1]
            spec = torch.nn.functional.pad(spec, (0, pad_amount))
        padded_specs.append(spec)

    # FIXED: Use torch.stack instead of torch.cat
    specs_batch = torch.stack(padded_specs, dim=0)  # (batch, channels, time)
    labels_batch = torch.stack(labels, dim=0)

    return specs_batch, labels_batch


# ============================================================================
# DATASET
# ============================================================================

class AudioDataset(Dataset):
    """Load preprocessed features with augmentation."""
    
    def __init__(self, data_dir, split='train', augment=True, aug_time_p=None, aug_freq_p=None, aug_noise_p=None, aug_stretch_p=None, aug_pitch_p=None, aug_snr_p=None, augment_prob=0.5):
        self.data_dir = Path(data_dir)
        self.split_dir = self.data_dir / split
        self.files = sorted(self.split_dir.glob('**/*.npz'))
        self.augment = augment and split == 'train'
        # Use provided augmentation probabilities if available, otherwise defaults from constants
        self.augmentation = AudioAugmentation(
            augment_prob=augment_prob,
            time_mask_p=(AUG_TIME_MASK_P if aug_time_p is None else aug_time_p),
            freq_mask_p=(AUG_FREQ_MASK_P if aug_freq_p is None else aug_freq_p),
            noise_p=(AUG_NOISE_P if aug_noise_p is None else aug_noise_p),
            stretch_p=(AUG_STRETCH_P if aug_stretch_p is None else aug_stretch_p),
            pitch_p=(AUG_PITCH_P if aug_pitch_p is None else aug_pitch_p),
            snr_p=(AUG_SNR_P if aug_snr_p is None else aug_snr_p),
        )
        
        if len(self.files) == 0:
            raise ValueError(f"No files found in {self.split_dir}")
        
        print(f"Loaded {len(self.files)} files from {split}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        try:
            data = np.load(self.files[idx])
        except Exception as e:
            # Corrupted NPZ - remove it from the list and retry next sample
            bad_file = self.files[idx]
            print(f"Warning: failed to load {bad_file}: {e} -- removing from dataset")
            try:
                self.files.pop(idx)
            except Exception:
                pass
            if len(self.files) == 0:
                raise RuntimeError(f"No valid files left in dataset after removing {bad_file}")
            # Try to fetch a different sample at the same index (now points to next file)
            return self.__getitem__(idx % len(self.files))
        # Prefer pretrained embeddings if present
        if 'embedding' in data:
            emb = torch.from_numpy(data['embedding']).float()
            # Return 1D embedding vector and labels
            if 'labels' in data:
                labels = torch.from_numpy(data['labels']).float()
            else:
                labels = torch.zeros(NUM_CLASSES, dtype=torch.float32)
            return emb, labels

        spectrogram = torch.from_numpy(data['spectrogram']).float()

        # Load labels if available, otherwise create dummy labels (all zeros)
        if 'labels' in data:
            # Labels in NPZs were stored as ordinal/integer (0..N).
            # Convert to binary presence indicator: positive if label > 0.
            lbl = np.asarray(data['labels'])
            lbl_bin = (lbl > 0).astype(np.float32)
            labels = torch.from_numpy(lbl_bin).float()
        else:
            labels = torch.zeros(NUM_CLASSES, dtype=torch.float32)

        # FIXED: Ensure 2D format (channels, time) - NO batch dimension
        if spectrogram.dim() == 1:
            spectrogram = spectrogram.unsqueeze(0)
        elif spectrogram.dim() == 3:
            spectrogram = spectrogram.squeeze(0)  # Remove batch dim

        # FIXED: Validate and fix channel count
        if spectrogram.shape[0] != TOTAL_CHANNELS:
            print(f"Warning: Expected {TOTAL_CHANNELS} channels, got {spectrogram.shape[0]}")
            if spectrogram.shape[0] < TOTAL_CHANNELS:
                pad_channels = TOTAL_CHANNELS - spectrogram.shape[0]
                # pad: (time_left, time_right, channel_top, channel_bottom)
                spectrogram = torch.nn.functional.pad(spectrogram, (0, 0, 0, pad_channels))
            else:
                spectrogram = spectrogram[:TOTAL_CHANNELS, :]

        # Apply augmentation if training
        if self.augment:
            spec_np = spectrogram.numpy()
            spec_np = self.augmentation(spec_np)
            spectrogram = torch.from_numpy(spec_np).float()

        return spectrogram, labels


# ============================================================================
# METRICS TRACKER
# ============================================================================

class MetricsTracker:
    """Track training and validation metrics."""
    
    STUTTER_CLASSES = [
        'Prolongation', 'Block', 'Sound Repetition', 
        'Word Repetition', 'Interjection'
    ]
    
    def __init__(self):
        self.metrics = {
            'train': [],
            'val': [],
            'thresholds': [],
            'optimized_thresholds': []
        }
        self.best_epoch_info = None
    
    def record(self, phase, loss, y_true, y_pred_probs, y_pred_binary, thresholds=None):
        """Record epoch metrics."""
        if len(y_true) == 0:
            return
        
        y_true = np.vstack(y_true) if isinstance(y_true[0], np.ndarray) else np.array(y_true)
        y_pred_probs = np.vstack(y_pred_probs) if isinstance(y_pred_probs[0], np.ndarray) else np.array(y_pred_probs)
        y_pred_binary = np.vstack(y_pred_binary) if isinstance(y_pred_binary[0], np.ndarray) else np.array(y_pred_binary)
        
        # Calculate per-label metrics for multilabel classification
        precision_per_class = []
        recall_per_class = []
        f1_per_class = []
        support_per_class = []
        roc_auc_list = []
        
        for i in range(y_true.shape[1]):
            y_true_i = y_true[:, i]
            y_pred_i = y_pred_binary[:, i]
            y_pred_probs_i = y_pred_probs[:, i]
            
            # Per-label metrics
            tp = np.sum((y_true_i == 1) & (y_pred_i == 1))
            fp = np.sum((y_true_i == 0) & (y_pred_i == 1))
            fn = np.sum((y_true_i == 1) & (y_pred_i == 0))
            tn = np.sum((y_true_i == 0) & (y_pred_i == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            precision_per_class.append(precision)
            recall_per_class.append(recall)
            f1_per_class.append(f1)
            support_per_class.append(np.sum(y_true_i))
            
            # ROC-AUC per label - use rank-based AUC (Mann-Whitney U)
            try:
                if len(np.unique(y_true_i)) > 1:  # Has both 0 and 1
                    n_pos = int(np.sum(y_true_i == 1))
                    n_neg = int(len(y_true_i) - n_pos)
                    if n_pos > 0 and n_neg > 0:
                        # compute ranks (1-based)
                        ranks = np.argsort(np.argsort(y_pred_probs_i)) + 1
                        sum_ranks_pos = np.sum(ranks[y_true_i == 1])
                        auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
                        roc_auc_list.append(float(auc))
                    else:
                        roc_auc_list.append(0.5)
                else:
                    roc_auc_list.append(0.5)
            except Exception:
                roc_auc_list.append(0.5)
        
        precision_per_class = np.array(precision_per_class)
        recall_per_class = np.array(recall_per_class)
        f1_per_class = np.array(f1_per_class)
        support_per_class = np.array(support_per_class)
        roc_auc_list = np.array(roc_auc_list)
        
        # Macro averaging (simple average across labels)
        f1_mac = np.mean(f1_per_class)
        precision_mac = np.mean(precision_per_class)
        recall_mac = np.mean(recall_per_class)
        
        # Weighted averaging (weighted by support)
        total_support = np.sum(support_per_class)
        if total_support > 0:
            f1_mic = np.sum(f1_per_class * support_per_class) / total_support
            precision_mic = np.sum(precision_per_class * support_per_class) / total_support
            recall_mic = np.sum(recall_per_class * support_per_class) / total_support
        else:
            f1_mic = precision_mic = recall_mic = 0.0
        
        roc_auc_macro = np.mean(roc_auc_list)
        hamming_loss = np.mean(y_true != y_pred_binary)
        
        epoch_metrics = {
            'loss': loss,
            'f1_macro': float(f1_mac),
            'f1_micro': float(f1_mic),
            'precision_macro': float(precision_mac),
            'recall_macro': float(recall_mac),
            'precision_micro': float(precision_mic),
            'recall_micro': float(recall_mic),
            'hamming_loss': float(hamming_loss),
            'roc_auc_macro': float(roc_auc_macro),
            'per_class': {
                self.STUTTER_CLASSES[i]: {
                    'precision': float(precision_per_class[i]),
                    'recall': float(recall_per_class[i]),
                    'f1': float(f1_per_class[i]),
                    'support': int(support_per_class[i]),
                    'roc_auc': float(roc_auc_list[i])
                }
                for i in range(len(self.STUTTER_CLASSES))
            }
        }
        
        self.metrics[phase].append(epoch_metrics)
        
        if phase == 'val' and thresholds is not None:
            self.metrics['optimized_thresholds'].append({
                'epoch': len(self.metrics['val']),
                'thresholds': {
                    self.STUTTER_CLASSES[i]: float(thresholds[i])
                    for i in range(len(self.STUTTER_CLASSES))
                }
            })
    
    def save(self, filepath):
        """Save metrics to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Saved metrics to {filepath}")


# ============================================================================
# TRAINER
# ============================================================================

class Trainer:
    """Model training with 90+ accuracy optimizations."""
    
    def __init__(self, model, train_loader, val_loader, device, use_ipex=False, model_name='improved_90plus', logger=None, training_timestamp=None, early_stop_patience=None, class_weights=None, features_dir='datasets/features', use_ema=False, ema_decay=0.999, use_label_smoothing=False, label_smoothing=0.1, grad_clip=1.0, accumulate_steps=1, sched_patience=None, loss_type='focal', learning_rate=1e-4, weight_decay=1e-5, focal_gamma=2.0, scheduler_type='reduce', max_lr=None, seed=None, aug_time_p=None, aug_freq_p=None, aug_noise_p=None, aug_stretch_p=None, aug_pitch_p=None, aug_snr_p=None, save_thresholds=False, thresh_min_precision=0.2, neutral_pos_weight=False):
        # MixUp strength (alpha) - applied in train_epoch if > 0
        self.mixup_alpha = float(0.0)
        self.model = model.to(device)
        self.device = device
        # Determine device type robustly for non-torch-device backends (e.g., DirectML)
        self.device_type = getattr(device, 'type', None)
        try:
            # torch.device has .type
            if self.device_type is None and isinstance(device, torch.device):
                self.device_type = device.type
        except Exception:
            pass
        if self.device_type is None:
            # Default to 'cpu' when unknown
            self.device_type = 'cpu'
        self.model_name = model_name
        self.logger = logger
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.training_timestamp = training_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create training-specific checkpoint directory
        self.checkpoint_dir = Path('Models/checkpoints') / f'training_{self.training_timestamp}'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training config
        self.learning_rate = float(learning_rate)
        # Scheduler and seed/config
        self.focal_gamma = float(focal_gamma)
        self.scheduler_type = str(scheduler_type)
        self.max_lr = float(max_lr) if max_lr is not None else None
        self.save_thresholds = bool(save_thresholds)
        # Threshold selection guard: require minimum precision when choosing thresholds
        self.thresh_min_precision = float(thresh_min_precision) if thresh_min_precision is not None else 0.2
        # If True and oversampling is used, force neutral pos_weight to avoid double-upweighting
        self.neutral_pos_weight = bool(neutral_pos_weight)
        # Keep augmentation parameters available for downstream use
        self.aug_time_p = aug_time_p
        self.aug_freq_p = aug_freq_p
        self.aug_noise_p = aug_noise_p
        self.aug_stretch_p = aug_stretch_p
        self.aug_pitch_p = aug_pitch_p
        self.aug_snr_p = aug_snr_p
        # Allow pipeline to override early stop patience
        self.early_stop_patience = early_stop_patience if early_stop_patience is not None else 50
        self.patience_counter = 0
        self.best_f1 = 0.0
        self.total_epochs = 0
        
        # Optimizers
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=float(weight_decay)
        )
        
        # Scheduler with warmup
        patience_val = int(sched_patience) if sched_patience is not None else SCHEDULER_PATIENCE
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=patience_val
        )

        # Store placeholder for OneCycle/other schedulers - will be configured in train()
        self._external_scheduler = None
        
        # Mixed precision: enable only on CUDA
        self.use_amp = (self.device_type == 'cuda')
        self.scaler = GradScaler() if self.use_amp else None

        # Optional Intel Extension for PyTorch (IPEX) optimization for CPU
        self.use_ipex = bool(use_ipex)
        if self.use_ipex:
            try:
                import intel_extension_for_pytorch as ipex
                try:
                    # ipex.optimize returns (model, optimizer) when optimizer provided
                    self.model, self.optimizer = ipex.optimize(self.model, optimizer=self.optimizer)
                except Exception:
                    # older/newer versions may have different returns; try safe call
                    ipex.optimize(self.model)
                print('Enabled IPEX optimizations for CPU')
            except Exception as e:
                print('IPEX not available or failed to initialize:', e)

        # Gradient accumulation support
        self.accumulate_steps = int(accumulate_steps) if accumulate_steps and int(accumulate_steps) > 0 else 1
        
        # Metrics
        self.metrics = MetricsTracker()
        
        # Class weights: compute dynamically from dataset if not provided
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
            print("Using provided class weights:")
        else:
            self.class_weights = self._compute_class_weights_from_features(features_dir)

        class_names = ['Prolongation', 'Block', 'Sound Repetition', 'Word Repetition', 'Interjection']
        print("Class weights (pos_weight for BCE):")
        for i, name in enumerate(class_names):
            print(f"  {name:20s}: {self.class_weights[i]:6.3f}")

        # Loss selection: support focal, label-smoothing or BCE
        self.use_label_smoothing = bool(use_label_smoothing)
        self.label_smoothing = float(label_smoothing)
        self.loss_type = loss_type
        if self.loss_type == 'bce':
            # Use BCEWithLogitsLoss; apply pos_weight only if provided (and not neutral)
            try:
                if self.class_weights is not None:
                    self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
                else:
                    self.criterion = torch.nn.BCEWithLogitsLoss()
                print('Using BCEWithLogitsLoss')
            except Exception:
                # Fallback
                self.criterion = torch.nn.BCEWithLogitsLoss()
        elif self.use_label_smoothing:
            self.criterion = LabelSmoothingBCELoss(smoothing=self.label_smoothing, pos_weight=self.class_weights)
            print(f"Using LabelSmoothingBCELoss(smoothing={self.label_smoothing})")
        else:
            # Pass focal gamma if provided
            try:
                self.criterion = UtilsFocalLoss(gamma=self.focal_gamma, pos_weight=self.class_weights)
            except Exception:
                self.criterion = UtilsFocalLoss(pos_weight=self.class_weights)
        
        # Thresholds - Will be optimized every epoch with smoothing
        self.optimal_thresholds = np.full(NUM_CLASSES, 0.5)
        self.previous_thresholds = np.full(NUM_CLASSES, 0.5)
        self.threshold_history = []  # Track last 3 thresholds for moving average
        
        # Threshold locking: Lock after epoch 2 (aggressive approach)
        self.thresholds_locked_epoch = -1  # When thresholds get locked

        # EMA (optional)
        self.use_ema = bool(use_ema)
        self.ema = None
        if self.use_ema:
            try:
                self.ema = ExponentialMovingAverage(self.model.parameters(), decay=float(ema_decay))
                print(f"Enabled EMA with decay={ema_decay}")
            except Exception as e:
                print(f"Warning: failed to initialize EMA: {e}")

        # Gradient clipping max norm
        self.grad_clip = float(grad_clip)
        # Ensure mixup attribute exists (may be set by caller)
        if not hasattr(self, 'mixup_alpha'):
            self.mixup_alpha = 0.0
        # Optionally set deterministic behavior if seed provided
        if seed is not None:
            try:
                s = int(seed)
                import random as _py_random
                np.random.seed(s)
                _py_random.seed(s)
                torch.manual_seed(s)
                try:
                    torch.cuda.manual_seed_all(s)
                except Exception:
                    pass
                # enforce deterministic cuDNN for reproducibility (best-effort)
                try:
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                except Exception:
                    pass
                print(f"Set deterministic seed = {s}")
            except Exception:
                pass
    
    def optimize_thresholds(self, y_true, y_pred_probs):
        """Find best thresholds for each class - CONSERVATIVE approach with smoothing."""
        num_classes = y_true.shape[1]
        optimal_thresholds = np.zeros(num_classes)
        
        for class_idx in range(num_classes):
            best_score = -1
            best_thresh = 0.5
            
            y_true_bin = y_true[:, class_idx].astype(int)
            
            if y_true_bin.sum() == 0:
                optimal_thresholds[class_idx] = self.previous_thresholds[class_idx]
                continue
            
            # Search using constants: select threshold that maximizes F1 per-class
            # Collect candidates that meet minimum precision (if configured)
            candidates = []
            best_precision_only = {'precision': -1.0, 'f1': -1.0, 'thresh': best_thresh}
            for threshold in np.arange(THRESH_SEARCH_START, THRESH_SEARCH_END + 1e-8, THRESH_SEARCH_STEP):
                y_pred = (y_pred_probs[:, class_idx] > threshold).astype(int)

                # Compute metrics
                tp = np.sum((y_true_bin == 1) & (y_pred == 1))
                fp = np.sum((y_true_bin == 0) & (y_pred == 1))
                fn = np.sum((y_true_bin == 1) & (y_pred == 0))

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

                # Use F1 as the selection metric to balance precision & recall robustly
                if (precision + recall) > 0:
                    score = 2.0 * (precision * recall) / (precision + recall)
                else:
                    score = 0.0

                # Track best precision-only candidate (used as fallback)
                if precision > best_precision_only['precision'] or (precision == best_precision_only['precision'] and score > best_precision_only['f1']):
                    best_precision_only = {'precision': precision, 'f1': score, 'thresh': threshold}

                # If this threshold meets the minimum precision requirement, consider it
                try:
                    pmin = float(self.thresh_min_precision)
                except Exception:
                    pmin = 0.0

                if precision >= pmin:
                    candidates.append((score, threshold))

            # Choose best among candidates (by F1). If none meet precision requirement, fall back
            # to the threshold that maximizes precision (to favor higher thresholds and avoid low-precision picks).
            if len(candidates) > 0:
                # pick candidate with highest F1 (score)
                candidates.sort(key=lambda x: x[0], reverse=True)
                best_score, best_thresh = candidates[0]
            else:
                best_thresh = best_precision_only['thresh']
            
            # SMOOTH thresholds with moving average to prevent oscillation
            # Use exponential decay: 60% new, 40% previous
            smoothed_thresh = (best_thresh * 0.6) + (self.previous_thresholds[class_idx] * 0.4)
            optimal_thresholds[class_idx] = smoothed_thresh
        
        # Save current thresholds for next epoch
        self.previous_thresholds = optimal_thresholds.copy()
        self.threshold_history.append(optimal_thresholds.copy())
        
        return optimal_thresholds

    def _compute_class_weights_from_features(self, features_dir):
        """Scan feature NPZ files to compute positive class weights for BCE pos_weight.
        pos_weight = (neg_count / pos_count) per class, clipped to [1.0, 50.0]
        """
        import numpy as _np
        from pathlib import Path as _Path
        feat_path = _Path(features_dir) / 'train'
        counts_pos = _np.zeros(NUM_CLASSES, dtype=_np.int64)
        total_files = 0
        if feat_path.exists():
            for f in feat_path.glob('*.npz'):
                try:
                    d = _np.load(f)
                    labels = d.get('labels')
                    if labels is None:
                        continue
                    labels = _np.asarray(labels)
                    # Treat any non-zero label as positive (binary presence)
                    labels_bin = (labels > 0).astype(int)
                    counts_pos += labels_bin
                    total_files += 1
                except Exception:
                    continue

        # Avoid division by zero: if no positives, set pos_count=1 to keep pos_weight large
        counts_pos = counts_pos.astype(float)
        pos = counts_pos
        neg = float(max(1, total_files)) - pos
        pos_weight = _np.where(pos > 0, neg / pos, _np.clip( (neg + 1.0), 1.0, 50.0))
        # Clip extreme values
        pos_weight = _np.clip(pos_weight, 1.0, 50.0)
        return torch.tensor(pos_weight, dtype=torch.float32).to(self.device)
    
    def train_epoch(self, epoch):
        """Train one epoch."""
        self.model.train()  # CRITICAL: Enable batch norm training mode
        total_loss = 0.0
        
        # GPU synchronization
        if self.device_type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        y_true_all = []
        y_pred_probs_all = []
        y_pred_binary_all = []
        
        pbar = tqdm(self.train_loader, desc=f"EPOCH {epoch+1} [TRAIN]")
        for batch_idx, (X, y) in enumerate(pbar):
            X = X.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            # Optional MixUp augmentation applied on-the-fly
            if self.mixup_alpha and self.mixup_alpha > 0.0:
                # sample lambda from Beta distribution
                lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                if lam < 0.0:
                    lam = 0.0
                # shuffle batch
                idx = torch.randperm(X.size(0))
                X_shuf = X[idx]
                y_shuf = y[idx]
                X = lam * X + (1.0 - lam) * X_shuf
                y = lam * y + (1.0 - lam) * y_shuf

            # Zero grads at accumulation boundaries
            if (batch_idx % self.accumulate_steps) == 0:
                self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                logits = self.model(X)
                loss = self.criterion(logits, y)
                # Scale loss by accumulation steps so gradient magnitudes remain consistent
                loss = loss / float(self.accumulate_steps)

            # Backward (AMP if enabled)
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Step the optimizer only on accumulation boundary or final batch
            is_last_step = ((batch_idx + 1) % self.accumulate_steps) == 0 or (batch_idx + 1) == len(self.train_loader)
            if is_last_step:
                # Unscale, clip, step and update if using AMP
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                    self.optimizer.step()

                # Update EMA after optimizer step
                if self.use_ema and self.ema is not None:
                    try:
                        self.ema.update()
                    except Exception:
                        pass
                # Step external per-batch scheduler (OneCycleLR)
                if getattr(self, '_step_per_batch', False) and getattr(self, '_external_scheduler', None) is not None:
                    try:
                        self._external_scheduler.step()
                    except Exception:
                        pass

            total_loss += (loss.item() * float(self.accumulate_steps))

            with torch.no_grad():
                probs = torch.sigmoid(logits).cpu()
                y_pred_probs_all.append(probs.numpy())
                y_pred_binary = (probs.numpy() > self.optimal_thresholds).astype(float)
                y_pred_binary_all.append(y_pred_binary)
                y_true_all.append(y.cpu().numpy())

            pbar.set_postfix({'loss': f'{(loss.item()*self.accumulate_steps):.4f}'})

            # Clear GPU cache every 10 batches
            if batch_idx % 10 == 0 and self.device_type == 'cuda':
                torch.cuda.empty_cache()
        
        # Final GPU sync
        if self.device_type == 'cuda':
            torch.cuda.synchronize()
        pbar.close()
        
        avg_loss = total_loss / len(self.train_loader)
        self.metrics.record('train', avg_loss, y_true_all, y_pred_probs_all, y_pred_binary_all)
        
        # Print training metrics with per-class details
        train_metrics = self.metrics.metrics['train'][-1]
        train_f1 = train_metrics['f1_macro']
        train_auc = train_metrics['roc_auc_macro']
        
        # Get current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        
        print(f"[TRAIN EPOCH {epoch+1}]")
        print(f"  Overall: F1={train_f1:.4f}, AUC={train_auc:.4f}, Loss={avg_loss:.4f}, LR={current_lr:.2e}")
        print(f"  Hamming Loss (avg label error): {train_metrics['hamming_loss']:.4f}")
        print(f"  Per-Class Metrics:")
        for class_name, metrics_dict in train_metrics['per_class'].items():
            print(f"    {class_name:18s} | P={metrics_dict['precision']:.3f} R={metrics_dict['recall']:.3f} F1={metrics_dict['f1']:.3f} AUC={metrics_dict['roc_auc']:.3f}")
        
        return avg_loss
    
    def validate(self, epoch):
        """Validate one epoch (single pass). Uses EMA averaged parameters if enabled."""
        # Prepare model (use EMA parameters if enabled)
        ctx = None
        if self.use_ema and self.ema is not None:
            ctx = self.ema.average_parameters()
            ctx.__enter__()

        try:
            self.model.eval()
            total_loss = 0.0
            y_true_all = []
            y_pred_probs_all = []

            pbar = tqdm(self.val_loader, desc=f"EPOCH {epoch+1} [VAL]")
            with torch.no_grad():
                for batch_idx, (X, y) in enumerate(pbar):
                    X = X.to(self.device)
                    y = y.to(self.device)

                    logits = self.model(X)
                    loss = self.criterion(logits, y)
                    total_loss += loss.item()

                    probs = torch.sigmoid(logits).cpu().numpy()
                    y_pred_probs_all.append(probs)
                    y_true_all.append(y.cpu().numpy())

                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})

                    # Clear GPU cache every 10 batches (only relevant on CUDA)
                    if batch_idx % 10 == 0 and self.device_type == 'cuda':
                        torch.cuda.empty_cache()
            pbar.close()

        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)

        avg_loss = total_loss / len(self.val_loader) if len(self.val_loader) > 0 else float('nan')
        y_true_array = np.vstack(y_true_all) if len(y_true_all) > 0 else np.zeros((0, NUM_CLASSES))
        y_pred_probs_array = np.vstack(y_pred_probs_all) if len(y_pred_probs_all) > 0 else np.zeros((0, NUM_CLASSES))
        
        # Threshold optimization: pick per-class thresholds by maximizing F1.
        # Do not aggressively lock thresholds early; keep adapting each epoch.
        self.optimal_thresholds = self.optimize_thresholds(y_true_array, y_pred_probs_array)
        # Enforce safe bounds to avoid extreme all-positive/all-negative behavior
        self.optimal_thresholds = np.clip(self.optimal_thresholds, 0.05, 0.95)
        print(f"✓ Thresholds (Epoch {epoch+1}): {[f'{t:.3f}' for t in self.optimal_thresholds]}")
        
        y_pred_binary_array = (y_pred_probs_array > self.optimal_thresholds).astype(float)
        
        self.metrics.record('val', avg_loss, [y_true_array], [y_pred_probs_array], [y_pred_binary_array], self.optimal_thresholds)
        
        val_metrics = self.metrics.metrics['val'][-1]
        val_f1 = val_metrics['f1_macro']
        
        # Print detailed validation results with per-class metrics
        print(f"[VAL EPOCH {epoch+1}]")
        print(f"  Overall: F1={val_f1:.4f}, Precision={val_metrics['precision_macro']:.4f}, Recall={val_metrics['recall_macro']:.4f}, ROC_AUC={val_metrics['roc_auc_macro']:.4f}")
        print(f"  Hamming Loss (avg label error): {val_metrics['hamming_loss']:.4f}")
        print(f"  Thresholds: {[f'{t:.3f}' for t in self.optimal_thresholds]}")
        print(f"  Per-Class Metrics (CLASS RECALL IS KEY!):")
        for class_name, metrics_dict in val_metrics['per_class'].items():
            recall_pct = metrics_dict['recall'] * 100
            print(f"    {class_name:18s} | P={metrics_dict['precision']:.3f} R={metrics_dict['recall']:.3f}({recall_pct:.1f}%) F1={metrics_dict['f1']:.3f} AUC={metrics_dict['roc_auc']:.3f} Supp={int(metrics_dict['support'])}")
        
        # Optionally save optimized thresholds for this epoch
        if getattr(self, 'save_thresholds', False):
            try:
                outp = self.checkpoint_dir / 'optimized_thresholds.json'
                json.dump({'epoch': epoch + 1, 'thresholds': self.optimal_thresholds.tolist()}, open(outp, 'w'))
                print(f"Saved optimized thresholds to {outp}")
            except Exception:
                pass

        # Checkpoints
        if val_f1 > self.best_f1:
            self.best_f1 = val_f1
            self.patience_counter = 0
            self.metrics.best_epoch_info = {
                'epoch': epoch + 1,
                'f1': val_f1,
                'precision': val_metrics['precision_macro'],
                'recall': val_metrics['recall_macro'],
                'auc': val_metrics['roc_auc_macro']
            }
            self.save_checkpoint(epoch, is_best=True)
        else:
            self.patience_counter += 1
            self.save_checkpoint(epoch, is_best=False)
        
        # Only step the ReduceLROnPlateau scheduler when using that scheduler
        if getattr(self, 'scheduler_type', 'reduce') == 'reduce':
            try:
                self.scheduler.step(val_f1)
            except Exception:
                pass
        
        return avg_loss, val_f1
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint to training-specific directory."""
        checkpoint_path = self.checkpoint_dir / f'{self.model_name}_epoch_{epoch + 1:03d}.pth'
        torch.save(self.model.state_dict(), checkpoint_path)
        
        if is_best:
            best_path = self.checkpoint_dir / f'{self.model_name}_best.pth'
            torch.save(self.model.state_dict(), best_path)
            print(f"  ✓✓ BEST MODEL FOUND! F1={self.best_f1:.4f} → {best_path}")
    
    def should_stop(self):
        return self.patience_counter >= self.early_stop_patience
    
    def train(self, num_epochs):
        """Full training loop."""
        print("=== TRAINING 90+ ACCURACY MODEL ===")
        print(f"Model: {self.model_name.upper()}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {len(next(iter(self.train_loader))[0])}")
        print(f"Start time: {datetime.now().isoformat()}")

        # Configure schedulers that require steps-per-epoch information (OneCycleLR, CosineAnnealing)
        self.total_epochs = int(num_epochs)
        self._step_per_batch = False
        if getattr(self, 'scheduler_type', 'reduce') == 'onecycle':
            try:
                steps_per_epoch = max(1, len(self.train_loader))
                max_lr = self.max_lr if getattr(self, 'max_lr', None) is not None else self.learning_rate
                self._external_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=max_lr, epochs=self.total_epochs, steps_per_epoch=steps_per_epoch)
                self._step_per_batch = True
                print(f"Configured OneCycleLR(max_lr={max_lr}, epochs={self.total_epochs}, steps_per_epoch={steps_per_epoch})")
            except Exception as e:
                print('Failed to configure OneCycleLR:', e)
        elif getattr(self, 'scheduler_type', 'reduce') == 'cosine':
            try:
                self._external_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.total_epochs)
                print(f"Configured CosineAnnealingLR(T_max={self.total_epochs})")
            except Exception as e:
                print('Failed to configure CosineAnnealingLR:', e)

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss, val_f1 = self.validate(epoch)
            
            # Print epoch summary
            train_metrics = self.metrics.metrics['train'][-1]
            val_metrics = self.metrics.metrics['val'][-1]
            
            # Show improvement indicator
            improvement = "↑" if val_f1 > self.best_f1 else "→" 
            
            print(f"  Summary: Train(F1:{train_metrics['f1_macro']:.3f}) {improvement} Val(F1:{val_f1:.3f}) | Best={self.best_f1:.4f} | Patience={self.patience_counter}/{self.early_stop_patience} | LR={self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Show class-wise improvements if best
            if val_f1 > self.best_f1:
                print(f"  ⭐ IMPROVED! Best recalls: ", end="")
                recalls = [(k, v['recall']) for k, v in val_metrics['per_class'].items()]
                for class_name, recall in sorted(recalls, key=lambda x: x[1], reverse=True)[:2]:
                    print(f"{class_name}({recall*100:.1f}%) ", end="")
                print()
            
            print('-' * 60)
            
            if self.should_stop():
                print(f"\n✓ Early stopping at epoch {epoch + 1}")
                break
        
        print("=== TRAINING COMPLETE ===")
        if self.metrics.best_epoch_info:
            info = self.metrics.best_epoch_info
            print(f"Best Model: Epoch {info['epoch']}")
            print(f"  F1={info['f1']:.4f}, Precision={info['precision']:.4f}, Recall={info['recall']:.4f}, AUC={info['auc']:.4f}")
            print(f"  Location: Models/checkpoints/{self.model_name}_best.pth")
        else:
            print(f"Best F1: {self.best_f1:.4f}")
        # single blank line separator
        print()
        
        # Final GPU cleanup
        torch.cuda.empty_cache()
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        return self.metrics


# ============================================================================
# LOGGING
# ============================================================================

class DualLogger:
    """Write to both console and file."""
    def __init__(self, filepath):
        self.console = sys.stdout
        self.file = open(filepath, 'w', encoding='utf-8')
    
    def write(self, msg):
        self.console.write(msg)
        self.file.write(msg)
        self.file.flush()
    
    def flush(self):
        try:
            self.console.flush()
        except Exception:
            pass
        try:
            if hasattr(self, 'file') and not self.file.closed:
                self.file.flush()
        except Exception:
            pass
    
    def close(self):
        try:
            if hasattr(self, 'file') and not self.file.closed:
                self.file.close()
        except Exception:
            pass


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train 90+ accuracy stuttering model')
    parser.add_argument('--epochs', type=int, default=60, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--data-dir', type=str, default='datasets/features', help='Data directory')
    parser.add_argument('--gpu', action='store_true', default=True, help='Use GPU')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'dml', 'ipex'], help='Device selection: auto/cpu/cuda/dml(i.e. DirectML)/ipex')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of DataLoader workers (default 2)')
    parser.add_argument('--omp-threads', type=int, default=4, help='Set OMP_NUM_THREADS / torch.set_num_threads')
    parser.add_argument('--use-ema', action='store_true', help='Enable EMA (Exponential Moving Average) for evaluation')
    parser.add_argument('--ema-decay', type=float, default=0.999, help='EMA decay (default 0.999)')
    parser.add_argument('--use-label-smoothing', action='store_true', help='Use label smoothing BCE loss instead of focal loss')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing factor (default 0.1)')
    parser.add_argument('--use-bce', action='store_true', help='Use plain BCEWithLogitsLoss (pos_weight applied only if provided)')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate to pass to model constructors (default 0.2)')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Max norm for gradient clipping')
    parser.add_argument('--oversample', type=str, default='none', choices=['none', 'rare', 'weight'], help='Oversampling strategy (none, rare, weight)')
    parser.add_argument('--arch', type=str, default='improved_90plus', choices=['improved_90plus', 'cnn_bilstm'], help='Model architecture to train')
    parser.add_argument('--auto-calibrate', action='store_true', help='Run threshold calibration automatically after training')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--sched-patience', type=int, default=None, help='Scheduler patience for ReduceLROnPlateau (overrides constants.SCHEDULER_PATIENCE)')
    parser.add_argument('--early-stop', type=int, default=None, help='Early stopping patience (overrides default Trainer setting)')
    parser.add_argument('--accumulate', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--mixup-alpha', type=float, default=0.0, help='MixUp alpha; 0 disables MixUp')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for deterministic runs')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Gamma parameter for focal loss')
    parser.add_argument('--loss-type', type=str, default=None, choices=['focal', 'bce', 'label_smoothing'], help='Loss type to use (overrides --use-bce/--use-label-smoothing)')
    parser.add_argument('--scheduler', type=str, default='reduce', choices=['reduce', 'onecycle', 'cosine'], help='LR scheduler to use')
    parser.add_argument('--max-lr', type=float, default=None, help='Max LR for OneCycleLR')
    # Augmentation overrides
    parser.add_argument('--aug-time-p', type=float, default=None, help='SpecAugment time-mask probability')
    parser.add_argument('--aug-freq-p', type=float, default=None, help='SpecAugment freq-mask probability')
    parser.add_argument('--aug-noise-p', type=float, default=None, help='Additive noise probability')
    parser.add_argument('--aug-stretch-p', type=float, default=None, help='Time-stretch probability')
    parser.add_argument('--aug-pitch-p', type=float, default=None, help='Pitch shift probability')
    parser.add_argument('--aug-snr-p', type=float, default=None, help='SNR noise augmentation probability')
    parser.add_argument('--save-thresholds', action='store_true', help='Persist optimized thresholds to checkpoint folder each epoch')
    parser.add_argument('--thresh-min-precision', type=float, default=0.2, help='Minimum per-class precision required when choosing thresholds (0-1)')
    parser.add_argument('--neutral-pos-weight', action='store_true', help='When oversampling, force neutral pos_weight to avoid double-upweighting')
    parser.add_argument('--sampler-replacement', action='store_true', help='Use replacement sampling for WeightedRandomSampler when oversampling')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to load model weights from (for fine-tuning)')
    
    args = parser.parse_args()
    
    # Create training timestamp - used for all output folders
    training_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup logging to file in training-specific folder
    output_dir = Path('output') / f'training_{training_timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f'training_{training_timestamp}.log'
    sys.stdout = DualLogger(str(log_file))
    # Configure logging verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        print('Verbose logging enabled')
    else:
        # Reduce noisy INFO logs from libraries during training
        logging.getLogger().setLevel(logging.WARNING)
    
    print(f"✓ Training session: {training_timestamp}")
    print(f"✓ Training log: {log_file}")
    print(f"✓ Output folder: {output_dir}")
    # Device selection logic (support optional DirectML and IPEX)
    use_ipex = False
    use_dml = False
    device = None
    if args.device == 'auto':
        if args.gpu and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    elif args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif args.device == 'cpu':
        device = torch.device('cpu')
    elif args.device == 'ipex':
        # Request CPU with IPEX optimization
        device = torch.device('cpu')
        use_ipex = True
    elif args.device == 'dml':
        # Try to use DirectML device (optional external package)
        try:
            import torch_directml
            device = torch_directml.device()
            use_dml = True
        except Exception:
            print('Warning: torch_directml not available; falling back to CPU')
            device = torch.device('cpu')

    print(f"Device: {device} (ipex={use_ipex} dml={use_dml})")

    # Apply threading settings (user-requested)
    try:
        os.environ['OMP_NUM_THREADS'] = str(int(args.omp_threads))
    except Exception:
        os.environ['OMP_NUM_THREADS'] = '4'
    try:
        torch.set_num_threads(int(args.omp_threads))
    except Exception:
        pass

    # Print effective runtime settings
    print(f"Settings: batch_size={args.batch_size}, num_workers={args.num_workers}, omp_threads={os.environ.get('OMP_NUM_THREADS')}")
    
    # Load data
    print("Loading datasets...")
    train_dataset = AudioDataset(
        args.data_dir,
        split='train',
        augment=True,
        aug_time_p=args.aug_time_p,
        aug_freq_p=args.aug_freq_p,
        aug_noise_p=args.aug_noise_p,
        aug_stretch_p=args.aug_stretch_p,
        aug_pitch_p=args.aug_pitch_p,
        aug_snr_p=args.aug_snr_p,
    )
    val_dataset = AudioDataset(
        args.data_dir,
        split='val',
        augment=False,
    )
    
    # Configure DataLoader workers and pin_memory
    cpu_count = os.cpu_count() or 1
    if args.num_workers is None:
        num_workers = max(0, min(8, cpu_count - 1))
    else:
        num_workers = max(0, args.num_workers)

    pin_memory = True if getattr(device, 'type', '') == 'cuda' else False

    # Use custom collate to handle variable-length spectrograms or embeddings
    # Optionally use oversampling for rare classes
    sampler = None
    if args.oversample in ('rare', 'weight'):
        # Compute inverse-frequency weights across all classes and assign per-file sample weight
        # This provides a balanced sampling for multilabel data by upweighting files
        # that contain rarer classes.
        print('Computing inverse-frequency sample weights for oversampling...')
        # Count positives per class
        class_pos = np.zeros((train_dataset.__len__(),), dtype=np.float32)
        counts_pos = np.zeros((NUM_CLASSES,), dtype=np.float64)
        file_labels = []
        for fidx, f in enumerate(train_dataset.files):
            try:
                d = np.load(f)
                lbl = d.get('labels')
                if lbl is None:
                    lbl = np.zeros((NUM_CLASSES,), dtype=np.int32)
                lbl = np.asarray(lbl)
                if lbl.size != NUM_CLASSES:
                    # If stored as integers (label ids), convert to binary presence
                    lbl = (lbl > 0).astype(np.int32)
                    if lbl.size < NUM_CLASSES:
                        # pad or truncate
                        tmp = np.zeros((NUM_CLASSES,), dtype=np.int32)
                        tmp[:min(len(lbl), NUM_CLASSES)] = lbl[:min(len(lbl), NUM_CLASSES)]
                        lbl = tmp
                else:
                    lbl = (lbl > 0).astype(np.int32)
                file_labels.append(lbl)
                counts_pos += lbl
            except Exception:
                file_labels.append(np.zeros((NUM_CLASSES,), dtype=np.int32))

        # avoid div by zero
        counts_pos = np.maximum(counts_pos, 1.0)
        inv_freq = (np.sum(counts_pos) / counts_pos)
        # Normalize inverse frequencies to have mean 1.0
        inv_freq = inv_freq / np.mean(inv_freq)

        weights = []
        for lbl in file_labels:
            # sample weight is average inv_freq of labels present; if none present use 1.0
            present = lbl.astype(bool)
            if np.any(present):
                w = float(np.mean(inv_freq[present]))
            else:
                w = 1.0
            weights.append(max(0.01, w))

        try:
            from torch.utils.data import WeightedRandomSampler
            # Use weighted sampling; allow replacement when requested via CLI to enable true oversampling
            try:
                replacement_flag = bool(args.sampler_replacement)
            except Exception:
                replacement_flag = False
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=replacement_flag)
            if replacement_flag:
                print('Using WeightedRandomSampler (with replacement) for oversampling')
            else:
                print('Using WeightedRandomSampler (no replacement) with inverse-frequency weighting')
        except Exception as e:
            print('Failed to create WeightedRandomSampler:', e)

    if sampler is not None:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_variable_length)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_variable_length)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_variable_length)
    
    # Model selection
    model_name = args.arch
    if args.arch == 'cnn_bilstm':
        try:
            from model_cnn_bilstm import CNNBiLSTM
            model = CNNBiLSTM(in_channels=123, n_classes=5, dropout=args.dropout)
            print('Using CNN+BiLSTM model')
        except Exception as e:
            print('Failed to import CNNBiLSTM, falling back to ImprovedStutteringCNN:', e)
            model = ImprovedStutteringCNN(n_channels=123, n_classes=5, dropout=args.dropout)
            model_name = 'improved_90plus'
    else:
        model = ImprovedStutteringCNN(n_channels=123, n_classes=5, dropout=args.dropout)
        model_name = 'improved_90plus'

    # Print the selected model name and exact parameter count (avoid hardcoded names)
    try:
        param_count = sum(p.numel() for p in model.parameters())
    except Exception:
        param_count = 0
    print(f"Model: {model_name} | Parameters: {param_count:,}")
    
    # Train with training timestamp
    # If oversampling is enabled, optionally avoid passing aggressive pos_weight to the loss
    # (oversampling already upweights rare classes). Use neutral class weights only when
    # the user requests it via --neutral-pos-weight to make this behavior opt-in.
    class_weights_arg = None
    try:
        import numpy as _np
        import torch as _torch
        if args.oversample == 'rare' and args.neutral_pos_weight:
            class_weights_arg = _torch.tensor(_np.ones((NUM_CLASSES,), dtype=_np.float32), dtype=_torch.float32).to(device)
            print('Oversampling enabled and --neutral-pos-weight set: overriding class pos_weight to neutral to avoid double-upweighting')
    except Exception:
        class_weights_arg = None

    # Extra-safe: if we constructed a sampler above, enforce neutral pos_weight only if user opted-in
    if 'sampler' in locals() and sampler is not None and args.neutral_pos_weight:
        try:
            import torch as _torch
            class_weights_arg = _torch.ones((NUM_CLASSES,), dtype=_torch.float32).to(device)
            print('Sampler present and --neutral-pos-weight set: enforcing neutral class pos_weight to avoid double-upweighting')
        except Exception:
            # If torch not available here for any reason, leave class_weights_arg as-is
            pass

    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        device,
        use_ipex=use_ipex,
        model_name=model_name,
        training_timestamp=training_timestamp,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        use_label_smoothing=args.use_label_smoothing,
        label_smoothing=args.label_smoothing,
        grad_clip=args.grad_clip,
        accumulate_steps=args.accumulate,
        sched_patience=args.sched_patience,
        early_stop_patience=(args.early_stop if args.early_stop is not None else None),
        loss_type=(args.loss_type if args.loss_type is not None else ('bce' if args.use_bce else 'focal')),
        class_weights=class_weights_arg,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        scheduler_type=args.scheduler,
        max_lr=args.max_lr,
        seed=args.seed,
        aug_time_p=args.aug_time_p,
        aug_freq_p=args.aug_freq_p,
        aug_noise_p=args.aug_noise_p,
        aug_stretch_p=args.aug_stretch_p,
        aug_pitch_p=args.aug_pitch_p,
        aug_snr_p=args.aug_snr_p,
        save_thresholds=args.save_thresholds,
        thresh_min_precision=args.thresh_min_precision,
        neutral_pos_weight=args.neutral_pos_weight,
    )
    # If user requested to resume/load model weights for fine-tuning, attempt to load before training
    if args.resume is not None:
        try:
            resume_path = Path(args.resume)
            if resume_path.exists():
                print(f"Loading checkpoint weights from {resume_path}")
                state = torch.load(str(resume_path), map_location=device)
                # Extract state_dict if wrapped in checkpoint dict
                sd = None
                if isinstance(state, dict):
                    if 'state_dict' in state:
                        sd = state['state_dict']
                    elif 'model_state' in state:
                        sd = state['model_state']
                    elif 'model' in state and isinstance(state['model'], dict):
                        sd = state['model']
                    else:
                        sd = state
                else:
                    sd = state

                # Try direct load, or strip 'module.' prefix if necessary
                try:
                    trainer.model.load_state_dict(sd)
                except Exception:
                    try:
                        new = {k.replace('module.', ''): v for k, v in sd.items()}
                        trainer.model.load_state_dict(new)
                    except Exception as e:
                        print('Failed to load checkpoint weights:', e)
                else:
                    print('Checkpoint weights loaded successfully')
            else:
                print(f'Resume checkpoint not found: {resume_path}')
        except Exception as e:
            print('Error while loading resume checkpoint:', e)
    # Apply mixup alpha after trainer initialization to avoid changing signature
    try:
        trainer.mixup_alpha = float(args.mixup_alpha)
        if trainer.mixup_alpha > 0.0:
            print(f'Enabled MixUp with alpha={trainer.mixup_alpha}')
    except Exception:
        trainer.mixup_alpha = 0.0
    metrics = trainer.train(args.epochs)
    
    # Save metrics in training-specific folder
    metrics_path = output_dir / f'{model_name}_metrics.json'
    metrics.save(metrics_path)
    
    # Create best model reference in main checkpoints folder
    best_model_path = trainer.checkpoint_dir / f'{model_name}_best.pth'
    best_link_path = Path('Models/checkpoints') / f'{model_name}_BEST_OVERALL.pth'
    
    if best_model_path.exists():
        # Copy to overall best location
        shutil.copy(str(best_model_path), str(best_link_path))
    
    print(f"\n✅ Training Complete!")
    print(f"├─ Checkpoint folder: {trainer.checkpoint_dir}")
    print(f"├─ Best model (this run): {best_model_path}")
    print(f"├─ Overall best model: {best_link_path}")
    print(f"├─ Metrics file: {metrics_path}")
    print(f"└─ Training log: {log_file}")
    
    # Optionally run calibration to produce `output/thresholds.json` and lock it
    if args.auto_calibrate:
        try:
            if best_model_path.exists():
                cmd = [sys.executable, 'Models/calibrate_thresholds.py', '--checkpoint', str(best_model_path), '--data-dir', args.data_dir, '--out', 'output/thresholds.json']
                print(f"Running calibration: {' '.join(cmd)}")
                subprocess.run(cmd, check=False)
                # touch lock file to indicate thresholds set
                try:
                    Path('output/thresholds.lock').write_text(datetime.now().isoformat())
                except Exception:
                    pass
            else:
                print('Best model not found; skipping auto-calibrate')
        except Exception as e:
            print('Auto-calibrate failed:', e)

    # Close logger
    if isinstance(sys.stdout, DualLogger):
        sys.stdout.close()


if __name__ == '__main__':
    main()
