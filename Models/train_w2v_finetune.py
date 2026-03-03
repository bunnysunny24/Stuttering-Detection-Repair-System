"""
Training script for end-to-end wav2vec2-large fine-tuning on stuttering detection.

PRODUCTION ARCHITECTURE — Hierarchical Detection:
  PRIMARY TASK: Binary detection (stutter vs fluent) — targets 90+ F1
  SECONDARY TASK: 5-class type classification — provides detail when stutter detected

  The model trains BOTH tasks jointly, but binary F1 is the PRIMARY metric
  used for model selection and early stopping. This is the key to 90+.

OPTIMIZATIONS:
  1. LABEL CLEANING: Remove Unsure/PoorAudio/NoSpeech/Music clips (~4300 noisy clips gone)
  2. MAJORITY VOTE (>=2): Only count positive if 2+ annotators agree (kills noise)
  3. 12 UNFROZEN LAYERS: 2x adaptation capacity, still fits in 3.2 GB
  4. BINARY-FIRST LOSS: binary_weight=1.0, multiclass_weight=0.5
  5. MIXUP: Audio-level interpolation augmentation
  6. SPEED PERTURBATION: 0.9-1.1x speed variation
  7. COSINE ANNEALING: Better LR schedule for fine-tuning
  8. R-DROP: KL-divergence consistency regularization
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
import shutil
from tqdm import tqdm
import warnings
import time
import gc
from contextlib import contextmanager

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from model_w2v_finetune import Wav2VecFineTuneClassifier
from constants import NUM_CLASSES, DEFAULT_SAMPLE_RATE
from utils import FocalLoss

STUTTER_CLASSES = ['Prolongation', 'Block', 'SoundRep', 'WordRep', 'Interjection']


# ============================================================================
# DATASET - Raw Audio with CLEAN labels
# ============================================================================

class RawAudioDataset(Dataset):
    """Load raw WAV files with CLEANED, majority-voted labels.

    Key improvements over v1:
      - Removes noisy clips (Unsure, PoorAudioQuality, NoSpeech, Music)
      - Majority vote: label positive only if >=2 annotators agree
      - Stores binary 'any_stutter' label for auxiliary head
    """

    def __init__(self, clips_dir, label_csvs, split_stems, split='train',
                 max_audio_len=None, sample_rate=16000, augment=False,
                 majority_vote=True, clean_labels=True, min_votes=2):
        import pandas as pd
        import scipy.io.wavfile as wav_io

        self.clips_dir = Path(clips_dir)
        self.sample_rate = sample_rate
        self.max_audio_len = max_audio_len
        self.augment = augment and (split == 'train')
        self._wav_io = wav_io

        stutter_cols = ['Prolongation', 'Block', 'SoundRep', 'WordRep', 'Interjection']
        noise_cols = ['Unsure', 'PoorAudioQuality', 'NoSpeech', 'Music']

        print(f"[{split}] Building CLEAN label lookup (majority_vote={majority_vote}, "
              f"clean={clean_labels}, min_votes={min_votes})...")

        # Load and concatenate all CSVs
        all_dfs = []
        for csv_path in label_csvs:
            all_dfs.append(pd.read_csv(csv_path))
        df = pd.concat(all_dfs, ignore_index=True)

        # STEP 1: Remove noisy clips
        if clean_labels:
            before = len(df)
            noise_mask = True
            for col in noise_cols:
                if col in df.columns:
                    noise_mask = noise_mask & (df[col] == 0)
            df = df[noise_mask].copy()
            removed = before - len(df)
            print(f"  Removed {removed} noisy clips ({', '.join(noise_cols)})")

        # STEP 2: Build label lookup with majority vote
        self.label_map = {}
        self.binary_map = {}  # stem -> float (any stutter)

        for _, row in df.iterrows():
            stem = f"{row['Show']}_{row['EpId']}_{row['ClipId']}"
            raw_labels = np.array([int(row[c]) for c in stutter_cols], dtype=np.float32)

            if majority_vote:
                labels_bin = (raw_labels >= min_votes).astype(np.float32)
            else:
                labels_bin = (raw_labels > 0).astype(np.float32)

            if stem in self.label_map:
                self.label_map[stem] = np.maximum(self.label_map[stem], labels_bin)
            else:
                self.label_map[stem] = labels_bin

        # Compute binary "any stutter" labels
        for stem, labels in self.label_map.items():
            self.binary_map[stem] = float(labels.max() > 0)

        # Filter to stems in this split with both WAV + labels
        self.samples = []
        missing_wav = 0
        missing_label = 0

        for stem in sorted(split_stems):
            wav_path = self.clips_dir / f"{stem}.wav"
            if not wav_path.exists():
                missing_wav += 1
                continue
            if stem not in self.label_map:
                missing_label += 1
                continue
            self.samples.append((
                str(wav_path),
                self.label_map[stem],
                self.binary_map[stem]
            ))

        if len(self.samples) == 0:
            print(f"[{split}] WARNING: 0 samples after filtering!")
            self.pos_counts = np.zeros(len(stutter_cols))
            self.neg_counts = np.zeros(len(stutter_cols))
            return

        all_labels = np.stack([s[1] for s in self.samples])
        all_binary = np.array([s[2] for s in self.samples])
        pos_counts = all_labels.sum(axis=0)
        total = len(self.samples)
        neg_counts = total - pos_counts

        print(f"[{split}] {len(self.samples)} samples "
              f"(missing WAV: {missing_wav}, missing label: {missing_label})")
        print(f"[{split}] Binary: {int(all_binary.sum())} stutter / "
              f"{int(total - all_binary.sum())} clean "
              f"({all_binary.sum()/total*100:.1f}% stutter)")
        print(f"[{split}] Per-class (majority vote={majority_vote}):")
        for i, name in enumerate(STUTTER_CLASSES):
            print(f"  {name:18s}: {int(pos_counts[i]):5d} pos / {int(neg_counts[i]):5d} neg "
                  f"({pos_counts[i]/total*100:.1f}%)")

        self.pos_counts = pos_counts
        self.neg_counts = neg_counts

    def get_pos_weight(self, clip_max=30.0):
        """Compute pos_weight for BCE: neg/pos, clipped."""
        pos = np.maximum(self.pos_counts, 1.0)
        pw = np.clip(self.neg_counts / pos, 1.0, clip_max)
        return torch.tensor(pw, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav_path, labels, binary_label = self.samples[idx]

        try:
            sr, audio = self._wav_io.read(wav_path)
        except Exception as e:
            print(f"Warning: failed to load {wav_path}: {e}")
            audio = np.zeros(self.sample_rate, dtype=np.float32)
            sr = self.sample_rate

        # Convert to float32 [-1, 1]
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample if needed
        if sr != self.sample_rate:
            duration = len(audio) / sr
            target_len = int(duration * self.sample_rate)
            if target_len > 0:
                audio = np.interp(
                    np.linspace(0, len(audio) - 1, target_len),
                    np.arange(len(audio)),
                    audio
                ).astype(np.float32)

        # Truncate
        if self.max_audio_len is not None and len(audio) > self.max_audio_len:
            if self.augment:
                start = np.random.randint(0, len(audio) - self.max_audio_len)
            else:
                start = (len(audio) - self.max_audio_len) // 2
            audio = audio[start:start + self.max_audio_len]

        # Min length for wav2vec2
        if len(audio) < 400:
            audio = np.pad(audio, (0, 400 - len(audio)), mode='constant')

        # Augmentations (training only)
        if self.augment:
            audio = self._augment_audio(audio)

        return (torch.from_numpy(audio).float(),
                torch.from_numpy(labels).float(),
                torch.tensor(binary_label, dtype=torch.float32))

    def _augment_audio(self, audio):
        """Aggressive waveform augmentations."""
        # Speed perturbation (0.9x - 1.1x) — critical for stuttering detection
        if np.random.random() < 0.5:
            speed = np.random.uniform(0.9, 1.1)
            orig_len = len(audio)
            new_len = int(orig_len / speed)
            if new_len > 100:
                audio = np.interp(
                    np.linspace(0, orig_len - 1, new_len),
                    np.arange(orig_len), audio
                ).astype(np.float32)
                # Pad or trim back to original length
                if len(audio) > orig_len:
                    audio = audio[:orig_len]
                elif len(audio) < orig_len:
                    audio = np.pad(audio, (0, orig_len - len(audio)), mode='constant')

        # Random gain
        if np.random.random() < 0.5:
            gain = np.random.uniform(0.6, 1.4)
            audio = audio * gain

        # Additive noise (SNR-based)
        if np.random.random() < 0.4:
            snr_db = np.random.uniform(10, 30)
            signal_power = np.mean(audio ** 2) + 1e-10
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = np.random.normal(0, np.sqrt(noise_power), len(audio)).astype(np.float32)
            audio = audio + noise

        # Time masking (zero out a portion)
        if np.random.random() < 0.3:
            mask_len = np.random.randint(1, max(2, len(audio) // 5))
            start = np.random.randint(0, max(1, len(audio) - mask_len))
            audio[start:start + mask_len] = 0.0

        # Random time shift
        if np.random.random() < 0.3:
            max_shift = max(1, int(len(audio) * 0.1))
            shift = np.random.randint(-max_shift, max_shift + 1)
            audio = np.roll(audio, shift)

        # Polarity inversion (harmless for content, helps generalization)
        if np.random.random() < 0.3:
            audio = -audio

        return audio


def collate_raw_audio(batch):
    """Collate variable-length audio with attention mask + binary labels."""
    audios = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    binary = [item[2] for item in batch]

    max_len = max(a.shape[0] for a in audios)

    padded = []
    masks = []
    for a in audios:
        pad_len = max_len - a.shape[0]
        if pad_len > 0:
            padded.append(F.pad(a, (0, pad_len)))
            mask = torch.ones(max_len, dtype=torch.long)
            mask[a.shape[0]:] = 0
            masks.append(mask)
        else:
            padded.append(a)
            masks.append(torch.ones(max_len, dtype=torch.long))

    return (torch.stack(padded),       # (B, T)
            torch.stack(masks),        # (B, T)
            torch.stack(labels),       # (B, 5)
            torch.stack(binary))       # (B,)


# ============================================================================
# EMA
# ============================================================================

class ExponentialMovingAverage:
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
            new_avg = (1.0 - self.decay) * p.data + self.decay * self.shadow[p]
            self.shadow[p].copy_(new_avg)

    @contextmanager
    def average_parameters(self):
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
# METRICS
# ============================================================================

class MetricsTracker:
    def __init__(self):
        self.metrics = {'train': [], 'val': []}
        self.best_epoch_info = None

    def record(self, phase, loss, y_true, y_pred_probs, y_pred_binary,
               binary_true=None, binary_probs=None):
        if len(y_true) == 0:
            return

        y_true = np.vstack(y_true) if isinstance(y_true[0], np.ndarray) else np.array(y_true)
        y_pred_probs = np.vstack(y_pred_probs) if isinstance(y_pred_probs[0], np.ndarray) else np.array(y_pred_probs)
        y_pred_binary = np.vstack(y_pred_binary) if isinstance(y_pred_binary[0], np.ndarray) else np.array(y_pred_binary)

        f1_per_class = []
        precision_per_class = []
        recall_per_class = []
        roc_auc_list = []

        for i in range(y_true.shape[1]):
            yt, yp, ypp = y_true[:, i], y_pred_binary[:, i], y_pred_probs[:, i]
            tp = np.sum((yt == 1) & (yp == 1))
            fp = np.sum((yt == 0) & (yp == 1))
            fn = np.sum((yt == 1) & (yp == 0))

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            precision_per_class.append(prec)
            recall_per_class.append(rec)
            f1_per_class.append(f1)

            try:
                n_pos = int(np.sum(yt == 1))
                n_neg = int(len(yt) - n_pos)
                if n_pos > 0 and n_neg > 0:
                    ranks = np.argsort(np.argsort(ypp)) + 1
                    auc = (np.sum(ranks[yt == 1]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
                    roc_auc_list.append(float(auc))
                else:
                    roc_auc_list.append(0.5)
            except Exception:
                roc_auc_list.append(0.5)

        # Binary detection metrics (PRIMARY — this is what gets 90+)
        binary_f1 = None
        binary_prec = None
        binary_rec = None
        binary_acc = None
        binary_auc = None
        if binary_true is not None and binary_probs is not None:
            bt = np.concatenate(binary_true) if isinstance(binary_true[0], np.ndarray) else np.array(binary_true)
            bp = np.concatenate(binary_probs) if isinstance(binary_probs[0], np.ndarray) else np.array(binary_probs)
            binary_pred = (bp > 0.5).astype(float)
            binary_acc = float(np.mean(bt == binary_pred))
            # Proper F1/P/R for binary detection
            tp = np.sum((bt == 1) & (binary_pred == 1))
            fp = np.sum((bt == 0) & (binary_pred == 1))
            fn = np.sum((bt == 1) & (binary_pred == 0))
            binary_prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            binary_rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            binary_f1 = float(2 * binary_prec * binary_rec / (binary_prec + binary_rec)) if (binary_prec + binary_rec) > 0 else 0.0
            # Binary AUC
            try:
                n_pos = int(np.sum(bt == 1))
                n_neg = int(len(bt) - n_pos)
                if n_pos > 0 and n_neg > 0:
                    ranks = np.argsort(np.argsort(bp)) + 1
                    binary_auc = float((np.sum(ranks[bt == 1]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
                else:
                    binary_auc = 0.5
            except Exception:
                binary_auc = 0.5

        epoch_metrics = {
            'loss': loss,
            # PRIMARY: Binary detection metrics (90+ target)
            'binary_f1': binary_f1,
            'binary_precision': binary_prec,
            'binary_recall': binary_rec,
            'binary_acc': binary_acc,
            'binary_auc': binary_auc,
            # SECONDARY: 5-class multi-label metrics
            'f1_macro': float(np.mean(f1_per_class)),
            'precision_macro': float(np.mean(precision_per_class)),
            'recall_macro': float(np.mean(recall_per_class)),
            'roc_auc_macro': float(np.mean(roc_auc_list)),
            'per_class': {
                STUTTER_CLASSES[i]: {
                    'precision': float(precision_per_class[i]),
                    'recall': float(recall_per_class[i]),
                    'f1': float(f1_per_class[i]),
                    'roc_auc': float(roc_auc_list[i])
                }
                for i in range(NUM_CLASSES)
            }
        }
        self.metrics[phase].append(epoch_metrics)

    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)


# ============================================================================
# THRESHOLD OPTIMIZATION
# ============================================================================

def optimize_thresholds(y_true, y_pred_probs, previous=None):
    """Find per-class thresholds maximizing F1 with fine search grid."""
    n_classes = y_true.shape[1]
    thresholds = np.full(n_classes, 0.5)

    for i in range(n_classes):
        yt = y_true[:, i].astype(int)
        yp = y_pred_probs[:, i]
        if yt.sum() == 0:
            if previous is not None:
                thresholds[i] = previous[i]
            continue

        best_f1 = -1
        best_t = 0.5
        # Fine grid search
        for t in np.arange(0.10, 0.90, 0.01):
            pred = (yp > t).astype(int)
            tp = np.sum((yt == 1) & (pred == 1))
            fp = np.sum((yt == 0) & (pred == 1))
            fn = np.sum((yt == 1) & (pred == 0))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        thresholds[i] = best_t

    if previous is not None:
        thresholds = 0.8 * thresholds + 0.2 * previous

    return np.clip(thresholds, 0.05, 0.95)


# ============================================================================
# TRAINER
# ============================================================================

class FineTuneTrainer:
    """Aggressive training loop with all optimizations."""

    def __init__(self, model, train_loader, val_loader, device, args):
        self.model = model.to(device)
        self.device = device
        self.args = args

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = Path('Models/checkpoints') / f'finetune_{self.timestamp}'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.train_loader = train_loader
        self.val_loader = val_loader

        # Discriminative LR optimizer
        self.param_groups = model.get_param_groups(
            backbone_lr=args.backbone_lr, head_lr=args.head_lr
        )
        self.optimizer = optim.AdamW(self.param_groups)

        # Cosine annealing with warm restarts
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=args.cosine_t0, T_mult=2, eta_min=1e-7
        )

        # Loss: Binary PRIMARY + multi-class SECONDARY
        pw = args.pos_weight.to(device) if hasattr(args, 'pos_weight') and args.pos_weight is not None else None
        self.criterion = FocalLoss(gamma=args.focal_gamma, pos_weight=pw)
        self.binary_criterion = nn.BCEWithLogitsLoss()
        self.binary_weight = args.binary_weight   # PRIMARY task weight (1.0)
        self.multiclass_weight = args.multiclass_weight  # SECONDARY task weight (0.5)

        # MixUp
        self.mixup_alpha = args.mixup_alpha

        # R-Drop KL divergence weight
        self.rdrop_weight = args.rdrop_weight

        # Training state — track BINARY F1 as primary (90+ target)
        self.best_binary_f1 = 0.0
        self.best_binary_auc = 0.0
        self.best_f1 = 0.0
        self.best_auc = 0.0
        self.optimal_binary_threshold = 0.5
        self.patience_counter = 0
        self.optimal_thresholds = np.full(NUM_CLASSES, 0.5)
        self.accumulate_steps = args.accumulate
        self.grad_clip = args.grad_clip
        self.early_stop_patience = args.early_stop

        # EMA
        self.ema = None
        if args.use_ema:
            self.ema = ExponentialMovingAverage(
                self.model.parameters(), decay=args.ema_decay
            )
            print(f"EMA: decay={args.ema_decay}")

        self.metrics = MetricsTracker()

        # Print config
        print(f"\nTrainer config:")
        print(f"  Micro-batch: {args.batch_size}")
        print(f"  Accumulate: {args.accumulate} -> effective batch {args.batch_size * args.accumulate}")
        print(f"  Backbone LR: {args.backbone_lr}")
        print(f"  Head LR: {args.head_lr}")
        print(f"  MixUp alpha: {self.mixup_alpha}")
        print(f"  R-Drop weight: {self.rdrop_weight}")
        print(f"  Binary weight: {self.binary_weight} (PRIMARY)")
        print(f"  Multiclass weight: {self.multiclass_weight} (SECONDARY)")
        print(f"  Cosine T0: {args.cosine_t0}")
        print(f"  Early stop: {self.early_stop_patience}")
        print(f"  Grad clip: {self.grad_clip}")

    def _mixup(self, audio, mask, labels, binary):
        """MixUp augmentation on raw audio."""
        if self.mixup_alpha <= 0:
            return audio, mask, labels, binary

        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        lam = max(lam, 1 - lam)  # Ensure lam >= 0.5

        idx = torch.randperm(audio.size(0))
        audio = lam * audio + (1 - lam) * audio[idx]
        labels = lam * labels + (1 - lam) * labels[idx]
        binary = lam * binary + (1 - lam) * binary[idx]

        return audio, mask, labels, binary

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        y_true_all, y_pred_probs_all, y_pred_bin_all = [], [], []
        binary_true_all, binary_probs_all = [], []

        pbar = tqdm(self.train_loader, desc=f"E{epoch+1} [TRAIN]")
        for batch_idx, (audio, mask, labels, binary) in enumerate(pbar):
            audio = audio.to(self.device)
            mask = mask.to(self.device)
            labels = labels.to(self.device)
            binary = binary.to(self.device)

            # MixUp
            audio, mask, labels, binary = self._mixup(audio, mask, labels, binary)

            # Zero grads at accumulation boundary
            if (batch_idx % self.accumulate_steps) == 0:
                self.optimizer.zero_grad()

            # Forward pass 1
            logits, binary_logit = self.model(audio, attention_mask=mask)

            # SECONDARY: Multi-class focal loss
            loss_cls = self.criterion(logits, labels)

            # PRIMARY: Binary detection loss
            loss_binary = self.binary_criterion(binary_logit.squeeze(-1), binary)

            # R-Drop: second forward pass + KL divergence consistency
            loss_rdrop = torch.tensor(0.0, device=self.device)
            if self.rdrop_weight > 0 and self.model.training:
                logits2, binary_logit2 = self.model(audio, attention_mask=mask)
                # KL divergence between two forward passes (dropout makes them different)
                p1 = torch.sigmoid(logits)
                p2 = torch.sigmoid(logits2)
                kl1 = F.binary_cross_entropy(p1, p2.detach(), reduction='mean')
                kl2 = F.binary_cross_entropy(p2, p1.detach(), reduction='mean')
                loss_rdrop = (kl1 + kl2) / 2.0
                # Average class loss from second pass
                loss_cls = (loss_cls + self.criterion(logits2, labels)) / 2.0

            total = (self.binary_weight * loss_binary +
                     self.multiclass_weight * loss_cls +
                     self.rdrop_weight * loss_rdrop) / float(self.accumulate_steps)

            total.backward()

            is_step = ((batch_idx + 1) % self.accumulate_steps == 0 or
                       (batch_idx + 1) == len(self.train_loader))
            if is_step:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.optimizer.step()
                # Step cosine scheduler per batch
                self.scheduler.step(epoch + batch_idx / len(self.train_loader))
                if self.ema is not None:
                    self.ema.update()

            total_loss += total.item() * self.accumulate_steps

            with torch.no_grad():
                probs = torch.sigmoid(logits).cpu().numpy()
                y_pred_probs_all.append(probs)
                y_pred_bin_all.append((probs > self.optimal_thresholds).astype(float))
                y_true_all.append(labels.cpu().numpy())
                binary_true_all.append(binary.cpu().numpy())
                binary_probs_all.append(torch.sigmoid(binary_logit.squeeze(-1)).cpu().numpy())

            pbar.set_postfix({'loss': f'{total.item()*self.accumulate_steps:.3f}'})

            if batch_idx % 20 == 0:
                gc.collect()

        pbar.close()
        avg_loss = total_loss / max(len(self.train_loader), 1)
        self.metrics.record('train', avg_loss, y_true_all, y_pred_probs_all,
                            y_pred_bin_all, binary_true_all, binary_probs_all)

        m = self.metrics.metrics['train'][-1]
        # PRIMARY metric first
        bin_str = ""
        if m['binary_f1'] is not None:
            bin_str = (f" | BINARY: F1={m['binary_f1']:.3f} "
                       f"P={m['binary_precision']:.3f} R={m['binary_recall']:.3f} "
                       f"Acc={m['binary_acc']:.3f} AUC={m['binary_auc']:.3f}")
        print(f"[TRAIN E{epoch+1}] Loss={avg_loss:.4f}{bin_str}")
        print(f"  5-class macro: F1={m['f1_macro']:.4f} AUC={m['roc_auc_macro']:.4f}")
        for c, cm in m['per_class'].items():
            print(f"  {c:18s} P={cm['precision']:.3f} R={cm['recall']:.3f} "
                  f"F1={cm['f1']:.3f} AUC={cm['roc_auc']:.3f}")
        return avg_loss

    def validate(self, epoch):
        ctx = None
        if self.ema is not None:
            ctx = self.ema.average_parameters()
            ctx.__enter__()

        try:
            self.model.eval()
            total_loss = 0.0
            y_true_all, y_pred_probs_all = [], []
            binary_true_all, binary_probs_all = [], []

            pbar = tqdm(self.val_loader, desc=f"E{epoch+1} [VAL]")
            with torch.no_grad():
                for audio, mask, labels, binary in pbar:
                    audio = audio.to(self.device)
                    mask = mask.to(self.device)
                    labels = labels.to(self.device)
                    binary = binary.to(self.device)

                    logits, binary_logit = self.model(audio, attention_mask=mask)
                    loss = self.criterion(logits, labels)
                    total_loss += loss.item()

                    probs = torch.sigmoid(logits).cpu().numpy()
                    y_pred_probs_all.append(probs)
                    y_true_all.append(labels.cpu().numpy())
                    binary_true_all.append(binary.cpu().numpy())
                    binary_probs_all.append(torch.sigmoid(binary_logit.squeeze(-1)).cpu().numpy())

            pbar.close()
        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)

        avg_loss = total_loss / max(len(self.val_loader), 1)
        y_true = np.vstack(y_true_all)
        y_probs = np.vstack(y_pred_probs_all)

        self.optimal_thresholds = optimize_thresholds(
            y_true, y_probs, self.optimal_thresholds
        )
        y_pred_bin = (y_probs > self.optimal_thresholds).astype(float)

        # Optimize binary threshold
        bt_all = np.concatenate(binary_true_all) if isinstance(binary_true_all[0], np.ndarray) else np.array(binary_true_all)
        bp_all = np.concatenate(binary_probs_all) if isinstance(binary_probs_all[0], np.ndarray) else np.array(binary_probs_all)
        best_bin_f1 = -1
        best_bin_t = 0.5
        for t in np.arange(0.20, 0.80, 0.01):
            pred = (bp_all > t).astype(float)
            tp = np.sum((bt_all == 1) & (pred == 1))
            fp = np.sum((bt_all == 0) & (pred == 1))
            fn = np.sum((bt_all == 1) & (pred == 0))
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            if f > best_bin_f1:
                best_bin_f1 = f
                best_bin_t = t
        self.optimal_binary_threshold = best_bin_t

        self.metrics.record('val', avg_loss, [y_true], [y_probs], [y_pred_bin],
                            binary_true_all, binary_probs_all)

        vm = self.metrics.metrics['val'][-1]
        val_f1 = vm['f1_macro']
        val_auc = vm['roc_auc_macro']
        val_binary_f1 = vm['binary_f1'] if vm['binary_f1'] is not None else 0.0
        val_binary_auc = vm['binary_auc'] if vm['binary_auc'] is not None else 0.5

        # Print PRIMARY (binary) metrics first
        print(f"")
        print(f"[VAL E{epoch+1}] Loss={avg_loss:.4f}")
        if vm['binary_f1'] is not None:
            print(f"  *** BINARY DETECTION (PRIMARY — 90+ target): ***")
            print(f"      F1={vm['binary_f1']:.4f}  P={vm['binary_precision']:.4f}  "
                  f"R={vm['binary_recall']:.4f}  Acc={vm['binary_acc']:.4f}  "
                  f"AUC={vm['binary_auc']:.4f}")
            print(f"      Optimal threshold: {self.optimal_binary_threshold:.2f} "
                  f"(F1@opt={best_bin_f1:.4f})")
        print(f"  5-class multi-label (secondary):")
        print(f"      Macro F1={val_f1:.4f}  AUC={val_auc:.4f}")
        thrs = [f'{t:.2f}' for t in self.optimal_thresholds]
        print(f"      Thresholds: {thrs}")
        for c, cm in vm['per_class'].items():
            print(f"      {c:18s} P={cm['precision']:.3f} R={cm['recall']:.3f} "
                  f"F1={cm['f1']:.3f} AUC={cm['roc_auc']:.3f}")

        # Track best by BINARY F1 (PRIMARY — this is the 90+ metric)
        improved = False
        if val_binary_f1 > self.best_binary_f1 or (val_binary_f1 == self.best_binary_f1 and val_binary_auc > self.best_binary_auc):
            self.best_binary_f1 = val_binary_f1
            self.best_binary_auc = val_binary_auc
            self.best_f1 = val_f1
            self.best_auc = val_auc
            self.patience_counter = 0
            improved = True
            self.metrics.best_epoch_info = {
                'epoch': epoch + 1,
                'binary_f1': val_binary_f1,
                'binary_auc': val_binary_auc,
                'binary_threshold': float(self.optimal_binary_threshold),
                'multiclass_f1': val_f1,
                'multiclass_auc': val_auc,
                'thresholds': self.optimal_thresholds.tolist()
            }
            self._save_checkpoint(epoch, is_best=True)
            print(f"  ** NEW BEST ** Binary F1={val_binary_f1:.4f} "
                  f"(5-class F1={val_f1:.4f}) at E{epoch+1}")
        else:
            self.patience_counter += 1
            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(epoch, is_best=False)

        print(f"  Patience: {self.patience_counter}/{self.early_stop_patience} "
              f"| Best Binary F1={self.best_binary_f1:.4f}")
        print('-' * 70)
        return avg_loss, val_binary_f1

    def _save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch + 1,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'best_binary_f1': self.best_binary_f1,
            'best_binary_auc': self.best_binary_auc,
            'binary_threshold': float(self.optimal_binary_threshold),
            'best_f1': self.best_f1,
            'best_auc': self.best_auc,
            'thresholds': self.optimal_thresholds.tolist(),
            'args': {k: str(v) for k, v in vars(self.args).items()
                     if not isinstance(v, torch.Tensor)},
        }
        if is_best:
            best_path = self.checkpoint_dir / 'w2v_finetune_best.pth'
            torch.save(checkpoint, best_path)
        else:
            path = self.checkpoint_dir / f'w2v_finetune_epoch_{epoch+1:03d}.pth'
            torch.save(checkpoint, path)

    def should_stop(self):
        return self.patience_counter >= self.early_stop_patience

    def train(self, num_epochs):
        print("=" * 60)
        print("WAV2VEC2-LARGE FINE-TUNING v2 — AGGRESSIVE OPTIMIZATION")
        print("=" * 60)
        print(f"Epochs: {num_epochs}")
        print(f"Effective batch: {self.args.batch_size * self.accumulate_steps}")
        print(f"Backbone LR: {self.args.backbone_lr} / Head LR: {self.args.head_lr}")
        print(f"Binary weight={self.binary_weight} (PRIMARY) | Multiclass weight={self.multiclass_weight}")
        print(f"MixUp={self.mixup_alpha} R-Drop={self.rdrop_weight}")
        print(f"Checkpoint: {self.checkpoint_dir}")
        print(f"Start: {datetime.now().isoformat()}")
        print("=" * 60)

        warmup_epochs = min(2, num_epochs)

        for epoch in range(num_epochs):
            # Linear warmup for first 2 epochs
            if epoch < warmup_epochs:
                frac = (epoch + 1) / warmup_epochs
                for i, pg in enumerate(self.optimizer.param_groups):
                    base_lr = self.args.backbone_lr if i == 0 else self.args.head_lr
                    pg['lr'] = base_lr * (0.1 + 0.9 * frac)

            t0 = time.time()
            self.train_epoch(epoch)
            _, val_f1 = self.validate(epoch)
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1} done in {elapsed/60:.1f} min")

            if self.should_stop():
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        if self.metrics.best_epoch_info:
            info = self.metrics.best_epoch_info
            print(f"")
            print(f"BEST RESULTS (Epoch {info['epoch']}):")
            print(f"  BINARY DETECTION:  F1={info['binary_f1']:.4f}  AUC={info['binary_auc']:.4f}  threshold={info['binary_threshold']:.2f}")
            print(f"  5-CLASS DETAIL:    F1={info['multiclass_f1']:.4f}  AUC={info['multiclass_auc']:.4f}")
            print(f"  Class thresholds:  {info['thresholds']}")
        print("=" * 60)
        return self.metrics


# ============================================================================
# LOGGING
# ============================================================================

class DualLogger:
    def __init__(self, filepath):
        self.console = sys.stdout
        self.file = open(filepath, 'w', encoding='utf-8')
    def write(self, msg):
        self.console.write(msg)
        self.file.write(msg)
        self.file.flush()
    def flush(self):
        try: self.console.flush()
        except Exception: pass
        try: self.file.flush()
        except Exception: pass
    def close(self):
        try: self.file.close()
        except Exception: pass


# ============================================================================
# MAIN
# ============================================================================

def get_split_stems(features_dir):
    """Get train/val split stems from existing feature extraction directories."""
    train_stems, val_stems = set(), set()
    for f in (Path(features_dir) / 'train').glob('*.npz'):
        if len(f.stem) > 5:
            train_stems.add(f.stem)
    for f in (Path(features_dir) / 'val').glob('*.npz'):
        if len(f.stem) > 5:
            val_stems.add(f.stem)
    print(f"Split: {len(train_stems)} train, {len(val_stems)} val")
    return train_stems, val_stems


def main():
    parser = argparse.ArgumentParser(description='Fine-tune wav2vec2-large v2 (aggressive)')

    # Data
    parser.add_argument('--clips-dir', default='datasets/clips/stuttering-clips/clips')
    parser.add_argument('--features-dir', default='datasets/features')
    parser.add_argument('--label-csvs', nargs='+',
                        default=['datasets/SEP-28k_labels.csv', 'datasets/fluencybank_labels.csv'])
    parser.add_argument('--no-clean', action='store_true',
                        help='Disable label cleaning (keep noisy clips)')
    parser.add_argument('--no-majority-vote', action='store_true',
                        help='Use any-annotator (>0) instead of majority (>=2)')
    parser.add_argument('--min-votes', type=int, default=2,
                        help='Minimum annotator votes for positive label')

    # Model
    parser.add_argument('--model-name', default='facebook/wav2vec2-large')
    parser.add_argument('--freeze-layers', type=int, default=12,
                        help='Freeze first N layers (12 = unfreeze last 12 of 24)')
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--lstm-hidden', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.3)

    # Training
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--accumulate', type=int, default=4)
    parser.add_argument('--backbone-lr', type=float, default=2e-5)
    parser.add_argument('--head-lr', type=float, default=5e-4)
    parser.add_argument('--grad-clip', type=float, default=0.5)
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    parser.add_argument('--cosine-t0', type=int, default=10,
                        help='Cosine annealing restart period (epochs)')
    parser.add_argument('--early-stop', type=int, default=15)
    parser.add_argument('--max-audio-len', type=int, default=None)

    # Regularization
    parser.add_argument('--mixup-alpha', type=float, default=0.3,
                        help='MixUp interpolation strength (0=off)')
    parser.add_argument('--rdrop-weight', type=float, default=0.1,
                        help='R-Drop KL divergence weight (0=off)')
    parser.add_argument('--binary-weight', type=float, default=1.0,
                        help='Binary detection loss weight (PRIMARY task)')
    parser.add_argument('--multiclass-weight', type=float, default=0.5,
                        help='Multi-class loss weight (SECONDARY task)')
    parser.add_argument('--use-ema', action='store_true', default=True)
    parser.add_argument('--ema-decay', type=float, default=0.999)

    # System
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--omp-threads', type=int, default=6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()

    # Setup threading
    os.environ['OMP_NUM_THREADS'] = str(args.omp_threads)
    os.environ['MKL_NUM_THREADS'] = str(args.omp_threads)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    torch.set_num_threads(args.omp_threads)

    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device('cpu')

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path('output') / f'finetune_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f'finetune_{timestamp}.log'
    sys.stdout = DualLogger(str(log_file))

    print(f"Log: {log_file}")
    print(f"Device: {device}")
    print(f"Threads: {torch.get_num_threads()}")

    # Data splits
    train_stems, val_stems = get_split_stems(args.features_dir)
    majority = not args.no_majority_vote
    clean = not args.no_clean

    train_dataset = RawAudioDataset(
        args.clips_dir, args.label_csvs, train_stems, 'train',
        max_audio_len=args.max_audio_len, sample_rate=DEFAULT_SAMPLE_RATE,
        augment=True, majority_vote=majority, clean_labels=clean,
        min_votes=args.min_votes
    )
    val_dataset = RawAudioDataset(
        args.clips_dir, args.label_csvs, val_stems, 'val',
        max_audio_len=args.max_audio_len, sample_rate=DEFAULT_SAMPLE_RATE,
        augment=False, majority_vote=majority, clean_labels=clean,
        min_votes=args.min_votes
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_raw_audio, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_raw_audio, pin_memory=False
    )

    # Model
    model = Wav2VecFineTuneClassifier(
        model_name=args.model_name, n_classes=NUM_CLASSES,
        freeze_layers=args.freeze_layers, hidden_dim=args.hidden_dim,
        lstm_hidden=args.lstm_hidden, dropout=args.dropout,
        use_gradient_checkpointing=True
    )

    args.pos_weight = train_dataset.get_pos_weight()
    print(f"pos_weight: {[round(x, 2) for x in args.pos_weight.tolist()]}")

    # Resume from checkpoint
    if args.resume:
        rp = Path(args.resume)
        if rp.exists():
            ckpt = torch.load(str(rp), map_location=device)
            if 'model_state' in ckpt:
                model.load_state_dict(ckpt['model_state'], strict=False)
            print(f"Resumed from {rp}")

    # Train
    trainer = FineTuneTrainer(model, train_loader, val_loader, device, args)
    metrics = trainer.train(args.epochs)

    # Save metrics and best model
    metrics.save(output_dir / 'finetune_metrics.json')

    best_src = trainer.checkpoint_dir / 'w2v_finetune_best.pth'
    best_dst = Path('Models/checkpoints') / 'w2v_finetune_BEST.pth'
    if best_src.exists():
        shutil.copy(str(best_src), str(best_dst))
        print(f"Best model -> {best_dst}")

    print(f"\nDone! Checkpoint: {trainer.checkpoint_dir}")

    if isinstance(sys.stdout, DualLogger):
        sys.stdout.close()


if __name__ == '__main__':
    main()
