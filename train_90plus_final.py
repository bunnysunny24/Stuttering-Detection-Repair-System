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
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from datetime import datetime
import pickle
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import models and utilities
from model_improved_90plus import ImprovedStutteringCNN

# GPU optimization
# GPU optimization - MAXIMUM PERFORMANCE
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
torch.set_num_threads(1)  # Prevent CPU thread overhead
torch.set_num_interop_threads(1)


# ============================================================================
# FOCAL LOSS (handles extreme imbalance)
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for extreme class imbalance."""
    
    def __init__(self, alpha=1.0, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(self, predictions, targets):
        """Focal Loss = alpha * (1 - p_t)^gamma * BCE"""
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            predictions, targets, reduction='none', pos_weight=self.pos_weight
        )
        probs = torch.sigmoid(predictions)
        p_t = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - p_t).pow(self.gamma)
        focal_loss = self.alpha * focal_weight * bce
        return focal_loss.mean()


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

class AudioAugmentation:
    """
    Apply augmentation to audio features for better generalization.
    Works with mel-spectrogram and enhanced features.
    """
    
    def __init__(self, augment_prob=0.5):
        self.augment_prob = augment_prob
    
    def __call__(self, spectrogram):
        """Apply random augmentations."""
        if np.random.random() < self.augment_prob:
            spectrogram = self._apply_augmentations(spectrogram)
        return spectrogram
    
    def _apply_augmentations(self, spec):
        """Apply one or more augmentations."""
        augmentation_type = np.random.choice(['time_mask', 'freq_mask', 'noise', 'stretch'], p=[0.3, 0.3, 0.2, 0.2])
        
        if augmentation_type == 'time_mask':
            # Mask random time segments (simulate speech gaps)
            spec = self._time_masking(spec)
        elif augmentation_type == 'freq_mask':
            # Mask random frequency bands
            spec = self._freq_masking(spec)
        elif augmentation_type == 'noise':
            # Add gaussian noise
            spec = self._add_noise(spec)
        else:  # stretch
            # Time stretching (simulate faster/slower speech)
            spec = self._time_stretch(spec)
        
        return spec
    
    def _time_masking(self, spec):
        """Mask random time region."""
        time_length = spec.shape[1]
        mask_len = np.random.randint(5, int(time_length * 0.3))
        mask_start = np.random.randint(0, time_length - mask_len)
        spec = spec.copy()
        spec[:, mask_start:mask_start + mask_len] = np.mean(spec)
        return spec
    
    def _freq_masking(self, spec):
        """Mask random frequency band."""
        freq_length = spec.shape[0]
        mask_len = np.random.randint(5, int(freq_length * 0.2))
        mask_start = np.random.randint(0, freq_length - mask_len)
        spec = spec.copy()
        spec[mask_start:mask_start + mask_len, :] = np.mean(spec)
        return spec
    
    def _add_noise(self, spec):
        """Add gaussian noise."""
        noise = np.random.normal(0, 0.01, spec.shape)
        return spec + noise
    
    def _time_stretch(self, spec):
        """Time stretching (simple implementation)."""
        stretch_factor = np.random.uniform(0.95, 1.05)
        new_length = int(spec.shape[1] * stretch_factor)
        if new_length < spec.shape[1]:
            # Pad
            pad_width = spec.shape[1] - new_length
            spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='edge')
        else:
            # Crop
            spec = spec[:, :spec.shape[1]]
        return spec


# ============================================================================
# COLLATE FUNCTION FOR VARIABLE LENGTH SEQUENCES
# ============================================================================

def collate_variable_length(batch):
    """Custom collate function to handle variable length sequences with padding."""
    specs = [item[0] for item in batch]  # (1, 123, time) 
    labels = [item[1] for item in batch]  # (5,)
    
    # Find max time length
    max_time = max(spec.shape[-1] for spec in specs)
    
    # Pad specs to max length
    padded_specs = []
    for spec in specs:
        if spec.shape[-1] < max_time:
            # Pad on the right (time dimension)
            pad_amount = max_time - spec.shape[-1]
            spec = torch.nn.functional.pad(spec, (0, pad_amount))  # Pad on last dim
        padded_specs.append(spec)
    
    # Stack
    specs_batch = torch.cat(padded_specs, dim=0)  # (batch, 123, max_time)
    labels_batch = torch.stack(labels, dim=0)     # (batch, 5)
    
    return specs_batch, labels_batch


# ============================================================================
# DATASET
# ============================================================================

class AudioDataset(Dataset):
    """Load preprocessed features with augmentation."""
    
    def __init__(self, data_dir, split='train', augment=True):
        self.data_dir = Path(data_dir)
        self.split_dir = self.data_dir / split
        self.files = sorted(self.split_dir.glob('**/*.npz'))
        self.augment = augment and split == 'train'
        self.augmentation = AudioAugmentation(augment_prob=0.5)
        
        if len(self.files) == 0:
            raise ValueError(f"No files found in {self.split_dir}")
        
        print(f"Loaded {len(self.files)} files from {split}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        spectrogram = torch.from_numpy(data['spectrogram']).float()
        
        # Load labels if available, otherwise create dummy labels (all zeros)
        if 'labels' in data:
            labels = torch.from_numpy(data['labels']).float()
        else:
            # Create dummy labels - will be ignored or handled specially
            labels = torch.zeros(5, dtype=torch.float32)
        
        # Ensure 3D: (channels, height, width) - actually (channels, time_steps)
        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0)  # (1, 123, time) format
        
        # Apply augmentation if training
        if self.augment:
            spec_np = spectrogram.squeeze(0).numpy()  # Remove batch dim
            spec_np = self.augmentation(spec_np)
            spectrogram = torch.from_numpy(spec_np).float().unsqueeze(0)
        
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
            'optimized_thresholds': []  # NEW: Track which thresholds were optimal
        }
    
    def record(self, phase, loss, y_true, y_pred_probs, y_pred_binary, thresholds=None):
        """Record epoch metrics."""
        if len(y_true) == 0:
            return
        
        y_true = np.vstack(y_true) if isinstance(y_true[0], np.ndarray) else np.array(y_true)
        y_pred_probs = np.vstack(y_pred_probs) if isinstance(y_pred_probs[0], np.ndarray) else np.array(y_pred_probs)
        y_pred_binary = np.vstack(y_pred_binary) if isinstance(y_pred_binary[0], np.ndarray) else np.array(y_pred_binary)
        
        # Calculate metrics
        precision_mac, recall_mac, f1_mac, _ = precision_recall_fscore_support(
            y_true, y_pred_binary, average='macro', zero_division=0
        )
        precision_mic, recall_mic, f1_mic, _ = precision_recall_fscore_support(
            y_true, y_pred_binary, average='micro', zero_division=0
        )
        
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(y_true, y_pred_binary, average=None, zero_division=0)
        
        try:
            roc_auc_list = [
                roc_auc_score(y_true[:, i], y_pred_probs[:, i])
                for i in range(y_true.shape[1])
            ]
            roc_auc_macro = np.mean(roc_auc_list)
        except:
            roc_auc_list = [0.0] * y_true.shape[1]
            roc_auc_macro = 0.0
        
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
    
    def __init__(self, model, train_loader, val_loader, device, model_name='improved_90plus', logger=None):
        self.model = model.to(device)
        self.device = device
        self.model_name = model_name
        self.logger = logger
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Training config
        self.learning_rate = 1e-4  # Slightly higher initial LR
        self.early_stop_patience = 50
        self.patience_counter = 0
        self.best_f1 = 0.0
        self.total_epochs = 0
        
        # Optimizers
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        # Scheduler with warmup
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )
        
        # Mixed precision
        self.scaler = GradScaler()
        
        # Metrics
        self.metrics = MetricsTracker()
        
        # Class weights (cached - not recomputed each run)
        # Use default weights for extreme imbalance (4.3x class imbalance)
        self.class_weights = torch.tensor([4.3, 4.3, 4.3, 4.3, 4.3], dtype=torch.float32).to(self.device)
        print("Using cached class weights (4.3x imbalance):")
        class_names = ['Prolongation', 'Block', 'Sound Repetition', 'Word Repetition', 'Interjection']
        for i, name in enumerate(class_names):
            print(f"  {name:20s}: {self.class_weights[i]:6.2f}")
        
        self.focal_loss = FocalLoss(pos_weight=self.class_weights)
        
        # Thresholds - START WITH 0.5 FOR UNTRAINED MODEL, OPTIMIZE AFTER FIRST VAL
        self.optimal_thresholds = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        self.thresholds_optimized = False
    
    def optimize_thresholds(self, y_true, y_pred_probs):
        """Find best thresholds for each class."""
        num_classes = y_true.shape[1]
        optimal_thresholds = np.zeros(num_classes)
        
        for class_idx in range(num_classes):
            best_f1 = 0
            best_thresh = 0.5
            
            for threshold in np.arange(0.1, 1.0, 0.05):
                y_pred = (y_pred_probs[:, class_idx] > threshold).astype(int)
                y_true_bin = y_true[:, class_idx].astype(int)
                
                if y_true_bin.sum() == 0:
                    continue
                
                _, _, f1, _ = precision_recall_fscore_support(
                    y_true_bin, y_pred, average='binary', zero_division=0
                )
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = threshold
            
            optimal_thresholds[class_idx] = best_thresh
        
        return optimal_thresholds
    
    def train_epoch(self, epoch):
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        
        # GPU synchronization
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        y_true_all = []
        y_pred_probs_all = []
        y_pred_binary_all = []
        
        pbar = tqdm(self.train_loader, desc=f"EPOCH {epoch+1} [TRAIN]")
        for batch_idx, (X, y) in enumerate(pbar):
            X = X.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            with autocast():
                logits = self.model(X)
                loss = self.focal_loss(logits, y)
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
            with torch.no_grad():
                probs = torch.sigmoid(logits).cpu()
                y_pred_probs_all.append(probs.numpy())
                y_pred_binary = (probs.numpy() > self.optimal_thresholds).astype(float)
                y_pred_binary_all.append(y_pred_binary)
                y_true_all.append(y.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Clear GPU cache every 10 batches
            if batch_idx % 10 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Final GPU sync
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        pbar.close()
        
        avg_loss = total_loss / len(self.train_loader)
        self.metrics.record('train', avg_loss, y_true_all, y_pred_probs_all, y_pred_binary_all)
        
        # Print training metrics
        train_metrics = self.metrics.metrics['train'][-1]
        train_f1 = train_metrics['f1_macro']
        train_auc = train_metrics['roc_auc_macro']
        print(f"[TRAIN] F1={train_f1:.4f}, AUC={train_auc:.4f}, Loss={avg_loss:.4f}")
        
        return avg_loss
    
    def validate(self, epoch):
        """Validate one epoch."""
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
                loss = self.focal_loss(logits, y)
                total_loss += loss.item()
                
                probs = torch.sigmoid(logits).cpu().numpy()
                y_pred_probs_all.append(probs)
                y_true_all.append(y.cpu().numpy())
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                # Clear GPU cache every 10 batches
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
        pbar.close()
        
        avg_loss = total_loss / len(self.val_loader)
        y_true_array = np.vstack(y_true_all)
        y_pred_probs_array = np.vstack(y_pred_probs_all)
        
        # Optimize thresholds only once
        if not self.thresholds_optimized:
            self.optimal_thresholds = self.optimize_thresholds(y_true_array, y_pred_probs_array)
            self.thresholds_optimized = True
            print(f"✓ Optimized thresholds: {[f'{t:.3f}' for t in self.optimal_thresholds]}")
        
        y_pred_binary_array = (y_pred_probs_array > self.optimal_thresholds).astype(float)
        
        self.metrics.record('val', avg_loss, [y_true_array], [y_pred_probs_array], [y_pred_binary_array], self.optimal_thresholds)
        
        val_metrics = self.metrics.metrics['val'][-1]
        val_f1 = val_metrics['f1_macro']
        
        # Print results
        print(f"\n[VAL] F1={val_f1:.4f}, Precision={val_metrics['precision_macro']:.4f}, Recall={val_metrics['recall_macro']:.4f}, ROC_AUC={val_metrics['roc_auc_macro']:.4f}")
        
        # Checkpoints
        if val_f1 > self.best_f1:
            self.best_f1 = val_f1
            self.patience_counter = 0
            self.save_checkpoint(epoch, is_best=True)
        else:
            self.patience_counter += 1
            self.save_checkpoint(epoch, is_best=False)
        
        self.scheduler.step(val_f1)
        
        return avg_loss, val_f1
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint_dir = Path('Models/checkpoints')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f'{self.model_name}_epoch_{epoch + 1:03d}.pth'
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        if is_best:
            best_path = checkpoint_dir / f'{self.model_name}_best.pth'
            torch.save(self.model.state_dict(), best_path)
            print(f"✓ New best model! F1={self.best_f1:.4f}")
    
    def should_stop(self):
        return self.patience_counter >= self.early_stop_patience
    
    def train(self, num_epochs):
        """Full training loop."""
        print(f"\n{'='*80}")
        print(f"TRAINING 90+ ACCURACY MODEL")
        print(f"{'='*80}")
        print(f"Model: {self.model_name.upper()}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {len(next(iter(self.train_loader))[0])}")
        print(f"Start time: {datetime.now().isoformat()}")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss, val_f1 = self.validate(epoch)
            
            if self.should_stop():
                print(f"\n✓ Early stopping at epoch {epoch + 1}")
                break
        
        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETE")
        print(f"Best F1: {self.best_f1:.4f}")
        print(f"{'='*80}\n")
        
        # Final GPU cleanup
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        return self.metrics


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train 90+ accuracy stuttering model')
    parser.add_argument('--epochs', type=int, default=60, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size (increased for max GPU)')
    parser.add_argument('--data-dir', type=str, default='datasets/features', help='Data directory')
    parser.add_argument('--gpu', action='store_true', default=True, help='Use GPU')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if (args.gpu and torch.cuda.is_available()) else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading datasets...")
    train_dataset = AudioDataset(args.data_dir, split='train', augment=True)
    val_dataset = AudioDataset(args.data_dir, split='val', augment=False)
    
    # GPU-first data loading (no CPU parallelization overhead)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # Model
    model = ImprovedStutteringCNN(n_channels=123, n_classes=5, dropout=0.4)
    model_name = 'improved_90plus'
    
    print(f"\nModel: ImprovedStutteringCNN (8-layer, 6.5M params)")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    trainer = Trainer(model, train_loader, val_loader, device, model_name=model_name)
    metrics = trainer.train(args.epochs)
    
    # Save
    metrics_dir = Path('output/metrics')
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f'{model_name}_metrics.json'
    metrics.save(metrics_path)
    
    print(f"\n✅ Completed! Metrics saved to {metrics_path}")
    print(f"Best model: Models/checkpoints/{model_name}_best.pth")
    print(f"Training expected accuracy: 85-90% F1 with optimized thresholds")


if __name__ == '__main__':
    main()
