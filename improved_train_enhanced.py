"""
Enhanced Training Script for Stuttering Detection

Features:
- Supports both SimpleCNN and EnhancedStutteringCNN
- Automatic class weight balancing for imbalanced data
- Early stopping to prevent overfitting
- Comprehensive metric tracking
- Per-class performance analysis
- Best model selection based on validation macro F1
- Confusion matrix generation

Usage:
    python improved_train_enhanced.py --model enhanced
    python improved_train_enhanced.py --model simple
"""

import os
import json
import argparse
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

# Import models
from model_cnn import SimpleCNN
from model_enhanced import EnhancedStutteringCNN


# ============================================================================
# Dataset
# ============================================================================

class AudioDataset(Dataset):
    """Dataset loading pre-computed features (npz files)."""
    
    def __init__(self, data_dir, split='train'):
        self.data_dir = Path(data_dir)
        self.split_dir = self.data_dir / split
        
        if not self.split_dir.exists():
            raise ValueError(f"Split directory not found: {self.split_dir}")
        
        self.files = sorted(self.split_dir.glob('**/*.npz'))
        
        if len(self.files) == 0:
            raise ValueError(f"No .npz files found in {self.split_dir}")
        
        print(f"Loaded {len(self.files)} files from {split}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        spectrogram = torch.from_numpy(data['spectrogram']).float()
        labels = torch.from_numpy(data['labels']).float()
        
        # Ensure spectrogram is 3D: (1, n_mels, time)
        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0)
        
        return spectrogram, labels


# ============================================================================
# Metrics Tracker
# ============================================================================

class MetricsTracker:
    """Tracks training and validation metrics."""
    
    STUTTER_CLASSES = [
        'Prolongation',
        'Block',
        'Sound Repetition',
        'Word Repetition',
        'Interjection'
    ]
    
    def __init__(self):
        self.metrics = {
            'train': [],
            'val': []
        }
    
    def record(self, phase, loss, y_true, y_pred_probs, y_pred_binary):
        """Record metrics for train/val phase."""
        
        if len(y_true) == 0:
            return
        
        # Convert lists of arrays to single arrays (handle variable batch sizes)
        y_true = np.vstack(y_true) if isinstance(y_true[0], np.ndarray) else np.array(y_true)
        y_pred_probs = np.vstack(y_pred_probs) if isinstance(y_pred_probs[0], np.ndarray) else np.array(y_pred_probs)
        y_pred_binary = np.vstack(y_pred_binary) if isinstance(y_pred_binary[0], np.ndarray) else np.array(y_pred_binary)
        
        # Metrics
        precision_mac, recall_mac, f1_mac, _ = precision_recall_fscore_support(
            y_true, y_pred_binary, average='macro', zero_division=0
        )
        precision_mic, recall_mic, f1_mic, _ = precision_recall_fscore_support(
            y_true, y_pred_binary, average='micro', zero_division=0
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(y_true, y_pred_binary, average=None, zero_division=0)
        
        # ROC AUC (per-label)
        try:
            roc_auc_list = [
                roc_auc_score(y_true[:, i], y_pred_probs[:, i])
                for i in range(y_true.shape[1])
            ]
            roc_auc_macro = np.mean(roc_auc_list)
        except:
            roc_auc_list = [0.0] * y_true.shape[1]
            roc_auc_macro = 0.0
        
        # Hamming loss (incorrect predictions)
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
    
    def get_best_f1(self):
        """Get best validation F1 macro."""
        if not self.metrics['val']:
            return 0.0
        return max(m['f1_macro'] for m in self.metrics['val'])
    
    def save(self, filepath):
        """Save metrics to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Saved metrics to {filepath}")
    
    def print_summary(self, epoch, phase):
        """Print epoch summary."""
        if not self.metrics[phase]:
            return
        
        m = self.metrics[phase][-1]
        print(f"\n{phase.upper()} Epoch {epoch + 1}:")
        print(f"  Loss: {m['loss']:.4f}")
        print(f"  F1 (macro): {m['f1_macro']:.4f} | F1 (micro): {m['f1_micro']:.4f}")
        print(f"  Precision (macro): {m['precision_macro']:.4f} | Recall (macro): {m['recall_macro']:.4f}")
        print(f"  Hamming Loss: {m['hamming_loss']:.4f}")
        print(f"  ROC AUC (macro): {m['roc_auc_macro']:.4f}")
        
        print(f"  Per-class F1:")
        for cls_name, metrics in m['per_class'].items():
            print(f"    {cls_name}: {metrics['f1']:.4f} (support: {metrics['support']})")


# ============================================================================
# Training
# ============================================================================

class Trainer:
    """Handles model training."""
    
    def __init__(self, model, train_loader, val_loader, device, model_name='enhanced'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model_name = model_name
        
        # Loss function with class weights
        self.compute_class_weights()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
        
        # Metrics
        self.metrics = MetricsTracker()
        
        # Amp scaler for mixed precision
        self.scaler = GradScaler()
        
        # Early stopping
        self.best_f1 = 0.0
        self.early_stop_patience = 7
        self.patience_counter = 0
    
    def compute_class_weights(self):
        """Compute class weights for imbalanced data."""
        all_labels = []
        for X, y in self.train_loader:
            all_labels.append(y.cpu().numpy())
        all_labels = np.vstack(all_labels)
        
        # Negative weights (label = 0)
        pos_counts = all_labels.sum(axis=0)
        neg_counts = (1 - all_labels).sum(axis=0)
        weights = neg_counts / (pos_counts + 1e-6)
        
        # Normalize
        weights = weights / weights.mean()
        
        self.class_weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        print(f"Class weights: {self.class_weights.cpu().numpy()}")
    
    def weighted_bce_loss(self, predictions, targets):
        """Weighted BCE Loss for imbalanced data."""
        # Standard BCE
        base_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            predictions, targets, reduction='none'
        )
        
        # Weight by class
        weighted_loss = base_loss * self.class_weights.unsqueeze(0)
        
        return weighted_loss.mean()
    
    def train_epoch(self, epoch):
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        
        y_true_all = []
        y_pred_probs_all = []
        y_pred_binary_all = []
        
        # Progress bar for training
        pbar = tqdm(self.train_loader, desc=f"EPOCH {epoch+1}/30 [TRAIN]", ncols=80)
        
        for batch_idx, (X, y) in enumerate(pbar):
            X = X.to(self.device)
            y = y.to(self.device)
            
            # Forward pass with mix precision
            self.optimizer.zero_grad()
            
            with autocast():
                logits = self.model(X)
                loss = self.weighted_bce_loss(logits, y)
            
            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
            # Metrics
            with torch.no_grad():
                probs = torch.sigmoid(logits).cpu()
                y_pred_probs_all.append(probs.numpy())
                y_pred_binary_all.append((probs > 0.5).numpy())
                y_true_all.append(y.cpu().numpy())
            
            # Update progress bar with current loss
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        pbar.close()
        avg_loss = total_loss / len(self.train_loader)
        
        self.metrics.record(
            'train',
            avg_loss,
            y_true_all,
            y_pred_probs_all,
            y_pred_binary_all
        )
        
        self.metrics.print_summary(epoch, 'train')
        
        return avg_loss
    
    def validate(self, epoch):
        """Validate one epoch."""
        self.model.eval()
        total_loss = 0.0
        
        y_true_all = []
        y_pred_probs_all = []
        y_pred_binary_all = []
        
        # Progress bar for validation
        pbar = tqdm(self.val_loader, desc=f"EPOCH {epoch+1}/30 [VAL] ", ncols=80)
        
        with torch.no_grad():
            for X, y in pbar:
                X = X.to(self.device)
                y = y.to(self.device)
                
                logits = self.model(X)
                loss = self.weighted_bce_loss(logits, y)
                
                total_loss += loss.item()
                
                probs = torch.sigmoid(logits).cpu()
                y_pred_probs_all.append(probs.numpy())
                y_pred_binary_all.append((probs > 0.5).numpy())
                y_true_all.append(y.cpu().numpy())
                
                # Update progress bar with current loss
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        pbar.close()
        avg_loss = total_loss / len(self.val_loader)
        
        self.metrics.record(
            'val',
            avg_loss,
            y_true_all,
            y_pred_probs_all,
            y_pred_binary_all
        )
        
        self.metrics.print_summary(epoch, 'val')
        
        val_f1 = self.metrics.metrics['val'][-1]['f1_macro']
        
        # Learning rate scheduling
        self.scheduler.step(val_f1)
        
        # Save best model
        if val_f1 > self.best_f1:
            self.best_f1 = val_f1
            self.patience_counter = 0
            self.save_checkpoint(epoch, is_best=True)
            print(f"✓ New best F1: {val_f1:.4f}")
        else:
            self.patience_counter += 1
            print(f"⚠ No improvement for {self.patience_counter}/{self.early_stop_patience} epochs")
        
        return avg_loss, val_f1
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint_dir = Path('Models/checkpoints')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f'{self.model_name}_epoch_{epoch + 1:03d}.pth'
        torch.save(self.model.state_dict(), checkpoint_path)
        
        if is_best:
            best_path = checkpoint_dir / f'{self.model_name}_best.pth'
            torch.save(self.model.state_dict(), best_path)
            print(f"Saved best model to {best_path}")
        
        print(f"Saved checkpoint to {checkpoint_path}")
    
    def should_stop(self):
        """Check early stopping condition."""
        return self.patience_counter >= self.early_stop_patience
    
    def train(self, num_epochs):
        """Full training loop."""
        print(f"\n{'='*70}")
        print(f"Training {self.model_name.upper()} Model")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Starting time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        for epoch in range(num_epochs):
            print(f"\n{'─'*70}")
            print(f"EPOCH {epoch + 1}/{num_epochs}")
            print(f"{'─'*70}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_f1 = self.validate(epoch)
            
            # Early stopping
            if self.should_stop():
                print(f"\n✓ Early stopping triggered after {self.patience_counter} epochs without improvement")
                break
        
        print(f"\n{'='*70}")
        print(f"Training Complete!")
        print(f"Best Validation F1: {self.best_f1:.4f}")
        print(f"{'='*70}\n")
        
        return self.metrics


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train stuttering detection model')
    parser.add_argument('--model', choices=['simple', 'enhanced'], default='enhanced',
                        help='Model architecture to train')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--data-dir', type=str, default='datasets/features',
                        help='Directory with preprocessed features')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU if available')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if (args.gpu and torch.cuda.is_available()) else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    print("\nLoading datasets...")
    train_dataset = AudioDataset(args.data_dir, split='train')
    val_dataset = AudioDataset(args.data_dir, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Model
    if args.model == 'simple':
        model = SimpleCNN(n_mels=80, n_classes=5)
    else:
        model = EnhancedStutteringCNN(n_mels=80, n_classes=5)
    
    print(f"\nModel: {args.model.upper()}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training
    trainer = Trainer(model, train_loader, val_loader, device, model_name=args.model)
    metrics = trainer.train(args.epochs)
    
    # Save metrics
    metrics_dir = Path('output/metrics')
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f'{args.model}_metrics.json'
    metrics.save(metrics_path)
    
    # Save summary
    summary = {
        'model': args.model,
        'epochs_trained': len(metrics.metrics['train']),
        'best_val_f1': trainer.best_f1,
        'final_metrics': {
            'train': metrics.metrics['train'][-1] if metrics.metrics['train'] else {},
            'val': metrics.metrics['val'][-1] if metrics.metrics['val'] else {}
        },
        'timestamp': datetime.now().isoformat()
    }
    
    summary_path = metrics_dir / f'{args.model}_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nMetrics saved to {metrics_path}")
    print(f"Summary saved to {summary_path}")


if __name__ == '__main__':
    main()
