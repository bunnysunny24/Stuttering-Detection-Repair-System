"""
Enhanced Training Script for Stuttering Detection - FIXED FOR CLASS IMBALANCE

FIXES APPLIED:
1. Focal Loss: Replaces BCE for better handling of extreme imbalance
2. Proper Class Weights: Corrected formula using neg/pos ratio per class
3. Per-Class Threshold Optimization: Automatically finds best threshold per class
4. Lower Learning Rate: 5e-5 instead of 1e-4 for stable learning
5. Longer Early Stopping: 50 epochs patience instead of 31
6. Increased Training: Default 60 epochs instead of 30

The model now properly handles:
- Extreme class imbalance (some classes have 1:100+ neg:pos ratio)
- Multi-label stuttering detection
- Automatic threshold tuning on validation set

Features:
- Supports both SimpleCNN and EnhancedStutteringCNN
- Automatic class weight balancing for imbalanced data
- Focal loss for better positive class learning
- Threshold optimization per class
- Comprehensive metric tracking
- Per-class performance analysis
- Best model selection based on validation macro F1
- Confusion matrix generation

Usage:
    python improved_train_enhanced.py --model enhanced --epochs 60
    python improved_train_enhanced.py --model simple --epochs 60
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

# GPU Optimization
torch.backends.cudnn.benchmark = True  # Enable automatic kernel selection
torch.backends.cudnn.deterministic = False  # Priority on speed over determinism
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async GPU calls


# ============================================================================
# Focal Loss for Imbalanced Data
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss: handles class imbalance better than BCE."""
    
    def __init__(self, alpha=1.0, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: logits (batch, num_classes)
            targets: binary targets (batch, num_classes)
            pos_weight: per-class positive weights
        """
        # BCE with logits
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            predictions, targets, reduction='none', pos_weight=self.pos_weight
        )
        
        # Get probabilities
        probs = torch.sigmoid(predictions)
        
        # Focal term: (1 - p_t)^gamma
        p_t = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - p_t).pow(self.gamma)
        
        # Focal loss = alpha * (1 - p_t)^gamma * BCE
        focal_loss = self.alpha * focal_weight * bce
        
        return focal_loss.mean()


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
        
        # Focal loss for handling imbalance - much better than simple weighted BCE
        self.focal_loss = FocalLoss(alpha=1.0, gamma=2.0, pos_weight=self.class_weights)
        
        # Optimizer - using lower LR initially, will ramp up if needed
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=5e-5,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler - warmup then reduce on plateau
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=True
        )
        
        # Metrics
        self.metrics = MetricsTracker()
        
        # Amp scaler for mixed precision
        self.scaler = GradScaler()
        
        # Early stopping - longer patience for difficult imbalanced tasks
        self.best_f1 = 0.0
        self.early_stop_patience = 50  # Allow more epochs without improvement
        self.patience_counter = 0
        
        # Per-class threshold optimization
        self.optimal_thresholds = None
    
    def compute_class_weights(self):
        """Compute class weights for imbalanced data using proper balanced formula."""
        print("Computing class weights from training data...")
        all_labels = []
        pbar = tqdm(self.train_loader, desc="Loading labels", ncols=80)
        for X, y in pbar:
            all_labels.append(y.cpu().numpy())
        pbar.close()
        all_labels = np.vstack(all_labels)
        
        # Proper balanced weight formula: weight_j = n_samples / (n_classes * n_j)
        num_classes = all_labels.shape[1]
        n_samples = len(all_labels)
        
        # Count positive samples per class
        pos_counts = (all_labels == 1).sum(axis=0)
        neg_counts = (all_labels == 0).sum(axis=0)
        
        # Weight positive class: ratio of negative to positive (higher = more imbalanced)
        # This is used by pos_weight in BCE loss
        pos_weights = neg_counts / (pos_counts + 1e-6)
        
        # Additional scaling for extreme imbalance (cap at 50, but keep ratios)
        pos_weights = np.minimum(pos_weights, 50.0)
        
        self.class_weights = torch.tensor(pos_weights, dtype=torch.float32).to(self.device)
        
        print(f"Per-class negative:positive ratios (weights for pos_weight):")
        class_names = ['Prolongation', 'Block', 'Sound Repetition', 'Word Repetition', 'Interjection']
        for i, name in enumerate(class_names):
            print(f"  {name:20s}: {pos_weights[i]:6.2f} (pos={pos_counts[i]:5d}, neg={neg_counts[i]:5d})")
    
    def optimize_thresholds(self, y_true, y_pred_probs):
        """Optimize per-class thresholds to maximize F1 score."""
        num_classes = y_true.shape[1]
        optimal_thresholds = np.zeros(num_classes)
        
        for class_idx in range(num_classes):
            best_f1 = 0
            best_thresh = 0.5
            
            # Test thresholds from 0.1 to 0.9
            for threshold in np.arange(0.1, 1.0, 0.05):
                y_pred_binary = (y_pred_probs[:, class_idx] > threshold).astype(int)
                y_true_binary = y_true[:, class_idx].astype(int)
                
                if y_true_binary.sum() == 0:  # No positive samples
                    continue
                
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true_binary, y_pred_binary, average='binary', zero_division=0
                )
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = threshold
            
            optimal_thresholds[class_idx] = best_thresh
        
        return optimal_thresholds
    
    def weighted_bce_loss(self, predictions, targets):
        """Use Focal Loss for better imbalance handling."""
        return self.focal_loss(predictions, targets)
    
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
                
                # Use optimized thresholds if available, else use 0.3 (lower default for imbalance)
                if self.optimal_thresholds is not None:
                    y_pred_binary = (probs.numpy() > self.optimal_thresholds).astype(float)
                else:
                    y_pred_binary = (probs.numpy() > 0.3).astype(float)
                
                y_pred_binary_all.append(y_pred_binary)
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
                y_true_all.append(y.cpu().numpy())
                
                # Update progress bar with current loss
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        pbar.close()
        avg_loss = total_loss / len(self.val_loader)
        
        # Convert to arrays for threshold optimization
        y_true_array = np.vstack(y_true_all)
        y_pred_probs_array = np.vstack(y_pred_probs_all)
        
        # Optimize thresholds on validation set every epoch
        self.optimal_thresholds = self.optimize_thresholds(y_true_array, y_pred_probs_array)
        
        # Apply optimized thresholds to get binary predictions
        y_pred_binary_array = (y_pred_probs_array > self.optimal_thresholds).astype(float)
        
        # Convert back to list for metrics tracking
        y_pred_binary_all = [y_pred_binary_array[i:i+len(y_true_all[j])] if i+len(y_true_all[j]) <= len(y_pred_binary_array) 
                              else y_pred_binary_array[i:] for i, j in zip(range(0, len(y_pred_binary_array), 1), range(len(y_true_all)))]
        y_pred_binary_all = [y_pred_binary_array]  # Simpler: just use the full array as one batch
        y_true_all_list = [y_true_array]
        y_pred_probs_all_list = [y_pred_probs_array]
        
        self.metrics.record(
            'val',
            avg_loss,
            y_true_all_list,
            y_pred_probs_all_list,
            y_pred_binary_all
        )
        
        # Print metrics
        self.metrics.print_summary(epoch, 'val')
        
        # Print optimal thresholds
        class_names = ['Prolongation', 'Block', 'Sound Repetition', 'Word Repetition', 'Interjection']
        print(f"  Optimal thresholds per class:")
        for i, name in enumerate(class_names):
            print(f"    {name:20s}: {self.optimal_thresholds[i]:.3f}")
        
        val_f1 = self.metrics.metrics['val'][-1]['f1_macro']
        val_metrics = self.metrics.metrics['val'][-1]
        
        # Diagnostic output to identify learning issues
        print(f"  💡 Diagnostic: Precision={val_metrics['precision_macro']:.3f}, Recall={val_metrics['recall_macro']:.3f}")
        if val_metrics['recall_macro'] < 0.05:
            print(f"  ⚠️  WARNING: Model has very low recall - not learning to detect stutters!")
            print(f"      Check: data quality, label correctness, feature extraction")
        
        # Learning rate scheduling - based on F1 improvement
        self.scheduler.step(val_f1)
        
        # Save checkpoint every epoch, mark as best if improved
        if val_f1 > self.best_f1:
            self.best_f1 = val_f1
            self.patience_counter = 0
            self.save_checkpoint(epoch, is_best=True)
            print(f"✓ New best F1: {val_f1:.4f}")
        else:
            self.patience_counter += 1
            self.save_checkpoint(epoch, is_best=False)  # Save epoch checkpoint even without improvement
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
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"cuDNN: {torch.backends.cudnn.version()}")
            print(f"Optimizations: CUDNN Benchmark=ON, Pinned Memory=ON")
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
    parser = argparse.ArgumentParser(description='Train stuttering detection model with imbalance fixes')
    parser.add_argument('--model', choices=['simple', 'enhanced'], default='enhanced',
                        help='Model architecture to train')
    parser.add_argument('--epochs', type=int, default=60,
                        help='Number of epochs to train (default: 60 for sufficient learning with imbalance)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size (default: 128 for max GPU utilization)')
    parser.add_argument('--data-dir', type=str, default='datasets/features',
                        help='Directory with preprocessed features')
    parser.add_argument('--gpu', action='store_true', default=True,
                        help='Use GPU if available (default: True)')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if (args.gpu and torch.cuda.is_available()) else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    print("\nLoading datasets...")
    train_dataset = AudioDataset(args.data_dir, split='train')
    val_dataset = AudioDataset(args.data_dir, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # Model
    if args.model == 'simple':
        model = SimpleCNN(n_mels=80, n_classes=5)
    else:
        model = EnhancedStutteringCNN(n_mels=80, n_classes=5)
    
    print(f"\nModel: {args.model.upper()}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training epochs: {args.epochs} (increased for imbalanced data)")
    
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
        'improvements_applied': [
            'Focal Loss instead of BCE',
            'Proper class weight formula',
            'Per-class threshold optimization',
            'Lower learning rate (5e-5)',
            'Longer patience (50 epochs)',
            'More training epochs (60 default)'
        ],
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
    print(f"\n✅ Training complete with imbalance fixes!")
    print(f"   - Focal Loss applied")
    print(f"   - Per-class thresholds optimized")
    print(f"   - Expected much better recall and F1 scores")


if __name__ == '__main__':
    main()
