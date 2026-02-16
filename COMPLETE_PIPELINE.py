"""
COMPLETE STUTTER DETECTION + REPAIR PIPELINE - IMPROVED DIAGNOSTICS
Enhanced with better import error reporting
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
import traceback
warnings.filterwarnings('ignore')

import torch
import logging
from typing import Optional, Dict, List, Any

# Add Models to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("IMPORT DIAGNOSTICS")
print("=" * 80)

# Import components with VERBOSE error reporting
FeatureExtractionPipeline = None
FeatureExtractionManager = None

print("\n1. Attempting to import extract_features.FeatureExtractionPipeline...")
try:
    from extract_features import FeatureExtractionPipeline
    print("   ✓ SUCCESS")
except Exception as e:
    print(f"   ✗ FAILED: {type(e).__name__}: {e}")
    traceback.print_exc(limit=2)
    FeatureExtractionPipeline = None

print("\n2. Attempting to import extract_features_90plus.FeatureExtractionManager...")
try:
    from extract_features_90plus import FeatureExtractionManager
    print("   ✓ SUCCESS")
except Exception as e:
    print(f"   ✗ FAILED: {type(e).__name__}: {e}")
    traceback.print_exc(limit=2)
    FeatureExtractionManager = None

# Some extractor implementations define FeatureExtractionPipeline inside extract_features_90plus
if FeatureExtractionManager is None:
    try:
        import importlib
        ef90 = importlib.import_module('extract_features_90plus')
        if hasattr(ef90, 'FeatureExtractionPipeline'):
            FeatureExtractionPipeline = getattr(ef90, 'FeatureExtractionPipeline')
            print("   → Found FeatureExtractionPipeline inside extract_features_90plus (compat)")
    except Exception:
        pass

print("\n3. Attempting to import train_90plus_final components...")
try:
    from train_90plus_final import Trainer, AudioDataset, collate_variable_length
    print("   ✓ SUCCESS")
except Exception as e:
    print(f"   ✗ FAILED: {type(e).__name__}: {e}")
    traceback.print_exc(limit=2)
    raise SystemExit("Cannot proceed without training components")

print("\n4. Attempting to import model_improved_90plus.ImprovedStutteringCNN...")
try:
    from model_improved_90plus import ImprovedStutteringCNN
    print("   ✓ SUCCESS")
except Exception as e:
    print(f"   ✗ FAILED: {type(e).__name__}: {e}")
    traceback.print_exc(limit=2)
    raise SystemExit("Cannot proceed without model")

print("\n5. Attempting to import enhanced_audio_preprocessor.EnhancedAudioPreprocessor...")
try:
    from enhanced_audio_preprocessor import EnhancedAudioPreprocessor
    print("   ✓ SUCCESS")
except Exception as e:
    print(f"   ✗ FAILED: {type(e).__name__}: {e}")
    traceback.print_exc(limit=2)
    EnhancedAudioPreprocessor = None

print("\n6. Attempting to import constants.TOTAL_CHANNELS...")
try:
    from constants import TOTAL_CHANNELS
    print(f"   ✓ SUCCESS (TOTAL_CHANNELS={TOTAL_CHANNELS})")
except Exception as e:
    print(f"   ✗ FAILED: {type(e).__name__}: {e}")
    TOTAL_CHANNELS = 123
    print(f"   → Using default: TOTAL_CHANNELS={TOTAL_CHANNELS}")

print("\n7. Attempting to import repair_advanced components...")
try:
    from repair_advanced import AdvancedStutterRepair, extract_stutter_analysis
    REPAIR_AVAILABLE = True
    print("   ✓ SUCCESS")
except ImportError as e:
    print(f"   ✗ FAILED: {e}")
    REPAIR_AVAILABLE = False
    print("   → Repair module not available - detection-only mode")

print("\n" + "=" * 80)
print("IMPORT SUMMARY")
print("=" * 80)
print(f"Feature Extraction: {'✓ Available' if (FeatureExtractionPipeline or FeatureExtractionManager) else '✗ MISSING'}")
print(f"Audio Preprocessor:  {'✓ Available' if EnhancedAudioPreprocessor else '✗ Missing'}")
print(f"Repair Module:       {'✓ Available' if REPAIR_AVAILABLE else '✗ Missing'}")
print("=" * 80)

if not (FeatureExtractionPipeline or FeatureExtractionManager):
    print("\n⚠️  WARNING: No feature extraction module available!")
    print("\nPossible solutions:")
    print("1. Check that extract_features_90plus.py exists in the Models directory")
    print("2. Verify all dependencies are installed (librosa, soundfile, etc.)")
    print("3. Run the diagnostic script: python diagnose_imports.py")
    print("4. Check the detailed error messages above")
    print("\nYou can still run training if features are already extracted.")
    print("=" * 80)

class PerClassMetrics:
    """Track and compute per-class metrics."""
    
    STUTTER_CLASSES = ['Prolongation', 'Block', 'Sound Repetition', 'Word Repetition', 'Interjection']
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.tp = np.zeros(5)  # True positives per class
        self.fp = np.zeros(5)  # False positives per class
        self.fn = np.zeros(5)  # False negatives per class
        self.tn = np.zeros(5)  # True negatives per class
    
    def update(self, predictions: np.ndarray, targets: np.ndarray, threshold: float = 0.5):
        """
        Update metrics with batch predictions.
        
        Args:
            predictions: (batch_size, 5) probability predictions
            targets: (batch_size, 5) binary targets
            threshold: Classification threshold
        """
        pred_binary = (predictions > threshold).astype(int)
        
        for i in range(5):
            self.tp[i] += np.sum((pred_binary[:, i] == 1) & (targets[:, i] == 1))
            self.fp[i] += np.sum((pred_binary[:, i] == 1) & (targets[:, i] == 0))
            self.fn[i] += np.sum((pred_binary[:, i] == 0) & (targets[:, i] == 1))
            self.tn[i] += np.sum((pred_binary[:, i] == 0) & (targets[:, i] == 0))
    
    def compute(self) -> Dict[str, Any]:
        """Compute per-class metrics."""
        precision = np.zeros(5)
        recall = np.zeros(5)
        f1 = np.zeros(5)
        
        for i in range(5):
            # Precision
            if self.tp[i] + self.fp[i] > 0:
                precision[i] = self.tp[i] / (self.tp[i] + self.fp[i])
            
            # Recall
            if self.tp[i] + self.fn[i] > 0:
                recall[i] = self.tp[i] / (self.tp[i] + self.fn[i])
            
            # F1
            if precision[i] + recall[i] > 0:
                f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        
        return {
            'per_class': {
                self.STUTTER_CLASSES[i]: {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1': float(f1[i]),
                    'support': int(self.tp[i] + self.fn[i]),
                    'tp': int(self.tp[i]),
                    'fp': int(self.fp[i]),
                    'fn': int(self.fn[i]),
                    'tn': int(self.tn[i])
                }
                for i in range(5)
            },
            'macro_avg': {
                'precision': float(np.mean(precision)),
                'recall': float(np.mean(recall)),
                'f1': float(np.mean(f1))
            },
            'weighted_avg': {
                'precision': float(np.average(precision, weights=self.tp + self.fn)),
                'recall': float(np.average(recall, weights=self.tp + self.fn)),
                'f1': float(np.average(f1, weights=self.tp + self.fn))
            }
        }
    
    def log_metrics(self, logger: logging.Logger, prefix: str = ""):
        """Log per-class metrics in a formatted table."""
        metrics = self.compute()
        
        logger.info(f"\n{prefix}Per-Class Metrics:")
        logger.info("=" * 100)
        logger.info(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10} {'TP':>8} {'FP':>8} {'FN':>8}")
        logger.info("-" * 100)
        
        for class_name, class_metrics in metrics['per_class'].items():
            logger.info(
                f"{class_name:<20} "
                f"{class_metrics['precision']:>10.4f} "
                f"{class_metrics['recall']:>10.4f} "
                f"{class_metrics['f1']:>10.4f} "
                f"{class_metrics['support']:>10} "
                f"{class_metrics['tp']:>8} "
                f"{class_metrics['fp']:>8} "
                f"{class_metrics['fn']:>8}"
            )
        
        logger.info("-" * 100)
        logger.info(
            f"{'Macro Avg':<20} "
            f"{metrics['macro_avg']['precision']:>10.4f} "
            f"{metrics['macro_avg']['recall']:>10.4f} "
            f"{metrics['macro_avg']['f1']:>10.4f}"
        )
        logger.info(
            f"{'Weighted Avg':<20} "
            f"{metrics['weighted_avg']['precision']:>10.4f} "
            f"{metrics['weighted_avg']['recall']:>10.4f} "
            f"{metrics['weighted_avg']['f1']:>10.4f}"
        )
        logger.info("=" * 100)


def setup_logging(output_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """
    Setup comprehensive logging system.
    
    Args:
        output_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        Configured logger
    """
    output_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('StutterPipeline')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler with color formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    
    # File handler for detailed logs
    log_file = output_dir / f'pipeline_{datetime.now():%Y%m%d_%H%M%S}.log'
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    
    # Error file handler
    error_file = output_dir / f'pipeline_errors_{datetime.now():%Y%m%d_%H%M%S}.log'
    error_handler = logging.FileHandler(error_file, encoding='utf-8')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_format)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    
    logger.info(f"Logging initialized - Level: {log_level}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Error log: {error_file}")
    
    return logger


def log_system_info(logger: logging.Logger):
    """Log system and environment information."""
    logger.info("=" * 80)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 80)
    
    # Python version
    logger.info(f"Python: {sys.version.split()[0]}")
    
    # PyTorch info
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"  GPU {i}: {props.name}")
            logger.info(f"    Memory: {props.total_memory / 1024**3:.2f} GB")
            logger.info(f"    Compute Capability: {props.major}.{props.minor}")
    
    # CPU info
    logger.info(f"CPU Cores: {os.cpu_count()}")
    
    # Modules
    logger.info(f"Feature Extraction: {'Available' if (FeatureExtractionPipeline or FeatureExtractionManager) else 'Missing'}")
    logger.info(f"Audio Preprocessor: {'Available' if EnhancedAudioPreprocessor else 'Missing'}")
    logger.info(f"Repair Module: {'Available' if REPAIR_AVAILABLE else 'Missing'}")
    
    logger.info("=" * 80)


def log_gpu_memory(logger: logging.Logger, stage: str = ""):
    """Log GPU memory usage."""
    if not torch.cuda.is_available():
        return
    
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        logger.debug(f"GPU {i} Memory [{stage}] - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")


class CompletePipeline:
    """End-to-end stutter detection and repair pipeline with comprehensive logging."""
    
    def __init__(self, output_dir='output', checkpoint_dir='Models/checkpoints', log_level='INFO'):
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'repaired_audio').mkdir(exist_ok=True)
        (self.output_dir / 'analysis').mkdir(exist_ok=True)
        (self.output_dir / 'metrics').mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = setup_logging(self.output_dir, log_level)
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Device: {self.device}")
        
        # Log system info
        log_system_info(self.logger)
        
        # Model state
        self.model = None
        self.best_model_path = None
        
        # Metrics
        self.metrics_tracker = PerClassMetrics()
        
        self.logger.info("Pipeline initialized successfully")
    
    def extract_features(self, sample_size: Optional[int] = None, skip_existing: bool = False) -> bool:
        """
        Phase 1: Extract features from audio files.
        
        Args:
            sample_size: Number of files to extract (None = all)
            skip_existing: Skip if features already exist
        
        Returns:
            Success status
        """
        self.logger.info("=" * 80)
        self.logger.info("PHASE 1: FEATURE EXTRACTION")
        self.logger.info("=" * 80)
        
        try:
            # Check existing features
            train_dir = Path('datasets/features/train')
            val_dir = Path('datasets/features/val')
            
            train_count = len(list(train_dir.glob('*.npz'))) if train_dir.exists() else 0
            val_count = len(list(val_dir.glob('*.npz'))) if val_dir.exists() else 0
            
            self.logger.info(f"Existing features - Train: {train_count}, Val: {val_count}")
            
            # Skip logic
            if skip_existing:
                if sample_size:
                    required_train = int(sample_size * 0.8)
                    required_val = int(sample_size * 0.2)
                    if train_count >= required_train and val_count >= required_val:
                        self.logger.info(f"Sufficient features exist (need train={required_train}, val={required_val})")
                        return True
                else:
                    if train_count > 20000 and val_count > 5000:
                        self.logger.info("Features already extracted for full dataset")
                        return True
            
            # Check audio directory
            audio_dir = Path('datasets/clips/stuttering-clips/clips')
            if not audio_dir.exists():
                self.logger.error(f"Audio directory not found: {audio_dir}")
                return False
            
            audio_count = len(list(audio_dir.glob('*.wav')))
            self.logger.info(f"Audio files available: {audio_count}")
            
            # Initialize extraction manager
            manager = None
            if FeatureExtractionPipeline is not None:
                try:
                    manager = FeatureExtractionPipeline()
                    self.logger.debug("Using FeatureExtractionPipeline")
                except Exception as e:
                    self.logger.debug(f"FeatureExtractionPipeline failed: {e}")
            
            if manager is None and FeatureExtractionManager is not None:
                manager = FeatureExtractionManager()
                self.logger.debug("Using FeatureExtractionManager")
            
            if manager is None:
                self.logger.error("No feature extraction implementation available")
                return False
            
            # Extraction parameters
            if sample_size:
                self.logger.info(f"Extracting {sample_size} sample files")
                self.logger.info(f"Estimated time: ~{sample_size // 10} minutes")
            else:
                self.logger.info("Extracting features from ALL files")
                self.logger.info("Estimated time: 6-8 hours on GPU")
            
            # Extract
            self.logger.info("Starting extraction...")
            start_time = datetime.now()
            
            if hasattr(manager, 'extract_all'):
                results = manager.extract_all(sample_size=sample_size, skip_existing=skip_existing)
            elif hasattr(manager, 'extract_all_features'):
                results = manager.extract_all_features(sample_size=sample_size, skip_existing=skip_existing)
            else:
                self.logger.error("Extraction manager missing extract method")
                return False
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            if results['total'] == 0:
                self.logger.error("No features were extracted")
                return False
            
            # Log results
            self.logger.info("=" * 80)
            self.logger.info("FEATURE EXTRACTION COMPLETE")
            self.logger.info("=" * 80)
            self.logger.info(f"Total extracted: {results['total']}")
            self.logger.info(f"Train set: {results['train']} files")
            self.logger.info(f"Val set: {results['val']} files")
            self.logger.info(f"Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
            if results['total'] > 0:
                self.logger.info(f"Average time per file: {elapsed/results['total']:.2f}s")
            self.logger.info("=" * 80)
            
            # Verify
            if hasattr(manager, 'verify_extraction'):
                try:
                    manager.verify_extraction()
                except Exception as e:
                    self.logger.warning(f"Verification failed: {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}", exc_info=True)
            return False
    
    def train_model(self, epochs: int = 60, batch_size: int = 32, early_stopping_patience: int = 50) -> bool:
        """
        Phase 2: Train detection model.
        
        Args:
            epochs: Maximum training epochs
            batch_size: Batch size
            early_stopping_patience: Epochs to wait before early stopping
        
        Returns:
            Success status
        """
        self.logger.info("=" * 80)
        self.logger.info("PHASE 2: MODEL TRAINING")
        self.logger.info("=" * 80)
        
        try:
            from torch.utils.data import DataLoader
            
            # Training config
            self.logger.info(f"Max epochs: {epochs}")
            self.logger.info(f"Batch size: {batch_size}")
            self.logger.info(f"Early stopping patience: {early_stopping_patience}")
            self.logger.info(f"Loss: Focal Loss (gamma=2.0)")
            self.logger.info(f"Device: {self.device}")
            
            # Check datasets
            if not Path('datasets/features/train').exists():
                self.logger.error("Training dataset not found - run feature extraction first")
                return False
            
            # Load datasets
            self.logger.info("Loading datasets...")
            train_dataset = AudioDataset('datasets/features', split='train', augment=True)
            val_dataset = AudioDataset('datasets/features', split='val', augment=False)
            
            self.logger.info(f"Train samples: {len(train_dataset)}")
            self.logger.info(f"Val samples: {len(val_dataset)}")
            
            if len(train_dataset) == 0 or len(val_dataset) == 0:
                self.logger.error("Empty datasets")
                return False
            
            # Check class distribution
            self.logger.info("Analyzing class distribution...")
            train_labels = []
            for i in range(min(1000, len(train_dataset))):  # Sample first 1000
                _, label = train_dataset[i]
                train_labels.append(label)
            train_labels = np.array(train_labels)
            
            self.logger.info("Training set class distribution (sample):")
            for i, class_name in enumerate(PerClassMetrics.STUTTER_CLASSES):
                count = np.sum(train_labels[:, i])
                pct = count / len(train_labels) * 100
                self.logger.info(f"  {class_name:<20}: {count:>5} samples ({pct:>5.1f}%)")
            
            # Create dataloaders
            num_workers = min(4, os.cpu_count() or 1)
            self.logger.debug(f"DataLoader workers: {num_workers}")
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=num_workers, 
                pin_memory=True if self.device.type == 'cuda' else False,
                collate_fn=collate_variable_length
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=num_workers, 
                pin_memory=True if self.device.type == 'cuda' else False,
                collate_fn=collate_variable_length
            )
            
            self.logger.info(f"Train batches: {len(train_loader)}")
            self.logger.info(f"Val batches: {len(val_loader)}")
            
            # Initialize model
            self.logger.info("Initializing model...")
            model = ImprovedStutteringCNN(n_channels=TOTAL_CHANNELS, n_classes=5, dropout=0.4)
            
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.logger.info(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
            
            log_gpu_memory(self.logger, "Before training")
            
            # Train
            self.logger.info("Starting training...")
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=self.device,
                model_name='improved_90plus',
                early_stop_patience=early_stopping_patience
            )
            
            metrics = trainer.train(num_epochs=epochs)
            
            log_gpu_memory(self.logger, "After training")
            
            # Get best model path
            self.best_model_path = trainer.checkpoint_dir / 'improved_90plus_best.pth'
            
            if not self.best_model_path.exists():
                self.logger.error("Best model file not found after training")
                return False
            
            # Log final metrics
            self.logger.info("=" * 80)
            self.logger.info("TRAINING COMPLETE")
            self.logger.info("=" * 80)
            self.logger.info(f"Best model: {self.best_model_path}")
            self.logger.info(f"Best F1 score: {trainer.best_f1:.4f}")
            self.logger.info(f"Final epoch: {trainer.current_epoch}")
            self.logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}", exc_info=True)
            return False
    
    def load_best_model(self, model_path: Optional[Path] = None) -> bool:
        """
        Load trained model.
        
        Args:
            model_path: Path to model file (None = auto-detect)
        
        Returns:
            Success status
        """
        if model_path is None:
            # Search for model candidates
            candidate_patterns = [
                self.best_model_path,
                self.checkpoint_dir / 'improved_90plus_best.pth',
                Path('Models/checkpoints/improved_90plus_best.pth'),
                Path('Models/checkpoints/improved_90plus_BEST_OVERALL.pth'),
            ]
            candidates = [p for p in candidate_patterns if p and p.exists()]
            
            if not candidates:
                self.logger.error("No trained model found")
                self.logger.info("Searched locations:")
                for path in candidate_patterns:
                    self.logger.info(f"  - {path}")
                return False
            
            # Pick newest file
            model_path = max(candidates, key=lambda p: p.stat().st_mtime)
            self.logger.info(f"Auto-detected model: {model_path}")
        
        try:
            self.logger.info(f"Loading model from: {model_path}")
            
            # Initialize model
            self.model = ImprovedStutteringCNN(
                n_channels=TOTAL_CHANNELS, 
                n_classes=5, 
                dropout=0.4
            ).to(self.device)
            
            # Load weights
            state_dict = torch.load(str(model_path), map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            self.best_model_path = model_path
            
            # Log model info
            total_params = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"Model loaded successfully ({total_params:,} parameters)")
            
            log_gpu_memory(self.logger, "After model load")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}", exc_info=True)
            return False
    
    def evaluate_model(self, data_loader, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Evaluate model on a dataset with per-class metrics.
        
        Args:
            data_loader: DataLoader for evaluation
            threshold: Classification threshold
        
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            self.logger.error("No model loaded")
            return {}
        
        self.logger.info("Running evaluation...")
        self.model.eval()
        
        metrics = PerClassMetrics()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, (features, targets) in enumerate(data_loader):
                features = features.to(self.device)
                targets = targets.numpy()
                
                # Forward pass
                logits = self.model(features)
                predictions = torch.sigmoid(logits).cpu().numpy()
                
                # Update metrics
                metrics.update(predictions, targets, threshold)
                
                all_predictions.append(predictions)
                all_targets.append(targets)
                
                if (batch_idx + 1) % 10 == 0:
                    self.logger.debug(f"Evaluated {batch_idx + 1}/{len(data_loader)} batches")
        
        # Compute final metrics
        results = metrics.compute()
        
        # Log metrics
        metrics.log_metrics(self.logger, "Evaluation ")
        
        # Save detailed metrics
        metrics_file = self.output_dir / 'metrics' / f'evaluation_{datetime.now():%Y%m%d_%H%M%S}.json'
        with open(metrics_file, 'w') as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Detailed metrics saved: {metrics_file}")
        
        return results
    
    def detect_and_repair_audio(self, audio_path: str, output_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Phase 3: Detect and repair stuttering in audio.
        
        Args:
            audio_path: Path to audio file
            output_name: Output filename prefix
        
        Returns:
            Detection/repair results
        """
        self.logger.info("=" * 80)
        self.logger.info("PHASE 3: DETECTION + REPAIR")
        self.logger.info("=" * 80)
        
        if not Path(audio_path).exists():
            self.logger.error(f"Audio file not found: {audio_path}")
            return None
        
        self.logger.info(f"Processing: {audio_path}")
        
        if not REPAIR_AVAILABLE:
            self.logger.warning("Repair module not available - detection only")
            return self._detect_only(audio_path, output_name)
        
        try:
            output_name = output_name or Path(audio_path).stem
            output_audio = self.output_dir / 'repaired_audio' / f"{output_name}_repaired.wav"
            output_json = self.output_dir / 'analysis' / f"{output_name}_analysis.json"
            
            if self.model is None or self.best_model_path is None:
                self.logger.error("No model loaded")
                return None
            
            # Initialize repair
            self.logger.info("Initializing repair module...")
            repair = AdvancedStutterRepair(model_path=str(self.best_model_path))
            
            # Repair audio
            self.logger.info("Repairing audio...")
            start_time = datetime.now()
            
            repaired_audio, stutter_regions = repair.repair_audio(
                audio_path, 
                output_path=str(output_audio)
            )
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            if repaired_audio is None:
                self.logger.error("Repair failed")
                return None
            
            # Generate analysis
            self.logger.info("Generating analysis...")
            analysis = extract_stutter_analysis(audio_path, str(output_json))
            
            # Log results
            self.logger.info("=" * 80)
            self.logger.info("DETECTION + REPAIR COMPLETE")
            self.logger.info("=" * 80)
            self.logger.info(f"Stutters detected: {len(stutter_regions)}")
            self.logger.info(f"Total stutter time: {analysis.get('total_stutter_time', 0):.2f}s")
            self.logger.info(f"Stuttering percentage: {analysis.get('stuttering_percentage', 0):.1f}%")
            self.logger.info(f"Processing time: {elapsed:.2f}s")
            self.logger.info(f"Repaired audio: {output_audio}")
            self.logger.info(f"Analysis: {output_json}")
            self.logger.info("=" * 80)
            
            return {
                'repaired_audio': str(output_audio),
                'analysis': analysis,
                'regions': stutter_regions,
                'processing_time': elapsed
            }
            
        except Exception as e:
            self.logger.error(f"Detection/repair failed: {e}", exc_info=True)
            return None
    
    def _detect_only(self, audio_path: str, output_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Detection-only mode when repair module is unavailable."""
        try:
            output_name = output_name or Path(audio_path).stem
            output_json = self.output_dir / 'analysis' / f"{output_name}_detection.json"
            
            self.logger.info("Running detection-only mode...")
            
            # Extract features
            preprocessor = EnhancedAudioPreprocessor()
            features = preprocessor.extract_features(audio_path)
            
            if features is None:
                self.logger.error("Feature extraction failed")
                return None
            
            self.logger.debug(f"Features shape: {features.shape}")
            
            # Run detection
            with torch.no_grad():
                features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
                logits = self.model(features_tensor)
                probs = torch.sigmoid(logits).cpu().numpy()[0]
            
            # Create detection report
            stutter_classes = PerClassMetrics.STUTTER_CLASSES
            detection_results = {
                'file': str(audio_path),
                'timestamp': datetime.now().isoformat(),
                'predictions': {
                    stutter_classes[i]: float(probs[i])
                    for i in range(5)
                },
                'detected_stutters': [
                    stutter_classes[i] for i in range(5) if probs[i] > 0.5
                ]
            }
            
            # Save results
            with open(output_json, 'w') as f:
                json.dump(detection_results, f, indent=2)
            
            # Log results
            self.logger.info("=" * 80)
            self.logger.info("DETECTION COMPLETE")
            self.logger.info("=" * 80)
            self.logger.info("Per-class probabilities:")
            for class_name, prob in detection_results['predictions'].items():
                status = "✓ DETECTED" if prob > 0.5 else ""
                self.logger.info(f"  {class_name:<20}: {prob:.4f} {status}")
            self.logger.info(f"Results saved: {output_json}")
            self.logger.info("=" * 80)
            
            return detection_results
            
        except Exception as e:
            self.logger.error(f"Detection failed: {e}", exc_info=True)
            return None
    
    def generate_report(self, results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate final pipeline report."""
        self.logger.info("=" * 80)
        self.logger.info("GENERATING PIPELINE REPORT")
        self.logger.info("=" * 80)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'repair_available': REPAIR_AVAILABLE,
            'model_path': str(self.best_model_path) if self.best_model_path else None,
            'pipeline_status': 'complete',
            'results': results or {}
        }
        
        report_path = self.output_dir / 'pipeline_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Report saved: {report_path}")
        return report
    
    def run_full_pipeline(self, 
                         extract_features: bool = True, 
                         train_model: bool = True, 
                         test_audio: Optional[str] = None,
                         sample_size: Optional[int] = None,
                         batch_size: int = 32,
                         epochs: int = 60) -> Optional[Dict[str, Any]]:
        """Run complete pipeline."""
        
        self.logger.info("=" * 80)
        self.logger.info("COMPLETE STUTTER DETECTION & REPAIR PIPELINE")
        self.logger.info("=" * 80)
        
        results = {}
        pipeline_start = datetime.now()
        
        # Phase 1: Features
        if extract_features:
            if not self.extract_features(sample_size=sample_size):
                self.logger.error("Pipeline failed at feature extraction")
                return None
        
        # Phase 2: Training
        if train_model:
            if not self.train_model(epochs=epochs, batch_size=batch_size):
                self.logger.error("Pipeline failed at training")
                return None
        
        # Phase 3: Load model
        if not self.load_best_model():
            self.logger.error("Pipeline failed at model loading")
            return None
        
        # Phase 4: Test on sample audio
        if test_audio:
            test_results = self.detect_and_repair_audio(test_audio, "test")
            if test_results:
                results['test_audio'] = test_results
        
        # Generate report
        self.generate_report(results)
        
        pipeline_elapsed = (datetime.now() - pipeline_start).total_seconds()
        
        self.logger.info("=" * 80)
        self.logger.info("PIPELINE COMPLETE")
        self.logger.info("=" * 80)
        self.logger.info(f"Total time: {pipeline_elapsed:.1f}s ({pipeline_elapsed/60:.1f} min)")
        self.logger.info("\nOutput locations:")
        self.logger.info(f"  Repaired audio: {self.output_dir / 'repaired_audio'}")
        self.logger.info(f"  Analysis: {self.output_dir / 'analysis'}")
        self.logger.info(f"  Metrics: {self.output_dir / 'metrics'}")
        self.logger.info(f"  Report: {self.output_dir / 'pipeline_report.json'}")
        self.logger.info("=" * 80)
        
        return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Complete Stutter Detection & Repair Pipeline (Debug Version)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with debug logging
  python COMPLETE_PIPELINE_DEBUG.py --log-level DEBUG
  
  # Quick test with sample data
  python COMPLETE_PIPELINE_DEBUG.py --sample-size 100 --epochs 10 --verbose
  
  # Extract features only
  python COMPLETE_PIPELINE_DEBUG.py --features-only --sample-size 500
  
  # Train only (features already extracted)
  python COMPLETE_PIPELINE_DEBUG.py --skip-features --verbose
  
  # Detect/repair only
  python COMPLETE_PIPELINE_DEBUG.py --skip-features --skip-training --test-file audio.wav
        """
    )
    
    # Mode selection
    parser.add_argument('--features-only', action='store_true', 
                       help='Extract features only')
    parser.add_argument('--train-only', action='store_true', 
                       help='Train model only')
    parser.add_argument('--repair-only', action='store_true', 
                       help='Repair audio only (requires --test-file)')
    
    # Pipeline control
    parser.add_argument('--skip-features', action='store_true', 
                       help='Skip feature extraction')
    parser.add_argument('--skip-training', action='store_true', 
                       help='Skip training (use existing model)')
    
    # Parameters
    parser.add_argument('--test-file', type=str, 
                       help='Audio file to process')
    parser.add_argument('--sample-size', type=int, 
                       help='Number of files to extract (for testing)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=60,
                       help='Training epochs (default: 60)')
    
    # Logging
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output (equivalent to --log-level DEBUG)')
    
    args = parser.parse_args()
    
    # Adjust log level
    log_level = 'DEBUG' if args.verbose else args.log_level
    
    # Initialize pipeline
    pipeline = CompletePipeline(log_level=log_level)
    
    # Run appropriate mode
    try:
        if args.features_only:
            pipeline.logger.info("MODE: FEATURE EXTRACTION ONLY")
            pipeline.extract_features(sample_size=args.sample_size)
            
        elif args.train_only:
            pipeline.logger.info("MODE: TRAINING ONLY")
            if not args.skip_features:
                pipeline.extract_features(sample_size=args.sample_size)
            pipeline.train_model(epochs=args.epochs, batch_size=args.batch_size)
            
        elif args.repair_only:
            pipeline.logger.info("MODE: DETECTION/REPAIR ONLY")
            if not args.test_file:
                pipeline.logger.error("--repair-only requires --test-file")
                sys.exit(1)
            if pipeline.load_best_model():
                pipeline.detect_and_repair_audio(args.test_file)
            else:
                pipeline.logger.error("Could not load model")
                sys.exit(1)
        else:
            pipeline.logger.info("MODE: FULL PIPELINE")
            pipeline.run_full_pipeline(
                extract_features=not args.skip_features,
                train_model=not args.skip_training,
                test_audio=args.test_file,
                sample_size=args.sample_size,
                batch_size=args.batch_size,
                epochs=args.epochs
            )
    
    except KeyboardInterrupt:
        pipeline.logger.warning("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        pipeline.logger.critical(f"Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()