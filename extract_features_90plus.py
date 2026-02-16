"""
PRODUCTION FEATURE EXTRACTION SCRIPT - DEBUG VERSION
Extracts 123-channel features with comprehensive logging and debugging

Enhanced Features:
- Structured logging with multiple levels
- Per-file extraction timing
- Memory usage tracking
- Feature quality metrics (SNR, silence detection)
- Detailed error categorization
- Progress checkpointing
- Audio validation diagnostics

Dataset Structure:
  datasets/clips/               - Raw audio files (.flac, .wav)
  datasets/*.csv                - Label files (fluencybank_labels.csv, SEP-28k_labels.csv, etc.)
  datasets/features/train/      - Training NPZ files (80%)
  datasets/features/val/        - Validation NPZ files (20%)
  datasets/corrupted_audio/     - Moved corrupted/empty files

Usage:
    python extract_features_debug.py --log-level DEBUG
    python extract_features_debug.py --sample-size 100 --verbose
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import warnings
import hashlib
import traceback
import logging
import json
import time
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any
from collections import defaultdict
warnings.filterwarnings('ignore')

# Add project directory to path (avoid creating Models/Models when running from Models/)
sys.path.insert(0, str(Path(__file__).parent))

try:
    from enhanced_audio_preprocessor import EnhancedAudioPreprocessor
    import soundfile as sf
except ImportError as e:
    print(f"ERROR: Required module not found: {e}")
    print("Ensure enhanced_audio_preprocessor.py is in current directory or Models/")
    sys.exit(1)


def setup_logging(output_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """
    Setup comprehensive logging system for feature extraction.
    
    Args:
        output_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        Configured logger
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger('FeatureExtraction')
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    
    # Main log file
    log_file = output_dir / f'extraction_{datetime.now():%Y%m%d_%H%M%S}.log'
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)-25s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    
    # Error log file
    error_file = output_dir / f'extraction_errors_{datetime.now():%Y%m%d_%H%M%S}.log'
    error_handler = logging.FileHandler(error_file, encoding='utf-8')
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(file_format)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    
    logger.info(f"Logging initialized - Level: {log_level}")
    logger.info(f"Main log: {log_file}")
    logger.info(f"Error log: {error_file}")
    
    return logger


class FeatureQualityMetrics:
    """Track and analyze feature quality metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.snr_values = []
        self.silence_ratios = []
        self.feature_norms = []
        self.time_lengths = []
        self.issues = defaultdict(int)
    
    def analyze_features(self, features: np.ndarray, audio_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Analyze feature quality.
        
        Args:
            features: (123, T) feature array
            audio_data: Optional raw audio for SNR computation
        
        Returns:
            Quality metrics dict
        """
        metrics = {}
        
        # Feature statistics
        metrics['shape'] = features.shape
        metrics['mean'] = float(np.mean(features))
        metrics['std'] = float(np.std(features))
        metrics['min'] = float(np.min(features))
        metrics['max'] = float(np.max(features))
        metrics['norm'] = float(np.linalg.norm(features))
        
        # Check for issues
        if np.any(np.isnan(features)):
            metrics['has_nan'] = True
            self.issues['nan_values'] += 1
        if np.any(np.isinf(features)):
            metrics['has_inf'] = True
            self.issues['inf_values'] += 1
        
        # Silence detection (channels with very low variance)
        channel_vars = np.var(features, axis=1)
        silent_channels = np.sum(channel_vars < 1e-8)
        metrics['silent_channels'] = int(silent_channels)
        if silent_channels > 60:  # More than half channels silent
            self.issues['mostly_silent'] += 1
        
        # SNR estimation (if audio available)
        if audio_data is not None:
            try:
                signal_power = np.mean(audio_data ** 2)
                if signal_power > 0:
                    metrics['snr_db'] = 10 * np.log10(signal_power / (1e-10 + np.var(audio_data - np.mean(audio_data))))
                    self.snr_values.append(metrics['snr_db'])
            except:
                pass
        
        # Track metrics
        self.feature_norms.append(metrics['norm'])
        self.time_lengths.append(features.shape[1])
        self.silence_ratios.append(silent_channels / 123.0)
        
        return metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.feature_norms:
            return {}
        
        return {
            'total_samples': len(self.feature_norms),
            'feature_norm': {
                'mean': float(np.mean(self.feature_norms)),
                'std': float(np.std(self.feature_norms)),
                'min': float(np.min(self.feature_norms)),
                'max': float(np.max(self.feature_norms))
            },
            'time_length': {
                'mean': float(np.mean(self.time_lengths)),
                'std': float(np.std(self.time_lengths)),
                'min': int(np.min(self.time_lengths)),
                'max': int(np.max(self.time_lengths))
            },
            'silence_ratio': {
                'mean': float(np.mean(self.silence_ratios)),
                'max': float(np.max(self.silence_ratios))
            },
            'snr_db': {
                'mean': float(np.mean(self.snr_values)),
                'std': float(np.std(self.snr_values))
            } if self.snr_values else None,
            'issues': dict(self.issues)
        }


class ExtractionCheckpoint:
    """Handle extraction checkpointing for resuming interrupted runs."""
    
    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.data = self._load()
    
    def _load(self) -> Dict[str, Any]:
        """Load checkpoint data."""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, 'r') as f:
                    raw = json.load(f)
                    # Ensure internal representation uses sets for fast membership/add
                    processed = set(raw.get('processed', [])) if isinstance(raw, dict) else set()
                    failed = set(raw.get('failed', [])) if isinstance(raw, dict) else set()
                    return {'processed': processed, 'failed': failed}
            except:
                return {'processed': set(), 'failed': set()}
        return {'processed': set(), 'failed': set()}
    
    def save(self):
        """Save checkpoint data."""
        try:
            # Convert sets to lists for JSON serialization
            save_data = {
                'processed': list(self.data['processed']),
                'failed': list(self.data['failed']),
                'timestamp': datetime.now().isoformat()
            }
            with open(self.checkpoint_path, 'w') as f:
                json.dump(save_data, f)
        except Exception as e:
            logging.warning(f"Failed to save checkpoint: {e}")
    
    def mark_processed(self, filename: str):
        """Mark file as successfully processed."""
        self.data['processed'].add(filename)
    
    def mark_failed(self, filename: str):
        """Mark file as failed."""
        self.data['failed'].add(filename)
    
    def is_processed(self, filename: str) -> bool:
        """Check if file already processed."""
        return filename in self.data['processed']


class FeatureExtractionPipeline:
    """Production pipeline for extracting 123-channel audio features with debugging."""
    
    def __init__(self, 
                 clips_dir: str = 'datasets/clips', 
                 output_dir: str = 'datasets/features',
                 label_dir: str = 'datasets',
                 sr: int = 16000,
                 log_level: str = 'INFO',
                 silence_warn_ratio: float = 0.80):
        """
        Initialize extraction pipeline.
        
        Args:
            clips_dir: Directory containing raw audio files
            output_dir: Directory for extracted features (train/val subdirs)
            label_dir: Directory containing label CSV files
            sr: Target sample rate (must be 16000 for current preprocessor)
            log_level: Logging level
        """
        self.clips_dir = Path(clips_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.label_dir = Path(label_dir).resolve()
        self.sr = sr
        
        # Create directory structure
        self.train_dir = self.output_dir / 'train'
        self.val_dir = self.output_dir / 'val'
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)
        
        # Corrupted audio staging area
        self.corrupt_dir = Path('datasets/corrupted_audio').resolve()
        self.corrupt_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logging(self.output_dir, log_level)
        
        # Checkpointing
        self.checkpoint = ExtractionCheckpoint(self.output_dir / 'extraction_checkpoint.json')
        
        # Quality metrics
        self.quality_metrics = FeatureQualityMetrics()
        
        # Timing statistics
        self.timing_stats = {
            'validation': [],
            'extraction': [],
            'saving': []
        }
        
        # Silence threshold for preprocessor warnings
        self.silence_warn_ratio = float(silence_warn_ratio)

        # Initialize preprocessor
        self.logger.info("Initializing EnhancedAudioPreprocessor...")
        try:
            self.preprocessor = EnhancedAudioPreprocessor(sr=sr, silence_warn_ratio=self.silence_warn_ratio)
            self.logger.info("Preprocessor initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize preprocessor: {e}", exc_info=True)
            raise
        
        # Label mappings
        self.label_map = {}
        self.load_labels()
        
        self.logger.info("Feature extraction pipeline initialized")
    
    def load_labels(self):
        """Load all label CSV files and build filename -> label mapping."""
        self.logger.info("=" * 80)
        self.logger.info("LOADING LABEL FILES")
        self.logger.info("=" * 80)
        
        # Find all potential label CSV files
        label_files = []
        
        # Priority order
        priority_files = ['labels.csv', 'fluencybank_labels.csv', 'SEP-28k_labels.csv']
        for fname in priority_files:
            fpath = self.label_dir / fname
            if fpath.exists():
                label_files.append(fpath)
                self.logger.debug(f"Found priority label file: {fname}")
        
        # Auto-detect other label files
        for pattern in ['*labels*.csv', '*_labels.csv']:
            for fpath in self.label_dir.glob(pattern):
                if fpath not in label_files:
                    label_files.append(fpath)
                    self.logger.debug(f"Found label file: {fpath.name}")
        
        if not label_files:
            self.logger.warning("No label CSV files found - using dummy zero labels")
            self.logger.warning(f"Searched in: {self.label_dir}")
            return
        
        self.logger.info(f"Found {len(label_files)} label files")
        
        # Parse each label file
        total_loaded = 0
        for label_file in label_files:
            count = self._parse_label_file(label_file)
            if count > 0:
                total_loaded += count
                self.logger.info(f"✓ {label_file.name}: {count:,} labels loaded")
            else:
                self.logger.warning(f"✗ {label_file.name}: No labels loaded")
        
        self.logger.info("=" * 80)
        self.logger.info(f"TOTAL LABELS LOADED: {total_loaded:,}")
        self.logger.info("=" * 80)
        
        # Analyze label distribution
        if total_loaded > 0:
            self._analyze_label_distribution()
    
    def _parse_label_file(self, csv_path: Path) -> int:
        """
        Parse a single label CSV file.
        
        Returns:
            Number of labels successfully loaded
        """
        try:
            self.logger.debug(f"Parsing {csv_path.name}...")
            df = pd.read_csv(csv_path)
            cols = [c.strip() for c in df.columns]
            count = 0
            
            self.logger.debug(f"Columns: {cols}")
            
            # Strategy 1: FluencyBank/SEP-28k format (Show, EpId, ClipId)
            if {'Show', 'EpId', 'ClipId'}.issubset(set(cols)):
                self.logger.debug("Using FluencyBank/SEP-28k format")
                for idx, row in df.iterrows():
                    show = str(row.get('Show', '')).strip()
                    epid = str(row.get('EpId', '')).strip()
                    clipid = str(row.get('ClipId', '')).strip()

                    if show and epid and clipid:
                        labels = self._make_label_vector(row)

                        # Primary key (as present in CSV)
                        primary = f"{show}_{epid}_{clipid}"
                        # Also register common zero-padded variants to match audio filename conventions
                        keys = {primary}
                        # If EpId looks numeric, add padded variants (2 and 3 digits)
                        try:
                            if epid.isdigit():
                                keys.add(f"{show}_{epid.zfill(2)}_{clipid}")
                                keys.add(f"{show}_{epid.zfill(3)}_{clipid}")
                            else:
                                # also add stripped-leading-zero variant
                                stripped = epid.lstrip('0') or epid
                                keys.add(f"{show}_{stripped}_{clipid}")
                        except Exception:
                            pass

                        for filename in keys:
                            # Avoid overwriting an existing mapping produced earlier
                            if filename not in self.label_map:
                                self.label_map[filename] = labels
                                count += 1

                        if count <= 3:  # Log first few examples
                            example_keys = list(keys)[:3]
                            self.logger.debug(f"  Example(s): {example_keys} -> {labels}")
                
                return count
            
            # Strategy 2: Standard format with 'filename' column
            if 'filename' in cols:
                self.logger.debug("Using standard filename format")
                for idx, row in df.iterrows():
                    filename = str(row.get('filename', '')).strip()
                    if filename:
                        filename = Path(filename).stem
                        labels = self._make_label_vector(row)
                        self.label_map[filename] = labels
                        count += 1
                        
                        if count <= 3:
                            self.logger.debug(f"  Example: {filename} -> {labels}")
                
                return count
            
            # Strategy 3: First column as filename (fallback)
            first_col = cols[0]
            self.logger.debug(f"Using first column '{first_col}' as filename")
            for idx, row in df.iterrows():
                filename = str(row.get(first_col, '')).strip()
                if filename:
                    filename = Path(filename).stem
                    labels = self._make_label_vector(row)
                    self.label_map[filename] = labels
                    count += 1
                    
                    if count <= 3:
                        self.logger.debug(f"  Example: {filename} -> {labels}")
            
            return count
            
        except Exception as e:
            self.logger.error(f"Error parsing {csv_path.name}: {e}", exc_info=True)
            return 0
    
    def _make_label_vector(self, row: pd.Series) -> np.ndarray:
        """
        Create label vector from DataFrame row.
        
        Returns:
            (5,) float32 array [Prolongation, Block, SoundRep, WordRep, Interjection]
        """
        return np.array([
            float(row.get('Prolongation', 0)),
            float(row.get('Block', 0)),
            float(row.get('SoundRep', 0)),
            float(row.get('WordRep', 0)),
            float(row.get('Interjection', 0))
        ], dtype=np.float32)
    
    def _analyze_label_distribution(self):
        """Analyze and log label distribution."""
        if not self.label_map:
            return
        
        label_names = ['Prolongation', 'Block', 'SoundRep', 'WordRep', 'Interjection']
        all_labels = np.array(list(self.label_map.values()))
        
        self.logger.info("\nLabel Distribution:")
        self.logger.info("-" * 60)
        
        for i, name in enumerate(label_names):
            count = np.sum(all_labels[:, i] > 0)
            pct = 100 * count / len(all_labels)
            self.logger.info(f"  {name:<15}: {count:>6,} ({pct:>5.2f}%)")
        
        files_with_any = np.sum(np.any(all_labels > 0, axis=1))
        pct_any = 100 * files_with_any / len(all_labels)
        self.logger.info(f"  {'Any stutter':<15}: {files_with_any:>6,} ({pct_any:>5.2f}%)")
        self.logger.info("-" * 60)
    
    def get_train_val_split(self, filename: str) -> str:
        """
        Deterministic train/val split using MD5 hash (80/20 split).
        
        Args:
            filename: File stem (without extension)
        
        Returns:
            'train' or 'val'
        """
        h = hashlib.md5(filename.encode('utf-8')).hexdigest()
        hash_val = int(h, 16) % 100
        return 'val' if hash_val < 20 else 'train'
    
    def validate_audio_file(self, audio_path: Path) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """
        Check if audio file is valid and readable.
        
        Returns:
            (is_valid, error_reason, audio_info)
        """
        start_time = time.time()
        
        try:
            # Check file size
            file_size = audio_path.stat().st_size
            if file_size == 0:
                return False, "empty_file", None
            
            # Read audio metadata
            info = sf.info(str(audio_path))
            
            audio_info = {
                'samplerate': info.samplerate,
                'frames': info.frames,
                'duration': info.frames / info.samplerate,
                'channels': info.channels,
                'format': info.format,
                'subtype': info.subtype
            }
            
            # Check frames
            if info.frames == 0:
                return False, "zero_frames", audio_info
            
            # Check duration (minimum 100ms)
            if audio_info['duration'] < 0.1:
                return False, f"too_short_{audio_info['duration']:.3f}s", audio_info
            
            # Check sample rate
            if info.samplerate != self.sr:
                self.logger.debug(
                    f"{audio_path.name}: Sample rate {info.samplerate}Hz "
                    f"(expected {self.sr}Hz) - will be resampled"
                )
            
            # Check for mono/stereo
            if info.channels > 2:
                self.logger.warning(f"{audio_path.name}: {info.channels} channels (will mix to mono)")
            
            elapsed = time.time() - start_time
            self.timing_stats['validation'].append(elapsed)
            
            return True, None, audio_info
            
        except Exception as e:
            error_msg = f"read_error: {str(e)[:100]}"
            return False, error_msg, None
    
    def move_to_corrupted(self, audio_path: Path, reason: str) -> bool:
        """
        Move corrupted audio file to staging directory.
        
        Returns:
            Success status
        """
        try:
            target = self.corrupt_dir / audio_path.name
            
            # Handle duplicate names
            counter = 1
            while target.exists():
                target = self.corrupt_dir / f"{audio_path.stem}_{counter}{audio_path.suffix}"
                counter += 1
            
            audio_path.rename(target)
            self.logger.info(f"Moved corrupted file: {audio_path.name} -> {target.name} (reason: {reason})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to move {audio_path.name}: {e}")
            return False
    
    def extract_single_file(self, audio_path: Path) -> Dict[str, Any]:
        """
        Extract features from a single audio file.
        
        Returns:
            Result dict with success status, features, labels, and metadata
        """
        filename = audio_path.stem
        
        # Validate audio
        start_validation = time.time()
        is_valid, error_reason, audio_info = self.validate_audio_file(audio_path)
        
        if not is_valid:
            # Move corrupted files
            if error_reason and error_reason.split('_')[0] in ('empty', 'zero', 'too'):
                self.move_to_corrupted(audio_path, error_reason)
            
            self.logger.debug(f"✗ {filename}: {error_reason}")
            self.checkpoint.mark_failed(filename)
            
            return {
                'success': False,
                'split': None,
                'features': None,
                'labels': None,
                'error': error_reason,
                'audio_info': audio_info
            }
        
        # Extract features
        try:
            start_extraction = time.time()
            features = self.preprocessor.extract_features(str(audio_path))
            extraction_time = time.time() - start_extraction
            self.timing_stats['extraction'].append(extraction_time)
            
            if features is None:
                error_msg = 'extraction_failed: preprocessor returned None'
                self.logger.warning(f"✗ {filename}: {error_msg}")
                self.checkpoint.mark_failed(filename)
                return {
                    'success': False,
                    'split': None,
                    'features': None,
                    'labels': None,
                    'error': error_msg,
                    'audio_info': audio_info
                }
            
            # Validate feature shape
            if features.ndim != 2 or features.shape[0] != 123:
                error_msg = f"invalid_shape: {features.shape} (expected (123, T))"
                self.logger.error(f"✗ {filename}: {error_msg}")
                self.checkpoint.mark_failed(filename)
                return {
                    'success': False,
                    'split': None,
                    'features': None,
                    'labels': None,
                    'error': error_msg,
                    'audio_info': audio_info
                }
            
            # Analyze feature quality
            quality = self.quality_metrics.analyze_features(features)
            
            # Log quality issues
            if quality.get('has_nan'):
                self.logger.warning(f"⚠ {filename}: Contains NaN values")
            if quality.get('has_inf'):
                self.logger.warning(f"⚠ {filename}: Contains Inf values")
            if quality.get('silent_channels', 0) > 60:
                self.logger.warning(
                    f"⚠ {filename}: {quality['silent_channels']}/123 channels silent"
                )
            
            # Get labels
            if filename in self.label_map:
                labels = self.label_map[filename]
                has_labels = True
            else:
                labels = np.zeros(5, dtype=np.float32)
                has_labels = False
            
            # Determine split
            split = self.get_train_val_split(filename)
            
            # Log detailed info for first few files
            if len(self.quality_metrics.feature_norms) <= 5:
                self.logger.debug(f"✓ {filename}:")
                self.logger.debug(f"    Shape: {features.shape}")
                self.logger.debug(f"    Split: {split}")
                self.logger.debug(f"    Labels: {labels} (mapped={has_labels})")
                self.logger.debug(f"    Quality: norm={quality['norm']:.2f}, "
                                f"silent={quality['silent_channels']}/123")
                self.logger.debug(f"    Timing: validation={start_extraction-start_validation:.3f}s, "
                                f"extraction={extraction_time:.3f}s")
            
            self.checkpoint.mark_processed(filename)
            
            return {
                'success': True,
                'split': split,
                'features': features,
                'labels': labels,
                'error': None,
                'audio_info': audio_info,
                'quality': quality,
                'timing': {
                    'validation': start_extraction - start_validation,
                    'extraction': extraction_time
                }
            }
            
        except Exception as e:
            error_msg = f"exception: {str(e)[:200]}"
            self.logger.error(f"✗ {filename}: {error_msg}", exc_info=True)
            self.checkpoint.mark_failed(filename)
            
            return {
                'success': False,
                'split': None,
                'features': None,
                'labels': None,
                'error': error_msg,
                'audio_info': audio_info
            }
    
    def get_existing_files(self) -> set:
        """
        Get set of already extracted filenames.
        
        Returns:
            Set of filenames (stems) that have been extracted
        """
        existing = set()
        for npz_path in self.train_dir.glob('*.npz'):
            existing.add(npz_path.stem)
        for npz_path in self.val_dir.glob('*.npz'):
            existing.add(npz_path.stem)
        return existing
    
    def find_audio_files(self) -> List[Path]:
        """
        Find all audio files in clips directory.
        
        Returns:
            Sorted list of audio file paths
        """
        self.logger.info(f"Scanning for audio files in: {self.clips_dir}")
        
        audio_extensions = ['.flac', '.wav', '.mp3', '.ogg']
        audio_files = []
        
        for ext in audio_extensions:
            # Root level
            files = list(self.clips_dir.glob(f'*{ext}'))
            if files:
                self.logger.debug(f"  Found {len(files)} {ext} files in root")
            audio_files.extend(files)
            
            # Subdirectories
            subdir_files = list(self.clips_dir.glob(f'**/*{ext}'))
            subdir_files = [f for f in subdir_files if f not in audio_files]
            if subdir_files:
                self.logger.debug(f"  Found {len(subdir_files)} {ext} files in subdirs")
            audio_files.extend(subdir_files)
        
        # Remove duplicates and sort
        audio_files = sorted(set(audio_files))
        
        self.logger.info(f"Total audio files found: {len(audio_files):,}")
        
        # Log file type distribution
        ext_counts = defaultdict(int)
        for f in audio_files:
            ext_counts[f.suffix] += 1
        
        self.logger.info("File type distribution:")
        for ext, count in sorted(ext_counts.items()):
            self.logger.info(f"  {ext}: {count:,} files")
        
        return audio_files
    
    def extract_all(self, 
                    sample_size: Optional[int] = None, 
                    skip_existing: bool = False) -> Dict[str, Any]:
        """
        Extract features from all audio files.
        
        Args:
            sample_size: Limit to N files for testing (None = all files)
            skip_existing: Skip files already extracted
        
        Returns:
            Extraction statistics
        """
        self.logger.info("=" * 80)
        self.logger.info("FEATURE EXTRACTION PIPELINE - 123 CHANNELS")
        self.logger.info("=" * 80)
        
        extraction_start = time.time()
        
        # Find audio files
        audio_files = self.find_audio_files()
        
        if len(audio_files) == 0:
            self.logger.error(f"No audio files found in {self.clips_dir}")
            return {'total': 0, 'train': 0, 'val': 0, 'failed': 0, 'skipped': 0}
        
        # Skip existing if requested
        skipped_count = 0
        if skip_existing:
            existing = self.get_existing_files()
            original_count = len(audio_files)
            audio_files = [f for f in audio_files if f.stem not in existing]
            skipped_count = original_count - len(audio_files)
            
            self.logger.info(f"Skipping {skipped_count:,} already extracted files")
            self.logger.info(f"Remaining: {len(audio_files):,} files")
        
        # Limit sample size
        if sample_size and sample_size > 0:
            audio_files = audio_files[:sample_size]
            self.logger.info(f"TEST MODE: Processing {len(audio_files):,} files")
        
        if len(audio_files) == 0:
            self.logger.info("All files already extracted")
            return {'total': 0, 'train': 0, 'val': 0, 'failed': 0, 'skipped': skipped_count}
        
        # Estimate time
        if sample_size:
            est_time_min = len(audio_files) * 2 / 60  # ~2s per file
            self.logger.info(f"Estimated time: ~{est_time_min:.1f} minutes")
        else:
            est_time_hours = len(audio_files) * 2 / 3600
            self.logger.info(f"Estimated time: ~{est_time_hours:.1f} hours")
        
        # Initialize statistics
        stats = {
            'success': 0,
            'failed': 0,
            'train': 0,
            'val': 0,
            'errors': defaultdict(int),
            'file_times': []
        }
        
        self.logger.info("=" * 80)
        self.logger.info("STARTING EXTRACTION")
        self.logger.info("=" * 80)
        
        # Process files with progress bar
        pbar = tqdm(audio_files, desc="Extracting", unit="file", ncols=100)
        
        for idx, audio_path in enumerate(pbar):
            file_start = time.time()
            
            result = self.extract_single_file(audio_path)
            
            if result['success']:
                # Save NPZ file (safe write + checkpoint)
                start_save = time.time()
                output_dir = self.train_dir if result['split'] == 'train' else self.val_dir
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{audio_path.stem}.npz"
                try:
                    np.savez_compressed(
                        str(output_path),
                        spectrogram=result['features'],
                        labels=result['labels']
                    )
                except Exception as e:
                    self.logger.error(f"Failed saving {output_path.name}: {e}")
                    stats['failed'] += 1
                    stats['errors']['save_error'] += 1
                    self.checkpoint.mark_failed(audio_path.stem)
                    pbar.set_postfix({'fail': stats['failed']}, refresh=False)
                    continue

                save_time = time.time() - start_save
                self.timing_stats['saving'].append(save_time)

                stats['success'] += 1
                stats[result['split']] += 1

                # Persist checkpoint for this file
                try:
                    self.checkpoint.mark_processed(audio_path.stem)
                    self.checkpoint.save()
                except Exception:
                    pass

                # Update progress
                pbar.set_postfix({
                    'OK': stats['success'],
                    'train': stats['train'],
                    'val': stats['val'],
                    'fail': stats['failed']
                }, refresh=False)
                
            else:
                stats['failed'] += 1
                error_key = result['error'].split(':')[0] if result['error'] else 'unknown'
                stats['errors'][error_key] += 1
                
                # Show first few errors in progress bar
                if stats['failed'] <= 3:
                    tqdm.write(f"⚠ {audio_path.name}: {result['error']}")
                
                pbar.set_postfix({
                    'fail': stats['failed'],
                    'err': error_key[:10]
                }, refresh=False)
            
            file_time = time.time() - file_start
            stats['file_times'].append(file_time)
            
            # Periodic checkpoint
            if (idx + 1) % 100 == 0:
                self.checkpoint.save()
                
                # Log progress
                if (idx + 1) % 1000 == 0:
                    self.logger.info(
                        f"Progress: {idx+1}/{len(audio_files)} "
                        f"({100*(idx+1)/len(audio_files):.1f}%) - "
                        f"Success: {stats['success']:,}, Failed: {stats['failed']:,}"
                    )
        
        pbar.close()
        
        # Final checkpoint
        self.checkpoint.save()
        
        # Compute statistics
        extraction_time = time.time() - extraction_start
        
        self.logger.info("=" * 80)
        self.logger.info("EXTRACTION COMPLETE")
        self.logger.info("=" * 80)
        
        # Print summary
        self._print_summary(stats, extraction_time)
        
        # Save detailed metrics
        self._save_metrics(stats, extraction_time)
        
        return {
            'total': stats['success'],
            'train': stats['train'],
            'val': stats['val'],
            'failed': stats['failed'],
            'skipped': skipped_count,
            'errors': dict(stats['errors']),
            'extraction_time': extraction_time
        }
    
    def _print_summary(self, stats: Dict[str, Any], extraction_time: float):
        """Print extraction summary."""
        self.logger.info(f"\n{'Successful:':<20} {stats['success']:>10,}")
        self.logger.info(f"{'  → Train:':<20} {stats['train']:>10,} → {self.train_dir}")
        self.logger.info(f"{'  → Validation:':<20} {stats['val']:>10,} → {self.val_dir}")
        self.logger.info(f"{'Failed:':<20} {stats['failed']:>10,}")
        
        if stats['errors']:
            self.logger.info("\nError Breakdown:")
            for error_type, count in sorted(stats['errors'].items(), key=lambda x: -x[1])[:10]:
                self.logger.info(f"  {error_type:<25}: {count:>6,}")
        
        if stats['file_times']:
            self.logger.info("\nTiming Statistics:")
            self.logger.info(f"  Total time:        {extraction_time/60:.1f} minutes")
            self.logger.info(f"  Avg per file:      {np.mean(stats['file_times']):.3f}s")
            self.logger.info(f"  Fastest file:      {np.min(stats['file_times']):.3f}s")
            self.logger.info(f"  Slowest file:      {np.max(stats['file_times']):.3f}s")
            self.logger.info(f"  Files per minute:  {len(stats['file_times'])/(extraction_time/60):.1f}")
        
        if self.timing_stats['validation']:
            self.logger.info("\nPipeline Stage Timing (average):")
            self.logger.info(f"  Validation:  {np.mean(self.timing_stats['validation']):.3f}s")
            self.logger.info(f"  Extraction:  {np.mean(self.timing_stats['extraction']):.3f}s")
            self.logger.info(f"  Saving:      {np.mean(self.timing_stats['saving']):.3f}s")
        
        # Feature quality summary
        quality_summary = self.quality_metrics.get_summary()
        if quality_summary:
            self.logger.info("\nFeature Quality Metrics:")
            self.logger.info(f"  Time length:  {quality_summary['time_length']['mean']:.0f} ± "
                           f"{quality_summary['time_length']['std']:.0f} frames "
                           f"(range: {quality_summary['time_length']['min']}-"
                           f"{quality_summary['time_length']['max']})")
            self.logger.info(f"  Feature norm: {quality_summary['feature_norm']['mean']:.2f} ± "
                           f"{quality_summary['feature_norm']['std']:.2f}")
            self.logger.info(f"  Silence:      {quality_summary['silence_ratio']['mean']*100:.1f}% "
                           f"channels (max: {quality_summary['silence_ratio']['max']*100:.1f}%)")
            
            if quality_summary.get('snr_db'):
                self.logger.info(f"  SNR:          {quality_summary['snr_db']['mean']:.1f} ± "
                               f"{quality_summary['snr_db']['std']:.1f} dB")
            
            if quality_summary['issues']:
                self.logger.info("\n  Quality Issues:")
                for issue, count in quality_summary['issues'].items():
                    self.logger.info(f"    {issue}: {count}")

        # Include preprocessor-level error counts (e.g., high silence warnings)
        try:
            if hasattr(self.preprocessor, 'error_counts') and self.preprocessor.error_counts:
                self.logger.info("\nPreprocessor Error Counts:")
                for err_type, cnt in sorted(self.preprocessor.error_counts.items(), key=lambda x: -x[1]):
                    self.logger.info(f"  {err_type:<30}: {cnt:>6}")
        except Exception:
            pass
        
        if stats['success'] > 0:
            avg_time_steps = quality_summary['time_length']['mean'] if quality_summary else 1200
            size_gb = stats['success'] * 123 * avg_time_steps * 4 / (1024**3)
            self.logger.info(f"\nEstimated disk usage: ~{size_gb:.2f} GB")
        
        self.logger.info("=" * 80)
    
    def _save_metrics(self, stats: Dict[str, Any], extraction_time: float):
        """Save detailed metrics to JSON."""
        metrics_file = self.output_dir / f'extraction_metrics_{datetime.now():%Y%m%d_%H%M%S}.json'
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'extraction_time_seconds': extraction_time,
            'statistics': {
                'success': stats['success'],
                'failed': stats['failed'],
                'train': stats['train'],
                'val': stats['val'],
                'errors': dict(stats['errors'])
            },
            'timing': {
                'validation': {
                    'mean': float(np.mean(self.timing_stats['validation'])) if self.timing_stats['validation'] else 0,
                    'std': float(np.std(self.timing_stats['validation'])) if self.timing_stats['validation'] else 0
                },
                'extraction': {
                    'mean': float(np.mean(self.timing_stats['extraction'])) if self.timing_stats['extraction'] else 0,
                    'std': float(np.std(self.timing_stats['extraction'])) if self.timing_stats['extraction'] else 0
                },
                'saving': {
                    'mean': float(np.mean(self.timing_stats['saving'])) if self.timing_stats['saving'] else 0,
                    'std': float(np.std(self.timing_stats['saving'])) if self.timing_stats['saving'] else 0
                }
            },
            'quality': self.quality_metrics.get_summary()
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"Detailed metrics saved: {metrics_file}")
    
    def verify_extraction(self):
        """Verify extracted features and compute statistics."""
        self.logger.info("=" * 80)
        self.logger.info("VERIFICATION & STATISTICS")
        self.logger.info("=" * 80)
        
        train_files = list(self.train_dir.glob('*.npz'))
        val_files = list(self.val_dir.glob('*.npz'))
        
        self.logger.info(f"\nFile Counts:")
        self.logger.info(f"  Train:      {len(train_files):>10,}")
        self.logger.info(f"  Validation: {len(val_files):>10,}")
        self.logger.info(f"  Total:      {(len(train_files) + len(val_files)):>10,}")
        
        if len(train_files) == 0 and len(val_files) == 0:
            self.logger.error("No extracted files found!")
            return False
        
        # Sample verification
        self.logger.info("\nVerifying Sample Files:")
        sample_files = (train_files[:5] if len(train_files) >= 5 else train_files) + \
                      (val_files[:3] if len(val_files) >= 3 else val_files)
        
        all_valid = True
        for npz_path in sample_files:
            try:
                data = np.load(npz_path)
                spec = data['spectrogram']
                labels = data.get('labels', None)
                
                if spec.shape[0] != 123:
                    self.logger.error(f"  ✗ {npz_path.name}: Wrong shape {spec.shape}")
                    all_valid = False
                else:
                    self.logger.info(
                        f"  ✓ {npz_path.name}: shape={spec.shape}, "
                        f"dtype={spec.dtype}, labels={labels}"
                    )
            except Exception as e:
                self.logger.error(f"  ✗ {npz_path.name}: {e}")
                all_valid = False
        
        # Compute label statistics
        self._compute_label_stats(train_files, val_files)
        
        self.logger.info("=" * 80)
        return all_valid
    
    def _compute_label_stats(self, train_files: List[Path], val_files: List[Path]):
        """Compute and display label distribution statistics."""
        label_names = ['Prolongation', 'Block', 'SoundRep', 'WordRep', 'Interjection']
        
        for split_name, files in [('Train', train_files), ('Val', val_files)]:
            if not files:
                continue
            
            self.logger.info(f"\n{split_name} Label Distribution:")
            self.logger.info("-" * 60)
            
            # Sample for speed
            sample_files = files if len(files) <= 5000 else \
                          list(np.random.choice(files, 5000, replace=False))
            
            positive_counts = np.zeros(5, dtype=int)
            files_with_any_positive = 0
            
            for npz_path in sample_files:
                try:
                    data = np.load(npz_path)
                    labels = data.get('labels', np.zeros(5))
                    positive_counts += (labels > 0).astype(int)
                    if labels.sum() > 0:
                        files_with_any_positive += 1
                except:
                    pass
            
            # Scale to full dataset
            scale_factor = len(files) / len(sample_files)
            positive_counts = (positive_counts * scale_factor).astype(int)
            files_with_any_positive = int(files_with_any_positive * scale_factor)
            
            for i, name in enumerate(label_names):
                pct = 100 * positive_counts[i] / len(files) if len(files) > 0 else 0
                self.logger.info(f"  {name:<15}: {positive_counts[i]:>6,} ({pct:>5.2f}%)")
            
            pct_any = 100 * files_with_any_positive / len(files) if len(files) > 0 else 0
            self.logger.info(f"  {'Any stutter':<15}: {files_with_any_positive:>6,} ({pct_any:>5.2f}%)")
            self.logger.info("-" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Extract 123-channel features with comprehensive debugging',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full extraction with debug logging
  python extract_features_debug.py --log-level DEBUG
  
  # Quick test with verbose output
  python extract_features_debug.py --sample-size 100 --verbose
  
  # Resume interrupted extraction
  python extract_features_debug.py --skip-existing
  
  # Custom paths with info logging
  python extract_features_debug.py --clips-dir /path/to/audio --log-level INFO
        """
    )
    
    parser.add_argument('--clips-dir', type=str, default='datasets/clips',
                       help='Directory containing audio files')
    parser.add_argument('--output-dir', type=str, default='datasets/features',
                       help='Output directory for features')
    parser.add_argument('--labels-dir', type=str, default='datasets',
                       help='Directory containing label CSV files')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Number of files to process (for testing)')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip files already extracted')
    parser.add_argument('--sample-rate', type=int, default=16000,
                       help='Target sample rate in Hz')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--silence-warn-ratio', type=float, default=0.80,
                       help='Silence warning threshold (0-1). Above this will be logged (INFO).')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output (DEBUG level)')
    
    args = parser.parse_args()
    
    # Adjust log level
    log_level = 'DEBUG' if args.verbose else args.log_level
    
    # Validate paths
    clips_path = Path(args.clips_dir)
    if not clips_path.exists():
        print(f"\n✗ ERROR: Clips directory not found: {clips_path}")
        print("  Use --clips-dir to specify correct path")
        return 1
    
    # Create pipeline
    try:
        pipeline = FeatureExtractionPipeline(
            clips_dir=args.clips_dir,
            output_dir=args.output_dir,
            label_dir=args.labels_dir,
                sr=args.sample_rate,
                log_level=log_level,
                silence_warn_ratio=args.silence_warn_ratio
        )
        
        # Extract features
        results = pipeline.extract_all(
            sample_size=args.sample_size,
            skip_existing=args.skip_existing
        )
        
        # Verify
        if results['total'] > 0:
            pipeline.verify_extraction()
        
        # Success
        if results['total'] > 0:
            pipeline.logger.info("\n✅ Feature extraction complete!")
            pipeline.logger.info(f"\nExtracted {results['total']:,} files:")
            pipeline.logger.info(f"  Train:      {results['train']:,}")
            pipeline.logger.info(f"  Validation: {results['val']:,}")
            if results['failed'] > 0:
                pipeline.logger.info(f"  Failed:     {results['failed']:,}")
            
            pipeline.logger.info("\nNext Steps:")
            pipeline.logger.info("  1. Train model:")
            pipeline.logger.info("     python COMPLETE_PIPELINE_DEBUG.py --skip-features --verbose")
            pipeline.logger.info("  2. Expected metrics:")
            pipeline.logger.info("     F1-score: 70-85% (per-class breakdown available)")
            
            return 0
        else:
            pipeline.logger.warning("No files were successfully extracted")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⚠ Extraction interrupted by user")
        return 1
    except Exception as e:
        print(f"\n✗ FATAL ERROR: {e}")
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())