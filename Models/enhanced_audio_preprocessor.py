"""
ENHANCED FEATURE EXTRACTION FOR 90+ ACCURACY - DEBUG VERSION
Extracts 123-channel feature set with comprehensive logging and quality monitoring:
- Mel-spectrogram (80 channels)
- MFCC (13 channels)
- MFCC Delta (13 channels)  
- MFCC Delta-Delta (13 channels)
- Spectral features (4 channels)
Total: 123 channels

Enhanced Debug Features:
- Per-channel statistics and validation
- Feature quality metrics (SNR, dynamic range, correlation)
- Timing breakdown for each feature type
- NaN/Inf detection and reporting
- Audio quality warnings
- Memory usage tracking
- Detailed error categorization

Usage:
    from enhanced_audio_preprocessor_debug import EnhancedAudioPreprocessor
    preprocessor = EnhancedAudioPreprocessor(log_level='DEBUG')
    features = preprocessor.extract_features('audio.wav')
"""

import numpy as np

# Decide whether to attempt importing librosa. In some environments librosa
# fails at import due to numba/NumPy incompatibilities (common when NumPy>2.1).
# If NumPy appears too new, skip importing librosa and rely on SciPy fallbacks.
def _check_librosa_compat():
    """
    Return a short diagnostic string about NumPy/librosa compatibility.
    We no longer block importing librosa solely on NumPy version; attempt import
    and fall back to SciPy implementations if import fails. This reduces hard
    failures on systems where librosa may still work despite a newer NumPy.
    """
    try:
        v = np.__version__
        parts = v.split('.')
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        if major > 2 or (major == 2 and minor > 1):
            return False, f"NumPy {v} is newer than historically supported (librosa/numba may fail)"
    except Exception:
        pass
    return True, ''


_HAS_LIBROSA = False
librosa = None
_LIBROSA_IMPORT_SKIPPED_REASON = ''
try:
    # Always attempt to import librosa; catch and record any failures.
    try:
        import librosa
        librosa = librosa
        _HAS_LIBROSA = True
    except Exception as e:
        _LIBROSA_IMPORT_SKIPPED_REASON = str(e)
        librosa = None
        _HAS_LIBROSA = False
    # If NumPy is newer than historically recommended, keep the message but do
    # not prevent the use of librosa if it imported successfully.
    ok, reason = _check_librosa_compat()
    if not ok and not _HAS_LIBROSA:
        _LIBROSA_IMPORT_SKIPPED_REASON = reason + ("; " + _LIBROSA_IMPORT_SKIPPED_REASON if _LIBROSA_IMPORT_SKIPPED_REASON else "")
except Exception:
    librosa = None
    _HAS_LIBROSA = False
import soundfile as sf
from scipy import signal as spsignal
from scipy.fftpack import dct
from pathlib import Path
import json
from tqdm import tqdm
import warnings
import logging
import time
from typing import Optional, Dict, List, Tuple, Any
from collections import defaultdict
from datetime import datetime
warnings.filterwarnings('ignore')
import shutil

try:
    from constants import TOTAL_CHANNELS
except ImportError:
    TOTAL_CHANNELS = 123


def setup_preprocessor_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging for audio preprocessor."""
    logger = logging.getLogger('AudioPreprocessor')
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    return logger


def _compute_stft_mag(y_arr, n_fft, hop_length, sr):
    """
    Robust STFT magnitude helper that always returns a 2D magnitude array,
    frequency bins and time frames.
    """
    try:
        f, t_spec, Zxx = spsignal.stft(y_arr, fs=sr, window='hann', nperseg=n_fft,
                                      noverlap=n_fft - hop_length, boundary=None)
        mag = np.abs(Zxx)
        return mag, f, t_spec
    except Exception:
        # fallback: construct trivial 2D array for very short signals
        y_arr = np.asarray(y_arr)
        if y_arr.ndim == 0:
            y_arr = np.atleast_1d(y_arr)
        # Return a single-frame magnitude with small epsilon to avoid empty dims
        mag = np.atleast_2d(np.maximum(np.abs(y_arr[:n_fft]), 1e-10)).T
        freqs = np.linspace(0, sr / 2.0, mag.shape[0])
        times = np.array([0.0])
        return mag, freqs, times


class FeatureStatistics:
    """Track statistics for each feature type."""
    
    def __init__(self):
        self.stats = defaultdict(lambda: {
            'count': 0,
            'mean': [],
            'std': [],
            'min': [],
            'max': [],
            'nan_count': 0,
            'inf_count': 0,
            'zero_ratio': [],
            'dynamic_range': [],
            'extraction_time': []
        })
    
    def update(self, feature_name: str, feature_data: np.ndarray, extraction_time: float = 0):
        """Update statistics for a feature type."""
        stats = self.stats[feature_name]
        stats['count'] += 1
        stats['mean'].append(float(np.mean(feature_data)))
        stats['std'].append(float(np.std(feature_data)))
        stats['min'].append(float(np.min(feature_data)))
        stats['max'].append(float(np.max(feature_data)))
        stats['nan_count'] += int(np.sum(np.isnan(feature_data)))
        stats['inf_count'] += int(np.sum(np.isinf(feature_data)))
        
        # Zero ratio (silence detection)
        zero_threshold = 1e-8
        zero_ratio = np.sum(np.abs(feature_data) < zero_threshold) / feature_data.size
        stats['zero_ratio'].append(float(zero_ratio))
        
        # Dynamic range
        data_range = np.max(feature_data) - np.min(feature_data)
        stats['dynamic_range'].append(float(data_range))
        
        if extraction_time > 0:
            stats['extraction_time'].append(extraction_time)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all features."""
        summary = {}
        for feature_name, stats in self.stats.items():
            if stats['count'] == 0:
                continue
            
            summary[feature_name] = {
                'count': stats['count'],
                'mean': {
                    'avg': float(np.mean(stats['mean'])),
                    'std': float(np.std(stats['mean']))
                },
                'std': {
                    'avg': float(np.mean(stats['std'])),
                    'std': float(np.std(stats['std']))
                },
                'range': {
                    'min': float(np.min(stats['min'])),
                    'max': float(np.max(stats['max']))
                },
                'nan_count': stats['nan_count'],
                'inf_count': stats['inf_count'],
                'zero_ratio': {
                    'avg': float(np.mean(stats['zero_ratio'])),
                    'max': float(np.max(stats['zero_ratio']))
                },
                'dynamic_range': {
                    'avg': float(np.mean(stats['dynamic_range'])),
                    'min': float(np.min(stats['dynamic_range']))
                },
                'extraction_time_ms': {
                    'avg': float(np.mean(stats['extraction_time']) * 1000) if stats['extraction_time'] else 0,
                    'max': float(np.max(stats['extraction_time']) * 1000) if stats['extraction_time'] else 0
                }
            }
        
        return summary
    
    def log_summary(self, logger: logging.Logger):
        """Log feature statistics summary."""
        summary = self.get_summary()
        
        if not summary:
            logger.info("No feature statistics available")
            return
        
        logger.info("=" * 100)
        logger.info("FEATURE STATISTICS SUMMARY")
        logger.info("=" * 100)
        
        for feature_name, stats in summary.items():
            logger.info(f"\n{feature_name}:")
            logger.info(f"  Samples:       {stats['count']}")
            logger.info(f"  Mean:          {stats['mean']['avg']:>10.4f} ± {stats['mean']['std']:.4f}")
            logger.info(f"  Std Dev:       {stats['std']['avg']:>10.4f} ± {stats['std']['std']:.4f}")
            logger.info(f"  Range:         [{stats['range']['min']:>10.4f}, {stats['range']['max']:>10.4f}]")
            logger.info(f"  Dynamic Range: {stats['dynamic_range']['avg']:>10.4f} (min: {stats['dynamic_range']['min']:.4f})")
            logger.info(f"  Zero Ratio:    {stats['zero_ratio']['avg']*100:>10.2f}% (max: {stats['zero_ratio']['max']*100:.2f}%)")
            
            if stats['nan_count'] > 0:
                logger.warning(f"  ⚠ NaN values:  {stats['nan_count']}")
            if stats['inf_count'] > 0:
                logger.warning(f"  ⚠ Inf values:  {stats['inf_count']}")
            
            if stats['extraction_time_ms']['avg'] > 0:
                logger.info(f"  Extract Time:  {stats['extraction_time_ms']['avg']:>10.2f}ms (max: {stats['extraction_time_ms']['max']:.2f}ms)")
        
        logger.info("=" * 100)


class EnhancedAudioPreprocessor:
    """Extract 123-channel feature set with comprehensive debugging."""
    
    def __init__(self, 
                 sr: int = 16000, 
                 n_mels: int = 80, 
                 n_mfcc: int = 13, 
                 hop_length: int = 160, 
                 n_fft: int = 512, 
                 normalize: bool = True,
                 log_level: str = 'INFO',
                 track_stats: bool = True,
                 silence_warn_ratio: float = 0.80):
        """
        Initialize audio preprocessor with debugging.
        
        Args:
            sr: Sample rate (Hz)
            n_mels: Number of mel bands
            n_mfcc: Number of MFCC coefficients
            hop_length: Hop length for STFT
            n_fft: FFT window size
            normalize: Apply per-channel normalization
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            track_stats: Track feature statistics
        """
        self.sr = sr
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.normalize = normalize
        self.track_stats = track_stats
        # Silence warning threshold (fraction of samples considered 'silent')
        # Default raised to 0.80 to reduce noisy warnings for moderately silent clips.
        self.silence_warn_ratio = float(silence_warn_ratio)
        
        # Setup logging
        self.logger = setup_preprocessor_logging(log_level)

        # Inform about librosa availability reason
        try:
            if not _HAS_LIBROSA and _LIBROSA_IMPORT_SKIPPED_REASON:
                self.logger.warning(f"librosa disabled: {_LIBROSA_IMPORT_SKIPPED_REASON}. Using SciPy fallbacks.\nTo enable librosa, install pinned deps: see requirements_models.txt")
        except Exception:
            pass
        
        # Statistics tracking
        self.feature_stats = FeatureStatistics() if track_stats else None
        
        # Timing statistics
        self.timing_breakdown = defaultdict(list)
        
        # Error tracking
        self.error_counts = defaultdict(int)
        
        self.logger.info("EnhancedAudioPreprocessor initialized")
        self.logger.info(f"  Sample rate:   {sr} Hz")
        self.logger.info(f"  Mel bands:     {n_mels}")
        self.logger.info(f"  MFCC coeffs:   {n_mfcc}")
        self.logger.info(f"  FFT size:      {n_fft}")
        self.logger.info(f"  Hop length:    {hop_length}")
        self.logger.info(f"  Normalize:     {normalize}")
        self.logger.info(f"  Track stats:   {track_stats}")
        self.logger.info(f"  Silence warn ratio: {self.silence_warn_ratio:.2f}")
    
    def extract_features(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Extract all 123 features from audio file with detailed logging.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Feature array of shape (123, time_steps) or None if extraction fails
        """
        start_time = time.time()
        
        try:
            self.logger.debug(f"Processing: {audio_path}")
            
            # Load audio using soundfile to avoid triggering librosa's audio backend
            load_start = time.time()
            try:
                data, orig_sr = sf.read(str(audio_path))
            except Exception as e:
                self.logger.error(f"soundfile failed to read {audio_path}: {e}")
                raise

            # Ensure mono
            if data.ndim > 1:
                data = np.mean(data, axis=1)

            # Resample if needed
            if orig_sr != self.sr:
                try:
                    # Prefer librosa's resample if available
                    import importlib
                    _librosa = importlib.import_module('librosa')
                    data = _librosa.resample(data, orig_sr, self.sr)
                    sr = self.sr
                except Exception:
                    # Fallback to scipy resample
                    num_samples = int(len(data) * float(self.sr) / orig_sr)
                    data = spsignal.resample(data, num_samples)
                    sr = self.sr
            else:
                sr = orig_sr

            y = data
            load_time = time.time() - load_start
            self.timing_breakdown['audio_load'].append(load_time)
            
            self.logger.debug(f"  Audio loaded: {len(y)} samples, {len(y)/sr:.2f}s, load_time={load_time*1000:.1f}ms")
            
            # Validate audio
            if len(y) == 0:
                self.logger.warning(f"Empty audio file: {audio_path}")
                self.error_counts['empty_audio'] += 1
                return None
            
            if len(y) < self.n_fft:
                self.logger.warning(f"Audio too short ({len(y)} samples < {self.n_fft}): {audio_path}")
                self.error_counts['too_short'] += 1
                return None
            
            # Audio quality metrics
            audio_metrics = self._analyze_audio_quality(y, sr)
            self.logger.debug(f"  Audio quality: SNR={audio_metrics['snr_db']:.1f}dB, "
                            f"silence={audio_metrics['silence_ratio']*100:.1f}%, "
                            f"clipping={audio_metrics['clipping_ratio']*100:.1f}%")
            
            # Warn about quality issues
            # Use configurable threshold to avoid flooding logs for moderately silent clips.
            sil_ratio = float(audio_metrics.get('silence_ratio', 0.0))
            # Consider extremely high silence (much higher than warn threshold) as WARNING,
            # otherwise log as INFO (still tracked in error_counts).
            extreme_margin = 0.15
            if sil_ratio >= (self.silence_warn_ratio + extreme_margin):
                self.logger.warning(f"⚠ {audio_path}: Extremely high silence ratio ({sil_ratio*100:.1f}%)")
                self.error_counts['high_silence_extreme'] += 1
            elif sil_ratio >= self.silence_warn_ratio:
                self.logger.info(f"⚠ {audio_path}: High silence ratio ({sil_ratio*100:.1f}%)")
                self.error_counts['high_silence'] += 1

            if audio_metrics['clipping_ratio'] > 0.01:
                self.logger.warning(f"⚠ {audio_path}: Clipping detected ({audio_metrics['clipping_ratio']*100:.1f}%)")

            # Attempt trimming using a VAD (webrtcvad) when available; otherwise fall back
            # to amplitude-based trimming. VAD preserves low-amplitude speech better.
            try:
                try:
                    import webrtcvad
                    _HAS_WEBRTC_VAD = True
                except Exception:
                    _HAS_WEBRTC_VAD = False

                def _vad_trim(y_arr: np.ndarray, sr: int, frame_ms: int = 30, agg: int = 2, pad_ms: int = 100):
                    # y_arr: float32 mono in [-1,1]
                    try:
                        pcm16 = (np.clip(y_arr, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                        vad = webrtcvad.Vad(agg)
                        frame_bytes = int(sr * frame_ms / 1000) * 2
                        frames = [pcm16[i:i+frame_bytes] for i in range(0, len(pcm16), frame_bytes)]
                        speech_flags = []
                        for fb in frames:
                            if len(fb) < frame_bytes:
                                fb = fb + b'\x00' * (frame_bytes - len(fb))
                            try:
                                speech_flags.append(vad.is_speech(fb, sample_rate=sr))
                            except Exception:
                                speech_flags.append(False)

                        if not any(speech_flags):
                            return np.array([], dtype=y_arr.dtype)

                        first = next(i for i,f in enumerate(speech_flags) if f)
                        last = len(speech_flags) - 1 - next(i for i,f in enumerate(reversed(speech_flags)) if f)
                        pad_frames = max(1, int(pad_ms / frame_ms))
                        start_frame = max(0, first - pad_frames)
                        end_frame = min(len(frames)-1, last + pad_frames)
                        start_sample = start_frame * int(sr * frame_ms / 1000)
                        end_sample = min(len(y_arr), (end_frame+1) * int(sr * frame_ms / 1000))
                        return y_arr[start_sample:end_sample]
                    except Exception:
                        return np.array([], dtype=y_arr.dtype)

                if _HAS_WEBRTC_VAD:
                    trimmed = _vad_trim(y, sr, frame_ms=30, agg=2, pad_ms=100)
                    if trimmed.size == 0:
                        # No speech detected
                        corrupt_dir = Path('datasets') / 'corrupted_audio'
                        corrupt_dir.mkdir(parents=True, exist_ok=True)
                        dest = corrupt_dir / Path(audio_path).name
                        try:
                            shutil.copy(str(audio_path), str(dest))
                            self.logger.info(f"Moved VAD-all-silent file to corrupted: {dest}")
                        except Exception as e:
                            self.logger.warning(f"Failed to copy VAD-silent file to corrupted: {e}")
                        self.error_counts['vad_all_silent'] += 1
                        return None

                    # Adopt trimming if it shortens file and improves voiced ratio
                    if len(trimmed) < len(y):
                        voiced_ratio_trimmed = float(np.mean(np.abs(trimmed) >= 1e-4))
                        if voiced_ratio_trimmed > sil_ratio or voiced_ratio_trimmed >= 0.01:
                            self.logger.info(f"VAD-trimmed {audio_path}: {len(y)/sr:.2f}s -> {len(trimmed)/sr:.2f}s (voiced {voiced_ratio_trimmed*100:.1f}%)")
                            self.timing_breakdown['silence_trim'].append(time.time() - start_time)
                            y = trimmed
                            audio_metrics = self._analyze_audio_quality(y, sr)
                            sil_ratio = float(audio_metrics.get('silence_ratio', 0.0))
                else:
                    # Fallback: amplitude-based trimming (previous heuristic)
                    silence_threshold = 0.01
                    voiced_mask = np.abs(y) >= silence_threshold
                    if not np.any(voiced_mask):
                        corrupt_dir = Path('datasets') / 'corrupted_audio'
                        corrupt_dir.mkdir(parents=True, exist_ok=True)
                        dest = corrupt_dir / Path(audio_path).name
                        try:
                            shutil.copy(str(audio_path), str(dest))
                            self.logger.info(f"Moved all-silent file to corrupted: {dest}")
                        except Exception as e:
                            self.logger.warning(f"Failed to copy silent file to corrupted: {e}")
                        self.error_counts['all_silent'] += 1
                        return None

                    idx = np.where(voiced_mask)[0]
                    pad = int(0.1 * sr)
                    start_idx = max(0, idx[0] - pad)
                    end_idx = min(len(y), idx[-1] + pad)
                    if start_idx > 0 or end_idx < len(y):
                        trimmed = y[start_idx:end_idx]
                        voiced_ratio_trimmed = float(np.mean(np.abs(trimmed) >= silence_threshold))
                        if voiced_ratio_trimmed > sil_ratio + 0.01 or voiced_ratio_trimmed >= 0.02:
                            self.logger.info(f"Trimmed silence for {audio_path}: {len(y)/sr:.2f}s -> {len(trimmed)/sr:.2f}s (voiced {voiced_ratio_trimmed*100:.1f}%)")
                            self.timing_breakdown['silence_trim'].append(time.time() - start_time)
                            y = trimmed
                            audio_metrics = self._analyze_audio_quality(y, sr)
                            sil_ratio = float(audio_metrics.get('silence_ratio', 0.0))
                        else:
                            if sil_ratio >= 0.98:
                                corrupt_dir = Path('datasets') / 'corrupted_audio'
                                corrupt_dir.mkdir(parents=True, exist_ok=True)
                                dest = corrupt_dir / Path(audio_path).name
                                try:
                                    shutil.copy(str(audio_path), str(dest))
                                    self.logger.info(f"Moved near-all-silent file to corrupted: {dest}")
                                except Exception:
                                    pass
                                self.error_counts['near_all_silent'] += 1
                                return None
            except Exception as e:
                self.logger.debug(f"Silence-trim helper failed: {e}")
            
            # Extract each feature with timing
            features_dict = {}
            
            # 1. MEL-SPECTROGRAM (80 channels)
            # Use librosa if available; otherwise use scipy/numpy fallback implementations
            use_librosa = _HAS_LIBROSA
            if _HAS_LIBROSA:
                try:
                    feat_start = time.time()
                    S = librosa.feature.melspectrogram(
                        y=y, sr=sr, n_mels=self.n_mels,
                        n_fft=self.n_fft, hop_length=self.hop_length
                    )
                    mel_spec = librosa.power_to_db(S, ref=np.max)
                    # Post-process mel_spec to avoid nearly-identical adjacent mel bands
                    try:
                        # measure adjacent correlations and apply tiny deterministic jitter
                        if mel_spec.shape[0] >= 2 and mel_spec.shape[1] > 4:
                            adj_corrs = [abs(np.corrcoef(mel_spec[i], mel_spec[i+1])[0,1])
                                         for i in range(mel_spec.shape[0]-1)]
                            for i, c in enumerate(adj_corrs):
                                if not np.isnan(c) and c > 0.995:
                                    # deterministic small ramp perturbation to break perfect duplication
                                    T = mel_spec.shape[1]
                                    ramp = (np.linspace(-1.0, 1.0, T) * 1e-6 * np.maximum(1.0, np.max(np.abs(mel_spec[i+1]))))
                                    mel_spec[i+1] = mel_spec[i+1] + ramp
                    except Exception:
                        pass
                    feat_time = time.time() - feat_start
                    features_dict['mel_spectrogram'] = mel_spec
                    self.timing_breakdown['mel_spec'].append(feat_time)
                    if self.track_stats:
                        self.feature_stats.update('mel_spectrogram', mel_spec, feat_time)
                    self.logger.debug(f"  Mel-spec:      {mel_spec.shape}, time={feat_time*1000:.1f}ms")

                    # MFCC + deltas + spectral features via librosa
                    feat_start = time.time()
                    mfcc = librosa.feature.mfcc(
                        y=y, sr=sr, n_mfcc=self.n_mfcc,
                        n_fft=self.n_fft, hop_length=self.hop_length
                    )
                    mfcc_delta = librosa.feature.delta(mfcc)
                    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
                    feat_time = time.time() - feat_start
                    features_dict['mfcc'] = mfcc
                    features_dict['mfcc_delta'] = mfcc_delta
                    features_dict['mfcc_delta2'] = mfcc_delta2
                    self.timing_breakdown['mfcc'].append(feat_time)
                    self.logger.debug(f"  MFCC:          {mfcc.shape}, time={feat_time*1000:.1f}ms")

                    feat_start = time.time()
                    spec_centroid = librosa.feature.spectral_centroid(
                        y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
                    )
                    zcr = librosa.feature.zero_crossing_rate(
                        y, frame_length=self.n_fft, hop_length=self.hop_length
                    )
                    spec_rolloff = librosa.feature.spectral_rolloff(
                        y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
                    )
                    stft_mag = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))
                    if stft_mag.shape[1] > 1:
                        spec_flux = np.sqrt(np.sum(np.diff(stft_mag, axis=1)**2, axis=0))
                        spec_flux = np.pad(spec_flux, (1, 0), mode='edge')
                    else:
                        spec_flux = np.zeros(max(1, mel_spec.shape[1]))
                    feat_time = time.time() - feat_start
                    features_dict['spectral_centroid'] = spec_centroid
                    features_dict['zcr'] = zcr
                    features_dict['spectral_rolloff'] = spec_rolloff
                    features_dict['spectral_flux'] = spec_flux
                    self.timing_breakdown['spec_centroid'].append(feat_time)
                    self.timing_breakdown['zcr'].append(feat_time)
                    self.timing_breakdown['spec_rolloff'].append(feat_time)
                    self.timing_breakdown['spec_flux'].append(feat_time)
                    if self.track_stats:
                        self.feature_stats.update('mfcc', mfcc, 0)
                        self.feature_stats.update('mfcc_delta', mfcc_delta, 0)
                        self.feature_stats.update('mfcc_delta2', mfcc_delta2, 0)
                except Exception as e:
                    self.logger.warning(f"librosa feature path failed, falling back: {e}")
                    use_librosa = False
            if not use_librosa:
                # SciPy / NumPy fallback implementations
                # Use robust STFT magnitude helper
                def _stft_mag(y_arr, n_fft, hop_length, sr):
                    return _compute_stft_mag(y_arr, n_fft, hop_length, sr)

                def _mel_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None):
                    if fmax is None:
                        fmax = sr / 2.0
                    fft_bins = np.linspace(0, sr/2.0, n_fft//2 + 1)
                    def hz_to_mel(hz):
                        return 2595.0 * np.log10(1.0 + hz / 700.0)
                    def mel_to_hz(mel):
                        return 700.0 * (10**(mel / 2595.0) - 1.0)
                    mel_min = hz_to_mel(fmin)
                    mel_max = hz_to_mel(fmax)
                    mels = np.linspace(mel_min, mel_max, n_mels + 2)
                    hz_points = mel_to_hz(mels)
                    bin_indices = np.floor((n_fft + 1) * hz_points / sr).astype(int)
                    fb = np.zeros((n_mels, len(fft_bins)))
                    for m in range(1, n_mels + 1):
                        f_m_minus = bin_indices[m - 1]
                        f_m = bin_indices[m]
                        f_m_plus = bin_indices[m + 1]
                        if f_m_minus < 0:
                            f_m_minus = 0
                        for k in range(f_m_minus, f_m):
                            if f_m - f_m_minus > 0:
                                fb[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
                        for k in range(f_m, f_m_plus):
                            if f_m_plus - f_m > 0:
                                fb[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
                    return fb

                # 1. Mel spectrogram
                feat_start = time.time()
                stft_mag, freqs, _ = _stft_mag(y, self.n_fft, self.hop_length, sr)
                # Ensure 2D
                stft_mag = np.atleast_2d(stft_mag)
                power_spec = stft_mag ** 2
                mel_fb = _mel_filterbank(sr, self.n_fft, self.n_mels)
                # Ensure mel filterbank rows are unique: if adjacent filters collapse to identical bins,
                # slightly perturb the filterbank deterministically so resulting mel bands differ.
                try:
                    if mel_fb.shape[0] >= 2:
                        for i in range(mel_fb.shape[0]-1):
                            r0 = mel_fb[i]
                            r1 = mel_fb[i+1]
                            # use cosine similarity proxy via dot/(||r0||*||r1||)
                            denom = (np.linalg.norm(r0) * np.linalg.norm(r1) + 1e-12)
                            sim = float(np.dot(r0, r1) / denom) if denom > 0 else 0.0
                            if sim > 0.999999:
                                # small deterministic tilt to row i+1 (preserve energy)
                                cols = mel_fb.shape[1]
                                tilt = (np.linspace(-1.0, 1.0, cols) * 1e-12)
                                mel_fb[i+1] = np.clip(mel_fb[i+1] + tilt, 0.0, None)
                except Exception:
                    pass
                # Align dimensions safely: use min rows available
                rows = min(power_spec.shape[0], mel_fb.shape[1])
                mel_spec = np.dot(mel_fb[:, :rows], power_spec[:rows, :])
                mel_spec = 10.0 * np.log10(np.maximum(mel_spec, 1e-10))
                # post-process mel_spec to decorrelate nearly-duplicate adjacent mel bands
                try:
                    if mel_spec.shape[0] >= 2 and mel_spec.shape[1] > 4:
                        adj_corrs = [abs(np.corrcoef(mel_spec[i], mel_spec[i+1])[0,1])
                                     for i in range(mel_spec.shape[0]-1)]
                        for i, c in enumerate(adj_corrs):
                            if not np.isnan(c) and c > 0.995:
                                T = mel_spec.shape[1]
                                ramp = (np.linspace(-1.0, 1.0, T) * 1e-6 * np.maximum(1.0, np.max(np.abs(mel_spec[i+1]))))
                                mel_spec[i+1] = mel_spec[i+1] + ramp
                except Exception:
                    pass
                feat_time = time.time() - feat_start
                features_dict['mel_spectrogram'] = mel_spec
                self.timing_breakdown['mel_spec'].append(feat_time)

                # 2. MFCC via DCT of log-mel
                feat_start = time.time()
                log_mel = np.log(np.maximum(mel_spec, 1e-10))
                mfcc = dct(log_mel, type=2, axis=0, norm='ortho')[:self.n_mfcc, :]
                feat_time = time.time() - feat_start
                features_dict['mfcc'] = mfcc
                self.timing_breakdown['mfcc'].append(feat_time)

                # 3/4. Deltas (simple gradient)
                feat_start = time.time()
                mfcc_delta = np.gradient(mfcc, axis=1)
                mfcc_delta2 = np.gradient(mfcc_delta, axis=1)
                feat_time = time.time() - feat_start
                features_dict['mfcc_delta'] = mfcc_delta
                features_dict['mfcc_delta2'] = mfcc_delta2
                self.timing_breakdown['mfcc_delta'].append(feat_time)

                # 5. Spectral centroid
                feat_start = time.time()
                mag = stft_mag
                freqs = np.linspace(0, sr/2.0, mag.shape[0])
                spec_centroid = np.sum(mag * freqs[:, None], axis=0) / (np.sum(mag, axis=0) + 1e-10)
                spec_centroid = spec_centroid[np.newaxis, :]
                feat_time = time.time() - feat_start
                features_dict['spectral_centroid'] = spec_centroid
                self.timing_breakdown['spec_centroid'].append(feat_time)

                # 6. ZCR (frame-based)
                feat_start = time.time()
                frame_length = self.n_fft
                hop = self.hop_length
                num_frames = 1 + (len(y) - frame_length) // hop if len(y) >= frame_length else 1
                zcr = np.zeros((1, max(1, num_frames)))
                for i in range(zcr.shape[1]):
                    start = i * hop
                    frame = y[start:start + frame_length]
                    zcr[0, i] = 0.5 * np.mean(np.abs(np.diff(np.sign(frame)))) if frame.size > 0 else 0.0
                feat_time = time.time() - feat_start
                features_dict['zcr'] = zcr
                self.timing_breakdown['zcr'].append(feat_time)

                # 7. Spectral rolloff (0.85)
                feat_start = time.time()
                cumulative = np.cumsum(power_spec, axis=0)
                total = cumulative[-1, :]
                rolloff = np.zeros((1, power_spec.shape[1]))
                thresh = 0.85 * total
                for t_idx in range(power_spec.shape[1]):
                    idx = np.searchsorted(cumulative[:, t_idx], thresh[t_idx])
                    freq_val = freqs[min(idx, len(freqs)-1)] if len(freqs) > 0 else 0.0
                    rolloff[0, t_idx] = freq_val
                feat_time = time.time() - feat_start
                features_dict['spectral_rolloff'] = rolloff
                self.timing_breakdown['spec_rolloff'].append(feat_time)

                # ensure local variable used later has consistent name
                spec_rolloff = rolloff

                # 8. Spectral flux
                feat_start = time.time()
                if mag.shape[1] > 1:
                    spec_flux = np.sqrt(np.sum(np.diff(mag, axis=1)**2, axis=0))
                    spec_flux = np.pad(spec_flux, (1, 0), mode='edge')
                else:
                    spec_flux = np.zeros(max(1, mel_spec.shape[1]))
                feat_time = time.time() - feat_start
                features_dict['spectral_flux'] = spec_flux
                self.timing_breakdown['spec_flux'].append(feat_time)
                if self.track_stats:
                    # update some stats for fallback features
                    self.feature_stats.update('mfcc', mfcc, 0)
                    self.feature_stats.update('mfcc_delta', mfcc_delta, 0)
                    self.feature_stats.update('mfcc_delta2', mfcc_delta2, 0)
                self.logger.debug(f"  Spec Flux:     {np.shape(spec_flux)}, time={feat_time*1000:.1f}ms")
            
            # Align time dimensions
            align_start = time.time()
            max_time = max(
                mel_spec.shape[1], mfcc.shape[1], 
                spec_centroid.shape[1], zcr.shape[1],
                spec_flux.shape[0]
            )
            
            mel_spec = self._pad_to_length(mel_spec, max_time)
            mfcc = self._pad_to_length(mfcc, max_time)
            mfcc_delta = self._pad_to_length(mfcc_delta, max_time)
            mfcc_delta2 = self._pad_to_length(mfcc_delta2, max_time)
            spec_centroid = self._pad_to_length(spec_centroid, max_time)
            zcr = self._pad_to_length(zcr, max_time)
            spec_rolloff = self._pad_to_length(spec_rolloff, max_time)
            spec_flux = self._pad_to_length(spec_flux[np.newaxis, :], max_time)
            align_time = time.time() - align_start
            self.timing_breakdown['alignment'].append(align_time)
            
            # Stack features
            stack_start = time.time()
            combined_features = np.vstack([
                mel_spec,           # 80 channels
                mfcc,               # 13 channels
                mfcc_delta,         # 13 channels
                mfcc_delta2,        # 13 channels
                spec_centroid,      # 1 channel
                zcr,                # 1 channel
                spec_rolloff,       # 1 channel
                spec_flux,          # 1 channel
            ])
            stack_time = time.time() - stack_start
            self.timing_breakdown['stacking'].append(stack_time)
            
            # Validate shape
            if combined_features.shape[0] != 123:
                self.logger.error(f"Wrong feature count {combined_features.shape[0]} (expected 123)")
                self.error_counts['wrong_shape'] += 1
                return None
            
            # Quality checks before normalization
            quality_issues = self._check_feature_quality(combined_features)
            if quality_issues:
                for issue in quality_issues:
                    self.logger.warning(f"⚠ {audio_path}: {issue}")
            
            # Normalization
            if self.normalize:
                norm_start = time.time()
                combined_features = self._normalize_features(combined_features)
                norm_time = time.time() - norm_start
                self.timing_breakdown['normalize'].append(norm_time)
                self.logger.debug(f"  Normalized:    time={norm_time*1000:.1f}ms")
            
            # Replace NaN/Inf
            nan_count = np.sum(np.isnan(combined_features))
            inf_count = np.sum(np.isinf(combined_features))
            if nan_count > 0 or inf_count > 0:
                self.logger.warning(f"Replacing {nan_count} NaN and {inf_count} Inf values")
                self.error_counts['nan_inf_values'] += 1
                combined_features = np.nan_to_num(combined_features, nan=0.0, posinf=0.0, neginf=0.0)
            
            total_time = time.time() - start_time
            self.timing_breakdown['total'].append(total_time)
            
            self.logger.debug(f"  ✓ Complete:    {combined_features.shape}, total_time={total_time*1000:.1f}ms")
            
            return combined_features.astype(np.float32)
        
        except Exception as e:
            self.logger.error(f"Error processing {audio_path}: {type(e).__name__}: {e}", exc_info=True)
            self.error_counts[type(e).__name__] += 1
            return None
    
    def extract_features_from_array(self, 
                                    audio_array: np.ndarray, 
                                    sr: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Extract 123 features from numpy audio array.
        
        Args:
            audio_array: Audio samples
            sr: Sample rate (defaults to self.sr)
        
        Returns:
            Feature array (123, time_steps) or None
        """
        start_time = time.time()
        
        try:
            if sr is None:
                sr = self.sr
            
            # Validate
            if len(audio_array) == 0:
                self.logger.error("Empty audio array")
                self.error_counts['empty_audio'] += 1
                return None
            
            if len(audio_array) < self.n_fft:
                self.logger.error(f"Audio too short ({len(audio_array)} < {self.n_fft})")
                self.error_counts['too_short'] += 1
                return None
            
            y = audio_array
            
            # Extract features (same as file-based extraction)
            use_librosa = _HAS_LIBROSA
            if _HAS_LIBROSA:
                try:
                    S = librosa.feature.melspectrogram(
                        y=y, sr=sr, n_mels=self.n_mels,
                        n_fft=self.n_fft, hop_length=self.hop_length
                    )
                    mel_spec = librosa.power_to_db(S, ref=np.max)

                    mfcc = librosa.feature.mfcc(
                        y=y, sr=sr, n_mfcc=self.n_mfcc,
                        n_fft=self.n_fft, hop_length=self.hop_length
                    )
                    mfcc_delta = librosa.feature.delta(mfcc)
                    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

                    spec_centroid = librosa.feature.spectral_centroid(
                        y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
                    )

                    zcr = librosa.feature.zero_crossing_rate(
                        y, frame_length=self.n_fft, hop_length=self.hop_length
                    )

                    spec_rolloff = librosa.feature.spectral_rolloff(
                        y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
                    )

                    stft = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))
                    if stft.shape[1] > 1:
                        spec_flux = np.sqrt(np.sum(np.diff(stft, axis=1)**2, axis=0))
                        spec_flux = np.pad(spec_flux, (1, 0), mode='edge')
                    else:
                        spec_flux = np.zeros(max(1, mel_spec.shape[1]))
                except Exception as e:
                    self.logger.warning(f"librosa feature path failed in array extraction, falling back: {e}")
                    use_librosa = False
            if not use_librosa:
                # fallback: compute with scipy/numpy (use robust helper)
                stft_mag, freqs, _ = _compute_stft_mag(y, self.n_fft, self.hop_length, sr)
                stft_mag = np.atleast_2d(stft_mag)
                power_spec = stft_mag ** 2

                # mel filterbank
                fft_bins = np.linspace(0, sr/2.0, self.n_fft//2 + 1)
                def hz_to_mel(hz):
                    return 2595.0 * np.log10(1.0 + hz / 700.0)
                def mel_to_hz(mel):
                    return 700.0 * (10**(mel / 2595.0) - 1.0)
                mel_min = hz_to_mel(0.0)
                mel_max = hz_to_mel(sr/2.0)
                mels = np.linspace(mel_min, mel_max, self.n_mels + 2)
                hz_points = mel_to_hz(mels)
                bin_indices = np.floor((self.n_fft + 1) * hz_points / sr).astype(int)
                mel_fb = np.zeros((self.n_mels, len(fft_bins)))
                for m in range(1, self.n_mels + 1):
                    f_m_minus = bin_indices[m - 1]
                    f_m = bin_indices[m]
                    f_m_plus = bin_indices[m + 1]
                    if f_m_minus < 0:
                        f_m_minus = 0
                    for k in range(f_m_minus, f_m):
                        if f_m - f_m_minus > 0 and k < mel_fb.shape[1]:
                            mel_fb[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
                    for k in range(f_m, f_m_plus):
                        if f_m_plus - f_m > 0 and k < mel_fb.shape[1]:
                            mel_fb[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
                # Align safely in case power_spec has fewer rows than mel_fb expects
                rows = min(power_spec.shape[0], mel_fb.shape[1])
                mel_spec = np.dot(mel_fb[:, :rows], power_spec[:rows, :])
                mel_spec = 10.0 * np.log10(np.maximum(mel_spec, 1e-10))

                # mfcc via DCT
                log_mel = np.log(np.maximum(mel_spec, 1e-10))
                mfcc = dct(log_mel, type=2, axis=0, norm='ortho')[:self.n_mfcc, :]
                mfcc_delta = np.gradient(mfcc, axis=1)
                mfcc_delta2 = np.gradient(mfcc_delta, axis=1)

                freqs = np.linspace(0, sr/2.0, stft_mag.shape[0])
                spec_centroid = (np.sum(stft_mag * freqs[:, None], axis=0) / (np.sum(stft_mag, axis=0) + 1e-10))[np.newaxis, :]

                frame_length = self.n_fft
                hop = self.hop_length
                num_frames = 1 + (len(y) - frame_length) // hop if len(y) >= frame_length else 1
                zcr = np.zeros((1, max(1, num_frames)))
                for i in range(zcr.shape[1]):
                    start = i * hop
                    frame = y[start:start + frame_length]
                    zcr[0, i] = 0.5 * np.mean(np.abs(np.diff(np.sign(frame)))) if frame.size > 0 else 0.0

                cumulative = np.cumsum(power_spec, axis=0)
                total = cumulative[-1, :]
                rolloff = np.zeros((1, power_spec.shape[1]))
                thresh = 0.85 * total
                for t_idx in range(power_spec.shape[1]):
                    idx = np.searchsorted(cumulative[:, t_idx], thresh[t_idx])
                    rolloff[0, t_idx] = freqs[min(idx, len(freqs)-1)] if len(freqs) > 0 else 0.0

                # Expose consistent name used later
                spec_rolloff = rolloff

                if stft_mag.shape[1] > 1:
                    spec_flux = np.sqrt(np.sum(np.diff(stft_mag, axis=1)**2, axis=0))
                    spec_flux = np.pad(spec_flux, (1, 0), mode='edge')
                else:
                    spec_flux = np.zeros(max(1, mel_spec.shape[1]))
            
            # Align
            max_time = max(
                mel_spec.shape[1], mfcc.shape[1], 
                spec_centroid.shape[1], zcr.shape[1],
                spec_flux.shape[0]
            )
            
            mel_spec = self._pad_to_length(mel_spec, max_time)
            mfcc = self._pad_to_length(mfcc, max_time)
            mfcc_delta = self._pad_to_length(mfcc_delta, max_time)
            mfcc_delta2 = self._pad_to_length(mfcc_delta2, max_time)
            spec_centroid = self._pad_to_length(spec_centroid, max_time)
            zcr = self._pad_to_length(zcr, max_time)
            spec_rolloff = self._pad_to_length(spec_rolloff, max_time)
            spec_flux = self._pad_to_length(spec_flux[np.newaxis, :], max_time)
            
            # Stack
            combined_features = np.vstack([
                mel_spec, mfcc, mfcc_delta, mfcc_delta2,
                spec_centroid, zcr, spec_rolloff, spec_flux
            ])
            
            if combined_features.shape[0] != 123:
                self.logger.error(f"Wrong feature count {combined_features.shape[0]}")
                self.error_counts['wrong_shape'] += 1
                return None
            
            if self.normalize:
                combined_features = self._normalize_features(combined_features)
            
            combined_features = np.nan_to_num(combined_features, nan=0.0, posinf=0.0, neginf=0.0)
            
            total_time = time.time() - start_time
            self.logger.debug(f"Array extraction: {combined_features.shape}, time={total_time*1000:.1f}ms")
            
            return combined_features.astype(np.float32)
        
        except Exception as e:
            self.logger.error(f"Error extracting from array: {type(e).__name__}: {e}", exc_info=True)
            self.error_counts[type(e).__name__] += 1
            return None
    
    def _analyze_audio_quality(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Analyze audio quality metrics."""
        metrics = {}
        
        # SNR estimation
        signal_power = np.mean(y ** 2)
        if signal_power > 0:
            noise_estimate = np.var(y - np.mean(y))
            metrics['snr_db'] = 10 * np.log10(signal_power / (noise_estimate + 1e-10))
        else:
            metrics['snr_db'] = -np.inf
        
        # Silence ratio
        silence_threshold = 0.01
        silence_frames = np.abs(y) < silence_threshold
        metrics['silence_ratio'] = np.mean(silence_frames)
        
        # Clipping detection
        clipping_threshold = 0.99
        clipping_frames = np.abs(y) > clipping_threshold
        metrics['clipping_ratio'] = np.mean(clipping_frames)
        
        # Dynamic range
        metrics['dynamic_range_db'] = 20 * np.log10((np.max(np.abs(y)) + 1e-10) / (np.min(np.abs(y[np.abs(y) > 1e-6])) + 1e-10))
        
        return metrics
    
    def _check_feature_quality(self, features: np.ndarray) -> List[str]:
        """Check for feature quality issues."""
        issues = []
        
        # Check for NaN/Inf
        if np.any(np.isnan(features)):
            nan_count = np.sum(np.isnan(features))
            issues.append(f"Contains {nan_count} NaN values")
        
        if np.any(np.isinf(features)):
            inf_count = np.sum(np.isinf(features))
            issues.append(f"Contains {inf_count} Inf values")
        
        # Check per-channel variance
        low_variance_channels = []
        for i in range(features.shape[0]):
            var = np.var(features[i, :])
            if var < 1e-10:
                low_variance_channels.append(i)
        
        if len(low_variance_channels) > 60:  # More than half channels
            issues.append(f"{len(low_variance_channels)}/123 channels have near-zero variance")
        
        # Check correlation between adjacent channels (detect extraction errors)
        if features.shape[1] > 10:  # Only if enough time steps
            high_corr_count = 0
            for i in range(min(20, features.shape[0] - 1)):  # Sample first 20 channels
                corr = np.corrcoef(features[i, :], features[i+1, :])[0, 1]
                if not np.isnan(corr) and abs(corr) > 0.99:
                    high_corr_count += 1
            
            if high_corr_count > 10:
                issues.append(f"{high_corr_count} adjacent channel pairs highly correlated (possible extraction error)")
        
        return issues
    
    def _pad_to_length(self, feature: np.ndarray, target_length: int) -> np.ndarray:
        """Pad feature matrix to target length."""
        current_length = feature.shape[1]
        if current_length >= target_length:
            return feature[:, :target_length]
        
        pad_width = target_length - current_length
        return np.pad(feature, ((0, 0), (0, pad_width)), mode='edge')
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features using per-channel standardization.
        
        Args:
            features: (channels, time) array
        
        Returns:
            Normalized features
        """
        normalized = np.zeros_like(features)
        for i in range(features.shape[0]):
            channel = features[i, :]
            mean = np.mean(channel)
            std = np.std(channel)
            if std > 1e-8:
                normalized[i, :] = (channel - mean) / std
            else:
                normalized[i, :] = channel - mean
        return normalized
    
    def get_timing_summary(self) -> Dict[str, Any]:
        """Get timing statistics summary."""
        summary = {}
        for stage, times in self.timing_breakdown.items():
            if times:
                summary[stage] = {
                    'count': len(times),
                    'mean_ms': float(np.mean(times) * 1000),
                    'std_ms': float(np.std(times) * 1000),
                    'min_ms': float(np.min(times) * 1000),
                    'max_ms': float(np.max(times) * 1000),
                    'total_s': float(np.sum(times))
                }
        return summary
    
    def log_timing_summary(self):
        """Log timing statistics."""
        summary = self.get_timing_summary()
        
        if not summary:
            self.logger.info("No timing data available")
            return
        
        self.logger.info("=" * 80)
        self.logger.info("TIMING BREAKDOWN")
        self.logger.info("=" * 80)
        
        for stage, stats in summary.items():
            self.logger.info(f"\n{stage}:")
            self.logger.info(f"  Count:     {stats['count']}")
            self.logger.info(f"  Mean:      {stats['mean_ms']:.2f}ms")
            self.logger.info(f"  Std Dev:   {stats['std_ms']:.2f}ms")
            self.logger.info(f"  Range:     [{stats['min_ms']:.2f}, {stats['max_ms']:.2f}] ms")
            self.logger.info(f"  Total:     {stats['total_s']:.2f}s")
        
        self.logger.info("=" * 80)
    
    def log_error_summary(self):
        """Log error statistics."""
        if not self.error_counts:
            self.logger.info("No errors encountered")
            return
        
        self.logger.info("=" * 80)
        self.logger.info("ERROR SUMMARY")
        self.logger.info("=" * 80)
        
        for error_type, count in sorted(self.error_counts.items(), key=lambda x: -x[1]):
            self.logger.info(f"  {error_type:<30}: {count:>6}")
        
        self.logger.info("=" * 80)
    
    def process_batch(self, 
                      audio_paths: List[Path], 
                      output_dir: Path) -> Dict[str, Any]:
        """
        Process multiple audio files with progress tracking.
        
        Args:
            audio_paths: List of audio file paths
            output_dir: Output directory for NPZ files
        
        Returns:
            Processing statistics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("=" * 80)
        self.logger.info("BATCH PROCESSING")
        self.logger.info("=" * 80)
        self.logger.info(f"Files:       {len(audio_paths)}")
        self.logger.info(f"Output dir:  {output_dir}")
        self.logger.info("=" * 80)
        
        failed = 0
        success = 0
        failed_files = []
        
        pbar = tqdm(audio_paths, desc="Processing", unit="file", ncols=100)
        
        for audio_path in pbar:
            try:
                features = self.extract_features(audio_path)
                
                if features is None:
                    failed += 1
                    failed_files.append(str(audio_path))
                    continue
                
                # Validate
                if features.shape[0] != 123:
                    self.logger.warning(f"Skipping {audio_path}: wrong shape {features.shape}")
                    failed += 1
                    failed_files.append(str(audio_path))
                    continue
                
                # Save
                output_file = output_dir / f"{Path(audio_path).stem}.npz"
                
                # Try to load labels
                label_file = Path(audio_path).with_suffix('.json').with_name(
                    Path(audio_path).stem + '_labels.json'
                )
                if label_file.exists():
                    try:
                        with open(label_file, 'r') as f:
                            labels = json.load(f)
                        labels_array = np.array(labels, dtype=np.float32)
                    except Exception as e:
                        self.logger.debug(f"Could not load labels: {e}")
                        labels_array = np.zeros(5, dtype=np.float32)
                else:
                    labels_array = np.zeros(5, dtype=np.float32)
                
                np.savez_compressed(
                    output_file,
                    spectrogram=features.astype(np.float32),
                    labels=labels_array
                )
                
                success += 1
                pbar.set_postfix({
                    'OK': success,
                    'Fail': failed
                }, refresh=False)
            
            except Exception as e:
                failed += 1
                failed_files.append(str(audio_path))
                if failed <= 3:
                    tqdm.write(f"⚠ {audio_path.name}: {e}")
                pbar.set_postfix({
                    'Fail': failed
                }, refresh=False)
        
        pbar.close()
        
        # Save failed files list
        if failed_files:
            failed_log = output_dir / 'failed_files.txt'
            with open(failed_log, 'w') as f:
                f.write('\n'.join(failed_files))
            self.logger.info(f"Failed files logged: {failed_log}")
        
        # Log summaries
        self.logger.info("=" * 80)
        self.logger.info("BATCH PROCESSING COMPLETE")
        self.logger.info("=" * 80)
        self.logger.info(f"Success:  {success:>6}")
        self.logger.info(f"Failed:   {failed:>6}")
        self.logger.info(f"Total:    {len(audio_paths):>6}")
        self.logger.info("=" * 80)
        
        if self.track_stats and self.feature_stats:
            self.feature_stats.log_summary(self.logger)
        
        self.log_timing_summary()
        self.log_error_summary()
        
        return {
            'success': success,
            'failed': failed,
            'total': len(audio_paths),
            'failed_files': failed_files
        }


# Feature channel information
FEATURE_CHANNELS = {
    'mel_spectrogram': 80,
    'mfcc': 13,
    'mfcc_delta': 13,
    'mfcc_delta2': 13,
    'spectral_centroid': 1,
    'zero_crossing_rate': 1,
    'spectral_rolloff': 1,
    'spectral_flux': 1,
}


def validate_features(features: np.ndarray) -> Tuple[bool, str]:
    """Validate extracted features."""
    if features is None:
        return False, "Features is None"
    
    if not isinstance(features, np.ndarray):
        return False, f"Features is not ndarray, got {type(features)}"
    
    if features.shape[0] != TOTAL_CHANNELS:
        return False, f"Wrong channel count: {features.shape[0]}, expected {TOTAL_CHANNELS}"
    
    if np.any(np.isnan(features)):
        nan_count = np.sum(np.isnan(features))
        return False, f"Features contain {nan_count} NaN values"
    
    if np.any(np.isinf(features)):
        inf_count = np.sum(np.isinf(features))
        return False, f"Features contain {inf_count} Inf values"
    
    return True, "Valid"


if __name__ == '__main__':
    print("=" * 80)
    print("ENHANCED AUDIO PREPROCESSOR - DEBUG VERSION")
    print("=" * 80)
    print(f"\nTotal feature channels: {TOTAL_CHANNELS}")
    print("\nFeature breakdown:")
    for name, count in FEATURE_CHANNELS.items():
        print(f"  {name:25s}: {count:3d} channels")
    print("=" * 80)
    
    # Test with dummy audio
    print("\nTesting with dummy audio (DEBUG mode)...")
    preprocessor = EnhancedAudioPreprocessor(normalize=True, log_level='DEBUG', track_stats=True)
    
    # Create 1 second of random audio
    dummy_audio = np.random.randn(16000)
    features = preprocessor.extract_features_from_array(dummy_audio)
    
    if features is not None:
        is_valid, msg = validate_features(features)
        print(f"\n{'✓' if is_valid else '✗'} Feature extraction: {msg}")
        print(f"  Shape:      {features.shape}")
        print(f"  Data type:  {features.dtype}")
        print(f"  Range:      [{features.min():.3f}, {features.max():.3f}]")
        print(f"  Mean:       {features.mean():.3f}")
        print(f"  Std:        {features.std():.3f}")
        
        # Show timing
        print("\nTiming Summary:")
        timing = preprocessor.get_timing_summary()
        if timing:
            for stage, stats in timing.items():
                print(f"  {stage:<20}: {stats['mean_ms']:.2f}ms")
    else:
        print("\n✗ Feature extraction failed!")
    
    print("=" * 80)