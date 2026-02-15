"""
ENHANCED FEATURE EXTRACTION FOR 90+ ACCURACY
Replaces simple mel-spectrogram with rich feature set:
- Mel-spectrogram (80 channels)
- MFCC (13 channels)
- MFCC Delta (13 channels)  
- MFCC Delta-Delta (13 channels)
- Spectral features (4 channels)
Total: 123 channels

This replaces: Models/preprocess_data.py feature extraction
"""

import numpy as np
import librosa
from pathlib import Path
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class EnhancedAudioPreprocessor:
    """Extract 123-channel feature set for stuttering detection."""
    
    def __init__(self, sr=16000, n_mels=80, n_mfcc=13, hop_length=160, n_fft=512):
        self.sr = sr
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.n_fft = n_fft
    
    def extract_features(self, audio_path):
        """
        Extract all 123 features from audio file.
        Returns: numpy array of shape (123, time_steps)
        """
        try:
            # Load audio
            y, sr = librosa.load(str(audio_path), sr=self.sr)
            
            if len(y) == 0:
                return None
            
            # 1. MEL-SPECTROGRAM (80 channels)
            S = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=self.n_mels, 
                n_fft=self.n_fft, hop_length=self.hop_length
            )
            mel_spec = librosa.power_to_db(S, ref=np.max)
            
            # 2. MFCC (13 channels)
            mfcc = librosa.feature.mfcc(
                y=y, sr=sr, n_mfcc=self.n_mfcc,
                n_fft=self.n_fft, hop_length=self.hop_length
            )
            
            # 3. MFCC DELTA - velocity of MFCC (13 channels)
            # Captures how quickly MFCC changes (important for stutters)
            mfcc_delta = librosa.feature.delta(mfcc)
            
            # 4. MFCC DELTA-DELTA - acceleration of MFCC (13 channels)
            # Captures rapid changes (very important for stutters!)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            
            # 5. SPECTRAL CENTROID (1 channel)
            # Frequency center of mass - important for consonant detection
            spec_centroid = librosa.feature.spectral_centroid(
                y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
            )
            
            # 6. ZERO-CROSSING RATE (1 channel)
            # Important for fricative detection (hissing sounds)
            zcr = librosa.feature.zero_crossing_rate(
                y, frame_length=self.n_fft, hop_length=self.hop_length
            )
            
            # 7. SPECTRAL ROLLOFF (1 channel)
            # Brightness of sound - helps with fricative classification
            spec_rolloff = librosa.feature.spectral_rolloff(
                y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
            )
            
            # 8. SPECTRAL FLUX (1 channel)
            # Detects abrupt changes (stutters!)
            stft = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))
            spec_flux = np.sqrt(np.sum(np.diff(stft, axis=1)**2, axis=0))
            spec_flux = np.pad(spec_flux, (1, 0), mode='edge')
            
            # Get maximum time dimension
            max_time = max(
                mel_spec.shape[1], mfcc.shape[1], 
                spec_centroid.shape[1], zcr.shape[1]
            )
            
            # Pad all features to same length
            mel_spec = self._pad_to_length(mel_spec, max_time)
            mfcc = self._pad_to_length(mfcc, max_time)
            mfcc_delta = self._pad_to_length(mfcc_delta, max_time)
            mfcc_delta2 = self._pad_to_length(mfcc_delta2, max_time)
            spec_centroid = self._pad_to_length(spec_centroid, max_time)
            zcr = self._pad_to_length(zcr, max_time)
            spec_rolloff = self._pad_to_length(spec_rolloff, max_time)
            spec_flux = self._pad_to_length(spec_flux[np.newaxis, :], max_time)
            
            # Stack all features: (80+13+13+13+1+1+1+1, time) = (123, time)
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
            
            return combined_features  # Shape: (123, time_steps)
        
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def extract_features_from_array(self, audio_array, sr=None):
        """
        Extract 123 features from numpy audio array.
        Useful for processing pre-loaded audio in real-time.
        
        Args:
            audio_array: numpy array of audio samples
            sr: sample rate (defaults to self.sr if None)
        
        Returns: numpy array of shape (123, time_steps)
        """
        try:
            if sr is None:
                sr = self.sr
            
            if len(audio_array) == 0:
                return None
            
            y = audio_array
            
            # 1. MEL-SPECTROGRAM (80 channels)
            S = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=self.n_mels, 
                n_fft=self.n_fft, hop_length=self.hop_length
            )
            mel_spec = librosa.power_to_db(S, ref=np.max)
            
            # 2. MFCC (13 channels)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            
            # 3. MFCC DELTA (13 channels)
            mfcc_delta = librosa.feature.delta(mfcc)
            
            # 4. MFCC DELTA-DELTA / ACCELERATION (13 channels)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            
            # 5. SPECTRAL CENTROID (1 channel)
            spec_centroid = librosa.feature.spectral_centroid(
                y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
            )
            
            # 6. ZERO-CROSSING RATE (1 channel)
            zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)
            
            # 7. SPECTRAL ROLLOFF (1 channel)
            spec_rolloff = librosa.feature.spectral_rolloff(
                y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
            )
            
            # 8. SPECTRAL FLUX (1 channel)
            stft = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))
            spec_flux = np.sqrt(np.sum(np.diff(stft, axis=1)**2, axis=0))
            spec_flux = np.pad(spec_flux, (1, 0), mode='edge')
            
            # Get maximum time dimension
            max_time = max(
                mel_spec.shape[1], mfcc.shape[1], 
                spec_centroid.shape[1], zcr.shape[1]
            )
            
            # Pad all features to same length
            mel_spec = self._pad_to_length(mel_spec, max_time)
            mfcc = self._pad_to_length(mfcc, max_time)
            mfcc_delta = self._pad_to_length(mfcc_delta, max_time)
            mfcc_delta2 = self._pad_to_length(mfcc_delta2, max_time)
            spec_centroid = self._pad_to_length(spec_centroid, max_time)
            zcr = self._pad_to_length(zcr, max_time)
            spec_rolloff = self._pad_to_length(spec_rolloff, max_time)
            spec_flux = self._pad_to_length(spec_flux[np.newaxis, :], max_time)
            
            # Stack all features: (80+13+13+13+1+1+1+1, time) = (123, time)
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
            
            return combined_features  # Shape: (123, time_steps)
        
        except Exception as e:
            print(f"Error extracting features from array: {e}")
            return None
    
    def _pad_to_length(self, feature, target_length):
        """Pad feature matrix to target length."""
        current_length = feature.shape[1]
        if current_length >= target_length:
            return feature[:, :target_length]
        
        pad_width = target_length - current_length
        return np.pad(feature, ((0, 0), (0, pad_width)), mode='edge')
    
    def process_batch(self, audio_paths, output_dir):
        """
        Process multiple audio files and save as NPZ.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        failed = 0
        success = 0
        
        pbar = tqdm(audio_paths, desc="Processing audio files")
        for audio_path in pbar:
            try:
                features = self.extract_features(audio_path)
                
                if features is None:
                    failed += 1
                    continue
                
                # Save as NPZ file
                output_file = output_dir / f"{Path(audio_path).stem}.npz"
                
                # Load corresponding labels if available
                label_file = Path(str(audio_path).replace('.wav', '_labels.json').replace('.flac', '_labels.json'))
                if label_file.exists():
                    with open(label_file, 'r') as f:
                        labels = json.load(f)
                    labels_array = np.array(labels, dtype=np.float32)
                else:
                    labels_array = np.zeros(5, dtype=np.float32)
                
                np.savez(
                    output_file,
                    spectrogram=features.astype(np.float32),
                    labels=labels_array
                )
                
                success += 1
                pbar.set_postfix({'Success': success, 'Failed': failed})
            
            except Exception as e:
                failed += 1
                pbar.set_postfix({'Success': success, 'Failed': failed, 'Error': str(e)[:30]})
        
        pbar.close()
        
        return {
            'success': success,
            'failed': failed,
            'total': len(audio_paths)
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

TOTAL_CHANNELS = sum(FEATURE_CHANNELS.values())
