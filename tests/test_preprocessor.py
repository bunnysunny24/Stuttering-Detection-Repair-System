import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from Models.enhanced_audio_preprocessor import EnhancedAudioPreprocessor


def test_dummy_audio():
    pre = EnhancedAudioPreprocessor(track_stats=False)
    dummy = np.random.randn(16000).astype(np.float32)
    feats = pre.extract_features_from_array(dummy)
    assert feats is not None, "Feature extraction returned None"
    assert feats.shape[0] == 123, f"Expected 123 channels, got {feats.shape[0]}"
    print('test_dummy_audio: OK', feats.shape)


def test_short_audio():
    pre = EnhancedAudioPreprocessor(track_stats=False)
    short = np.random.randn(200).astype(np.float32)
    feats = pre.extract_features_from_array(short)
    # short audio should be rejected (too short)
    assert feats is None, 'Short audio should return None'
    print('test_short_audio: OK')


if __name__ == '__main__':
    test_dummy_audio()
    test_short_audio()
