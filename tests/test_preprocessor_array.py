import numpy as np
from Models.enhanced_audio_preprocessor import EnhancedAudioPreprocessor


def test_preprocessor_various_lengths():
    pre = EnhancedAudioPreprocessor(sr=16000, n_fft=512, hop_length=160, track_stats=False)
    # normal length (1s)
    arr = np.random.randn(16000)
    f = pre.extract_features_from_array(arr, sr=16000)
    assert f is not None
    assert f.shape[0] == 123

    # exactly n_fft length
    arr2 = np.random.randn(512)
    f2 = pre.extract_features_from_array(arr2, sr=16000)
    # may be None if considered too short by policy; ensure no crash
    assert (f2 is None) or (isinstance(f2, np.ndarray) and f2.shape[0] == 123)


if __name__ == '__main__':
    test_preprocessor_various_lengths()
    print('test_preprocessor_array: OK')
