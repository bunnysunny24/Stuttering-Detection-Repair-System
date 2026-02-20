import numpy as np
from Models.calibrate_thresholds import calibrate_thresholds


def test_calibrate_thresholds_synth():
    # synthetic probs and labels for 5 classes
    N = 200
    C = 5
    rng = np.random.RandomState(0)
    probs = rng.rand(N, C)
    # create imbalanced labels
    labels = (probs > 0.95).astype(int)
    thresholds, metrics = calibrate_thresholds(probs, labels)
    assert len(thresholds) == C
    assert isinstance(metrics, dict)


if __name__ == '__main__':
    test_calibrate_thresholds_synth()
    print('test_calibrate_thresholds_synth: OK')
