# Central constants for the stuttering pipeline
# Keep small, stable values here to avoid duplication across modules
TOTAL_CHANNELS = 123  # 80 mel + 13 mfcc + 13 delta + 13 delta2 + 4 spectral
NUM_CLASSES = 5
DEFAULT_SAMPLE_RATE = 16000
# Augmentation probabilities
AUG_TIME_MASK_P = 0.6
AUG_FREQ_MASK_P = 0.5
AUG_NOISE_P = 0.4
AUG_STRETCH_P = 0.25
# Additional augmentations
AUG_PITCH_P = 0.25
AUG_SNR_P = 0.25
# Threshold optimization
# Wider range to find optimal per-class thresholds (was 0.5-0.95, too narrow)
THRESH_SEARCH_START = 0.25
THRESH_SEARCH_END = 0.85
THRESH_SEARCH_STEP = 0.025
# Scheduler and training
SCHEDULER_PATIENCE = 7
THRESHOLD_OPT_EPOCHS = 5
