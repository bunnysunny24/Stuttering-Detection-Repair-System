# Central constants for the stuttering pipeline
# Keep small, stable values here to avoid duplication across modules
TOTAL_CHANNELS = 123  # 80 mel + 13 mfcc + 13 delta + 13 delta2 + 4 spectral
NUM_CLASSES = 5
DEFAULT_SAMPLE_RATE = 16000
# Augmentation probabilities
AUG_TIME_MASK_P = 0.3
AUG_FREQ_MASK_P = 0.3
AUG_NOISE_P = 0.2
AUG_STRETCH_P = 0.2
# Threshold optimization
THRESH_SEARCH_START = 0.2
THRESH_SEARCH_END = 0.9
THRESH_SEARCH_STEP = 0.02
# Scheduler and training
SCHEDULER_PATIENCE = 5
THRESHOLD_OPT_EPOCHS = 5
