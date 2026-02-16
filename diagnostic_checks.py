import sys
from pathlib import Path
import traceback
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

outputs = []

# 1) Label mapping check
try:
    import extract_features_90plus as ef90
    pipeline = ef90.FeatureExtractionPipeline(clips_dir=str(ROOT.parent / 'datasets' / 'clips' / 'stuttering-clips' / 'clips'), output_dir=str(ROOT.parent / 'datasets' / 'features'), label_dir=str(ROOT.parent / 'datasets'))
    label_keys = list(pipeline.label_map.keys())[:50]
    # list some audio stems
    audio_dir = Path(pipeline.clips_dir)
    audio_stems = [p.stem for p in sorted(audio_dir.glob('**/*.wav'))][:50]
    # unmatched counts
    label_set = set(label_keys)
    audio_set = set(audio_stems)
    matched = len(label_set & audio_set)
    outputs.append(f"Label keys (sample 50): {label_keys[:10]}... total_labels={len(pipeline.label_map)}")
    outputs.append(f"Audio stems (sample 50): {audio_stems[:10]}...")
    outputs.append(f"Matched label/audio stems: {matched} / {min(len(label_set), len(audio_set))}")
except Exception as e:
    outputs.append('Label mapping check failed: ' + repr(e))
    outputs.append(traceback.format_exc())

# 2) Feature extraction smoke test
try:
    from enhanced_audio_preprocessor import EnhancedAudioPreprocessor
    pre = EnhancedAudioPreprocessor(sr=16000)
    # pick 5 audio files
    clips_root = Path(ROOT.parent / 'datasets' / 'clips' / 'stuttering-clips' / 'clips')
    wavs = sorted(clips_root.glob('**/*.wav'))[:5]
    feat_info = []
    for p in wavs:
        try:
            f = pre.extract_features(str(p))
            if f is None:
                feat_info.append((p.name, 'None'))
            else:
                feat_info.append((p.name, f.shape, 'nan', int((f!=f).sum()), 'inf', int((~np.isfinite(f)).sum())))
        except Exception as e:
            feat_info.append((p.name, 'error', repr(e)))
    outputs.append('Feature smoke test results:')
    for info in feat_info:
        outputs.append(str(info))
except Exception as e:
    outputs.append('Feature extraction check failed: ' + repr(e))
    outputs.append(traceback.format_exc())

# 3) Model forward pass
try:
    import torch
    from constants import TOTAL_CHANNELS
    from model_improved_90plus import ImprovedStutteringCNN
    model = ImprovedStutteringCNN(n_channels=TOTAL_CHANNELS, n_classes=5)
    model.eval()
    T = 128
    x = torch.randn(1, TOTAL_CHANNELS, T)
    with torch.no_grad():
        out = model(x)
    outputs.append(f'Model forward: input (1,{TOTAL_CHANNELS},{T}) -> output {tuple(out.shape)}')
except Exception as e:
    outputs.append('Model forward check failed: ' + repr(e))
    outputs.append(traceback.format_exc())

# Print concise report
print('\n'.join(str(o) for o in outputs))
