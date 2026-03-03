"""
Production Stuttering Detection Pipeline.

Usage:
    # Detect stuttering in a single file:
    python Models/detect.py --audio path/to/audio.wav

    # Process a directory:
    python Models/detect.py --audio path/to/folder/ --output results.json

    # Use custom checkpoint:
    python Models/detect.py --audio file.wav --checkpoint Models/checkpoints/w2v_finetune_BEST.pth

Output JSON format:
    {
        "file": "audio.wav",
        "duration_sec": 3.2,
        "binary_detection": {
            "is_stuttered": true,
            "confidence": 0.94
        },
        "stutter_types": {
            "Prolongation": {"detected": false, "confidence": 0.12},
            "Block": {"detected": true, "confidence": 0.87},
            "SoundRep": {"detected": false, "confidence": 0.05},
            "WordRep": {"detected": false, "confidence": 0.09},
            "Interjection": {"detected": true, "confidence": 0.73}
        },
        "segments": [  // For long audio with sliding window
            {"start_sec": 0.0, "end_sec": 3.0, "is_stuttered": true, "confidence": 0.94, ...}
        ]
    }
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
import warnings
import time

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

from model_w2v_finetune import Wav2VecFineTuneClassifier
from constants import NUM_CLASSES, DEFAULT_SAMPLE_RATE

STUTTER_CLASSES = ['Prolongation', 'Block', 'SoundRep', 'WordRep', 'Interjection']


def load_audio(path, target_sr=16000):
    """Load audio file, convert to mono float32 at target sample rate."""
    import scipy.io.wavfile as wav_io

    sr, audio = wav_io.read(str(path))

    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        duration = len(audio) / sr
        target_len = int(duration * target_sr)
        if target_len > 0:
            audio = np.interp(
                np.linspace(0, len(audio) - 1, target_len),
                np.arange(len(audio)),
                audio
            ).astype(np.float32)
        sr = target_sr

    return audio, sr


class StutterDetector:
    """Production stuttering detection with hierarchical output.

    Binary detection (stutter vs fluent) → 90+ F1
    Type classification (5 types) → detail for detected stutters
    Sliding window for long audio → per-segment results
    """

    def __init__(self, checkpoint_path, device='cpu',
                 model_name='facebook/wav2vec2-large',
                 freeze_layers=12, hidden_dim=256, lstm_hidden=128):
        self.device = torch.device(device)

        print(f"Loading model from {checkpoint_path}...")
        self.model = Wav2VecFineTuneClassifier(
            model_name=model_name, n_classes=NUM_CLASSES,
            freeze_layers=freeze_layers, hidden_dim=hidden_dim,
            lstm_hidden=lstm_hidden, dropout=0.0,  # No dropout at inference
            use_gradient_checkpointing=False  # No gc at inference
        )

        ckpt = torch.load(str(checkpoint_path), map_location=self.device)
        if 'model_state' in ckpt:
            self.model.load_state_dict(ckpt['model_state'], strict=False)
        else:
            self.model.load_state_dict(ckpt, strict=False)

        self.model.to(self.device)
        self.model.eval()

        # Load thresholds from checkpoint
        self.binary_threshold = ckpt.get('binary_threshold', 0.5)
        self.class_thresholds = np.array(
            ckpt.get('thresholds', [0.5] * NUM_CLASSES), dtype=np.float32
        )

        print(f"  Binary threshold: {self.binary_threshold:.2f}")
        print(f"  Class thresholds: {[f'{t:.2f}' for t in self.class_thresholds]}")
        if 'best_binary_f1' in ckpt:
            print(f"  Training best binary F1: {ckpt['best_binary_f1']:.4f}")
        print("  Model ready.")

    @torch.no_grad()
    def detect(self, audio_path, window_sec=3.0, hop_sec=1.5, min_audio_sec=0.5):
        """Detect stuttering in an audio file.

        Args:
            audio_path: Path to WAV file
            window_sec: Sliding window size in seconds (for long audio)
            hop_sec: Hop between windows in seconds
            min_audio_sec: Minimum audio length to process

        Returns:
            dict with binary detection, type classification, and per-segment results
        """
        audio, sr = load_audio(audio_path, DEFAULT_SAMPLE_RATE)
        duration = len(audio) / sr

        if duration < min_audio_sec:
            return {
                'file': str(audio_path),
                'duration_sec': round(duration, 3),
                'error': f'Audio too short ({duration:.2f}s < {min_audio_sec}s)',
                'binary_detection': {'is_stuttered': False, 'confidence': 0.0},
                'stutter_types': {c: {'detected': False, 'confidence': 0.0} for c in STUTTER_CLASSES}
            }

        window_samples = int(window_sec * sr)
        hop_samples = int(hop_sec * sr)

        # For short audio (< 2x window), process as single segment
        if len(audio) <= window_samples * 1.5:
            result = self._detect_segment(audio)
            return {
                'file': str(audio_path),
                'duration_sec': round(duration, 3),
                'binary_detection': {
                    'is_stuttered': bool(result['binary_prob'] > self.binary_threshold),
                    'confidence': round(float(result['binary_prob']), 4)
                },
                'stutter_types': {
                    STUTTER_CLASSES[i]: {
                        'detected': bool(result['class_probs'][i] > self.class_thresholds[i]),
                        'confidence': round(float(result['class_probs'][i]), 4)
                    }
                    for i in range(NUM_CLASSES)
                }
            }

        # Sliding window for long audio
        segments = []
        all_binary_probs = []
        all_class_probs = []

        for start in range(0, max(1, len(audio) - window_samples + 1), hop_samples):
            end = min(start + window_samples, len(audio))
            segment_audio = audio[start:end]

            if len(segment_audio) < sr * min_audio_sec:
                continue

            result = self._detect_segment(segment_audio)

            start_sec = round(start / sr, 3)
            end_sec = round(end / sr, 3)

            seg_info = {
                'start_sec': start_sec,
                'end_sec': end_sec,
                'is_stuttered': bool(result['binary_prob'] > self.binary_threshold),
                'confidence': round(float(result['binary_prob']), 4),
                'stutter_types': {
                    STUTTER_CLASSES[i]: {
                        'detected': bool(result['class_probs'][i] > self.class_thresholds[i]),
                        'confidence': round(float(result['class_probs'][i]), 4)
                    }
                    for i in range(NUM_CLASSES)
                }
            }
            segments.append(seg_info)
            all_binary_probs.append(result['binary_prob'])
            all_class_probs.append(result['class_probs'])

        # Aggregate: take max confidence across segments (if ANY segment has stutter, report it)
        if len(all_binary_probs) > 0:
            agg_binary = float(np.max(all_binary_probs))
            agg_class = np.max(np.stack(all_class_probs), axis=0)
        else:
            agg_binary = 0.0
            agg_class = np.zeros(NUM_CLASSES)

        return {
            'file': str(audio_path),
            'duration_sec': round(duration, 3),
            'binary_detection': {
                'is_stuttered': bool(agg_binary > self.binary_threshold),
                'confidence': round(agg_binary, 4)
            },
            'stutter_types': {
                STUTTER_CLASSES[i]: {
                    'detected': bool(agg_class[i] > self.class_thresholds[i]),
                    'confidence': round(float(agg_class[i]), 4)
                }
                for i in range(NUM_CLASSES)
            },
            'segments': segments,
            'n_segments': len(segments),
            'n_stuttered_segments': sum(1 for s in segments if s['is_stuttered'])
        }

    def _detect_segment(self, audio):
        """Run model on a single audio segment."""
        if len(audio) < 400:
            audio = np.pad(audio, (0, 400 - len(audio)), mode='constant')

        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        mask = torch.ones(1, audio_tensor.shape[1], dtype=torch.long, device=self.device)

        logits, binary_logit = self.model(audio_tensor, attention_mask=mask)

        class_probs = torch.sigmoid(logits).cpu().numpy()[0]
        binary_prob = torch.sigmoid(binary_logit).cpu().numpy()[0, 0]

        return {
            'class_probs': class_probs,
            'binary_prob': float(binary_prob)
        }

    def detect_batch(self, audio_paths, **kwargs):
        """Process multiple audio files."""
        results = []
        for i, path in enumerate(audio_paths):
            try:
                r = self.detect(path, **kwargs)
                results.append(r)
                status = 'STUTTER' if r['binary_detection']['is_stuttered'] else 'FLUENT'
                conf = r['binary_detection']['confidence']
                print(f"  [{i+1}/{len(audio_paths)}] {Path(path).name}: "
                      f"{status} ({conf:.2f})")
            except Exception as e:
                results.append({
                    'file': str(path),
                    'error': str(e),
                    'binary_detection': {'is_stuttered': False, 'confidence': 0.0}
                })
                print(f"  [{i+1}/{len(audio_paths)}] {Path(path).name}: ERROR: {e}")
        return results


def print_result(result, verbose=False):
    """Pretty-print a detection result."""
    fname = Path(result['file']).name
    duration = result.get('duration_sec', 0)

    if 'error' in result and result['error']:
        print(f"  {fname} ({duration:.1f}s): ERROR - {result['error']}")
        return

    bd = result['binary_detection']
    status = 'STUTTERED' if bd['is_stuttered'] else 'FLUENT'
    conf = bd['confidence']

    print(f"\n  {fname} ({duration:.1f}s): {status} (confidence: {conf:.1%})")

    if bd['is_stuttered'] or verbose:
        print(f"  Stutter types:")
        for name, info in result.get('stutter_types', {}).items():
            marker = 'X' if info['detected'] else ' '
            print(f"    [{marker}] {name:18s} {info['confidence']:.1%}")

    if 'segments' in result:
        n_seg = result.get('n_segments', 0)
        n_stut = result.get('n_stuttered_segments', 0)
        print(f"  Segments: {n_stut}/{n_seg} stuttered")


def main():
    parser = argparse.ArgumentParser(
        description='Production Stuttering Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Models/detect.py --audio test.wav
  python Models/detect.py --audio datasets/clips/stuttering-clips/clips/ --output results.json
  python Models/detect.py --audio test.wav --checkpoint Models/checkpoints/w2v_finetune_BEST.pth
        """
    )
    parser.add_argument('--audio', required=True,
                        help='Path to WAV file or directory of WAV files')
    parser.add_argument('--checkpoint', default='Models/checkpoints/w2v_finetune_BEST.pth',
                        help='Model checkpoint path')
    parser.add_argument('--output', default=None,
                        help='Output JSON file path (default: print to console)')
    parser.add_argument('--model-name', default='facebook/wav2vec2-large')
    parser.add_argument('--freeze-layers', type=int, default=12)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--lstm-hidden', type=int, default=128)
    parser.add_argument('--window-sec', type=float, default=3.0,
                        help='Sliding window size for long audio (seconds)')
    parser.add_argument('--hop-sec', type=float, default=1.5,
                        help='Hop between windows (seconds)')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--omp-threads', type=int, default=6)

    args = parser.parse_args()

    os.environ['OMP_NUM_THREADS'] = str(args.omp_threads)
    os.environ['MKL_NUM_THREADS'] = str(args.omp_threads)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    torch.set_num_threads(args.omp_threads)

    # Check checkpoint exists
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        print("Train a model first with: powershell -File scripts/run_finetune_pipeline.ps1")
        sys.exit(1)

    # Initialize detector
    detector = StutterDetector(
        checkpoint_path=ckpt_path,
        model_name=args.model_name,
        freeze_layers=args.freeze_layers,
        hidden_dim=args.hidden_dim,
        lstm_hidden=args.lstm_hidden
    )

    # Collect audio files
    audio_path = Path(args.audio)
    if audio_path.is_file():
        audio_files = [audio_path]
    elif audio_path.is_dir():
        audio_files = sorted(audio_path.glob('*.wav'))
        print(f"Found {len(audio_files)} WAV files in {audio_path}")
    else:
        print(f"ERROR: {audio_path} not found")
        sys.exit(1)

    if not audio_files:
        print("No WAV files found")
        sys.exit(1)

    # Run detection
    print(f"\n{'='*60}")
    print(f"  STUTTERING DETECTION — {len(audio_files)} file(s)")
    print(f"{'='*60}")

    t0 = time.time()
    results = detector.detect_batch(
        audio_files, window_sec=args.window_sec, hop_sec=args.hop_sec
    )
    elapsed = time.time() - t0

    # Print results
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")

    n_stuttered = sum(1 for r in results if r['binary_detection']['is_stuttered'])
    n_fluent = len(results) - n_stuttered
    print(f"\n  Total: {len(results)} files | "
          f"Stuttered: {n_stuttered} | Fluent: {n_fluent}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/max(len(results),1):.2f}s/file)")

    for r in results:
        print_result(r, verbose=args.verbose)

    # Save JSON
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'checkpoint': str(ckpt_path),
                'n_files': len(results),
                'n_stuttered': n_stuttered,
                'n_fluent': n_fluent,
                'results': results
            }, f, indent=2)
        print(f"\n  Results saved to {output_path}")

    print()


if __name__ == '__main__':
    main()
