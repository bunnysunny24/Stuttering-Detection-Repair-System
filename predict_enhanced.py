"""
Enhanced Inference Script - Supports both SimpleCNN and EnhancedStutteringCNN

Features:
- Switch between models easily
- Batch processing for efficiency
- Confidence scoring
- Model comparison
- Better error handling

Usage:
    # Run inference with enhanced model
    python predict_enhanced.py --model enhanced --input audio.wav --output output.json
    
    # Batch processing
    python predict_enhanced.py --model enhanced --batch-dir audio_clips/ --output-dir results/
    
    # Compare models
    python predict_enhanced.py --compare --input audio.wav
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import sys

# Import models
from model_cnn import SimpleCNN
from model_enhanced import EnhancedStutteringCNN
from features import waveform_to_logmel
from asr_whisper import transcribe_to_words
from repair import repair_audio
import soundfile as sf
import librosa


class StutterDetector:
    """Unified interface for stuttering detection."""
    
    STUTTER_CLASSES = [
        'Prolongation',
        'Block',
        'Sound Repetition',
        'Word Repetition',
        'Interjection'
    ]
    
    def __init__(self, model_path, model_type='enhanced', device='cpu'):
        """
        Initialize detector.
        
        Args:
            model_path: Path to trained model weights
            model_type: 'enhanced' or 'simple'
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        self.model_type = model_type
        self.model = self._load_model(model_path)
        print(f"✓ Loaded {model_type} model from {model_path}")
    
    def _load_model(self, model_path):
        """Load model architecture and weights."""
        if self.model_type == 'enhanced':
            model = EnhancedStutteringCNN(n_mels=80, n_classes=5)
        else:
            model = SimpleCNN(n_mels=80, n_classes=5)
        
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        
        return model
    
    def detect_stuttering(self, audio_path, threshold=0.3, window_size=256, hop_size=128):
        """
        Detect stuttering in audio file.
        
        Args:
            audio_path: Path to audio file
            threshold: Classification threshold (0-1) - 0.3 for enhanced model
            window_size: Window size in frames (256 = ~2.56s at 16kHz/160hop)
            hop_size: Hop size in frames
        
        Returns:
            dict with detection results
        """
        # Load audio
        y, sr = librosa.load(str(audio_path), sr=16000)
        duration = len(y) / sr
        
        # Compute features (should match training preprocessing)
        X_mel = waveform_to_logmel(y, sr=sr, n_mels=80)
        
        # Use frame-based window (not time-based) to match training
        n_frames = X_mel.shape[1]
        target_frames = window_size  # Use 256 frames like training
        
        # Sliding window inference
        detections = []
        step = hop_size  # Use 128 frame hop
        
        for start_frame in range(0, max(1, n_frames - target_frames + 1), step):
            end_frame = min(start_frame + target_frames, n_frames)
            window = X_mel[:, start_frame:end_frame]
            
            # Pad if necessary
            if window.shape[1] < target_frames:
                padding = target_frames - window.shape[1]
                window = np.pad(window, ((0, 0), (0, padding)), mode='constant')
            
            # Convert to tensor - format: (batch=1, channels=1, mels=80, time=256)
            X_tensor = torch.from_numpy(window).unsqueeze(0).unsqueeze(0).float()
            if X_tensor.shape != (1, 1, 80, target_frames):
                # Ensure correct shape
                X_tensor = torch.nn.functional.pad(X_tensor, (0, max(0, target_frames - X_tensor.shape[3])))
                X_tensor = X_tensor[:, :, :, :target_frames]
            X_tensor = X_tensor.to(self.device)
            
            # Get predictions
            with torch.no_grad():
                logits = self.model(X_tensor)
                probs = torch.sigmoid(logits).cpu().numpy()[0]
            
            # Determine time position
            time_start = start_frame * 160 / sr
            time_end = min((end_frame) * 160 / sr, duration)
            
            # Check for stuttering
            is_stuttering = (probs > threshold).astype(int)
            
            detection = {
                'time_start': float(time_start),
                'time_end': float(time_end),
                'duration': float(time_end - time_start),
                'is_stuttering': bool(is_stuttering.any()),
                'probabilities': {
                    self.STUTTER_CLASSES[i]: float(probs[i])
                    for i in range(len(self.STUTTER_CLASSES))
                },
                'classes': [
                    self.STUTTER_CLASSES[i]
                    for i in range(len(self.STUTTER_CLASSES))
                    if is_stuttering[i]
                ],
                'confidence': float(max(probs))
            }
            
            detections.append(detection)
        
        # Get ASR transcription
        try:
            text, words_data = transcribe_to_words(str(audio_path), model_size='base')
            asr_data = {'text': text, 'words': words_data}
        except Exception as e:
            print(f"⚠ ASR failed: {e}")
            asr_data = None
        
        result = {
            'audio_file': str(audio_path),
            'duration': float(duration),
            'sr': int(sr),
            'model_type': self.model_type,
            'threshold': float(threshold),
            'window_size': float(window_size),
            'hop_size': float(hop_size),
            'timestamp': datetime.now().isoformat(),
            'detections': detections,
            'asr': asr_data,
            'summary': {
                'total_windows': len(detections),
                'stuttering_windows': sum(1 for d in detections if d['is_stuttering']),
                'stuttering_percentage': 100.0 * sum(1 for d in detections if d['is_stuttering']) / max(1, len(detections)),
                'per_class_count': {
                    cls_name: sum(1 for d in detections for c in d['classes'] if c == cls_name)
                    for cls_name in self.STUTTER_CLASSES
                }
            }
        }
        
        return result


def main():
    parser = argparse.ArgumentParser(description='Detect stuttering in audio files')
    parser.add_argument('--model', choices=['simple', 'enhanced'], default='enhanced',
                        help='Model type')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to model weights (auto-detected if None)')
    parser.add_argument('--input', type=str, default=None,
                        help='Input audio file')
    parser.add_argument('--batch-dir', type=str, default=None,
                        help='Directory with audio files for batch processing')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for batch results')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Detection threshold (0-1) - use 0.3 for enhanced model')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                        help='Device to use')
    parser.add_argument('--compare', action='store_true',
                        help='Compare both models on same input')
    
    args = parser.parse_args()
    
    # Auto-detect model path
    if args.model_path is None:
        checkpoint_dir = Path('Models/checkpoints')
        model_name = f'{args.model}_best.pth' if args.model else 'enhanced_best.pth'
        args.model_path = checkpoint_dir / model_name
        
        if not args.model_path.exists():
            print(f"⚠ Model not found: {args.model_path}")
            print(f"Looking for checkpoint files...")
            checkpoints = list(checkpoint_dir.glob(f'{args.model}*.pth'))
            if checkpoints:
                args.model_path = sorted(checkpoints)[-1]  # Use latest
                print(f"Using: {args.model_path}")
            else:
                print("✗ No model found")
                return
    
    # Device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠ CUDA not available, using CPU")
        device = 'cpu'
    else:
        device = args.device
    
    # Single file inference
    if args.input:
        detector = StutterDetector(args.model_path, model_type=args.model, device=device)
        
        print(f"\nProcessing: {args.input}")
        result = detector.detect_stuttering(args.input, threshold=args.threshold)
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"DETECTION SUMMARY")
        print(f"{'='*70}")
        print(f"Duration: {result['duration']:.2f}s")
        print(f"Stuttering frames: {result['summary']['stuttering_windows']}/{result['summary']['total_windows']}")
        print(f"Stuttering %: {result['summary']['stuttering_percentage']:.1f}%")
        print(f"\nPer-class counts:")
        for cls_name, count in result['summary']['per_class_count'].items():
            print(f"  {cls_name}: {count}")
        
        # Save results
        output_path = args.output or 'detection_result.json'
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n✓ Saved to {output_path}")
    
    # Batch processing
    elif args.batch_dir:
        detector = StutterDetector(args.model_path, model_type=args.model, device=device)
        batch_dir = Path(args.batch_dir)
        output_dir = Path(args.output_dir or 'batch_results')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        audio_files = sorted(batch_dir.glob('*.wav')) + sorted(batch_dir.glob('*.mp3'))
        
        print(f"\nProcessing {len(audio_files)} files from {batch_dir}")
        
        results = []
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}] {audio_file.name}")
            try:
                result = detector.detect_stuttering(audio_file, threshold=args.threshold)
                results.append(result)
                
                # Save individual result
                output_path = output_dir / f"{audio_file.stem}_detection.json"
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"  ✓ Saved to {output_path}")
                print(f"  Stuttering: {result['summary']['stuttering_percentage']:.1f}%")
            except Exception as e:
                print(f"  ✗ Error: {e}")
        
        # Save summary
        summary_path = output_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump({
                'total_files': len(audio_files),
                'processed': len(results),
                'results': results
            }, f, indent=2)
        
        print(f"\n✓ Batch processing complete. Results saved to {output_dir}")


if __name__ == '__main__':
    main()
