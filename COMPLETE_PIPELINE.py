"""
COMPLETE STUTTER DETECTION + REPAIR PIPELINE
All-in-one solution for detecting and fixing stuttering speech

PIPELINE FLOW:
1. Extract Features (16-20 sentences, ~30min each, or all 30k files)
2. Train Detection Model (90%+ accuracy)
3. Test on sample audio
4. Detect stutters
5. Repair audio (vocoder-based)
6. Generate metrics report

USAGE:
    python COMPLETE_PIPELINE.py
    
Optional (fully-customizable):
    python COMPLETE_PIPELINE.py --features-only --repair-only --test-file audio.wav
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import torch
import librosa
import soundfile as sf

# Add Models to path
sys.path.insert(0, str(Path(__file__).parent))

from extract_features_90plus import FeatureExtractionManager
from train_90plus_final import Trainer
from model_improved_90plus import ImprovedStutteringCNN
from enhanced_audio_preprocessor import EnhancedAudioPreprocessor
from repair_advanced import AdvancedStutterRepair, extract_stutter_analysis


class CompletePipeline:
    """End-to-end stutter detection and repair pipeline."""
    
    def __init__(self, output_dir='output', checkpoint_dir='Models/checkpoints'):
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_file = self.output_dir / 'pipeline_log.txt'
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'repaired_audio').mkdir(exist_ok=True)
        (self.output_dir / 'analysis').mkdir(exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.best_model_path = None
        
        print(f"Pipeline initialized. Device: {self.device}")
        self._log(f"Pipeline started at {datetime.now()}")
    
    def _log(self, message):
        """Log message to file and console."""
        print(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now()}: {message}\n")
    
    def extract_features(self, sample_size=None):
        """Phase 1: Extract features from audio files."""
        self._log("\n" + "="*60)
        self._log("PHASE 1: FEATURE EXTRACTION")
        self._log("="*60)
        
        try:
            # Check if features already exist
            train_dir = Path('datasets/features/train')
            val_dir = Path('datasets/features/val')
            
            train_count = len(list(train_dir.glob('*.npz'))) if train_dir.exists() else 0
            val_count = len(list(val_dir.glob('*.npz'))) if val_dir.exists() else 0
            
            if train_count > 20000 and val_count > 5000:
                self._log(f"[SKIP] Features already extracted!")
                self._log(f"  Train: {train_count} files")
                self._log(f"  Val: {val_count} files")
                self._log(f"  Total: {train_count + val_count} files")
                return True
            
            manager = FeatureExtractionManager()
            
            if sample_size:
                self._log(f"Extracting {sample_size} sample files (quick test)...")
            else:
                self._log("Extracting features from all 30,036 files (FULL PIPELINE)...")
                self._log("Estimated time: 6-8 hours on GPU")
            
            manager.extract_all_features(sample_size=sample_size)
            manager.verify_extraction()
            
            self._log("[OK] Feature extraction complete!")
            return True
        except Exception as e:
            self._log(f"✗ Feature extraction failed: {e}")
            return False
    
    def train_model(self, epochs=60, early_stopping_patience=50):
        """Phase 2: Train detection model."""
        self._log("\n" + "="*60)
        self._log("PHASE 2: MODEL TRAINING")
        self._log("="*60)
        
        try:
            from torch.utils.data import DataLoader
            from train_90plus_final import AudioDataset, collate_variable_length
            
            self._log(f"Training with Focal Loss, Focal gamma=2.0")
            self._log(f"Max epochs: {epochs}, Early stop patience: {early_stopping_patience}")
            self._log(f"Estimated time: 10-12 hours on GPU")
            
            # Load datasets
            train_dataset = AudioDataset('datasets/features', split='train', augment=True)
            val_dataset = AudioDataset('datasets/features', split='val', augment=False)
            
            # Create dataloaders with custom collate for variable length sequences
            train_loader = DataLoader(
                train_dataset, 
                batch_size=256, 
                shuffle=True, 
                num_workers=0, 
                pin_memory=True,
                collate_fn=collate_variable_length
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=256, 
                shuffle=False, 
                num_workers=0, 
                pin_memory=True,
                collate_fn=collate_variable_length
            )
            
            # Initialize model
            model = ImprovedStutteringCNN()
            
            # Train
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=self.device,
                model_name='improved_90plus'
            )
            
            trainer.train(num_epochs=epochs)
            
            self.best_model_path = self.checkpoint_dir / 'improved_90plus_best.pth'
            self._log(f"[OK] Training complete! Best model: {self.best_model_path}")
            return True
        except Exception as e:
            self._log(f"✗ Training failed: {e}")
            import traceback
            self._log(traceback.format_exc())
            return False
    
    def load_best_model(self, model_path=None):
        """Load trained model."""
        if model_path is None:
            if self.best_model_path is None:
                model_path = self.checkpoint_dir / 'improved_90plus_best.pth'
            else:
                model_path = self.best_model_path
        
        try:
            self.model = ImprovedStutteringCNN().to(self.device)
            self.model.load_state_dict(torch.load(str(model_path), map_location=self.device))
            self.model.eval()
            self._log(f"[OK] Loaded model: {model_path}")
            return True
        except Exception as e:
            self._log(f"✗ Failed to load model: {e}")
            return False
    
    def detect_and_repair_audio(self, audio_path, output_name=None):
        """Phase 3: Detect and repair stuttering in audio."""
        self._log("\n" + "="*60)
        self._log("PHASE 3: DETECTION + REPAIR")
        self._log("="*60)
        
        if not Path(audio_path).exists():
            self._log(f"✗ Audio file not found: {audio_path}")
            return None
        
        try:
            output_name = output_name or Path(audio_path).stem
            output_audio = self.output_dir / 'repaired_audio' / f"{output_name}_repaired.wav"
            output_json = self.output_dir / 'analysis' / f"{output_name}_analysis.json"
            
            self._log(f"Processing: {audio_path}")
            
            # Initialize repair with trained model
            repair = AdvancedStutterRepair(model_path=str(self.best_model_path))
            
            # Repair audio
            repaired_audio, stutter_regions = repair.repair_audio(
                audio_path, 
                output_path=str(output_audio)
            )
            
            if repaired_audio is None:
                return None
            
            # Generate analysis
            analysis = extract_stutter_analysis(audio_path, str(output_json))
            
            # Summary
            self._log(f"\n[OK] Repair complete!")
            self._log(f"  Stutters detected: {len(stutter_regions)}")
            self._log(f"  Total stutter time: {analysis['total_stutter_time']}s")
            self._log(f"  Stuttering: {analysis['stuttering_percentage']}%")
            self._log(f"  Repaired audio: {output_audio}")
            self._log(f"  Analysis: {output_json}")
            
            return {
                'repaired_audio': str(output_audio),
                'analysis': analysis,
                'regions': stutter_regions
            }
        except Exception as e:
            self._log(f"✗ Detection/repair failed: {e}")
            return None
    
    def generate_report(self, results=None):
        """Generate final metrics report."""
        self._log("\n" + "="*60)
        self._log("PIPELINE REPORT")
        self._log("="*60)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'pipeline_status': 'complete',
            'results': results or {}
        }
        
        report_path = self.output_dir / 'pipeline_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self._log(f"\n[OK] Report saved: {report_path}")
        return report
    
    def run_full_pipeline(self, 
                         extract_features=True, 
                         train_model=True, 
                         test_audio=None,
                         sample_size=None):
        """Run complete pipeline."""
        
        self._log("\n" + "="*60)
        self._log("COMPLETE STUTTER DETECTION & REPAIR")
        self._log("="*60)
        
        results = {}
        
        # Phase 1: Features
        if extract_features:
            if not self.extract_features(sample_size=sample_size):
                return None
        
        # Phase 2: Training
        if train_model:
            if not self.train_model():
                return None
        
        # Phase 3: Load model
        if not self.load_best_model():
            return None
        
        # Phase 4: Test on sample audio
        if test_audio:
            test_results = self.detect_and_repair_audio(test_audio, "test")
            if test_results:
                results['test_audio'] = test_results
        
        # Generate report
        self.generate_report(results)
        
        self._log("\n" + "="*60)
        self._log("[OK] PIPELINE COMPLETE!")
        self._log("="*60)
        self._log("\nNext steps:")
        self._log("1. Download repaired audio from: output/repaired_audio/")
        self._log("2. Check metrics in: output/analysis/")
        self._log("3. Review report: output/pipeline_report.json")
        
        return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Complete Stutter Detection & Repair Pipeline')
    parser.add_argument('--features-only', action='store_true', help='Extract features only')
    parser.add_argument('--train-only', action='store_true', help='Train model only')
    parser.add_argument('--repair-only', action='store_true', help='Repair audio only')
    parser.add_argument('--test-file', type=str, help='Audio file to repair')
    parser.add_argument('--sample-size', type=int, help='Number of files to extract (for testing)')
    parser.add_argument('--skip-features', action='store_true', help='Skip feature extraction')
    parser.add_argument('--skip-training', action='store_true', help='Skip training (use existing model)')
    
    args = parser.parse_args()
    
    pipeline = CompletePipeline()
    
    if args.features_only:
        pipeline.extract_features(sample_size=args.sample_size)
    elif args.train_only:
        pipeline.load_best_model()
        pipeline.train_model()
    elif args.repair_only:
        if not args.test_file:
            print("Error: --repair-only requires --test-file")
            sys.exit(1)
        pipeline.load_best_model()
        pipeline.detect_and_repair_audio(args.test_file)
    else:
        # Full pipeline
        pipeline.run_full_pipeline(
            extract_features=not args.skip_features,
            train_model=not args.skip_training,
            test_audio=args.test_file,
            sample_size=args.sample_size
        )


if __name__ == '__main__':
    main()
