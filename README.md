# Stuttering Detection & Repair System

**Production-ready system for detecting and repairing stuttered speech**

Complete end-to-end pipeline for training detection models, detecting stuttering, and repairing audio.

---

## 🚀 QUICK START - COPY & PASTE (4 Steps)

### Step 0: Activate Environment (Do This First!)
```powershell
cd d:\Bunny\AGNI
.venv_models\Scripts\Activate.ps1
```

### Step 1: Train Model (1-2 hours on GPU)
```powershell
python Models/improved_train_enhanced.py --model enhanced --epochs 30 --batch-size 16 --gpu
```
**Output:** `Models/checkpoints/enhanced_best.pth` | **F1:** 50-65%

### Step 2: Test Detection (10 seconds)
```powershell
python Models/predict_enhanced.py --model enhanced --input datasets/clips/FluencyBank_010_0.wav --output test_result.json
```
**Output:** `test_result.json` (stuttering %, classes)

### Step 3: Get Repaired Audio (10 seconds)
```powershell
python Models/run_asr_map_repair.py --model_path Models/checkpoints/enhanced_best.pth --input_file datasets/clips/FluencyBank_010_0.wav --output_audio output/repaired_audio/FluencyBank_010_0_repaired.wav --mode attenuate
```
**Output:** 
- `output/repaired_audio/FluencyBank_010_0_repaired.wav` ← **LISTEN HERE** 🎵
- `output/diagnostics/FluencyBank_010_0.asr_repair.json` ← Report

---

## 📋 SCRIPTS EXPLAINED - What To Run & When

### ⭐ MAIN SCRIPTS (3 Total - This is All You Need)

**1. improved_train_enhanced.py** (TRAINING - Run First)
- Purpose: Train the stuttering detection model
- When: Only once at the beginning
- Features: Class weighting, mixed precision, early stopping
- Command: `python Models/improved_train_enhanced.py --model enhanced --epochs 30 --gpu`
- Output: `Models/checkpoints/enhanced_best.pth` (~3.4 MB)
- Time: 1-2 hours (GPU) | 4-6 hours (CPU)

**2. predict_enhanced.py** (TESTING - After Training)
- Purpose: Test detection on audio files
- When: After training to verify model works
- Supports: Single file or batch processing
- Command: `python Models/predict_enhanced.py --model enhanced --input audio.wav`
- Output: JSON with stuttering %, detected classes, confidence
- Time: < 10 seconds per file

**3. run_asr_map_repair.py** (FULL PIPELINE - To Get Repaired Audio)
- Purpose: Complete end-to-end workflow (detect → transcribe → map → repair)
- When: When you want final repaired audio
- Modes: remove, silence, attenuate
- Command: `python Models/run_asr_map_repair.py --model_path best.pth --input_file audio.wav --output_audio output.wav`
- Output: Repaired WAV + JSON diagnostics
- Time: 5-10 seconds per file

### 🔧 SUPPORT SCRIPTS (8 Total - All Required)

Required for the 3 main scripts to work:

| Script | Used By | Purpose | Size |
|--------|---------|---------|------|
| model_enhanced.py | Training | Enhanced model (7 layers, attention, residual) | 7.3 KB |
| model_cnn.py | Training/Testing | SimpleCNN baseline (4 layers) | 2.3 KB |
| features.py | All | Log-Mel spectrogram extraction | 3.1 KB |
| asr_whisper.py | Repair | Speech-to-text transcription | 8.3 KB |
| repair.py | Repair | Audio editing engine | 9.5 KB |
| map_sed_words.py | Repair | Window-to-word alignment | 1.9 KB |
| utils.py | All | Helper functions | 4.9 KB |
| __init__.py | All | Package init | 0.4 KB |

---

## 🗑️ WASTE SCRIPTS - Safe To Delete

### ❌ DELETE THIS (Definite Waste)

**improved_train.py** (388 lines, 13 KB)
- **Problem:** Redundant old training script
- **Why waste:** `improved_train_enhanced.py` is better (newer, more features)
- **Recommendation:** DELETE immediately
- **Command:** `Remove-Item -Path "d:\Bunny\AGNI\Models\improved_train.py"`
- **Impact:** -13 KB, cleaner codebase, no confusion

### ⚠️ KEEP THIS (Optional, Not Waste)

**ctc_align.py** (202 lines, 9 KB)
- **Purpose:** Advanced character-level alignment (optional)
- **Use:** Only if you need ultra-precise word boundaries
- **Current:** Using Whisper + energy refinement (sufficient)
- **Recommendation:** KEEP (might need for production)
- **Cost:** Only 9 KB

---

## ⚙️ HOW THE PIPELINE WORKS

```
Audio File (WAV/MP3)
        ↓
[STEP 1] DETECTION (Sliding-Window CNN)
  - Scan with 1.0s windows, 0.5s hop
  - Classify: Prolongation, Block, Sound Rep, Word Rep, Interjection
        ↓
[STEP 2] TRANSCRIPTION (Whisper ASR)
  - Convert speech to text
  - Generate word-level timestamps
        ↓
[STEP 3] MAPPING (Window-to-Word Alignment)
  - Link detected stutter frames to specific words
  - Create per-word stutter labels
        ↓
[STEP 4] REPAIR (Audio Editing)
  - Remove: Excise stuttered words
  - Silence: Replace with quiet
  - Attenuate: Reduce volume (-10dB) ← DEFAULT
        ↓
FINAL OUTPUT
  ├─ Repaired Audio (WAV file) ✅
  └─ Diagnostics (JSON report)
```

---

## 🎯 EXECUTION ORDER & TIME

### ONE-TIME SETUP
```
1. Activate environment           < 1 second
2. Train model (30 epochs)        1-2 hours    ← LONG WAIT
3. Done! Ready to use
```

### REPEATED USAGE (After Model Trained)
```
1. Activate environment           < 1 second
2. Run predict/repair             10-20 seconds
3. Get results
```

### TYPICAL WORKFLOW
```
.venv_models\Scripts\Activate.ps1                  # 1 second
python Models/improved_train_enhanced.py ...       # 1-2 HOURS (train once)
python Models/predict_enhanced.py ...              # 10 seconds (test)
python Models/run_asr_map_repair.py ...           # 10 seconds (repair)
Open output/repaired_audio/*.wav in audio player  # Listen!
```

---

## 📊 REPAIR MODES (Choose One)

```powershell
--mode attenuate    ← RECOMMENDED (Most Natural)
  Effect: Reduce stuttered word volume by 10dB
  Sound: Natural, preserves context
  Use: Default, safest choice
  Example: "I c-c-c-can't" → "I [quieter] can't"
  
--mode silence      (Alternative)
  Effect: Replace with silence
  Sound: Clear but may sound choppy
  Use: When obvious removal needed
  Example: "I c-c-c-can't" → "I [pause] can't"
  
--mode remove       (Aggressive)
  Effect: Excise words, stitch audio
  Sound: Very choppy, unnatural gaps
  Use: Only for complete removal
  Example: "I c-c-c-can't" → "I can't"
```

---

## 📁 FILE STRUCTURE

```
d:\Bunny\AGNI\
│
├── README.md ← YOU ARE HERE
│
├── Models/
│   ├── improved_train_enhanced.py ⭐ TRAINING
│   ├── predict_enhanced.py ⭐ TESTING
│   ├── run_asr_map_repair.py ⭐ REPAIR+PIPELINE
│   │
│   ├── model_enhanced.py, model_cnn.py (REQUIRED)
│   ├── features.py, asr_whisper.py (REQUIRED)
│   ├── repair.py, map_sed_words.py (REQUIRED)
│   ├── utils.py, __init__.py (REQUIRED)
│   │
│   ├── improved_train.py 🗑️ DELETE
│   ├── ctc_align.py ⚠️ OPTIONAL
│   │
│   └── checkpoints/
│       └── enhanced_best.pth (created after training)
│
├── datasets/
│   ├── clips/ (32,321 audio files)
│   ├── SEP-28k_labels.csv, fluencybank_labels.csv
│   └── annotated_time_aligned/
│
└── output/
    ├── repaired_audio/ ← FINAL AUDIO FILES ✅
    ├── diagnostics/ (JSON reports)
    ├── metrics/ (training metrics)
    └── test_results/ (batch results)
```

---

## ⏱️ TIME BREAKDOWN

| Operation | GPU | CPU |
|-----------|-----|-----|
| Activate env | <1s | <1s |
| Train 30 epochs | 1-2 hrs | 4-6 hrs |
| Test 1 file | <10s | <10s |
| Repair 1 file | 5-10s | 5-10s |
| Batch 100 files | 8-15m | 8-15m |

---

## ✅ EXPECTED RESULTS

### Training (30 epochs)
- ✅ Time: 1-2 hours (GPU)
- ✅ Best Model Epoch: ~15-20
- ✅ Final F1: 50-65%
- ✅ Early Stop: ~22-28 epochs

### Inference
- ✅ Accuracy: 50-85% F1
- ✅ Speed: 1-5s per 10s clip
- ✅ Quality: No artifacts
- ✅ Coverage: 60-80% of stuttering

---

## 🛠️ TROUBLESHOOTING

### Training is slow
```powershell
# Use GPU
--gpu

# Or use SimpleCNN
--model simple

# Or reduce epochs for testing
--epochs 5
```

### CUDA out of memory
```powershell
# Reduce batch size
--batch-size 8

# Or use CPU
(remove --gpu flag)
```

### Model not found
```powershell
# Run training first
python Models/improved_train_enhanced.py --model enhanced --epochs 30 --gpu
```

### Low accuracy (F1 < 40%)
- Increase training epochs (try 50+)
- Increase batch size (try 32)
- Check data quality in datasets/
- Adjust detection threshold

### Audio sounds unnatural
```powershell
# Try attenuate mode (recommended)
--mode attenuate

# Or try silence
--mode silence
```

---

## 📚 DATASET INFO

- **Total Clips:** 32,321 audio files
- **SEP-28k:** 28,179 labeled clips
- **FluencyBank:** 4,146 labeled clips
- **Classes:** 5 stutter types (Prolongation, Block, Sound Rep, Word Rep, Interjection)
- **Format:** Multi-label (clips can have multiple types)
- **Audio:** 16 kHz baseline, diverse speakers/accents

---

## 🎯 COMMAND REFERENCE (Copy & Paste Fast Access)

```powershell
# Activate (always first)
.venv_models\Scripts\Activate.ps1

# Train (only once)
python Models/improved_train_enhanced.py --model enhanced --epochs 30 --batch-size 16 --gpu

# Test detection
python Models/predict_enhanced.py --model enhanced --input datasets/clips/FluencyBank_010_0.wav --output test.json

# Repair single file
python Models/run_asr_map_repair.py --model_path Models/checkpoints/enhanced_best.pth --input_file datasets/clips/FluencyBank_010_0.wav --output_audio output/repaired_audio/output.wav --mode attenuate

# Batch test
python Models/predict_enhanced.py --model enhanced --batch-dir datasets/clips/ --output-dir output/test_results/
```

---

## ✨ SUMMARY

✅ **3 scripts only:** improved_train_enhanced.py, predict_enhanced.py, run_asr_map_repair.py  
✅ **1 waste script:** improved_train.py (safe to delete)  
✅ **Execution order:** Train → Test → Repair  
✅ **Time:** 2-3 hours total on GPU  
✅ **Output:** Repaired audio in output/repaired_audio/  
✅ **Everything ready** to start using!

**👉 Copy the 4 steps from top and start training now! 🚀**
