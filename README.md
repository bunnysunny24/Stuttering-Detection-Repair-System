# AGNI - Stuttering Detection & Repair System

> Hierarchical stuttering detection using end-to-end wav2vec2-large fine-tuning. Binary detection (stutter vs fluent, 90+ F1 target) as PRIMARY task, 5-class type classification as SECONDARY. Includes frozen-feature BiLSTM+CNN classifiers, self-training label refinement, ensemble learning, test-time augmentation, production inference pipeline, and vocoder-based stutter repair.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Requirements](#2-system-requirements)
3. [Installation & Environment Setup](#3-installation--environment-setup)
4. [Dataset](#4-dataset)
5. [Feature Extraction](#5-feature-extraction)
   - 5.1 [123-Channel Spectrogram Features](#51-123-channel-spectrogram-features)
   - 5.2 [Temporal wav2vec2 Features (Primary)](#52-temporal-wav2vec2-features-primary)
6. [Model Architectures](#6-model-architectures)
   - 6.1 [Wav2VecFineTuneClassifier (Production - 317.6M params)](#61-wav2vecfinetuneclassifier-production---3176m-params)
   - 6.2 [TemporalBiLSTMClassifier (Frozen Features - 2.07M params)](#62-temporalbilstmclassifier-frozen-features---207m-params)
   - 6.3 [TemporalStutterClassifier (870K params)](#63-temporalstutterclassifier-870k-params)
   - 6.4 [CNNBiLSTM (Baseline)](#64-cnnbilstm-baseline)
   - 6.5 [ImprovedStutteringCNN](#65-improvedstutteringcnn)
   - 6.6 [ImprovedStutteringCNNLarge](#66-improvedstutteringcnnlarge)
   - 6.7 [ImprovedStutteringCNNLargeSE](#67-improvedstutteringcnnlargese)
7. [Training System](#7-training-system)
   - 7.1 [End-to-End Fine-Tuning (Production)](#71-end-to-end-fine-tuning-production)
   - 7.2 [Frozen-Feature Training](#72-frozen-feature-training)
   - 7.3 [Data Loading & Augmentation](#73-data-loading--augmentation)
   - 7.4 [Loss Functions](#74-loss-functions)
   - 7.5 [Optimizers & Schedulers](#75-optimizers--schedulers)
   - 7.6 [Training Techniques](#76-training-techniques)
8. [Production Inference](#8-production-inference)
9. [Calibration & Evaluation](#9-calibration--evaluation)
10. [Self-Training Label Refinement](#10-self-training-label-refinement)
11. [Ensemble Evaluation with TTA](#11-ensemble-evaluation-with-tta)
12. [Stutter Repair](#12-stutter-repair)
13. [Pipeline Scripts](#13-pipeline-scripts)
14. [Tools & Utilities](#14-tools--utilities)
15. [Tests](#15-tests)
16. [Constants & Configuration](#16-constants--configuration)
17. [Directory Structure](#17-directory-structure)
18. [Training History & Results](#18-training-history--results)
19. [Troubleshooting](#19-troubleshooting)

---

## 1. Project Overview

AGNI is an end-to-end system for:

1. **Detecting** 5 types of stuttering in conversational audio (multi-label classification)
2. **Repairing** detected stutter regions using spectral inpainting and vocoder-based reconstruction

**Stutter Classes (5):**

| Index | Class | Description |
|-------|-------|-------------|
| 0 | Prolongation | Extended duration of a sound ("sssssnake") |
| 1 | Block | Silent pause/stoppage in speech flow |
| 2 | SoundRep | Sound repetition ("b-b-b-ball") |
| 3 | WordRep | Word repetition ("I I I want") |
| 4 | Interjection | Filler words ("um", "uh", "like") |

**Approach:** Hierarchical detection system:
- **PRIMARY (90+ target):** Binary stutter detection (stutter vs fluent) using end-to-end wav2vec2-large fine-tuning
- **SECONDARY:** 5-class type classification (Prolongation/Block/SoundRep/WordRep/Interjection)
- **Frozen-feature path:** Extract temporal features from wav2vec2-base (frozen), train BiLSTM+CNN+Attention classifier
- **Label quality:** Majority vote (>=2 annotators), noise removal (Unsure/PoorAudio/NoSpeech/Music)

### Key Files

| File | Purpose |
|------|---------|
| `Models/model_w2v_finetune.py` | **Production model:** End-to-end wav2vec2-large with binary + 5-class heads (317.6M params) |
| `Models/train_w2v_finetune.py` | **Production training:** Hierarchical loss, binary F1 model selection, threshold optimization |
| `Models/detect.py` | **Production inference:** Single file / directory / sliding window detection |
| `scripts/run_finetune_pipeline.ps1` | **Production pipeline:** Complete training launch script |
| `Models/model_temporal_bilstm.py` | Frozen-feature model: BiLSTM + Dilated CNN + Multi-Head Attention |
| `Models/train_90plus_final.py` | Frozen-feature training loop (~2039 lines) |
| `Models/extract_wav2vec2_temporal.py` | Temporal wav2vec2-base feature extraction |
| `Models/self_train_refine.py` | Self-training label refinement |
| `Models/ensemble_eval.py` | Ensemble + TTA evaluation |
| `Models/calibrate_thresholds.py` | Temperature scaling + per-class threshold optimization |
| `Models/repair_advanced.py` | Vocoder-based stutter repair (725 lines) |
| `scripts/run_90plus_pipeline.ps1` | Frozen-feature 6-step training pipeline |

---

## 2. System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10+ | 3.11 |
| RAM | 16 GB | 32+ GB |
| CPU | 4 cores | 8+ threads |
| Storage | 50 GB free | 100+ GB |
| GPU | Not required | CUDA GPU for fine-tuning |
| OS | Windows 10/11 | Any (tested on Windows) |

**Tested Configuration:**
- Intel i7-1165G7 (4 cores / 8 threads, 2.8 GHz)
- 40 GB DDR4-3200
- CPU-only (Intel Iris Xe integrated)
- Windows, conda env `agni311`

---

## 3. Installation & Environment Setup

```powershell
# Create conda environment
conda create -n agni311 python=3.11 -y
conda activate agni311

# Install PyTorch (CPU)
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
pip install transformers scipy numpy scikit-learn matplotlib tqdm librosa

# Note: torchaudio is NOT used (broken on some CPU builds).
# Audio loading uses scipy.io.wavfile + scipy.signal.resample_poly instead.
```

See `requirements_complete.txt` for the full pinned dependency list.

---

## 4. Dataset

**Sources:** SEP-28k + FluencyBank corpora

### Raw Dataset

| Split | Files | Source |
|-------|-------|--------|
| Train | ~25,445 | 80% of labeled clips |
| Val | ~6,470 | 20% of labeled clips |
| **Total** | **~31,915** | |

### Cleaned Dataset (Production)

After label cleaning and majority vote filtering:

| Cleaning Step | Removed | Remaining |
|---------------|---------|-----------|
| Raw total | — | ~31,915 |
| Noise removal (Unsure/PoorAudio/NoSpeech/Music) | ~4,526 | ~27,389 |
| Majority vote (≥2 of 3 annotators agree) | further | ~25,374 |
| **Final Split** | | |
| Train (cleaned) | | **20,193** |
| Val (cleaned) | | **5,181** |

**Binary Balance (cleaned train):** 52.3% stutter / 47.7% fluent

**Per-Class Distribution (majority vote, train):**

| Class | Count |
|-------|-------|
| Prolongation | 2,018 |
| Block | 2,474 |
| SoundRep | 1,772 |
| WordRep | 2,186 |
| Interjection | 4,634 |

**Raw Audio:** `datasets/clips/stuttering-clips/clips/*.wav` (16 kHz, mono WAV)

**Label Sources:**
- `datasets/SEP-28k_labels.csv` - 3-annotator labels for SEP-28k
- `datasets/fluencybank_labels.csv` - FluencyBank labels
- `datasets/SEP-28k_episodes.csv` - Episode metadata
- `datasets/fluencybank_episodes.csv` - Episode metadata
- `datasets/label_audio_map.json` - Mapping from label stems to audio file paths

**Labels:** Multi-label binary - each sample can have multiple stutter types simultaneously. Original annotations from 3 annotators are binarized via majority vote (≥2 agree → 1.0). Noise labels (Unsure, PoorAudio, NoSpeech, Music, DifficultToUnderstand) are used for sample exclusion, not as classification targets.

**Feature Directories:**

| Directory | Format | Used By |
|-----------|--------|---------|
| `datasets/features/{train,val}/*.npz` | `spectrogram (123, T)` + `labels (5,)` | CNN models |
| `datasets/features_w2v_temporal/{train,val}/*.npz` | `temporal_embedding (768, T)` + `labels (5,)` | BiLSTM/CNN temporal models |
| `datasets/features_w2v_temporal_refined/{train,val}/*.npz` | Same format, cleaned labels | Stage 2 training |
| N/A (raw audio directly) | WAV files loaded at runtime | Wav2VecFineTuneClassifier |

---

## 5. Feature Extraction

### 5.1. 123-Channel Spectrogram Features

**Script:** `Models/extract_features_90plus.py` (class `FeatureExtractionPipeline`, 1287 lines)
**Preprocessor:** `Models/enhanced_audio_preprocessor.py` (class `EnhancedAudioPreprocessor`)

Extracts a 123-channel feature stack from raw audio:

| Channels | Feature | Library |
|----------|---------|---------|
| 0-79 (80) | Mel spectrogram | librosa / SciPy fallback |
| 80-92 (13) | MFCCs | DCT of mel spectrogram |
| 93-105 (13) | MFCC deltas | 1st derivative |
| 106-118 (13) | MFCC delta-deltas | 2nd derivative |
| 119 (1) | Spectral centroid | Frequency-weighted mean |
| 120 (1) | Spectral rolloff | 85th percentile frequency |
| 121 (1) | Zero-crossing rate | Sign changes per frame |
| 122 (1) | Spectral flux | Frame-to-frame magnitude change |

**Parameters:**
- Sample rate: 16,000 Hz
- FFT size: 1024
- Hop length: 512
- Mel bands: 80
- MFCCs: 13

**Processing Pipeline:**
1. Load WAV -> resample to 16 kHz -> mono
2. VAD trimming (amplitude-based fallback)
3. STFT -> mel spectrogram -> MFCCs -> deltas -> spectral features
4. NaN/Inf sanitization
5. Save as NPZ: `spectrogram (123, T)` + `labels (5,)`

**Usage:**
```powershell
python Models/extract_features_90plus.py
```

### 5.2. Temporal wav2vec2 Features (Primary)

**Script:** `Models/extract_wav2vec2_temporal.py`
**Model:** `facebook/wav2vec2-base` (HuggingFace, 94M parameters, frozen)

Extracts temporal frame-level features from wav2vec2's transformer encoder using a multi-layer weighted average.

**Audio Loading (scipy-based - no torchaudio):**
```python
scipy.io.wavfile.read()          # Load WAV
scipy.signal.resample_poly()     # Resample to 16kHz
# int16/int32/float64 -> float32, stereo -> mono
```

**Multi-Layer Weighted Average (layers 7-12):**

The top 6 transformer layers are combined with learned-prior weights:

| Layer | Raw Weight | Normalized | Rationale |
|-------|-----------|------------|-----------|
| 7 | 0.10 | ~0.10 | Lower-level acoustic |
| 8 | 0.12 | ~0.12 | |
| 9 | 0.15 | ~0.15 | Mid-level features |
| 10 | 0.18 | ~0.18 | |
| 11 | 0.20 | ~0.20 | Higher-level linguistic |
| 12 | 0.25 | ~0.25 | Most task-relevant |

**Implementation:**
```python
model.config.output_hidden_states = True
outputs = model(input_values)
hidden_states = outputs.hidden_states  # tuple of 13 tensors (embedding + 12 layers)

# Weighted combination of layers 7-12
layer_weights = {7: 0.10, 8: 0.12, 9: 0.15, 10: 0.18, 11: 0.20, 12: 0.25}
weighted_sum = sum(w * hidden_states[layer] for layer, w in layer_weights.items())
weighted_avg = weighted_sum / sum(layer_weights.values())
```

**Output Format:**
```
datasets/features_w2v_temporal/{train,val}/<clip_id>.npz
  +-- temporal_embedding: float32 (768, T)   # T <= 200 frames (~4s at 50fps)
  +-- labels:             float32 (5,)       # Binary stutter labels
```

**Label Loading:** Uses a pickle cache (`datasets/features/_label_cache.pkl`) to avoid re-reading 31k+ NPZ files for label extraction. Cache is auto-built on first run.

**CLI Arguments:**
```
--model          facebook/wav2vec2-base    # HuggingFace model name
--clips-dir      datasets/clips/...       # Raw audio directory
--features-dir   datasets/features        # Source for labels + train/val split
--output-dir     datasets/features_w2v_temporal
--max-frames     200                      # Max temporal frames (truncate/pad)
--batch-size     8                        # Inference batch size
--layer-mode     weighted-avg             # 'weighted-avg' or 'last'
```

**CPU Optimization:**
```python
torch.set_num_threads(os.cpu_count())      # All logical CPUs
torch.set_num_interop_threads(os.cpu_count())
os.environ['OMP_NUM_THREADS'] = str(N_CORES)
os.environ['MKL_NUM_THREADS'] = str(N_CORES)
```

**Resumable:** Skips already-extracted files (checks for existing `.npz`).

---

## 6. Model Architectures

### 6.1. Wav2VecFineTuneClassifier (Production - 317.6M params)

**File:** `Models/model_w2v_finetune.py` (389 lines)
**Training:** `Models/train_w2v_finetune.py` (~1049 lines)

The production model. End-to-end fine-tuning of wav2vec2-large with hierarchical classification:
- **BINARY head (PRIMARY):** Stutter vs fluent detection (90+ F1 target)
- **5-class heads (SECONDARY):** Per-type stutter classification

**Key design:**
- wav2vec2-large backbone (24 transformer layers, 1024-dim)
- Bottom 12 layers frozen, top 12 fine-tuned (161.7M trainable / 317.6M total = 50.9%)
- Weighted layer combination across all 25 hidden states (learned weights)
- BiLSTM + Dilated CNN + Wide Conv fusion
- Multi-Head Attention Pooling with per-class specialization
- Separate binary head + 5 per-class heads

**Constructor:**
```python
Wav2VecFineTuneClassifier(
    model_name='facebook/wav2vec2-large',
    n_classes=5,
    freeze_layers=12,         # Freeze bottom 12 transformer layers
    lstm_hidden=128,          # Per-direction LSTM hidden size
    lstm_layers=2,
    use_gradient_checkpointing=True
)
```

**Layer-by-Layer:**

| Layer | Output Shape | Details |
|-------|-------------|---------|
| Raw audio input | `(B, T_samples)` | 16 kHz mono WAV |
| wav2vec2-large encoder | `(B, T_frames, 1024)` × 25 layers | 24 transformer layers + CNN features |
| Weighted layer combination | `(B, T_frames, 1024)` | Learned softmax weights over all 25 hidden states |
| LayerNorm + Dropout(0.1) | `(B, T_frames, 1024)` | Normalize combined features |
| BiLSTM (2 layers, 128/dir) | `(B, T_frames, 256)` | Sequential temporal modeling |
| DilatedCNN (d=1,2,4) | `(B, T_frames, 128)` | Multi-scale local patterns |
| WideConv (kernel=7) | `(B, T_frames, 64)` | Broad context capture |
| Fusion: cat + project | `(B, T_frames, 256)` | Concatenate BiLSTM+CNN+WideConv → 256-dim |
| Multi-Head Attention Pool (5 heads) | `(B, 5, 256)` | Per-class attention over temporal axis |
| Per-class classifier × 5 | `(B, 5)` (logits) | `Linear(256→64→1)` per class |
| Binary head | `(B, 1)` (logit) | `Linear(256→64→1)` from mean-pooled attention |

**Forward pass returns:** `(logits, binary_logit)` where logits is `(B, 5)` and binary_logit is `(B, 1)`.

**Parameter groups** (for differential learning rates):
```python
groups = model.get_param_groups(backbone_lr=2e-5, head_lr=5e-4)
# group[0]: wav2vec2 backbone (unfrozen layers) at backbone_lr
# group[1]: classifier heads + fusion layers at head_lr
```

---

### 6.2. TemporalBiLSTMClassifier (Frozen Features - 2.07M params)

**File:** `Models/model_temporal_bilstm.py`
**Architecture choice:** `--arch temporal_bilstm`

The strongest frozen-feature model. Combines BiLSTM sequential modeling with dilated CNNs for multi-scale temporal patterns and multi-head attention pooling with per-class specialization.

**Constructor:**
```python
TemporalBiLSTMClassifier(
    input_dim=768,      # wav2vec2 feature dim
    n_classes=5,        # Number of stutter types
    hidden_dim=256,     # CNN hidden dimension
    lstm_hidden=128,    # LSTM hidden size (x2 for bidirectional)
    lstm_layers=2,      # LSTM depth
    dropout=0.3         # Dropout rate
)
```

**Complete Layer Table:**

| # | Layer | Type | Dimensions | Notes |
|---|-------|------|------------|-------|
| 1 | `proj.0` | Conv1d | 768 -> 256, k=1 | Projection, no bias |
| 2 | `proj.1` | BatchNorm1d | 256 | |
| 3 | `proj.2` | GELU | - | |
| 4 | `proj.3` | Dropout | 0.15 | |
| 5 | `lstm` | LSTM | input=256, hidden=128, layers=2, bidirectional=True | Output: 256 (128x2) |
| 6 | `lstm_norm` | LayerNorm | 256 | Post-LSTM normalization |
| 7 | `lstm_drop` | Dropout | 0.3 | |
| 8 | `temporal_blocks.0` | ConvBlock | 256 -> 256, k=3, dilation=1 | Residual + BN + GELU |
| 9 | `temporal_blocks.1` | ConvBlock | 256 -> 256, k=3, dilation=2 | Receptive field grows |
| 10 | `temporal_blocks.2` | ConvBlock | 256 -> 256, k=3, dilation=4 | |
| 11 | `wide_conv.0` | Conv1d | 256 -> 128, k=7, pad=3 | Wide kernel path |
| 12 | `wide_conv.1` | BatchNorm1d | 128 | |
| 13 | `wide_conv.2` | GELU | - | |
| 14 | `wide_conv.3` | Dropout | 0.3 | |
| 15 | `fusion.0` | Conv1d | 384 -> 256, k=1 | Merge dilated + wide |
| 16 | `fusion.1` | BatchNorm1d | 256 | |
| 17 | `fusion.2` | GELU | - | |
| 18 | `fusion.3` | Dropout | 0.3 | |
| 19 | `attention_pool` | MultiHeadAttentionPool | 5 heads, each: Linear(256->64)->Tanh->Linear(64->1) | 1 head per class |
| 20-24 | `class_heads[0-4]` | Sequential x 5 | Linear(256->64)->GELU->Dropout(0.15)->Linear(64->1) | Per-class output |

**Each ConvBlock contains:**
```
Conv1d(C_in, C_out, kernel=3, dilation=d, padding=d, bias=False)
-> BatchNorm1d(C_out)
-> GELU
-> Dropout(0.3)
+ Residual skip (1x1 Conv1d if C_in != C_out)
```

**Forward Pass Flow:**
```
Input: (B, 768, T)
  |
  +-- 1. Conv1d projection -> (B, 256, T)
  |
  +-- 2. Transpose -> (B, T, 256) -> BiLSTM -> (B, T, 256)
  |     -> LayerNorm -> Dropout -> Transpose -> (B, 256, T)
  |
  +-- 3. Dilated ConvBlocks (d=1,2,4) -> h_temporal (B, 256, T)
  |
  +-- 4. Wide Conv (k=7) on LSTM output -> h_wide (B, 128, T)
  |
  +-- 5. Concatenate [h_temporal, h_wide] -> (B, 384, T) -> Fusion -> (B, 256, T)
  |
  +-- 6. Multi-Head Attention Pool (5 heads, 1 per class) -> 5x (B, 256)
  |
  +-- 7. Per-class heads -> (B, 5) logits
```

**Weight Initialization:**
- LSTM `weight_ih`: Xavier uniform
- LSTM `weight_hh`: Orthogonal
- LSTM forget gate bias: 1.0 (encourages remembering)
- Conv1d: Kaiming normal
- Linear: Xavier normal
- BatchNorm: weight=1, bias=0

**Unique checkpoint keys:** `lstm_norm.weight`, `lstm_norm.bias` (used for architecture auto-detection)

---

### 6.3. TemporalStutterClassifier (870K params)

**File:** `Models/model_temporal_w2v.py`
**Architecture choice:** `--arch temporal_w2v`

Lighter alternative without LSTM - pure dilated CNN.

**Constructor:**
```python
TemporalStutterClassifier(input_dim=768, n_classes=5, hidden_dim=256, dropout=0.3)
```

**Layer Table:**

| # | Layer | Type | Dimensions |
|---|-------|------|------------|
| 1 | `proj.0` | Conv1d | 768 -> 256, k=1, no bias |
| 2 | `proj.1` | BatchNorm1d | 256 |
| 3 | `proj.2` | GELU | - |
| 4 | `temporal_blocks.0` | ConvBlock | 256 -> 256, k=3, dil=1 |
| 5 | `temporal_blocks.1` | ConvBlock | 256 -> 256, k=3, dil=2 |
| 6 | `temporal_blocks.2` | ConvBlock | 256 -> 256, k=3, dil=4 |
| 7-10 | `wide_conv` | Conv1d+BN+GELU+Drop | 256 -> 128, k=7, pad=3 |
| 11-14 | `fusion` | Conv1d+BN+GELU+Drop | 384 -> 256, k=1 |
| 15 | `attention_pool` | AttentionPool | Linear(256->64)->Tanh->Linear(64->1) |
| 16-20 | `class_heads[0-4]` | Sequential x 5 | 256->64->GELU->Drop(0.15)->64->1 |

**Forward:** Same as BiLSTM model but without the LSTM stage - goes directly from projection to dilated ConvBlocks.

**Receptive field:** 15 frames = ~300ms at 50fps wav2vec2 output rate.

---

### 6.4. CNNBiLSTM (Baseline)

**File:** `Models/model_cnn_bilstm.py`
**Architecture choice:** `--arch cnn_bilstm`

Simple baseline CNN + BiLSTM for 123-channel spectrogram features.

**Constructor:**
```python
CNNBiLSTM(in_channels=123, cnn_out=128, lstm_hidden=256, n_classes=5, num_layers=1, dropout=0.3)
```

**Layer Table:**

| # | Layer | Type | Dimensions |
|---|-------|------|------------|
| 1 | `conv1` | Conv1d | 123 -> 128, k=5, pad=2 |
| 2 | `bn1` | BatchNorm1d | 128 |
| 3 | `conv2` | Conv1d | 128 -> 128, k=5, pad=2 |
| 4 | `bn2` | BatchNorm1d | 128 |
| 5 | `lstm` | LSTM | input=128, hidden=256, layers=1, bidirectional=True |
| 6 | `classifier.0` | Linear | 512 -> 256 |
| 7 | `classifier.1` | ReLU | - |
| 8 | `classifier.2` | Dropout | 0.3 |
| 9 | `classifier.3` | Linear | 256 -> 5 |

**Forward:** Conv1->ReLU->BN->Conv2->ReLU->BN->permute->LSTM->mean pool->classifier

---

### 6.5. ImprovedStutteringCNN

**File:** `Models/model_improved_90plus.py`
**Architecture choice:** `--arch improved_90plus`

8-block residual CNN with squeeze-excitation attention.

**Constructor:**
```python
ImprovedStutteringCNN(n_channels=123, n_classes=5, dropout=0.4)
```

**Residual Blocks (each: 2x Conv1d + BN + Dropout + skip connection):**

| Block | In -> Out | Stride |
|-------|----------|--------|
| 1 | 123 -> 64 | 1 |
| 2 | 64 -> 128 | 2 |
| 3 | 128 -> 256 | 1 |
| 4 | 256 -> 256 | 2 |
| 5 | 256 -> 512 | 2 |
| 6 | 512 -> 512 | 1 |
| 7 | 512 -> 256 | 1 |
| 8 | 256 -> 128 | 1 |

**Classifier Head:**
- `AttentionBlock(128)`: SE attention with reduction=16 -> Linear(128->8)->ReLU->Linear(8->128)->Sigmoid
- AdaptiveAvgPool1d + AdaptiveMaxPool1d -> concatenate -> `(B, 256)`
- Linear(256->64) -> ReLU -> Dropout(0.4) -> Linear(64->5)

---

### 6.6. ImprovedStutteringCNNLarge

**File:** `Models/model_improved_90plus_large.py`
**Architecture choice:** `--arch improved_90plus_large`

Largest model. Multi-scale dilated entry, SE-augmented ResBlocks, Transformer encoder, AttentivePooling, per-class heads.

**Constructor:**
```python
ImprovedStutteringCNNLarge(n_channels=123, n_classes=5, dropout=0.35, d_model=256)
```

**Multi-Scale Entry (3 parallel dilated convolutions):**
- Conv1d(123->64, k=3, dil=1) + BN + ReLU
- Conv1d(123->64, k=3, dil=2) + BN + ReLU
- Conv1d(123->64, k=3, dil=4) + BN + ReLU
- Fuse: Conv1d(192->256, k=1) + BN + ReLU

**8 SE-ResidualBlocks (each includes SEBlock with reduction=8):**

| Block | In -> Out | Stride |
|-------|----------|--------|
| 1 | 256 -> 160 | 1 |
| 2 | 160 -> 320 | 2 |
| 3 | 320 -> 640 | 1 |
| 4 | 640 -> 640 | 2 |
| 5 | 640 -> 1280 | 2 |
| 6 | 1280 -> 1280 | 1 |
| 7 | 1280 -> 640 | 1 |
| 8 | 640 -> 320 | 1 |

**Post-blocks:**
- AttentionBlock(320)
- Conv1d projection: 320 -> 256
- TransformerEncoder: 1 layer, d_model=256, nhead=4, dim_ff=512, dropout=0.35
- AttentionPool: learned query-key dot-product (attn_dim=128)
- 5x class heads: Linear(256->64)->ReLU->Dropout->Linear(64->1)

---

### 6.7. ImprovedStutteringCNNLargeSE

**File:** `Models/model_improved_90plus_se.py`
**Architecture choice:** `--arch improved_90plus_se`

SE-enhanced variant with larger channel widths.

**Constructor:**
```python
ImprovedStutteringCNNLargeSE(n_channels=123, n_classes=5, dropout=0.35)
```

**8 ResidualBlockSE blocks (SEBlock reduction=16):**

| Block | In -> Out | Stride |
|-------|----------|--------|
| 1 | 123 -> 128 | 1 |
| 2 | 128 -> 256 | 2 |
| 3 | 256 -> 512 | 1 |
| 4 | 512 -> 512 | 2 |
| 5 | 512 -> 1024 | 2 |
| 6 | 1024 -> 1024 | 1 |
| 7 | 1024 -> 512 | 1 |
| 8 | 512 -> 256 | 1 |

**Head:** AttentionBlock(256) -> avg+max pool -> cat -> Linear(512->128)->ReLU->Dropout->Linear(128->5)

---

## 7. Training System

### 7.1. End-to-End Fine-Tuning (Production)

**File:** `Models/train_w2v_finetune.py` (~1049 lines)

Hierarchical training with binary detection as PRIMARY task and 5-class classification as SECONDARY.

**Key classes:**
- `RawAudioDataset` - Loads raw WAV files, applies label cleaning + majority vote, returns `(audio_tensor, labels_5class, binary_label)`
- `FineTuneTrainer` - Full training loop with hierarchical loss, binary threshold optimization, EMA
- `ExponentialMovingAverage` - EMA of model weights with configurable decay
- `MetricsTracker` - Tracks all binary + multi-class metrics per epoch

**Hierarchical Loss:**
```python
total_loss = binary_weight * loss_binary + multiclass_weight * loss_multiclass
# Defaults: binary_weight=1.0, multiclass_weight=0.5
# Model selection uses BINARY F1 for checkpointing (not multi-class)
```

**Binary threshold optimization:** Grid search over [0.20, 0.80] step 0.05 every validation epoch to find optimal binary decision threshold.

**Data augmentation (raw audio):**
- Speed perturbation (0.9x - 1.1x)
- Time masking (zero out random segments)
- Polarity inversion (random sign flip)
- Additive Gaussian noise
- MixUp at batch level (configurable alpha)

**Optimized training config:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| Backbone LR | 2e-5 | For unfrozen wav2vec2 layers |
| Head LR | 5e-4 | For classifier heads |
| Batch size | 24 | Optimized for i7-1165G7 CPU |
| Accumulate | 1 | Effective batch = 24 |
| Freeze layers | 12 | Bottom 12 of 24 transformer layers |
| Max audio length | 48000 | 3 seconds at 16kHz |
| MixUp alpha | 0.3 | Beta distribution parameter |
| R-Drop | 0 | Disabled (doubles compute) |
| binary-weight | 1.0 | PRIMARY task |
| multiclass-weight | 0.5 | SECONDARY task |
| OMP threads | 6 | Optimal for 4-core/8-thread CPU |
| num_workers | 0 | Windows spawn overhead makes >0 slower |

**CLI (key arguments):**
```
--backbone-lr       2e-5        wav2vec2 backbone learning rate
--head-lr           5e-4        Classifier head learning rate
--freeze-layers     12          Number of transformer layers to freeze
--max-audio-len     48000       Max audio samples (truncate longer)
--binary-weight     1.0         Binary loss weight (PRIMARY)
--multiclass-weight 0.5         Multiclass loss weight (SECONDARY)
--mixup-alpha       0.3         MixUp alpha (0 = disabled)
--rdrop-weight      0           R-Drop regularization weight
--use-ema           True        Exponential Moving Average
--ema-decay         0.999       EMA decay factor
```

---

### 7.2. Frozen-Feature Training

**File:** `Models/train_90plus_final.py` (~2039 lines)

### 7.3. Data Loading & Augmentation

**`AudioDataset` class** - Supports 3 data formats:

| Format | NPZ Key | Shape | Used By |
|--------|---------|-------|---------|
| Temporal wav2vec2 | `temporal_embedding` | `(768, T)` | BiLSTM, temporal CNN |
| Embedding | `embedding` | `(D,)` e.g. 1536 | MLP |
| Spectrogram | `spectrogram` | `(123, T)` | CNN models |

**Preprocessing per format:**
- **Temporal:** Per-channel z-score normalization. Time masking augmentation (masks up to 20% of frames with zeros).
- **Spectrogram:** Full augmentation pipeline (see below) + per-channel z-score normalization.
- **Labels:** Binarized: `value > 0 -> 1.0`

**`AudioAugmentation` class** - Augmentations applied independently with individual probabilities:

| Augmentation | Default Prob | Method |
|-------------|-------------|--------|
| Time masking | 0.6 | Masks up to 30% of time frames with channel mean |
| Frequency masking | 0.5 | Masks up to 20% of frequency bins with channel mean |
| Gaussian noise | 0.4 | Additive, sigma = 0.01 |
| SNR noise | 0.25 | Random SNR 10-30 dB |
| Pitch shift | 0.25 | Roll frequency axis +/-5% bins |
| Time stretch | 0.25 | Interpolate to factor 0.95-1.05 |

All augmentation probabilities are independently configurable via CLI (`--aug-time-p`, `--aug-freq-p`, etc.).

**`collate_variable_length` function** - Pads variable-length 2D features `(C, T)` to max time dimension in batch using `torch.nn.functional.pad`. Handles 1D, 2D, and 3D tensor inputs.

**MixUp Augmentation** - Applied at batch level during training:
```python
# Beta distribution sampling
lam = np.random.beta(alpha, alpha)
idx = torch.randperm(batch_size)
mixed_x = lam * x + (1 - lam) * x[idx]
mixed_y = lam * y + (1 - lam) * y[idx]
```

### 7.4. Loss Functions

**`FocalLoss`** (default, from `Models/utils.py`):
```
loss = BCE(logits, targets) * (1 - p_t)^gamma * alpha
```
- `gamma`: 2.0 (default) - focuses on hard examples
- `alpha`: 1.0
- `pos_weight`: Auto-computed per class as `neg_count / pos_count`, clipped to [1.0, 50.0]

**`LabelSmoothingBCELoss`:**
```
smoothed_target = target * (1 - smoothing) + 0.5 * smoothing
loss = BCE(logits, smoothed_target)
```
- `smoothing`: 0.05-0.1

**`BCEWithLogitsLoss`:** Standard binary cross-entropy with optional `pos_weight`.

### 7.5. Optimizers & Schedulers

**Optimizer:** AdamW

| Parameter | Default | Pipeline Config |
|-----------|---------|-----------------|
| Learning rate | 1e-4 | 3e-4 |
| Weight decay | 1e-5 | 5e-4 |
| Betas | (0.9, 0.999) | default |

**Schedulers:**

| Type | Configuration | When |
|------|--------------|------|
| `reduce` (default) | ReduceLROnPlateau(mode='max', factor=0.5, patience=7) | Reduces LR when val F1 plateaus |
| `cosine` | 3-epoch LinearLR warmup -> CosineAnnealingLR(T_max=epochs-3, eta_min=1e-6) | Smooth decay |
| `onecycle` | OneCycleLR with `--max-lr` | Aggressive warmup + annealing |

**LR Warmup:** For `reduce` scheduler, manual linear warmup over 3 epochs from 1% -> 100% of target LR.

### 7.6. Training Techniques

| Technique | Implementation | CLI Flag |
|-----------|---------------|----------|
| **Focal Loss** | Downweights easy examples via (1-p)^gamma | `--focal-gamma 2.0` |
| **Label Smoothing** | Smooths binary targets toward 0.5 | `--use-label-smoothing --label-smoothing 0.05` |
| **MixUp** | Beta-distribution interpolation of inputs/labels | `--mixup-alpha 0.2` |
| **Oversampling** | WeightedRandomSampler with inverse-frequency weights | `--oversample rare` |
| **SWA** | Stochastic Weight Averaging via `torch.optim.swa_utils` | `--use-swa --swa-start 60 --swa-lr 1e-5` |
| **EMA** | Exponential Moving Average of model weights | `--use-ema --ema-decay 0.999` |
| **Gradient Clipping** | `torch.nn.utils.clip_grad_norm_` | `--grad-clip 1.0` |
| **Gradient Accumulation** | Accumulates gradients over N batches | `--accumulate N` |
| **Threshold Optimization** | Per-class F1 grid search every epoch [0.25, 0.85] step 0.025 | `--save-thresholds` |
| **Mixed Precision (AMP)** | `torch.cuda.amp.GradScaler` - CUDA only | Automatic on GPU |
| **Layer Freezing** | Freeze layers by prefix for N epochs, then unfreeze | `--freeze-prefix --freeze-epochs` |
| **IPEX** | Intel Extension for PyTorch optimization | `--device ipex` |
| **DirectML** | AMD/Intel GPU via DirectML | `--device dml` |
| **Early Stopping** | Stops when val F1 hasn't improved for N epochs | `--early-stop 25` |
| **Deterministic Seeds** | Sets Python, NumPy, Torch random seeds | `--seed 42` |

**SWA Details:**
```python
swa_model = torch.optim.swa_utils.AveragedModel(model)
swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=1e-5)
# Starts averaging at epoch --swa-start
# Final: update_bn(train_loader, swa_model) to fix BatchNorm stats
```

**Class Weight Computation:**
```python
pos_weight[c] = clamp(neg_count[c] / pos_count[c], min=1.0, max=50.0)
# Neutralized to 1.0 when --oversample is used (avoids double-upweighting)
```

**Checkpointing:** Full state dict saved every epoch:
```python
{
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'scheduler_state_dict': ...,
    'scaler_state_dict': ...,      # AMP scaler (CUDA only)
    'ema_shadow': ...,             # EMA weights
    'rng_states': {                # For exact reproducibility
        'python': ...,
        'numpy': ...,
        'torch': ...,
        'cuda': ...
    },
    'epoch': ...,
    'best_f1': ...,
    'metrics': ...
}
```

### 7.7. All CLI Arguments (Frozen-Feature Training)

```
Training Arguments:
  --epochs              60          Number of training epochs
  --batch-size          64          Batch size
  --data-dir            datasets/features    Feature directory
  --lr                  1e-4        Learning rate
  --weight-decay        1e-5        AdamW weight decay
  --dropout             0.2         Model dropout rate
  --grad-clip           1.0         Max gradient norm
  --seed                None        Random seed for reproducibility
  --resume              None        Resume from checkpoint path

Architecture:
  --arch                improved_90plus    Model architecture
      choices: improved_90plus, improved_90plus_large, improved_90plus_se,
               cnn_bilstm, embedding_mlp, temporal_w2v, temporal_bilstm

Loss & Regularization:
  --focal-gamma         2.0         Focal loss gamma
  --loss-type           None        focal / bce / label_smoothing
  --use-label-smoothing False       Enable label smoothing
  --label-smoothing     0.1         Smoothing factor
  --use-bce             False       Use plain BCE loss
  --neutral-pos-weight  False       Set all pos_weight to 1.0

Augmentation:
  --mixup-alpha         0.0         MixUp alpha (0 = disabled)
  --aug-time-p          None        Time masking probability
  --aug-freq-p          None        Frequency masking probability
  --aug-noise-p         None        Gaussian noise probability
  --aug-stretch-p       None        Time stretch probability
  --aug-pitch-p         None        Pitch shift probability
  --aug-snr-p           None        SNR noise probability

Scheduler:
  --scheduler           reduce      reduce / onecycle / cosine
  --max-lr              None        Max LR for OneCycleLR
  --sched-patience      None        ReduceLROnPlateau patience

Training Techniques:
  --use-swa             False       Enable Stochastic Weight Averaging
  --swa-start           80          Epoch to start SWA
  --swa-lr              1e-5        SWA learning rate
  --use-ema             False       Enable EMA
  --ema-decay           0.999       EMA decay factor
  --accumulate          1           Gradient accumulation steps
  --oversample          none        none / rare / weight
  --sampler-replacement False       Sample with replacement
  --early-stop          None        Early stopping patience
  --freeze-prefix       None        Layer name prefix to freeze
  --freeze-up-to        None        Freeze layers up to this name
  --freeze-epochs       None        Unfreeze after N epochs

Calibration & Thresholds:
  --auto-calibrate      False       Auto-calibrate thresholds
  --save-thresholds     False       Save per-class thresholds
  --thresh-min-precision 0.2        Minimum precision for thresholds

System:
  --gpu                 True        Use GPU if available
  --device              auto        auto / cpu / cuda / dml / ipex
  --num-workers         2           DataLoader worker processes
  --omp-threads         4           OpenMP threads
  --verbose             False       Extra logging
```

---

## 8. Production Inference

**File:** `Models/detect.py` (290 lines)

Production-ready detection pipeline for single files, directories, or long audio segments. Uses the trained Wav2VecFineTuneClassifier checkpoint for hierarchical detection.

**Key features:**
- Loads model checkpoint, applies optimal binary threshold
- Sliding window inference for long audio (configurable window/stride)
- Outputs JSON with per-segment and per-type probabilities
- Supports batch processing of directories
- CLI interface:

```sh
python Models/detect.py --audio input.wav --output results.json
python Models/detect.py --audio_dir my_wavs/ --output_dir results/
```

**Output:**
- For each audio file: binary stutter detection, per-type probabilities, segment-level results for long files
- JSON format for easy downstream use

---

## 9. Calibration & Evaluation

### Threshold Calibration

**File:** `Models/calibrate_thresholds.py`

Two-stage calibration on validation data:

1. **Temperature Scaling:** Learns a single scalar temperature T to calibrate predicted probabilities:
   ```
   calibrated_logits = logits / T
   ```
   Optimized via Adam (lr=0.01, 300 steps) on BCE NLL. Temperature clamped to [0.05, 10.0].

2. **Per-Class Threshold Search:** After temperature scaling, grid-searches optimal threshold per class:
   ```
   search range: [0.01, 0.99], step 0.01
   metric: per-class F1 score
   ```

**Output:** `output/thresholds.json`
```json
{
    "thresholds": [0.45, 0.48, 0.52, 0.38, 0.41],
    "per_class_metrics": { "..." : "..." },
    "temperature": 1.23
}
```

Auto-detects 7 model architectures from checkpoint state dict keys:
- `lstm_norm.` -> TemporalBiLSTMClassifier
- `proj.` + `temporal_blocks.` -> TemporalStutterClassifier
- `ms_conv1.` -> ImprovedStutteringCNNLarge
- `block1.se.` -> ImprovedStutteringCNNLargeSE
- `block1.conv1.` -> ImprovedStutteringCNN
- `conv1.` + `lstm.` -> CNNBiLSTM

### Validation Evaluation

**File:** `Models/eval_validation.py`

Loads checkpoint + val data, computes:
- Per-class AUC-ROC and Average Precision
- Generates plots: ROC curves, PR curves, probability histograms, confusion matrices
- Writes `metrics.json` with all results

**Output:** `output/eval_<model_name>/`

---

## 10. Self-Training Label Refinement

**File:** `Models/self_train_refine.py`

Uses a trained model's confident predictions to fix noisy labels in the training data. Addresses the ~20% inter-annotator disagreement in SEP-28k.

**Algorithm:**
```
For each sample (feature, label) in train and val:
    prediction = model(feature)  # sigmoid probability

    For each class c:
        if prediction[c] > confidence_high (0.90) AND label[c] == 0:
            -> Flip label to 1  (model is confident this IS present)

        if prediction[c] < confidence_low (0.10) AND label[c] == 1:
            -> Flip label to 0  (model is confident this is NOT present)

    Save feature (unchanged) + refined label to output directory
```

**CLI:**
```
--checkpoint        Path to trained model checkpoint
--data-dir          Input feature directory
--output-dir        Output directory for refined features
--confidence-high   0.90    Threshold for flipping 0->1
--confidence-low    0.10    Threshold for flipping 1->0
```

Reports per-class flip statistics (how many labels were changed in each direction).

---

## 11. Ensemble Evaluation with TTA

**File:** `Models/ensemble_eval.py`

Combines three techniques for maximum accuracy:

### Multi-Model Ensemble
Loads N model checkpoints (trained with different seeds), averages their predicted probabilities:
```python
avg_probs = mean([model_1(x), model_2(x), model_3(x)])
```

### Test-Time Augmentation (TTA)
For each sample, creates N time-shifted versions via circular shift, averages predictions:
```python
def tta_predict(model, x, n_shifts=5):
    preds = [model(x)]  # Original
    for i in range(n_shifts):
        shift = (i + 1) * (T // (n_shifts + 1))
        x_shifted = torch.roll(x, shifts=shift, dims=-1)  # Circular
        preds.append(model(x_shifted))
    return mean(preds)
```

### Per-Class Threshold Optimization
Grid search on [0.1, 0.9] step 0.01 to maximize F1 per class.

**CLI:**
```
--checkpoints    Path(s) to model checkpoints (nargs +)
--data-dir       Feature directory
--tta-shifts     5       Number of TTA time shifts
--output-dir     output/ensemble_eval
```

**Output:**
- `ensemble_results.json` - Full metrics, per-class F1, macro F1, AUC
- `ensemble_thresholds.json` - Optimized thresholds (also copied to `output/thresholds.json`)

---

## 12. Stutter Repair

**File:** `Models/repair_advanced.py` (725 lines)
**Class:** `AdvancedStutterRepair`

Detects stutter regions in audio using the trained model, then repairs them using:

1. **Phase reconstruction** - Griffin-Lim algorithm or phase vocoder
2. **Spectral inpainting** - Interpolates magnitude spectrum across stutter boundaries
3. **Vocoder reconstruction** - Synthesizes clean speech from modified spectral representation

Falls back to SciPy-only processing when librosa is unavailable.

---

## 13. Pipeline Scripts

### Main Pipeline: `scripts/run_90plus_pipeline.ps1` (345 lines)

The complete 6-step training pipeline:

```
STEP 1: Extract multi-layer wav2vec2 temporal features (layers 7-12)
    +-- python Models/extract_wav2vec2_temporal.py --layer-mode weighted-avg

STEP 2: Stage 1 Training (seed 42, original labels)
    +-- python Models/train_90plus_final.py --arch temporal_bilstm --seed 42 ...

STEP 3: Self-Training Label Refinement
    +-- python Models/self_train_refine.py --confidence-high 0.90 --confidence-low 0.10

STEP 4: Stage 2 Training (seed 42, refined labels)
    +-- python Models/train_90plus_final.py --data-dir refined/ --seed 42 ...

STEP 5: Ensemble Training (seeds 123, 777 on refined labels)
    +-- python Models/train_90plus_final.py --seed 123 ...
    +-- python Models/train_90plus_final.py --seed 777 ...

STEP 6: Ensemble + TTA Evaluation
    +-- python Models/ensemble_eval.py --tta-shifts 5 --checkpoints [3 best checkpoints]
```

**Pipeline Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Architecture | `temporal_bilstm` |
| Epochs | 100 |
| Batch size | 128 |
| Learning rate | 3e-4 |
| Weight decay | 5e-4 |
| Scheduler | cosine (3-epoch warmup) |
| Focal gamma | 2.0 |
| MixUp alpha | 0.2 |
| Label smoothing | 0.05 |
| SWA start | epoch 60 |
| SWA LR | 1e-5 |
| Early stopping | 25 epochs |
| Gradient clipping | 1.0 |
| Oversampling | rare |
| DataLoader workers | 4 |
| PyTorch threads | 8 |

**CPU Optimization (set in pipeline):**
```powershell
$env:OMP_NUM_THREADS = "8"
$env:MKL_NUM_THREADS = "8"
$env:NUMEXPR_MAX_THREADS = "8"
$env:OPENBLAS_NUM_THREADS = "8"
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c  # High Performance
```

**Usage:**
```powershell
cd D:\Bunny\AGNI
conda activate agni311
.\scripts\run_90plus_pipeline.ps1
```

### Production Fine-Tuning Pipeline: `scripts/run_finetune_pipeline.ps1` (168 lines)

End-to-end pipeline for hierarchical detection system (wav2vec2-large fine-tuning):

```
STEP 1: Launches training with all optimized settings for CPU (see table below)
STEP 2: Monitors system resources, sets High Performance power plan
STEP 3: Handles all CLI arguments, environment variables, and logging
STEP 4: Runs `Models/train_w2v_finetune.py` with correct config
```

**Pipeline Hyperparameters:**
```
Model: Wav2VecFineTuneClassifier
Backbone LR: 2e-5
Head LR: 5e-4
Batch size: 24
Accumulate: 1
Freeze layers: 12
Max audio len: 48000
MixUp alpha: 0.3
R-Drop: 0
binary-weight: 1.0
multiclass-weight: 0.5
OMP threads: 6
num_workers: 0
Early stopping: 25
```

**CPU Optimization (set in pipeline):**
```powershell
$env:OMP_NUM_THREADS = "6"
$env:MKL_NUM_THREADS = "6"
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c  # High Performance
```

**Usage:**
```powershell
cd D:\Bunny\AGNI
conda activate agni311
.\scripts\run_finetune_pipeline.ps1
```

### Other Pipeline Scripts

| Script | Description |
|--------|-------------|
| `scripts/run_temporal_pipeline.ps1` | Extract temporal features -> train TemporalStutterClassifier -> calibrate -> evaluate |
| `scripts/run_wav2vec2_pipeline.ps1` | Extract wav2vec2 embeddings -> train MLP -> calibrate -> evaluate |
| `scripts/run_experiments.ps1` | Single training run with specific hyperparameters |
| `scripts/experiment_suite.ps1` | Generates suite of HP experiments |

### Data Utility Scripts

| Script | Description |
|--------|-------------|
| `scripts/fix_label_audio_match.py` | Audits label-to-audio file mapping, creates symlinks for mismatches |
| `scripts/make_speaker_split.py` | Creates speaker-holdout train/val/test splits |
| `scripts/inspect_problem_files.py` | Generates waveform + spectrogram PNGs for problem audio |
| `scripts/debug_extraction_trace.py` | Debugs feature extraction for specific files |
| `scripts/estimate_batch_size.py` | Estimates safe batch size from available memory |
| `scripts/probe_max_batch.py` | Binary-searches max batch size using synthetic model |
| `scripts/probe_real_model.py` | Binary-searches max batch size using real CNNBiLSTM |

---

## 14. Tools & Utilities

All in `tools/`:

### Analysis & Ranking

| Tool | Description |
|------|-------------|
| `analyze_runs.py` | Scans all training runs, ranks top 10 by val F1 macro |
| `check_best.py` | Prints best epoch F1/P/R/AUC for a specific run |
| `check_metrics.py` | Prints best/last F1 for the 8 most recent runs |
| `check_thresholds.py` | Inspects threshold records from a training run |
| `find_best_run.py` | Finds best run by chosen metric, locates checkpoint |

### Hyperparameter Search

| Tool | Description |
|------|-------------|
| `hp_sweep_random.py` | Random hyperparameter search: spawns training subprocesses with sampled LR, WD, batch, etc. |
| `hp_sweep_analyze.py` | Analyzes HP sweep results, prints top N trials |
| `collect_hp_sweep_results.py` | Collects all HP sweep results into `hp_sweep_results.json` |

### Data Quality

| Tool | Description |
|------|-------------|
| `inspect_dataset.py` | Inspects NPZ files: label distributions, shapes, NaN/Inf counts |
| `compute_class_weights.py` | Computes per-class label counts and BCE pos_weight |
| `standardize_labels.py` | Standardizes multi-level labels (2, 3) -> binary (0/1) |

### Error Analysis & Label Correction

| Tool | Description |
|------|-------------|
| `export_top_errors.py` | Exports top FP/FN errors to folders + CSV for human review |
| `generate_corrections_template.py` | Creates label correction template from error analysis |
| `apply_label_corrections.py` | Applies manual label corrections from CSV to NPZ files |
| `make_apply_csv_from_template.py` | Converts correction template to applicable format |

### Ensemble

| Tool | Description |
|------|-------------|
| `ensemble_checkpoints.py` | Ensembles multiple checkpoints by averaging probabilities |

---

## 15. Tests

All in `tests/`:

### Unit Tests (5)

| Test | Description |
|------|-------------|
| `test_augmentation.py` | Tests AudioAugmentation with zero-filled spectrogram |
| `test_calibrate_thresholds_synth.py` | Tests threshold calibration with synthetic random data |
| `test_cnn_bilstm.py` | Forward-pass test for CNNBiLSTM (123 channels -> 5 outputs) |
| `test_preprocessor.py` | Tests EnhancedAudioPreprocessor with dummy audio, rejects short audio |
| `test_preprocessor_array.py` | Tests preprocessor with various audio lengths |

**Run tests:**
```powershell
cd D:\Bunny\AGNI
conda activate agni311
python -m pytest tests/ -v
```

### Diagnostic Scripts (20, prefixed `tmp_`)

| Script | Description |
|--------|-------------|
| `tmp_apply_labels.py` | Re-applies label mappings to existing NPZ files |
| `tmp_check_preproc.py` | Quick inspection of preprocessor defaults |
| `tmp_check_silence.py` | Scans NPZ files for high-silence samples |
| `tmp_count_corrupted_files.py` | Lists corrupted audio files |
| `tmp_count_labels.py` | Counts per-class positive labels |
| `tmp_diagnose_extract_import.py` | Tests extract_features_90plus import |
| `tmp_import_test.py` | Verifies EnhancedAudioPreprocessor import |
| `tmp_import_train_test.py` | Verifies train_90plus_final parses/imports |
| `tmp_inspect_npz.py` | Reads one NPZ file and prints keys/shapes |
| `tmp_label_stats.py` | Per-class positive counts and fractions |
| `tmp_make_small_features.py` | Copies 200+50 NPZ files for fast iteration |
| `tmp_model_check.py` | Instantiates CNNBiLSTM, prints param count, dummy forward |
| `tmp_monitor.ps1` | Monitors CPU%, RAM, top Python processes |
| `tmp_quick_eval.py` | Quick eval of a checkpoint with threshold loading |
| `tmp_reprocess_problematic.py` | Re-extracts features from problematic samples |
| `tmp_smoke_input_test.py` | End-to-end smoke: dataset -> DataLoader -> model forward |
| `tmp_smoke_training_no_torch.py` | NumPy-only BCE loss sanity check |
| `tmp_test_checkpoint.py` | Inspects extraction checkpoint JSON |
| `tmp_trainer_smoke.py` | 3-epoch smoke training with 200+50 samples |
| `tmp_verify_extraction.py` | Full extraction verification: NaN/Inf/shape checks |

---

## 16. Constants & Configuration

**File:** `Models/constants.py`

| Constant | Value | Description |
|----------|-------|-------------|
| `TOTAL_CHANNELS` | 123 | 80 mel + 13 MFCC + 13 delta + 13 delta-delta + 4 spectral |
| `NUM_CLASSES` | 5 | Prolongation, Block, SoundRep, WordRep, Interjection |
| `DEFAULT_SAMPLE_RATE` | 16000 | Audio sample rate |
| `AUG_TIME_MASK_P` | 0.6 | Default time masking probability |
| `AUG_FREQ_MASK_P` | 0.5 | Default frequency masking probability |
| `AUG_NOISE_P` | 0.4 | Default Gaussian noise probability |
| `AUG_STRETCH_P` | 0.25 | Default time stretch probability |
| `AUG_PITCH_P` | 0.25 | Default pitch shift probability |
| `AUG_SNR_P` | 0.25 | Default SNR noise probability |
| `THRESH_SEARCH_START` | 0.25 | Threshold search range start |
| `THRESH_SEARCH_END` | 0.85 | Threshold search range end |
| `THRESH_SEARCH_STEP` | 0.025 | Threshold search step size |
| `SCHEDULER_PATIENCE` | 7 | ReduceLROnPlateau patience |
| `THRESHOLD_OPT_EPOCHS` | 5 | Epochs between threshold optimization |

---

## 17. Directory Structure

```
D:\Bunny\AGNI\
|-- README.md                           # This file
|-- requirements_complete.txt           # Pinned dependencies
|
|-- datasets/
|   |-- SEP-28k_labels.csv             # SEP-28k 3-annotator labels
|   |-- SEP-28k_episodes.csv           # Episode metadata
|   |-- fluencybank_labels.csv         # FluencyBank labels
|   |-- fluencybank_episodes.csv       # Episode metadata
|   |-- label_audio_map.json           # Label stem -> audio path mapping
|   |-- clips/
|   |   +-- stuttering-clips/clips/    # Raw WAV audio files (16kHz)
|   |-- features/                      # 123-channel spectrogram NPZs
|   |   |-- train/*.npz                # ~25,445 files
|   |   |-- val/*.npz                  # ~6,470 files
|   |   +-- _label_cache.pkl           # Label cache for fast loading
|   |-- features_w2v_temporal/         # wav2vec2 temporal NPZs (768, T)
|   |   |-- train/*.npz
|   |   +-- val/*.npz
|   |-- features_w2v_temporal_refined/ # Self-training refined labels
|   |   |-- train/*.npz
|   |   +-- val/*.npz
|   |-- corrupted_audio/               # Staged corrupted files
|   +-- problematic_samples/           # Flagged problem files
|
|-- Models/
|   |-- __init__.py
|   |-- constants.py                   # Project constants
|   |-- utils.py                       # FocalLoss, multilabel_f1, diagnostics
|   |
|   |-- # --- Feature Extraction ---
|   |-- enhanced_audio_preprocessor.py # 123-channel feature extractor
|   |-- extract_features_90plus.py     # Batch spectrogram extraction pipeline
|   |-- extract_features.py            # Re-exports extract_features_90plus
|   |-- extract_wav2vec2_temporal.py   # wav2vec2 temporal feature extraction
|   |
|   |-- # --- Model Architectures ---
|   |-- model_w2v_finetune.py           # Wav2Vec2-large fine-tuning model (production)
|   |-- train_w2v_finetune.py           # Hierarchical training system (production)
|   |-- detect.py                       # Production inference pipeline
|   |-- model_temporal_bilstm.py       # TemporalBiLSTMClassifier (2.07M, primary)
|   |-- model_temporal_w2v.py          # TemporalStutterClassifier (870K)
|   |-- model_cnn_bilstm.py            # CNNBiLSTM (baseline)
|   |-- model_improved_90plus.py       # ImprovedStutteringCNN
|   |-- model_improved_90plus_large.py # ImprovedStutteringCNNLarge
|   |-- model_improved_90plus_se.py    # ImprovedStutteringCNNLargeSE
|   |
|   |-- # --- Training & Evaluation ---
|   |-- train_90plus_final.py          # Training system (~2039 lines)
|   |-- calibrate_thresholds.py        # Temperature scaling + threshold search
|   |-- eval_validation.py             # Validation evaluation + plots
|   |-- self_train_refine.py           # Self-training label refinement
|   |-- ensemble_eval.py               # Ensemble + TTA evaluation
|   |
|   |-- # --- Diagnostics ---
|   |-- diagnose_best_checkpoint.py    # Per-sample FP/FN analysis
|   |-- diagnostic_checks.py           # Pipeline integrity checks
|   |-- inspect_probs.py               # Raw probability inspection
|   |-- inspect_val_stats.py           # Validation set statistics
|   |-- produce_evaluation_plots.py    # ROC, PR, confusion matrix plots
|   |
|   |-- # --- Repair ---
|   |-- repair_advanced.py             # Vocoder-based stutter repair (725 lines)
|   |-- augment_repetitions.py         # Synthetic word repetition augmentation
|   |-- COMPLETE_PIPELINE.py           # End-to-end orchestration
|   |
|   |-- checkpoints/                   # Saved model checkpoints
|   |   +-- training_YYYYMMDD_HHMMSS/  # Per-run checkpoint directories
|   |       |-- <arch>_best.pth
|   |       +-- <arch>_epoch_<N>.pth
|   +-- annotator/
|       +-- annotator.html             # Browser-based Whisper alignment annotator
|
|-- scripts/
|   |-- run_90plus_pipeline.ps1        # Main 6-step training pipeline
|   |-- run_temporal_pipeline.ps1      # Temporal CNN pipeline
|   |-- run_wav2vec2_pipeline.ps1      # wav2vec2 embedding pipeline
|   |-- run_experiments.ps1            # Single training run
|   |-- experiment_suite.ps1           # HP experiment generator
|   |-- fix_label_audio_match.py       # Label-to-audio mapping audit
|   |-- make_speaker_split.py          # Speaker-holdout splits
|   |-- inspect_problem_files.py       # Problem file visualization
|   |-- debug_extraction_trace.py      # Extraction debugging
|   |-- estimate_batch_size.py         # Batch size estimator
|   |-- probe_max_batch.py             # Max batch finder (synthetic)
|   +-- probe_real_model.py            # Max batch finder (real model)
|
|-- tools/
|   |-- analyze_runs.py                # Rank runs by F1
|   |-- check_best.py                  # Best epoch details
|   |-- check_metrics.py               # Recent runs summary
|   |-- check_thresholds.py            # Threshold inspection
|   |-- find_best_run.py               # Find best run + checkpoint
|   |-- hp_sweep_random.py             # Random HP search
|   |-- hp_sweep_analyze.py            # Analyze HP sweep results
|   |-- collect_hp_sweep_results.py    # Collect HP sweep data
|   |-- inspect_dataset.py             # Dataset quality inspection
|   |-- compute_class_weights.py       # Class weight computation
|   |-- standardize_labels.py          # Label standardization
|   |-- export_top_errors.py           # Error analysis export
|   |-- generate_corrections_template.py # Correction template
|   |-- apply_label_corrections.py     # Apply label corrections
|   |-- make_apply_csv_from_template.py # Correction CSV conversion
|   +-- ensemble_checkpoints.py        # Quick checkpoint ensemble
|
|-- tests/                             # Unit tests + diagnostic scripts
|   |-- test_augmentation.py
|   |-- test_calibrate_thresholds_synth.py
|   |-- test_cnn_bilstm.py
|   |-- test_preprocessor.py
|   |-- test_preprocessor_array.py
|   +-- tmp_*.py                       # 20 diagnostic/temporary scripts
|
|-- Online_test/
|   +-- I Have a Stutter 60 Second Docs.mp3  # Sample test audio
|
|-- output/
|   |-- thresholds.json                # Current best thresholds
|   |-- ensemble_eval/                 # Ensemble evaluation results
|   |-- training_YYYYMMDD_HHMMSS/      # Per-run metrics & logs
|   +-- eval_*/                        # Evaluation outputs
|
+-- ffmpeg/                            # FFmpeg binary (for audio conversion)
```

---

## 17. Training History & Results

### Best Results by Architecture

| Architecture | Best Val F1 | Epochs | Config | Notes |
|-------------|------------|--------|--------|-------|
| CNNBiLSTM | 0.604 | 16 (early stop) | batch=96, lr=auto, oversample=rare, EMA | Baseline on 123-ch spectrograms |
| ImprovedStutteringCNN | 0.539 | 60 | batch=64, lr=1e-4 | AUC ~0.75, per-class AUCs: [0.77, 0.67, 0.77, 0.72, 0.83] |
| MLP on wav2vec2 embeddings | 0.554 | - | Mean+std pooling destroys temporal info | Not competitive |
| TemporalBiLSTM | TBD | - | Currently running pipeline | Expected: 0.75-0.85 |

### Best CNN-BiLSTM Run Details

```
Checkpoint: Models/checkpoints/training_20260219_015326/cnn_bilstm_best.pth
Command: python Models/train_90plus_final.py --epochs 100 --batch-size 96
         --arch cnn_bilstm --oversample rare --auto-calibrate --verbose
         --num-workers 2 --omp-threads 4 --accumulate 4 --use-ema
         --sched-patience 5 --early-stop 15

Per-class AUCs: [0.6174, 0.6124, 0.6261, 0.6144, 0.6428]
Per-class APs:  [0.3979, 0.5078, 0.3056, 0.2418, 0.4819]
```

### ImprovedStutteringCNN Run Details

```
60 epochs, best val F1 = 0.5391
Precision: 0.4217, Recall: 0.7706
Per-class AUCs: [0.77, 0.67, 0.77, 0.72, 0.83]
```

### Current Pipeline (Expected)

Training the TemporalBiLSTMClassifier with:
- Multi-layer wav2vec2 temporal features (layers 7-12 weighted average)
- Self-training label refinement
- 3-model ensemble (seeds 42, 123, 777)
- Test-time augmentation (5 shifts)
- Per-class threshold optimization

**Expected metrics:**

| Metric | Expected Range |
|--------|---------------|
| Macro F1 | 0.75 - 0.85 |
| Mean class accuracy | 0.85 - 0.92 |
| AUC-ROC | 0.88 - 0.95 |

---

## 18. Troubleshooting

### Common Issues

**torchaudio import error:**
```
ModuleNotFoundError: No module named 'torchaudio._internal.fb'
```
Solution: We don't use torchaudio. All audio loading uses `scipy.io.wavfile` + `scipy.signal.resample_poly`.

**PowerShell encoding error (em-dash U+2014):**
```
The string is missing the terminator: "
```
Solution: Ensure all `.ps1` files use ASCII-only characters. No em-dashes, smart quotes, or non-ASCII.

**Slow label loading (31k+ NPZ reads):**
Solution: Fixed with pickle cache at `datasets/features/_label_cache.pkl`. Delete cache to force rebuild.

**Out of memory during training:**
Solution: Reduce `--batch-size` (try 64 or 32). Check with `scripts/estimate_batch_size.py`.

**fbgemm.dll error on Windows:**
Solution: Use CPU-only PyTorch: `pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu`

**Extraction resuming / skipping files:**
Solution: The extraction script skips existing `.npz` files. Delete `datasets/features_w2v_temporal/` to force full re-extraction.

### Performance Tips

1. **Power Plan:** Set to High Performance: `powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c`
2. **Thread count:** Set `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to your logical CPU count
3. **Batch size:** With 40 GB RAM, batch 128 is safe for temporal features
4. **DataLoader workers:** 4 workers for 4-core CPU
5. **Page file:** Set to at least 8 GB for 40 GB RAM systems

---

## 19. Other Issues

**PowerShell parsing error with parentheses in double-quoted strings:**
```
"(90+ F1)" is interpreted as a PowerShell expression
```
Solution: Use single quotes for any string containing parentheses or plus signs in PowerShell scripts.

**OMP thread/env mismatch:**
```
OMP_NUM_THREADS=8, --omp-threads=6
```
Solution: Always set both to the same value (6 is optimal for i7-1165G7 4c/8t).

**R-Drop doubles compute:**
Solution: Set `--rdrop-weight 0` to disable for 50% speedup on CPU.

**num_workers > 0 is slower on Windows:**
Solution: Use `--num-workers 0` for best CPU performance (Windows spawn overhead).

**Label noise / low F1:**
Solution: Use majority vote (≥2 annotators) and remove all Unsure/PoorAudio/NoSpeech/Music labels for cleanest training set.

**Training is CPU-bound at 95%:**
Solution: No config can fix this; only faster CPU will help. RAM is not the bottleneck.

---

## License

Internal research project.

---

*Last updated: March 3, 2026*
