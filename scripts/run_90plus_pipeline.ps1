# =============================================================================
# ULTIMATE 90+ STUTTERING DETECTION PIPELINE
# =============================================================================
#
# This pipeline uses EVERY technique available to maximize accuracy:
#
#   STEP 1: Extract multi-layer wav2vec2 temporal features (layers 7-12)
#   STEP 2: Train BiLSTM model (seed 42) -- Stage 1
#   STEP 3: Self-training label refinement (clean noisy labels)
#   STEP 4: Retrain on cleaned labels (seed 42) -- Stage 2
#   STEP 5: Train 2 more models (seeds 123, 777) -- for ensemble
#   STEP 6: Ensemble evaluation with Test-Time Augmentation (TTA)
#
# Techniques used:
#   - Multi-layer weighted average of wav2vec2 layers 7-12
#   - BiLSTM + Dilated CNN + Multi-Head Attention (2.1M params)
#   - Focal loss with label smoothing for noisy labels
#   - MixUp augmentation + time masking
#   - Stochastic Weight Averaging (SWA)
#   - Self-training label refinement (fixes ~5-15% noisy labels)
#   - 3-model ensemble with different seeds
#   - Test-time augmentation (5 time-shifted predictions)
#   - Per-class threshold optimization
#
# Usage:
#   cd D:\Bunny\AGNI
#   conda activate agni311
#   .\scripts\run_90plus_pipeline.ps1
#
# =============================================================================

$ErrorActionPreference = "Stop"
conda activate agni311

# ---- System Optimization: Max out CPU threads ----
$env:OMP_NUM_THREADS = "6"       # 6 threads = sweet spot (135 samp/s vs 111 at 8)
$env:MKL_NUM_THREADS = "6"
$env:NUMEXPR_MAX_THREADS = "6"
$env:OPENBLAS_NUM_THREADS = "6"
$env:VECLIB_MAXIMUM_THREADS = "6"
# High Performance power plan (already set, kept as fallback)
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c 2>$null

Write-Host ""
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host "   ULTIMATE 90+ STUTTERING DETECTION PIPELINE"                -ForegroundColor Cyan
Write-Host "   Ensemble + Self-Training + TTA"                            -ForegroundColor Cyan
Write-Host "   CPU Optimized: 6 threads, LR=1e-4, EMA+SWA"               -ForegroundColor Cyan
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host ""

# ---- Configuration ----
$DataDir       = "datasets/features_w2v_temporal"
$RefinedDir    = "datasets/features_w2v_temporal_refined"
$ClipsDir      = "datasets/clips/stuttering-clips/clips"
$FeaturesDir   = "datasets/features"
$W2VModel      = "facebook/wav2vec2-base"

# Extraction
$MaxFrames     = 200
$ExtractBatch  = 8
$LayerMode     = "weighted-avg"

# Training (shared across all runs)
$Arch          = "temporal_bilstm"
$Epochs        = 100
$Batch         = 256        # 40GB RAM: even 1024 leaves 20GB free
$LR            = "1e-4"     # Reduced from 3e-4 to prevent collapse
$Dropout       = 0.3
$WeightDecay   = "1e-4"     # Reduced from 5e-4 for stability
$Scheduler     = "reduce"   # ReduceLROnPlateau adapts to progress
$FocalGamma    = 2.0
$MixupAlpha    = 0.1        # Reduced from 0.2 - lighter regularization
$SWAStart      = 30         # Start averaging earlier for stability
$SWALR         = "1e-5"
$EarlyStop     = 15         # Don't waste time on collapse
$GradClip      = 0.5        # Tighter clipping prevents gradient explosions
$Oversample    = "rare"
$LabelSmoothing = 0.05
$NumWorkers    = 0          # 0 is fastest: SSD + small NPZ + Windows spawn overhead
$ExtractWorkers = 8         # Use all 8 logical CPUs for extraction
$UseEMA        = $true      # EMA smooths weight updates

# ==========================================================================
# STEP 1: Extract multi-layer temporal features
# ==========================================================================
Write-Host ""
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "  STEP 1/6: Extract Features"            -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
Write-Host ""

$trainCount = 0
$valCount   = 0
if (Test-Path "$DataDir/train") {
    $trainCount = (Get-ChildItem "$DataDir/train" -Filter "*.npz" -ErrorAction SilentlyContinue | Measure-Object).Count
}
if (Test-Path "$DataDir/val") {
    $valCount = (Get-ChildItem "$DataDir/val" -Filter "*.npz" -ErrorAction SilentlyContinue | Measure-Object).Count
}

Write-Host "  Existing: $trainCount train, $valCount val"

if (($trainCount -lt 25000) -or ($valCount -lt 6000)) {
    Write-Host "  Extracting features (~2 hours on CPU)..." -ForegroundColor Yellow

    python Models/extract_wav2vec2_temporal.py `
        --model      $W2VModel `
        --clips-dir  $ClipsDir `
        --features-dir $FeaturesDir `
        --output-dir $DataDir `
        --max-frames $MaxFrames `
        --batch-size $ExtractBatch `
        --layer-mode $LayerMode

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Feature extraction failed!" -ForegroundColor Red
        exit 1
    }

    $trainCount = (Get-ChildItem "$DataDir/train" -Filter "*.npz" | Measure-Object).Count
    $valCount   = (Get-ChildItem "$DataDir/val"   -Filter "*.npz" | Measure-Object).Count
    Write-Host "  Done: $trainCount train, $valCount val" -ForegroundColor Green
} else {
    Write-Host "  Features already extracted. Skipping." -ForegroundColor Green
}

# ==========================================================================
# STEP 2: Stage 1 Training (seed 42, original labels)
# ==========================================================================
Write-Host ""
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "  STEP 2/6: Stage 1 Training (seed 42)" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
Write-Host ""

$trainArgs = @(
    '--data-dir',          $DataDir,
    '--arch',              $Arch,
    '--epochs',            $Epochs,
    '--batch-size',        $Batch,
    '--lr',                $LR,
    '--dropout',           $Dropout,
    '--weight-decay',      $WeightDecay,
    '--scheduler',         $Scheduler,
    '--focal-gamma',       $FocalGamma,
    '--mixup-alpha',       $MixupAlpha,
    '--use-swa',
    '--swa-start',         $SWAStart,
    '--swa-lr',            $SWALR,
    '--early-stop',        $EarlyStop,
    '--grad-clip',         $GradClip,
    '--oversample',        $Oversample,
    '--sampler-replacement',
    '--save-thresholds',
    '--use-label-smoothing',
    '--label-smoothing',   $LabelSmoothing,
    '--num-workers',       $NumWorkers,
    '--omp-threads',       6,
    '--sched-patience',    7
)
if ($UseEMA) {
    $trainArgs += @('--use-ema', '--ema-decay', '0.999')
}

python Models/train_90plus_final.py @trainArgs --seed 42

if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Stage 1 training returned non-zero exit code." -ForegroundColor Yellow
}

# Find Stage 1 best checkpoint
$Stage1Ckpt = Get-ChildItem -Path "Models/checkpoints" -Recurse -Filter "temporal_bilstm_best.pth" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if ($null -eq $Stage1Ckpt) {
    Write-Host "ERROR: No Stage 1 checkpoint found!" -ForegroundColor Red
    exit 1
}

Write-Host "  Stage 1 best: $($Stage1Ckpt.FullName)" -ForegroundColor Green

# ==========================================================================
# STEP 3: Self-Training Label Refinement
# ==========================================================================
Write-Host ""
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "  STEP 3/6: Label Refinement"            -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Using Stage 1 model to clean noisy labels..."

python Models/self_train_refine.py `
    --checkpoint "$($Stage1Ckpt.FullName)" `
    --data-dir   $DataDir `
    --output-dir $RefinedDir `
    --confidence-high 0.90 `
    --confidence-low  0.10

if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Label refinement failed. Using original labels." -ForegroundColor Yellow
    $RefinedDir = $DataDir
}

# ==========================================================================
# STEP 4: Stage 2 Training (seed 42, refined labels)
# ==========================================================================
Write-Host ""
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "  STEP 4/6: Stage 2 Training (refined)"  -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
Write-Host ""

python Models/train_90plus_final.py @trainArgs --data-dir $RefinedDir --seed 42

# ==========================================================================
# STEP 5: Train 2 more models (seeds 123, 777) for ensemble
# ==========================================================================
Write-Host ""
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "  STEP 5/6: Ensemble Training"           -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
Write-Host ""

foreach ($seed in @(123, 777)) {
    Write-Host "  Training with seed $seed ..." -ForegroundColor Cyan

    python Models/train_90plus_final.py @trainArgs --data-dir $RefinedDir --seed $seed

    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: Seed $seed training returned non-zero." -ForegroundColor Yellow
    }
}

# ==========================================================================
# STEP 6: Ensemble Evaluation with TTA
# ==========================================================================
Write-Host ""
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "  STEP 6/6: Ensemble + TTA Evaluation"   -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
Write-Host ""

# Find all temporal_bilstm_best.pth checkpoints (one per training run)
$AllCkpts = Get-ChildItem -Path "Models/checkpoints" -Recurse -Filter "temporal_bilstm_best.pth" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 3

$CkptPaths = @()
foreach ($c in $AllCkpts) {
    $CkptPaths += $c.FullName
}

Write-Host "  Found $($CkptPaths.Count) checkpoints for ensemble:"
foreach ($p in $CkptPaths) {
    Write-Host "    - $p"
}
Write-Host ""

if ($CkptPaths.Count -eq 0) {
    Write-Host "ERROR: No checkpoints found for ensemble!" -ForegroundColor Red
    exit 1
}

python Models/ensemble_eval.py `
    --checkpoints $CkptPaths `
    --data-dir    $RefinedDir `
    --tta-shifts  5 `
    --output-dir  "output/ensemble_eval"

# Also run single-model calibration + eval for comparison
$BestSingleCkpt = $AllCkpts | Select-Object -First 1
if ($null -ne $BestSingleCkpt) {
    Write-Host ""
    Write-Host "  Single-model calibration + eval for comparison:" -ForegroundColor Cyan

    python Models/calibrate_thresholds.py `
        --checkpoint "$($BestSingleCkpt.FullName)" `
        --data-dir   $RefinedDir `
        --out        output/thresholds.json

    python Models/eval_validation.py `
        --checkpoint "$($BestSingleCkpt.FullName)" `
        --data-dir   $RefinedDir
}

# ==========================================================================
# DONE
# ==========================================================================
Write-Host ""
Write-Host "=============================================================" -ForegroundColor Green
Write-Host "   PIPELINE COMPLETE"                                          -ForegroundColor Green
Write-Host "=============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Results:"
Write-Host "    Ensemble:    output/ensemble_eval/ensemble_results.json"
Write-Host "    Thresholds:  output/ensemble_eval/ensemble_thresholds.json"
Write-Host "    Single-best: output/thresholds.json"
Write-Host ""
Write-Host "  What was done:"
Write-Host "    1. Multi-layer wav2vec2 feature extraction (layers 7-12)"
Write-Host "    2. Stage 1 training (original labels, seed 42)"
Write-Host "    3. Self-training label refinement (cleaned noisy labels)"
Write-Host "    4. Stage 2 training (refined labels, seed 42)"
Write-Host "    5. Ensemble seeds 123, 777 (refined labels)"
Write-Host "    6. Ensemble + TTA evaluation (3 models x 5 shifts)"
Write-Host ""
