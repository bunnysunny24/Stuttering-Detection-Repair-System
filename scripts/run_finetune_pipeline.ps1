# =============================================================================
# HIERARCHICAL STUTTERING DETECTION — PRODUCTION PIPELINE v3
# Binary detection (90+ F1) as PRIMARY + 5-class type as SECONDARY
# End-to-end wav2vec2-large fine-tuning on CPU
# =============================================================================
#
# ARCHITECTURE — Hierarchical Detection:
#   PRIMARY:   Binary stutter detection (stutter vs fluent) → 90+ F1 target
#   SECONDARY: 5-class type (Prolongation/Block/SoundRep/WordRep/Interjection)
#
# OPTIMIZATIONS:
#   - 12 unfrozen layers — 2x adaptation capacity, 3.2 GB RAM
#   - Label cleaning: removes Unsure/PoorAudio/NoSpeech/Music clips
#   - Majority vote (>=2 annotators) — kills single-annotator noise
#   - Binary detection as primary loss (weight=1.0 vs multiclass=0.5)
#   - Model selection on BINARY F1 (not 5-class macro F1)
#   - MixUp augmentation (alpha=0.3) on raw audio
#   - R-Drop consistency regularization
#   - Speed perturbation, time masking, polarity inversion augmentations
#   - Cosine annealing with warm restarts (T0=10, Tmult=2)
#   - Binary threshold optimization in validation
#
# Expected per epoch: ~2-3.5 hrs on i7-1165G7
# Memory: ~3.2 GB peak (batch=8, 12 unfrozen layers)
#
# Usage: powershell -ExecutionPolicy Bypass -File scripts/run_finetune_pipeline.ps1
# =============================================================================

$ErrorActionPreference = "Continue"

# Set High Performance power plan
try {
    $highPerfGuid = (powercfg /list | Select-String "High performance" | ForEach-Object { ($_ -split '\s+')[3] })
    if ($highPerfGuid) {
        powercfg /setactive $highPerfGuid
        Write-Host "[OK] Set High Performance power plan" -ForegroundColor Green
    }
} catch {
    Write-Host "[WARN] Could not set power plan" -ForegroundColor Yellow
}

# Environment
$env:OMP_NUM_THREADS = "6"
$env:MKL_NUM_THREADS = "6"
$env:TOKENIZERS_PARALLELISM = "false"

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host '  HIERARCHICAL STUTTERING DETECTION - PRODUCTION v3' -ForegroundColor Cyan
Write-Host '  PRIMARY: Binary detection (90+ F1 target)' -ForegroundColor Cyan
Write-Host '  SECONDARY: 5-class type classification' -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# Activate conda environment
try {
    conda activate agni311
    Write-Host "[OK] Activated conda env: agni311" -ForegroundColor Green
} catch {
    Write-Host "[WARN] Could not activate conda env" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[INFO] wav2vec2-large (~1.2 GB) will be auto-downloaded on first run" -ForegroundColor Yellow
Write-Host "[INFO] Subsequent runs use the HuggingFace cache" -ForegroundColor Yellow
Write-Host ""

# =============================================================================
# STAGE 1: Fine-tune wav2vec2-large with all optimizations
# =============================================================================

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  STAGE 1: Hierarchical fine-tuning (binary PRIMARY)" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

$trainArgs = @(
    "Models/train_w2v_finetune.py"

    # Data - CLEAN labels with majority vote
    "--clips-dir", "datasets/clips/stuttering-clips/clips"
    "--features-dir", "datasets/features"
    "--label-csvs", "datasets/SEP-28k_labels.csv", "datasets/fluencybank_labels.csv"
    "--min-votes", "2"
    # Label cleaning ON by default (removes ~4300 bad clips)
    # Majority vote ON by default (kills single-annotator noise)

    # Model - 12 unfrozen layers (was 6)
    "--model-name", "facebook/wav2vec2-large"
    "--freeze-layers", "12"
    "--hidden-dim", "256"
    "--lstm-hidden", "128"
    "--dropout", "0.3"

    # Training - optimized for CPU throughput
    "--epochs", "60"
    "--batch-size", "24"
    "--accumulate", "1"
    "--backbone-lr", "2e-5"
    "--head-lr", "5e-4"
    "--grad-clip", "0.5"
    "--focal-gamma", "2.0"
    "--cosine-t0", "10"
    "--early-stop", "15"
    "--max-audio-len", "48000"

    # Regularization - R-Drop OFF (saves 2nd forward pass = 50% speedup)
    "--mixup-alpha", "0.3"
    "--rdrop-weight", "0"
    "--use-ema"
    "--ema-decay", "0.999"

    # Hierarchical loss weights — binary PRIMARY
    "--binary-weight", "1.0"
    "--multiclass-weight", "0.5"

    # System (optimized for i7-1165G7 + 40GB RAM)
    "--num-workers", "0"
    "--omp-threads", "6"
    "--seed", "42"
)

Write-Host ""
Write-Host "Training command:" -ForegroundColor Yellow
Write-Host "  python $($trainArgs -join ' ')" -ForegroundColor Gray
Write-Host ""
Write-Host "Key settings:" -ForegroundColor Yellow
Write-Host '  - BINARY detection = PRIMARY task (weight=1.0, 90+ F1 target)' -ForegroundColor Gray
Write-Host '  - 5-class types = SECONDARY task (weight=0.5)' -ForegroundColor Gray
Write-Host '  - Model selection by BINARY F1 (not macro F1)' -ForegroundColor Gray
Write-Host '  - 12 unfrozen layers, clean labels + majority vote' -ForegroundColor Gray
Write-Host '  - MixUp=0.3, R-Drop=OFF (2x faster), cosine annealing' -ForegroundColor Gray
Write-Host '  - batch=24, accumulate=1, effective=24, early stop=15' -ForegroundColor Gray
Write-Host '  - max-audio-len=48000 (3s cap), ~10 GB peak' -ForegroundColor Gray
Write-Host ""
Write-Host "Expected: ~7-9 hrs per epoch, 60 epochs max" -ForegroundColor Yellow
Write-Host "Memory: ~7-8 GB peak" -ForegroundColor Yellow
Write-Host ""

$startTime = Get-Date
python @trainArgs
$exitCode = $LASTEXITCODE
$elapsed = (Get-Date) - $startTime

Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "[OK] Training completed in $($elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Green
} else {
    Write-Host "[FAIL] Training failed with exit code $exitCode after $($elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Red
}

# =============================================================================
# DONE
# =============================================================================

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  PIPELINE COMPLETE" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Check results:" -ForegroundColor Yellow
Write-Host "  - Checkpoints: Models/checkpoints/finetune_*/" -ForegroundColor Gray
Write-Host "  - Best model:  Models/checkpoints/w2v_finetune_BEST.pth" -ForegroundColor Gray
Write-Host "  - Metrics:     output/finetune_*/finetune_metrics.json" -ForegroundColor Gray
Write-Host "  - Logs:        output/finetune_*/finetune_*.log" -ForegroundColor Gray
Write-Host "" 
Write-Host "Run detection on new audio:" -ForegroundColor Yellow
Write-Host "  python Models/detect.py --audio path/to/audio.wav" -ForegroundColor Gray
Write-Host "  python Models/detect.py --audio path/to/folder/ --output results.json" -ForegroundColor Gray
Write-Host ""
