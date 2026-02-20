# Run multiple training/eval experiments for quick comparison (PowerShell)
# Usage: .\run_experiments.ps1 [-Epochs 1]
# Edit the variables below to tweak experiments without changing the Python script invocation
# Defaults tuned from quick experiments (Experiment B was best)
$Epochs = 60
$BatchSize = 256
$NumWorkers = 8
$OmpThreads = 8

# Training hyperparameters you can tweak
$Arch = "cnn_bilstm"
$UseEMA = $true
$EMA_Decay = 0.999
$Dropout = 0.3
$Verbose = $true
$Accumulate = 1
$EarlyStop = 8
$SchedPatience = 5
$AutoCalibrate = $true
$MixupAlpha = 0.2
$LR = 3e-4
$WeightDecay = 1e-5
$LossType = "focal"
$FocalGamma = 2.0
$Scheduler = "reduce"
$Seed = 42
$SaveThresholds = $true
# Default to CPU to avoid IPEX import failures on systems without IPEX
$Device = "cpu"

# Oversampling / sampler options
$Oversample = "none"    # set to 'none' or 'rare' (default: none -> use pos_weight)
$NeutralPosWeight = $false
$SamplerReplacement = $false

# Misc
$VerboseFlag = "--verbose"

$py = "python"

# Export recommended thread environment vars for BLAS/OpenMP
$env:OMP_NUM_THREADS = $OmpThreads
$env:MKL_NUM_THREADS = $OmpThreads

# Helper: conditional flags
function flag-if([bool]$cond, [string]$flag) {
    if ($cond) { return $flag } else { return "" }
}

# Build flags string from variables
$useEmaFlag = flag-if $UseEMA "--use-ema"
$verboseFlag = flag-if $Verbose "--verbose"
$autoCalFlag = flag-if $AutoCalibrate "--auto-calibrate"
$saveThreshFlag = flag-if $SaveThresholds "--save-thresholds"
$neutralPosFlag = flag-if $NeutralPosWeight "--neutral-pos-weight"
$samplerReplacementFlag = flag-if $SamplerReplacement "--sampler-replacement"

$baseParams = "-u Models/train_90plus_final.py --epochs $Epochs --batch-size $BatchSize --arch $Arch $useEmaFlag --ema-decay $EMA_Decay --dropout $Dropout $verboseFlag --num-workers $NumWorkers --omp-threads $OmpThreads --accumulate $Accumulate --early-stop $EarlyStop --sched-patience $SchedPatience $autoCalFlag --mixup-alpha $MixupAlpha --lr $LR --weight-decay $WeightDecay --loss-type $LossType --focal-gamma $FocalGamma --scheduler $Scheduler --seed $Seed $saveThreshFlag --device $Device"

# Experiment A: Oversample rare + neutral pos_weight (recommended if using oversampling)
Write-Host "=== EXPERIMENT A: oversample=rare + neutral-pos-weight ==="
$cmdA = "$py $baseParams --oversample rare $neutralPosFlag $samplerReplacementFlag"
Write-Host $cmdA
Invoke-Expression $cmdA

# Evaluate the most-recent checkpoint produced by the last run (best model)
Write-Host "Running evaluation for Experiment A best checkpoint (if present)"
$ckptDir = Get-ChildItem -Path Models\checkpoints -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($ckptDir) {
    $best = Join-Path $ckptDir.FullName "cnn_bilstm_best.pth"
    if (Test-Path $best) {
        Write-Host "Evaluating checkpoint: $best"
        & $py Models/eval_validation.py --checkpoint $best --data-dir datasets/features
    } else {
        Write-Host "Best checkpoint not found at $best"
    }
}

# Experiment B: No oversampling, use pos_weight computed from dataset
Write-Host "=== EXPERIMENT B: oversample=none (use pos_weight) ==="
$cmdB = "$py $baseParams --oversample none"
Write-Host $cmdB
Invoke-Expression $cmdB

# Experiment C: Full exact command the user provided (uses base params + oversample+neutral+sampler)
Write-Host "=== EXPERIMENT C: Full user-specified run ==="
$cmdC = "$py $baseParams --oversample $Oversample $neutralPosFlag $samplerReplacementFlag"
Write-Host $cmdC
Invoke-Expression $cmdC

Write-Host "Done. Review outputs under output/ and Models/checkpoints/."
Write-Host "Tip: increase -Epochs to 60 for full training runs."
