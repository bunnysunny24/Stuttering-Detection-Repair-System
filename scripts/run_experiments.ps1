# Baseline training runner: single focused run to improve model (no oversampling)
# Usage: .\run_experiments.ps1

# --- User-editable hyperparameters ---
$Epochs = 60
$BatchSize = 256
$NumWorkers = 8
$OmpThreads = 8

# Model / training settings
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
$MaxLR = 3e-3
$Seed = 42
$SaveThresholds = $true
$Device = "cpu"

# Oversampling: baseline uses none (do not oversample)
$Oversample = "none"
$NeutralPosWeight = $false
$SamplerReplacement = $false

$py = "python"

# BLAS / threading suggestions
$env:OMP_NUM_THREADS = $OmpThreads
$env:MKL_NUM_THREADS = $OmpThreads

function flag-if([bool]$cond, [string]$flag) {
    if ($cond) { return $flag } else { return "" }
}

$useEmaFlag = flag-if $UseEMA "--use-ema"
$verboseFlag = flag-if $Verbose "--verbose"
$autoCalFlag = flag-if $AutoCalibrate "--auto-calibrate"
$saveThreshFlag = flag-if $SaveThresholds "--save-thresholds"
$neutralPosFlag = flag-if $NeutralPosWeight "--neutral-pos-weight"
$samplerReplacementFlag = flag-if $SamplerReplacement "--sampler-replacement"

# Build and run baseline training command (no oversample)
$baseCmd = "$py -u Models/train_90plus_final.py --epochs $Epochs --batch-size $BatchSize --arch $Arch $useEmaFlag --ema-decay $EMA_Decay --dropout $Dropout $verboseFlag --num-workers $NumWorkers --omp-threads $OmpThreads --accumulate $Accumulate --early-stop $EarlyStop --sched-patience $SchedPatience $autoCalFlag --mixup-alpha $MixupAlpha --lr $LR --weight-decay $WeightDecay --loss-type $LossType --focal-gamma $FocalGamma --scheduler $Scheduler --max-lr $MaxLR --seed $Seed $saveThreshFlag --device $Device --oversample none"

Write-Host "Running baseline training (no oversample) with $Epochs epochs"
Write-Host $baseCmd
Invoke-Expression $baseCmd

# Helper to find latest checkpoint folder
function Get-LatestCheckpointFolder() {
    $ckptDir = Get-ChildItem -Path Models\checkpoints -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    return $ckptDir
}

# After training: run calibration (if best checkpoint found) and evaluation
Write-Host 'Training finished - running calibration and evaluation if checkpoint present'
$latest = Get-LatestCheckpointFolder
if ($latest) {
    $best = Join-Path $latest.FullName "cnn_bilstm_best.pth"
    if (Test-Path $best) {
        Write-Host "Calibrating thresholds for $best -> output/thresholds.json"
        & $py Models/calibrate_thresholds.py --checkpoint $best --data-dir datasets/features --out output/thresholds.json
        Write-Host "Evaluating best checkpoint: $best"
        & $py Models/eval_validation.py --checkpoint $best --data-dir datasets/features
        Write-Host "Baseline complete. Best checkpoint: $best"
    } else {
        Write-Host "No best checkpoint found in $($latest.FullName)"
    }
} else {
    Write-Host 'No checkpoint folders found under Models/checkpoints'
}

Write-Host 'Done. Review outputs under output/ and Models/checkpoints.'