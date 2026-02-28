# Simple runner: set params, train, calibrate, evaluate

# --- User-editable hyperparameters (from best run) ---
$Epochs = 120
$BatchSize = 128
$NumWorkers = 8
$OmpThreads = 8

# Model / training settings (use best model)
$Arch = "improved_90plus_large"
$UseEMA = $true
$EMA_Decay = 0.999
$Dropout = 0.3
$MixupAlpha = 0.2
$LR = 5e-5
$WeightDecay = 1e-5
$Seed = 42
$SaveThresholds = $true
$Device = "cpu"

$py = "python"

# threading
$env:OMP_NUM_THREADS = $OmpThreads
$env:MKL_NUM_THREADS = $OmpThreads

Write-Host "Running training: Arch=$Arch, Epochs=$Epochs, BatchSize=$BatchSize"

# Manual resume: set `$ResumePath` at top of this script to resume from a checkpoint.
# If empty, training will start fresh. Example to resume:
# $ResumePath = 'Models/checkpoints/training_20260225_023137/improved_90plus_large_best.pth'
$ResumePath = $null

$ResumeArg = ""
if ($ResumePath -and (Test-Path $ResumePath)) {
    Write-Host "User-specified resume path: $ResumePath"
    $ResumeArg = "--resume `"$ResumePath`""
} else {
    Write-Host 'No resume path set; starting training from scratch.'
}

$trainCmd = "$py -u Models/train_90plus_final.py --epochs $Epochs --batch-size $BatchSize --arch $Arch --use-ema --ema-decay $EMA_Decay --dropout $Dropout --num-workers $NumWorkers --omp-threads $OmpThreads --mixup-alpha $MixUpAlpha --accumulate 2 --loss-type focal --focal-gamma 1.5 --lr $LR --weight-decay $WeightDecay --seed $Seed --save-thresholds --device $Device --oversample none $ResumeArg"
Write-Host $trainCmd
Invoke-Expression $trainCmd
Write-Host $trainCmd
Invoke-Expression $trainCmd

# Helper to find best checkpoint (simple preference for *_best.pth)
function Get-BestCheckpoint() {
    if (-not (Test-Path Models\checkpoints)) { return $null }
    $best = Get-ChildItem -Path Models\checkpoints -Recurse -Filter '*_best.pth' -File | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($best) { return $best.FullName }
    $any = Get-ChildItem -Path Models\checkpoints -Recurse -Filter '*.pth' -File | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($any) { return $any.FullName }
    return $null
}

Write-Host 'Training finished - running calibration and evaluation if checkpoint present'
$best = Get-BestCheckpoint
if ($best -and (Test-Path $best)) {
    Write-Host "Found best checkpoint: $best"
    & $py Models/calibrate_thresholds.py --checkpoint $best --data-dir datasets/features --out output/thresholds.json
    & $py Models/eval_validation.py --checkpoint $best --data-dir datasets/features
    Write-Host "Done. Check outputs under output/ and Models/checkpoints."
} else {
    Write-Host "No checkpoint found under Models/checkpoints"
}
