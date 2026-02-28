<#
Experiment suite generator for AGNI training experiments.
This script writes ready-to-run commands for the prioritized experiments described
and can optionally execute them if you set `$RunNow = $false` to `$true`.

Usage: .\scripts\experiment_suite.ps1  (edit variables at top)
#>

param()

# === Configuration - edit these values ===
$RunNow = $true        # if $true the script will run commands sequentially (use with care)
$Python = "$env:CONDA_PREFIX\python.exe"  # uses current conda env python if run inside env
$BestCheckpoint = "Models/checkpoints/training_20260221_023548/cnn_bilstm_best.pth"
$DataDir = "datasets/features"
$EpochsFineTune = 10
$Batch = 192
$NumWorkers = 4
$OutCmdFile = "output/experiment_commands.txt"
$LogDir = "output/experiments_logs"

Write-Host "Generating experiment commands -> $OutCmdFile"
New-Item -ItemType Directory -Path (Split-Path $OutCmdFile) -Force | Out-Null

New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

$commands = @()

##############################
# Fine-tune variants
##############################

# 1a) Baseline fine-tune: OneCycleLR + focal (original suggestion)
$commands += "# Fine-tune A: OneCycleLR + focal (baseline)"
$commands += "$Python Models/train_90plus_final.py --arch cnn_bilstm --resume $BestCheckpoint --epochs $EpochsFineTune --batch-size $Batch --scheduler onecycle --max-lr 0.003 --loss-type focal --focal-gamma 2 --use-ema --num-workers $NumWorkers --verbose"

# 1b) Conservative fine-tune: lower max-lr, MixUp + SpecAugment
$commands += "# Fine-tune B: lower max-lr + MixUp + SpecAugment (recommended to reduce overfit)"
$commands += "$Python Models/train_90plus_final.py --arch cnn_bilstm --resume $BestCheckpoint --epochs $EpochsFineTune --batch-size $Batch --scheduler onecycle --max-lr 0.0005 --loss-type focal --focal-gamma 2 --mixup-alpha 0.2 --aug-time-p 0.45 --aug-freq-p 0.45 --use-ema --num-workers $NumWorkers --verbose"

# 1c) Regularized fine-tune: weight-decay + label smoothing
$commands += "# Fine-tune C: weight-decay + label-smoothing"
$commands += "$Python Models/train_90plus_final.py --arch cnn_bilstm --resume $BestCheckpoint --epochs $EpochsFineTune --batch-size $Batch --scheduler onecycle --max-lr 0.0005 --weight-decay 1e-4 --use-label-smoothing --label-smoothing 0.1 --use-ema --num-workers $NumWorkers --verbose"

# 1d) Longer low-LR fine-tune (safer) - small LR for more epochs
$commands += "# Fine-tune D: longer low-LR fine-tune"
$commands += "$Python Models/train_90plus_final.py --arch cnn_bilstm --resume $BestCheckpoint --epochs 20 --batch-size 128 --scheduler onecycle --max-lr 0.0003 --loss-type focal --focal-gamma 2 --mixup-alpha 0.1 --aug-time-p 0.35 --aug-freq-p 0.35 --use-ema --num-workers $NumWorkers --verbose"

# 2) Hyperparameter sweep (random short trials)
$commands += "# HP sweep: focused random search (max-lr, weight-decay, mixup, batch) - short trials"
$commands += "$Python tools/hp_sweep_random.py --trials 32 --epochs 3 --concurrency 4 --num-workers $NumWorkers --search-space 'lr:1e-5,3e-5,1e-4,3e-4,1e-3 wd:0.0,1e-6,1e-5,1e-4 mixup_alpha:0.0,0.05,0.1,0.2 batch:64,128,256 oversample:none,weight loss_type:focal,bce'"

# 3) Ensembling top-3 checkpoints (template) - choose checkpoints list and run
$commands += "# Ensemble: evaluate averaged probabilities for top-3 checkpoints (edited to use BEST_OVERALL and two recent bests)"
$commands += "python tools/ensemble_checkpoints.py --checkpoints Models/checkpoints/cnn_bilstm_BEST_OVERALL.pth Models/checkpoints/training_20260221_112440/cnn_bilstm_best.pth Models/checkpoints/training_20260221_023548/cnn_bilstm_best.pth --data-dir $DataDir --batch-size 64"

# 4) Data augmentation experiments
$commands += "# Augment A: stronger SpecAugment + MixUp (30 epochs)"
$commands += "$Python Models/train_90plus_final.py --epochs 30 --batch-size $Batch --aug-time-p 0.45 --aug-freq-p 0.45 --mixup-alpha 0.2 --use-ema --num-workers $NumWorkers --verbose"
$commands += "# Augment B: SpecAugment + stronger noise + stretch"
$commands += "$Python Models/train_90plus_final.py --epochs 30 --batch-size $Batch --aug-time-p 0.40 --aug-freq-p 0.40 --aug-noise-p 0.2 --aug-stretch-p 0.2 --mixup-alpha 0.1 --use-ema --num-workers $NumWorkers --verbose"

# 5) Oversampling experiments
$commands += "# Oversample A: weight oversample with replacement + neutral pos_weight"
$commands += "$Python Models/train_90plus_final.py --epochs 30 --batch-size 128 --oversample weight --sampler-replacement --neutral-pos-weight --use-ema --num-workers $NumWorkers --verbose"
$commands += "# Oversample B: rare oversample (no replacement)"
$commands += "$Python Models/train_90plus_final.py --epochs 30 --batch-size 128 --oversample rare --use-ema --num-workers $NumWorkers --verbose"

# Small utility: compute class positive counts and suggested pos_weight (verify sampler vs pos_weight)
$commands += "# Compute class weights from features"
$commands += "$Python tools/compute_class_weights.py --data-dir $DataDir"

# Recommended oversample flow: run compute_class_weights then run oversample training with neutral pos_weight
$commands += "# Recommended oversample flow: compute weights then train (use --neutral-pos-weight to avoid double-upweighting)"
$commands += "$Python tools/compute_class_weights.py --data-dir $DataDir"
$commands += "$Python Models/train_90plus_final.py --epochs 30 --batch-size 128 --oversample weight --sampler-replacement --neutral-pos-weight --use-ema --num-workers $NumWorkers --verbose"

# 6) Capacity increase: larger model
$commands += "# Capacity A: train improved_90plus_large (higher capacity)"
$commands += "$Python Models/train_90plus_final.py --epochs 60 --batch-size 64 --arch improved_90plus_large --dropout 0.35 --use-ema --num-workers $NumWorkers --verbose"
$commands += "# Capacity B: train improved_90plus_large but freeze early layers for fine-tune (example uses new --freeze-up-to)"
$commands += "$Python Models/train_90plus_final.py --epochs 30 --batch-size 64 --arch improved_90plus_large --dropout 0.35 --freeze-up-to 3 --use-ema --num-workers $NumWorkers --verbose"
$commands += "# Capacity B2: freeze first 3 modules for 5 epochs then unfreeze (--freeze-epochs example)"
$commands += "$Python Models/train_90plus_final.py --epochs 40 --batch-size 64 --arch improved_90plus_large --dropout 0.35 --freeze-up-to 3 --freeze-epochs 5 --use-ema --num-workers $NumWorkers --verbose"
$commands += "# Capacity C: conservative train improved_90plus_large (lower LR, more dropout)"
$commands += "$Python Models/train_90plus_final.py --epochs 40 --batch-size 64 --arch improved_90plus_large --dropout 0.4 --scheduler reduce --lr 5e-5 --weight-decay 1e-5 --mixup-alpha 0.1 --use-ema --num-workers $NumWorkers --verbose"
$commands += "# Capacity D: train SE-augmented model (improved_90plus_se)"
$commands += "$Python Models/train_90plus_final.py --epochs 40 --batch-size 64 --arch improved_90plus_se --dropout 0.35 --scheduler reduce --lr 5e-5 --weight-decay 1e-5 --mixup-alpha 0.1 --use-ema --num-workers $NumWorkers --verbose"

# 7) Pseudo-label / self-training steps (multi-step)
$commands += "# Pseudo-step 1: produce probabilities on unlabeled set (use ensemble or single ckpt)"
$commands += "python tools/ensemble_checkpoints.py --checkpoints Models/checkpoints/training_20260221_023548/cnn_bilstm_best.pth --data-dir datasets/features_unlabeled --batch-size 128"
# 7) Pseudo-labeling (removed - requires custom script)
# Pseudo-labeling steps were removed because they require a helper script `tools/create_pseudo_labels.py`.
# If you implement that script, re-add commands here to generate pseudo labels and retrain.

# 8) Manual audit and dataset inspection utilities
$commands += "# Inspect dataset and stats (label distribution, corrupted files)"
$commands += "python tools/inspect_dataset.py --root datasets/features"
$commands += "# Produce evaluation plots for a checkpoint (PR/ROC):"
$commands += "python Models/produce_evaluation_plots.py --metrics-json output/training_20260221_023548/cnn_bilstm_metrics.json --out output/plots_run1"

# 10) Scheduler comparison: ReduceLROnPlateau variant (less aggressive than OneCycle for fine-tuning)
$commands += "# Scheduler: ReduceLROnPlateau fine-tune (safer than OneCycle)"
$commands += "$Python Models/train_90plus_final.py --arch cnn_bilstm --resume $BestCheckpoint --epochs $EpochsFineTune --batch-size $Batch --scheduler reduce --lr 3e-4 --weight-decay 1e-5 --loss-type focal --focal-gamma 2 --mixup-alpha 0.1 --aug-time-p 0.35 --aug-freq-p 0.35 --use-ema --num-workers $NumWorkers --verbose"

# 9) Promote best checkpoint to production (manual confirm before running)
$commands += "# Promote best checkpoint to production (edit path if desired)"
$commands += "Copy-Item -Path Models/checkpoints/training_20260221_023548/cnn_bilstm_best.pth -Destination Models/checkpoints/cnn_bilstm_PRODUCTION.pth -Force"

# 11) Summarize all runs and pick best checkpoint
$commands += "# Summarize runs and pick best checkpoint"
$commands += "$Python tools/find_best_run.py --metrics-root output --top 5"

# Write commands to file
[System.IO.File]::WriteAllLines($OutCmdFile, $commands)
Write-Host "Wrote $(($commands).Count) lines to $OutCmdFile"

if ($RunNow) {
    Write-Host "Running commands sequentially (RunNow = true). Be prepared to monitor logs."
    foreach ($c in $commands) {
        if ($c.Trim().StartsWith('#') -or [string]::IsNullOrWhiteSpace($c)) { continue }
        Write-Host "Running: $c"
        iex $c
    }
} else {
    Write-Host "RunNow is false. Review $OutCmdFile and run commands you want." 
}

Write-Host "Experiment suite prepared. Edit $OutCmdFile to adjust commands before running."
