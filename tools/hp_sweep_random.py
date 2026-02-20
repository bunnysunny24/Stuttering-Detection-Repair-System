import subprocess
import random
import time
import json
import os
import sys
import threading
import argparse
from pathlib import Path
import numpy as np
import concurrent.futures

OUT = Path('output')
# use same python executable as caller
BASE_CMD = [sys.executable, 'Models/train_90plus_final.py']

# Lock to serialize writes to the results file when using threads
WRITE_LOCK = threading.Lock()


def parse_search_space(spec_str):
    """Parse a search-space spec like "lr:1e-4,3e-4,1e-3 wd:1e-6,1e-5" into a dict."""
    space = {}
    if not spec_str:
        return space
    parts = spec_str.split()
    for p in parts:
        if ':' not in p:
            continue
        k, v = p.split(':', 1)
        vals = v.split(',')
        parsed = []
        for vv in vals:
            vv = vv.strip()
            # try parse numeric
            try:
                if vv.lower().startswith('1e') or 'e' in vv.lower() or '.' in vv:
                    parsed.append(float(vv))
                else:
                    parsed.append(int(vv))
            except Exception:
                parsed.append(vv)
        space[k] = parsed
    return space


def sample_from_space(space, default_random=True):
    cfg = {}
    # continuous/logspace sampling for lr/wd if not enumerated
    if 'lr' in space:
        cfg['lr'] = random.choice(space['lr'])
    elif default_random:
        cfg['lr'] = 10 ** random.uniform(-4, -2)
    if 'wd' in space:
        cfg['wd'] = random.choice(space['wd'])
    elif default_random:
        cfg['wd'] = 10 ** random.uniform(-6, -3)
    # batch
    if 'batch' in space:
        cfg['batch'] = random.choice(space['batch'])
    else:
        cfg['batch'] = 128
    # dropout
    if 'dropout' in space:
        cfg['dropout'] = random.choice(space['dropout'])
    else:
        cfg['dropout'] = 0.2
    # mixup
    if 'mixup_alpha' in space:
        cfg['mixup_alpha'] = random.choice(space['mixup_alpha'])
    else:
        cfg['mixup_alpha'] = 0.0
    # oversample
    if 'oversample' in space:
        cfg['oversample'] = random.choice(space['oversample'])
    else:
        cfg['oversample'] = 'none'
    # loss type
    if 'loss_type' in space:
        cfg['loss_type'] = random.choice(space['loss_type'])
    else:
        cfg['loss_type'] = 'focal'
    return cfg


def run_trial(trial_idx, space, short_epochs, default_num_workers):
    cfg = sample_from_space(space)
    lr = cfg.get('lr')
    wd = cfg.get('wd')
    batch = cfg.get('batch')
    dropout = cfg.get('dropout')
    mixup = cfg.get('mixup_alpha')
    oversample = cfg.get('oversample')
    loss_type = cfg.get('loss_type')

    cmd = BASE_CMD + [
        '--epochs', str(short_epochs),
        '--batch-size', str(batch),
        '--dropout', str(dropout),
        '--early-stop', '2',
        '--sched-patience', '2',
        '--num-workers', str(default_num_workers),
        '--use-ema',
        '--verbose',
        '--lr', str(lr),
        '--weight-decay', str(wd)
    ]

    # loss flags
    if isinstance(loss_type, str):
        if loss_type.lower() == 'bce':
            cmd.append('--use-bce')
        elif loss_type.lower() == 'label_smoothing' or loss_type.lower() == 'label-smoothing':
            cmd.append('--use-label-smoothing')
            # forward a default smoothing value
            cmd += ['--label-smoothing', '0.1']
    # oversample
    if oversample and str(oversample).lower() != 'none':
        cmd += ['--oversample', str(oversample)]
    # mixup
    if mixup and float(mixup) > 0.0:
        cmd += ['--mixup-alpha', str(mixup)]

    lr_s = f"{lr:.6g}" if lr is not None else "None"
    wd_s = f"{wd:.6g}" if wd is not None else "None"
    print(f'Running trial {trial_idx+1} lr={lr_s} wd={wd_s} batch={batch} dropout={dropout} mixup={mixup} oversample={oversample} loss={loss_type}')
    # record start time to find the correct output folder for this trial
    start_time = time.time()
    proc = subprocess.run(cmd)

    # find folders created/modified after we started the trial
    time.sleep(0.5)
    candidates = [p for p in OUT.glob('training_*') if p.is_dir() and p.stat().st_mtime >= start_time - 1.0]
    if not candidates:
        # fallback: take latest folder (best-effort)
        folders = sorted([p for p in OUT.glob('training_*') if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
        if not folders:
            print('No output folder found for trial', trial_idx+1)
            result = {'trial': trial_idx+1, 'lr': lr, 'wd': wd, 'batch': batch, 'metrics_file': None, 'val_f1': None, 'rc': proc.returncode}
            _write_result_atomic(result)
            return result
        latest = folders[0]
    else:
        latest = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    metrics_files = list(latest.glob('*_metrics.json'))
    metrics_file = None
    val_f1 = None
    if metrics_files:
        metrics_file = str(metrics_files[0])
        try:
            metrics = json.load(open(metrics_file))
            val_entries = metrics.get('val', [])
            val_f1 = val_entries[-1].get('f1_macro') if len(val_entries) > 0 else None
        except Exception:
            val_f1 = None
    else:
        print('No metrics json in', latest)

    # include training log if present for easier debugging
    log_files = list(latest.glob('*.log'))
    log_file = str(log_files[0]) if log_files else None

    # write trial metadata into the training folder for downstream collection/debug
    result = {
        'trial': trial_idx+1,
        'lr': lr,
        'wd': wd,
        'batch': batch,
        'dropout': dropout,
        'mixup_alpha': mixup,
        'oversample': oversample,
        'loss_type': loss_type,
        'cmd': ' '.join(cmd),
        'metrics_file': metrics_file,
        'val_f1': val_f1,
        'log_file': log_file,
        'rc': proc.returncode
    }
    print('Trial result:', result)
    # attempt to write trial metadata into the training folder
    try:
        if latest is not None and latest.exists():
            meta_path = latest / 'trial_meta.json'
            with open(meta_path, 'w', encoding='utf-8') as mf:
                json.dump(result, mf, indent=2)
    except Exception as e:
        print('Failed to write trial_meta.json:', e)

    _write_result_atomic(result)
    return result


def _write_result_atomic(result):
    """Append a single trial result to OUT/hp_sweep_results.json atomically."""
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'hp_sweep_results.json'
    with WRITE_LOCK:
        data = []
        if target.exists():
            try:
                data = json.load(open(target))
            except Exception:
                data = []
        data.append(result)
        tmp = OUT / ('.hp_sweep_results.json.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(target)


def main():
    p = argparse.ArgumentParser(description='Random HP sweep helper')
    p.add_argument('--trials', type=int, default=int(os.environ.get('HP_TRIALS', '6')), help='Number of trials')
    p.add_argument('--epochs', type=int, default=int(os.environ.get('HP_EPOCHS', '3')), help='Epochs per short trial')
    p.add_argument('--concurrency', type=int, default=int(os.environ.get('HP_CONCURRENCY', '2')), help='Concurrent trials')
    p.add_argument('--num-workers', type=int, default=int(os.environ.get('HP_NUM_WORKERS', '0')), help='Default num_workers forwarded to training')
    p.add_argument('--search-space', type=str, default=None, help='Search space spec: "lr:1e-4,3e-4,1e-3 wd:1e-6,1e-5"')
    args = p.parse_args()

    trials = args.trials
    short_epochs = args.epochs
    concurrency = max(1, args.concurrency)
    default_num_workers = max(0, args.num_workers)
    space = parse_search_space(args.search_space) if args.search_space else {}

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(run_trial, i, space, short_epochs, default_num_workers) for i in range(trials)]
        for f in concurrent.futures.as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                print('Trial failed with exception:', e)

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUT / 'hp_sweep_results.json', 'w'), indent=2)
    print('Saved hp sweep results to', OUT / 'hp_sweep_results.json')


if __name__ == '__main__':
    main()
