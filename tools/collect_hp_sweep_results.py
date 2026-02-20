#!/usr/bin/env python3
"""Collect HP sweep results from output/training_* folders and write hp_sweep_results.json

Scans each training folder, reads *_metrics.json if present to get last val f1,
otherwise parses the training log for 'BEST MODEL FOUND! F1=' lines. Attempts to
extract lr, wd, batch from the log as well.
"""
import json
from pathlib import Path
import re

OUT = Path('output')
res = []
for d in sorted(OUT.glob('training_*')):
    entry = {'folder': str(d), 'metrics_file': None, 'val_f1': None, 'lr': None, 'wd': None, 'batch': None, 'log_file': None}
    # find metrics json
    mfiles = list(d.glob('*_metrics.json'))
    if mfiles:
        entry['metrics_file'] = str(mfiles[0])
        try:
            m = json.load(open(mfiles[0]))
            val = m.get('val', [])
            if val:
                entry['val_f1'] = val[-1].get('f1_macro')
        except Exception:
            pass
    # parse log
    logs = list(d.glob('*.log'))
    if logs:
        lf = logs[0]
        entry['log_file'] = str(lf)
        txt = lf.read_text(encoding='utf-8', errors='ignore')
        # lr pattern like LR=3.00e-04 or LR=1.00e-03
        m = re.search(r'LR=(\d+\.\d+e[\-\+]\d+)', txt)
        if m:
            entry['lr'] = float(m.group(1))
        # settings line
        m2 = re.search(r'Settings: batch_size=(\d+), num_workers=(\d+), omp_threads=(\d+)', txt)
        if m2:
            entry['batch'] = int(m2.group(1))
        # try to find best F1 in log
        m3 = re.search(r'BEST MODEL FOUND! F1=([0-9\.]+)', txt)
        if m3 and entry['val_f1'] is None:
            try:
                entry['val_f1'] = float(m3.group(1))
            except Exception:
                pass
    res.append(entry)

OUT.mkdir(parents=True, exist_ok=True)
target = OUT / 'hp_sweep_results.json'
with open(target, 'w', encoding='utf-8') as f:
    json.dump(res, f, indent=2)
print('Wrote', target)
