"""Analyze ALL training runs to find the best config and identify bottlenecks."""
import json, glob, os

all_runs = []
for mf in sorted(glob.glob('output/training_*/*metrics*.json')):
    try:
        with open(mf) as f:
            data = json.load(f)
        vals = data.get('val', [])
        if not vals:
            continue
        best = max(vals, key=lambda x: x.get('f1_macro', 0))
        bi = [i for i, v in enumerate(vals) if v['f1_macro'] == best['f1_macro']][0] + 1
        run_dir = os.path.basename(os.path.dirname(mf))
        arch = os.path.basename(mf).replace('_metrics.json', '')
        all_runs.append({
            'dir': run_dir,
            'arch': arch,
            'best_epoch': bi,
            'total_epochs': len(vals),
            'f1': best['f1_macro'],
            'precision': best['precision_macro'],
            'recall': best['recall_macro'],
            'auc': best['roc_auc_macro'],
            'per_class': best.get('per_class', {}),
        })
    except Exception as e:
        pass

# Sort by F1 descending
all_runs.sort(key=lambda x: x['f1'], reverse=True)

print(f"Total training runs analyzed: {len(all_runs)}")
print(f"\n{'='*80}")
print("TOP 10 TRAINING RUNS BY VAL F1_MACRO:")
print(f"{'='*80}")
for i, r in enumerate(all_runs[:10], 1):
    print(f"{i:2d}. {r['dir']} [{r['arch']}]")
    print(f"    F1={r['f1']:.4f}  P={r['precision']:.4f}  R={r['recall']:.4f}  AUC={r['auc']:.4f}  (best@ep{r['best_epoch']}/{r['total_epochs']})")
    if r['per_class']:
        for name, m in r['per_class'].items():
            print(f"      {name:20s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

# Check if any training logs exist with the hyperparameters used
print(f"\n{'='*80}")
print("CHECKING TRAINING LOGS FOR TOP RUNS:")
print(f"{'='*80}")
for r in all_runs[:3]:
    log_dir = os.path.join('output', r['dir'])
    logs = glob.glob(os.path.join(log_dir, '*.log'))
    for logf in logs[:1]:
        try:
            with open(logf, encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()[:50]
            for line in lines:
                l = line.strip()
                if any(kw in l.lower() for kw in ['setting', 'lr=', 'dropout', 'batch', 'device', 'arch', 'model:', 'parameter', 'class weight', 'focal', 'loss', 'ema', 'swa', 'scheduler', 'accumulate', 'mixup', 'oversamp', 'epoch']):
                    print(f"  [{r['dir']}] {l}")
        except:
            pass
