import json

with open('output/training_20260225_023137/improved_90plus_large_metrics.json') as f:
    data = json.load(f)
vals = data['val']
best = max(vals, key=lambda x: x['f1_macro'])
bi = [i for i, v in enumerate(vals) if v['f1_macro'] == best['f1_macro']][0] + 1
print(f"Best epoch: {bi}")
print(f"F1_macro={best['f1_macro']:.4f}  P={best['precision_macro']:.4f}  R={best['recall_macro']:.4f}  AUC={best['roc_auc_macro']:.4f}")
print("Per-class:")
for name, m in best['per_class'].items():
    print(f"  {name:20s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} AUC={m['roc_auc']:.3f} Support={m['support']}")

# Also check the label distribution in training data
import numpy as np, glob
counts = np.zeros(5)
total = 0 
for f in glob.glob('datasets/features/train/*.npz')[:5000]:
    try:
        d = np.load(f)
        lbl = d.get('labels')
        if lbl is not None:
            lbl = np.asarray(lbl)
            counts += (lbl > 0).astype(float)
            total += 1
    except:
        pass
print(f"\nLabel distribution (out of {total} scanned samples):")
class_names = ['Prolongation', 'Block', 'Sound Repetition', 'Word Repetition', 'Interjection']
for i, name in enumerate(class_names):
    pct = 100 * counts[i] / total if total > 0 else 0
    print(f"  {name:20s}: {int(counts[i]):5d} ({pct:.1f}%)")
