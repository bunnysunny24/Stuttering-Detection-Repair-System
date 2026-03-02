"""Check threshold values from the best training run."""
import json
with open('output/training_20260225_023137/improved_90plus_large_metrics.json') as f:
    data = json.load(f)
# Check optimized_thresholds
threshs = data.get('optimized_thresholds', [])
if threshs:
    print(f"Total threshold records: {len(threshs)}")
    for t in threshs[-5:]:
        print(f"  Epoch {t.get('epoch','?')}: {t.get('thresholds',{})}")
else:
    print("No threshold records in metrics.")

# Check the best epoch val metrics more carefully
vals = data['val']
best_idx = max(range(len(vals)), key=lambda i: vals[i]['f1_macro'])
best = vals[best_idx]
print(f"\nBest epoch: {best_idx+1}")
print(f"Loss: {best.get('loss', '?')}")
print(f"Hamming loss: {best.get('hamming_loss', '?')}")
print(f"F1_micro: {best.get('f1_micro', '?')}")
print(f"F1_macro: {best.get('f1_macro', '?')}")

# Also check the last 5 epochs to see if model was still improving
print("\nLast 5 epochs F1 progression:")
for i in range(max(0, len(vals)-5), len(vals)):
    v = vals[i]
    print(f"  Epoch {i+1}: F1_macro={v['f1_macro']:.4f} Loss={v.get('loss','?'):.4f}" if isinstance(v.get('loss'), (int,float)) else f"  Epoch {i+1}: F1_macro={v['f1_macro']:.4f}")
