import json, glob, os

dirs = sorted(glob.glob('output/training_*'), reverse=True)[:8]
for d in dirs:
    mfiles = glob.glob(os.path.join(d, '*metrics*'))
    for mf in mfiles:
        try:
            with open(mf) as f:
                data = json.load(f)
            vals = data.get('val', [])
            if vals:
                # Find best epoch by f1_macro
                best = max(vals, key=lambda x: x.get('f1_macro', 0))
                last = vals[-1]
                ep_count = len(vals)
                print(f"{os.path.basename(d)}: BEST F1={best['f1_macro']:.4f} (P={best['precision_macro']:.4f} R={best['recall_macro']:.4f} AUC={best['roc_auc_macro']:.4f}) | Last F1={last['f1_macro']:.4f} | {ep_count} epochs")
        except Exception as e:
            print(f"{d}: ERROR {e}")
