from pathlib import Path
import numpy as np
from Models.extract_features_90plus import FeatureExtractionPipeline

p = FeatureExtractionPipeline(clips_dir='datasets/problematic_samples', output_dir='datasets/features_problematic', log_level='INFO')
files = list(Path('datasets/problematic_samples').glob('*.wav'))
summary = {'processed':0,'failed':0,'moved_corrupted':[],'trimmed':[]}
for f in files:
    res = p.extract_single_file(f)
    if res['success']:
        print(f"OK: {f.name} (split={res['split']})")
        summary['processed'] += 1
    else:
        print(f"FAIL: {f.name} -> {res.get('error')}")
        summary['failed'] += 1
        if res.get('error') and 'corrupted' in str(res.get('error')).lower():
            summary['moved_corrupted'].append(str(f.name))

print('\nSUMMARY:')
print(summary)
np.savez('output/problem_inspect/reprocess_summary.npz', **summary)
print('Saved summary -> output/problem_inspect/reprocess_summary.npz')
