import json
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, r'D:/Bunny/AGNI')
from Models.extract_features_90plus import FeatureExtractionPipeline

out = Path('output/debug_reports')
out.mkdir(parents=True, exist_ok=True)

p = FeatureExtractionPipeline(clips_dir='datasets/clips', output_dir='datasets/features', log_level='INFO')

# Run verification
valid = p.verify_extraction()

# Collect preprocessor error counts and quality summary
preproc_errors = getattr(p.preprocessor, 'error_counts', {})
quality_summary = p.quality_metrics.get_summary() or {}

# Scan NPZs for NaN/Inf and wrong shapes
train_files = list(Path('datasets/features/train').glob('*.npz'))
val_files = list(Path('datasets/features/val').glob('*.npz'))
all_files = train_files + val_files

issues = {'nan_files':[], 'inf_files':[], 'wrong_shape':[], 'count': len(all_files)}
for f in all_files:
    try:
        data = np.load(f)
        spec = data['spectrogram']
        if spec.shape[0] != 123:
            issues['wrong_shape'].append(str(f))
        if np.any(np.isnan(spec)):
            issues['nan_files'].append(str(f))
        if np.any(np.isinf(spec)):
            issues['inf_files'].append(str(f))
    except Exception as e:
        issues.setdefault('load_errors', []).append({'file': str(f), 'error': str(e)})

report = {
    'verify_passed': bool(valid),
    'total_extracted_files': issues['count'],
    'preprocessor_errors': dict(preproc_errors),
    'quality_summary': quality_summary,
    'issues': {
        'nan_files_count': len(issues['nan_files']),
        'inf_files_count': len(issues['inf_files']),
        'wrong_shape_count': len(issues['wrong_shape']),
        'load_errors': issues.get('load_errors', [])[:10]
    }
}

out_file = out / 'extraction_verification.json'
with open(out_file, 'w') as f:
    json.dump(report, f, indent=2)

print('Saved verification report ->', out_file)
print('Summary:')
print(' verify_passed=', report['verify_passed'])
print(' total_extracted_files=', report['total_extracted_files'])
print(' nan_files=', report['issues']['nan_files_count'])
print(' inf_files=', report['issues']['inf_files_count'])
print(' wrong_shape=', report['issues']['wrong_shape_count'])
print(' preprocessor_errors=', report['preprocessor_errors'])
