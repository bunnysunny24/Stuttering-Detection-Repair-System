"""Debug extraction trace for specific audio files.

Usage:
  conda activate agni
  python scripts/debug_extraction_trace.py --files datasets/problematic_samples/*.wav --max-chk 10

Prints: sample rate, length, feature shape, per-channel variance, adjacent-channel correlations (first 50 pairs), and top highly-correlated pairs.
"""
import argparse
import numpy as np
from pathlib import Path
import soundfile as sf
import json
import sys
import os

# Ensure repo root is on sys.path so imports like `Models.enhanced_audio_preprocessor`
# and local modules resolve regardless of current working directory.
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from Models.enhanced_audio_preprocessor import EnhancedAudioPreprocessor
except Exception:
    # fallback to direct import when running from scripts/ with PYTHONPATH set
    from enhanced_audio_preprocessor import EnhancedAudioPreprocessor


def adj_correlations(feat: np.ndarray, max_pairs: int = 50):
    # compute correlation between adjacent channel pairs
    chans = min(feat.shape[0]-1, max_pairs)
    corrs = []
    for i in range(chans):
        a = feat[i, :]
        b = feat[i+1, :]
        # skip constant arrays
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            corr = 0.0
        else:
            corr = np.corrcoef(a, b)[0, 1]
            if np.isnan(corr):
                corr = 0.0
        corrs.append((i, i+1, float(corr)))
    return corrs


def summarize_file(preproc: EnhancedAudioPreprocessor, path: Path, max_pairs=50):
    print('\n===', path.name)
    try:
        data, sr = sf.read(str(path))
        if data.ndim > 1:
            data = np.mean(data, axis=1)
    except Exception as e:
        print('read error:', e)
        return

    print('samples=', len(data), 'sr=', sr, 'duration(s)=', len(data)/sr)
    feat = preproc.extract_features_from_array(data, sr=sr)
    if feat is None:
        print('extract_features returned None')
        return

    print('feature shape:', feat.shape, 'dtype:', feat.dtype)
    variances = np.var(feat, axis=1)
    print('per-channel variance: min=%.3e median=%.3e max=%.3e' % (np.min(variances), np.median(variances), np.max(variances)))
    zero_var = np.sum(variances < 1e-12)
    print('channels with near-zero variance:', int(zero_var))

    corrs = adj_correlations(feat, max_pairs=max_pairs)
    # sort by abs correlation desc
    corrs_sorted = sorted(corrs, key=lambda x: -abs(x[2]))
    print('\nTop adjacent-channel correlations (abs desc, showing first 10):')
    for i, j, c in corrs_sorted[:10]:
        print(f'  {i}-{j}: corr={c:.4f}')

    # show example channel pair wave ranges
    high = [t for t in corrs_sorted if abs(t[2]) > 0.99]
    if high:
        print('\nHighly correlated pairs (>0.99):')
        for i, j, c in high[:10]:
            print(f'  {i}-{j}: corr={c:.4f}')
    else:
        print('\nNo extremely-high adjacent correlations found')

    # save small JSON report next to output directory
    report = {
        'file': str(path),
        'samples': len(data),
        'sr': sr,
        'feat_shape': feat.shape,
        'zero_var_channels': int(zero_var),
        'top_adj_corr': corrs_sorted[:20]
    }
    out = Path('output') / 'debug_reports'
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / (path.stem + '_debug.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print('Saved report ->', report_path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--files', nargs='+', required=True, help='One or more WAV files (globs are supported when expanded by shell)')
    p.add_argument('--max-chk', type=int, default=50, help='Max adjacent pairs to check')
    args = p.parse_args()

    preproc = EnhancedAudioPreprocessor(track_stats=False)
    # Expand globs manually if needed
    file_paths = []
    for pattern in args.files:
        file_paths.extend([Path(p) for p in sorted(Path('.').glob(pattern))])

    if not file_paths:
        print('No files found for patterns:', args.files)
        sys.exit(1)

    for fp in file_paths:
        summarize_file(preproc, fp, max_pairs=args.max_chk)
