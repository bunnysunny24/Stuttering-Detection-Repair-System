import json
from pathlib import Path
import numpy as np


def inspect_folder(folder: Path):
    report = {
        'total_files': 0,
        'missing_labels': 0,
        'label_shapes': {},
        'label_value_counts': {},
        'spectrogram_shapes': {},
        'channel_mismatch_files': [],
        'nan_inf_counts': 0,
        'examples': []
    }

    for p in sorted(folder.glob('**/*.npz')):
        report['total_files'] += 1
        try:
            data = np.load(p)
        except Exception as e:
            report['examples'].append({'file': str(p), 'error': f'load_error:{e}'})
            continue

        spec = data.get('spectrogram')
        labels = data.get('labels')

        if labels is None:
            report['missing_labels'] += 1
        else:
            ls = tuple(labels.shape)
            report['label_shapes'][str(ls)] = report['label_shapes'].get(str(ls), 0) + 1
            # record unique values
            try:
                vals = np.unique(labels)
                for v in vals.tolist():
                    k = str(v)
                    report['label_value_counts'][k] = report['label_value_counts'].get(k, 0) + 1
            except Exception:
                pass

        if spec is not None:
            ss = tuple(spec.shape)
            report['spectrogram_shapes'][str(ss)] = report['spectrogram_shapes'].get(str(ss), 0) + 1
            # channel mismatch (expect 123 channels)
            if spec.shape[0] != 123:
                report['channel_mismatch_files'].append({'file': str(p), 'spec_shape': ss})
            # nan/inf check
            try:
                if np.isnan(spec).any() or np.isinf(spec).any():
                    report['nan_inf_counts'] += 1
                    report['examples'].append({'file': str(p), 'error': 'nan_or_inf'})
            except Exception:
                pass

    return report


def main(root='datasets/features'):
    root = Path(root)
    out = {
        'root': str(root),
        'train': {},
        'val': {}
    }
    for split in ['train', 'val']:
        d = root / split
        if d.exists():
            out[split] = inspect_folder(d)
        else:
            out[split] = {'error': 'missing_folder'}

    out_path = Path('output') / 'dataset_inspection.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2)
    print('Wrote dataset inspection to', out_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='datasets/features')
    args = parser.parse_args()
    main(args.root)
