import csv
from pathlib import Path
import numpy as np


def main(csv_path: Path):
    if not csv_path.exists():
        raise SystemExit(f'CSV not found: {csv_path}')

    with csv_path.open('r', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            file = Path(r['file'])
            if not file.exists():
                print('missing', file)
                continue
            # expected CSV columns: file, label_idx, new_value
            try:
                label_idx = int(r.get('label_idx', -1))
                new_val = int(r.get('new_value', -1))
            except Exception:
                print('invalid row', r)
                continue
            if label_idx < 0 or new_val not in (0, 1):
                print('skipping invalid correction', r)
                continue
            try:
                data = dict(np.load(file))
                labels = data.get('labels')
                if labels is None:
                    print('no labels in', file)
                    continue
                labels = labels.astype(int)
                labels[label_idx] = new_val
                # write back
                np.savez_compressed(file, spectrogram=data.get('spectrogram'), labels=labels)
                print('updated', file, 'label', label_idx, '->', new_val)
            except Exception as e:
                print('failed to update', file, e)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='CSV with columns file,label_idx,new_value')
    args = parser.parse_args()
    main(Path(args.csv))
