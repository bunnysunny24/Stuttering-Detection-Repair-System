import json
from pathlib import Path
import csv
import shutil
import numpy as np


def main(diagnostics_path: Path, out_dir: Path):
    diag = json.loads(diagnostics_path.read_text())
    top = diag.get('top_errors', {})
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'top_errors_review.csv'

    rows = []
    for key, items in top.items():
        # key format: class_{i}_FP or class_{i}_FN
        parts = key.split('_')
        if len(parts) < 3:
            continue
        try:
            cls = int(parts[1])
        except Exception:
            continue
        etype = parts[2]
        dest_sub = out_dir / f'class_{cls}' / etype
        dest_sub.mkdir(parents=True, exist_ok=True)
        for it in items:
            src = Path(it['file'])
            prob = float(it.get('prob', 0.0))
            if not src.exists():
                # try converting slashes
                src = Path(str(it['file']).replace('\\', '/'))
            if not src.exists():
                rows.append({'file': it['file'], 'class': cls, 'error_type': etype, 'prob': prob, 'note': 'missing'})
                continue
            # copy file to review folder
            dest = dest_sub / src.name
            try:
                shutil.copy2(src, dest)
            except Exception:
                rows.append({'file': str(src), 'class': cls, 'error_type': etype, 'prob': prob, 'note': 'copy_failed'})
                continue
            # read true label from npz
            try:
                data = np.load(src)
                labels = data['labels'].astype(int).tolist() if 'labels' in data else None
            except Exception:
                labels = None
            rows.append({'file': str(src), 'copied_to': str(dest), 'class': cls, 'error_type': etype, 'prob': prob, 'labels': labels})

    # write CSV
    with csv_path.open('w', newline='', encoding='utf-8') as fh:
        fieldnames = ['file', 'copied_to', 'class', 'error_type', 'prob', 'labels', 'note']
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, '') for k in fieldnames})

    print('Exported top errors and copies to', out_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--diagnostics', default='output/diag_cnn_bilstm_best/diagnostics.json')
    parser.add_argument('--out', default='output/review_top_errors')
    args = parser.parse_args()
    main(Path(args.diagnostics), Path(args.out))
