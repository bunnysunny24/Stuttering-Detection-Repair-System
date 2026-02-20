"""
Audit and optionally generate a mapping between label stems and audio files.
Usage:
    python scripts/fix_label_audio_match.py --datasets-dir datasets --out mapping.json --fix-links

Options:
  --datasets-dir: path to datasets folder containing CSV label files and clips/
  --out: output JSON mapping of label_id -> audio_path (or null if missing)
  --fix-links: attempt to create symlinks in datasets/clips to match label stems (Windows requires admin or Developer Mode)

This script only reports issues by default. Use --fix-links carefully.
"""
import argparse
from pathlib import Path
import csv
import json


def load_label_stems(label_csv_path):
    stems = set()
    with open(label_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        # Heuristic: first column contains filename/stem
        for row in reader:
            if not row:
                continue
            stem = Path(row[0]).stem
            stems.add(stem)
    return stems


def find_audio_files(clips_dir):
    p = Path(clips_dir)
    files = list(p.glob('**/*.wav'))
    stems = {f.stem: str(f) for f in files}
    return stems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets-dir', default='datasets')
    parser.add_argument('--out', default='datasets/label_audio_map.json')
    parser.add_argument('--fix-links', action='store_true')
    args = parser.parse_args()

    datasets = Path(args.datasets_dir)
    label_files = list(datasets.glob('*_labels.csv'))
    if not label_files:
        print('No label CSVs found in', datasets)
        return

    # Collect all label stems
    all_label_stems = set()
    label_details = {}
    for lf in label_files:
        stems = load_label_stems(lf)
        label_details[lf.name] = len(stems)
        all_label_stems.update(stems)

    print('Label files found:')
    for k, v in label_details.items():
        print(f'  {k}: {v} stems')

    # Collect audio stems
    clips_dir = datasets / 'clips' / 'stuttering-clips' / 'clips'
    audio_map = find_audio_files(clips_dir)
    print(f'Found {len(audio_map)} audio files under {clips_dir}')

    matched = []
    unmatched_labels = []
    for stem in sorted(all_label_stems):
        if stem in audio_map:
            matched.append((stem, audio_map[stem]))
        else:
            unmatched_labels.append(stem)

    print(f'Matched labels: {len(matched)}')
    print(f'Unmatched labels: {len(unmatched_labels)}')

    # Save mapping
    mapping = {stem: audio_map.get(stem) for stem in sorted(all_label_stems)}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'mapping': mapping, 'matched': len(matched), 'unmatched': len(unmatched_labels)}, f, indent=2)
    print('Wrote mapping to', out_path)

    if args.fix_links:
        # Attempt to create symlinks for unmatched labels if possible (experimental)
        from shutil import copy2
        for stem in unmatched_labels:
            # Try to find close matches by prefix/suffix
            candidates = [s for s in audio_map.keys() if stem in s or s in stem]
            if candidates:
                src = Path(audio_map[candidates[0]])
                dst = clips_dir / (stem + src.suffix)
                try:
                    if not dst.exists():
                        # create hard copy as safer default on Windows
                        copy2(src, dst)
                        print(f'Copied {src} -> {dst} (attempted fix)')
                except Exception as e:
                    print(f'Failed to copy {src} -> {dst}: {e}')

    # Print a short list of unmatched samples
    if unmatched_labels:
        print('\nSample unmatched labels (10):')
        for s in unmatched_labels[:10]:
            print('  ', s)


if __name__ == '__main__':
    main()
