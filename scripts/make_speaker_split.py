"""
Create speaker-holdout train/val/test splits based on speaker IDs.

Usage:
  python scripts/make_speaker_split.py --csv datasets/fluencybank_episodes.csv --audio-column audio_file --speaker-column speaker_id --out datasets/splits --seed 42 --ratios 0.8,0.1,0.1

This writes JSON files: out/train.txt, out/val.txt, out/test.txt containing audio stems (without extension), and copies matching NPZs from datasets/features to datasets/features_speaker_split/{train,val,test}.
"""
import argparse
from pathlib import Path
import csv
import random
import json
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--audio-column', default='audio_file')
    parser.add_argument('--speaker-column', default='speaker_id')
    parser.add_argument('--out', default='datasets/splits')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ratios', default='0.8,0.1,0.1')
    parser.add_argument('--features-dir', default='datasets/features')
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    features_dir = Path(args.features_dir)

    ratios = [float(x) for x in args.ratios.split(',')]
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise SystemExit('Ratios must sum to 1.0')

    # Read CSV and map speaker -> list of stems
    mapping = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) == 0:
        raise SystemExit('CSV appears empty')

    for r in rows:
        if args.audio_column not in r or args.speaker_column not in r:
            continue
        audio = r[args.audio_column].strip()
        speaker = r[args.speaker_column].strip()
        if not audio or not speaker:
            continue
        stem = Path(audio).stem
        mapping.setdefault(speaker, []).append(stem)

    speakers = list(mapping.keys())
    random.seed(args.seed)
    random.shuffle(speakers)

    n = len(speakers)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    train_speakers = speakers[:n_train]
    val_speakers = speakers[n_train:n_train+n_val]
    test_speakers = speakers[n_train+n_val:]

    def gather_stems(spk_list):
        s = []
        for sp in spk_list:
            s.extend(mapping.get(sp, []))
        return sorted(list(set(s)))

    train_stems = gather_stems(train_speakers)
    val_stems = gather_stems(val_speakers)
    test_stems = gather_stems(test_speakers)

    (out_dir / 'train.txt').write_text('\n'.join(train_stems))
    (out_dir / 'val.txt').write_text('\n'.join(val_stems))
    (out_dir / 'test.txt').write_text('\n'.join(test_stems))

    print(f'Wrote splits to {out_dir} with {len(train_stems)} train, {len(val_stems)} val, {len(test_stems)} test stems')

    # Optional: create feature copies into features_speaker_split
    target_base = features_dir.parent / f'{features_dir.name}_speaker_split'
    for split_name, stems in [('train', train_stems), ('val', val_stems), ('test', test_stems)]:
        tgt = target_base / split_name
        tgt.mkdir(parents=True, exist_ok=True)
        copied = 0
        for stem in stems:
            # find matching npz in features_dir subfolders (train/val/test)
            for npz in features_dir.rglob(f'{stem}.npz'):
                try:
                    shutil.copy2(str(npz), str(tgt / npz.name))
                    copied += 1
                except Exception:
                    pass
        print(f'Copied {copied} NPZs for {split_name} to {tgt}')

    print('Speaker-holdout split complete')


if __name__ == '__main__':
    main()
