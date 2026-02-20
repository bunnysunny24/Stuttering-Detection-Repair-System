import argparse
import os
import shutil
import numpy as np


def process_file(path, backup=True):
    try:
        with np.load(path, allow_pickle=True) as data:
            keys = list(data.keys())
            # find labels array
            labels_key = None
            for k in keys:
                arr = data[k]
                if hasattr(arr, 'shape') and tuple(arr.shape) == (5,):
                    labels_key = k
                    break
            if labels_key is None:
                return False, 0

            labels = data[labels_key]
            orig = labels.copy()
            # map 2->1 and 3->1, keep 0/1 as-is
            mapped = labels.copy()
            mapped = np.where(np.isin(mapped, [2, 3]), 1, mapped)
            # ensure binary (any non-1 becomes 0)
            mapped = np.where(mapped == 1, 1, 0).astype(np.int8)

            if np.array_equal(orig, mapped):
                return False, 0

            # backup original file if requested
            if backup:
                bak = path + '.orig'
                if not os.path.exists(bak):
                    shutil.copy2(path, bak)

            # prepare save dict preserving arrays
            save_dict = {}
            for k in keys:
                if k == labels_key:
                    save_dict[k] = mapped
                else:
                    save_dict[k] = data[k]

            # overwrite npz
            np.savez(path, **save_dict)
            return True, int((orig != mapped).sum())
    except Exception as e:
        print(f"ERROR processing {path}: {e}")
        return False, 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, help='root directory with train/val feature npz files')
    p.add_argument('--backup', action='store_true', help='create .orig backups')
    p.add_argument('--dry-run', action='store_true', help='only report changes, do not write')
    args = p.parse_args()

    total_files = 0
    changed_files = 0
    total_label_changes = 0

    for split in ['train', 'val']:
        split_dir = os.path.join(args.root, split)
        if not os.path.isdir(split_dir):
            print(f"Skipping missing split: {split_dir}")
            continue
        for fname in os.listdir(split_dir):
            if not fname.endswith('.npz'):
                continue
            path = os.path.join(split_dir, fname)
            total_files += 1
            if args.dry_run:
                try:
                    with np.load(path, allow_pickle=True) as data:
                        labels = None
                        for k in data.keys():
                            if tuple(getattr(data[k], 'shape', ())) == (5,):
                                labels = data[k]
                                break
                        if labels is None:
                            continue
                        # check if mapping needed
                        if np.any(np.isin(labels, [2, 3])):
                            print(f"WILL CHANGE: {path} contains values {np.unique(labels)}")
                            changed_files += 1
                except Exception as e:
                    print(f"ERROR reading {path}: {e}")
                continue

            changed, nchanges = process_file(path, backup=args.backup)
            if changed:
                print(f"Updated {path}: changed {nchanges} label entries")
                changed_files += 1
                total_label_changes += nchanges

    print(f"Processed {total_files} files, updated {changed_files} files, total label entries changed: {total_label_changes}")


if __name__ == '__main__':
    main()
