#!/usr/bin/env python3
"""
Scan training output metrics JSON files and report the best run by a chosen validation metric.
Writes a short summary and prints the best checkpoint path if found.
"""
import argparse
import json
from pathlib import Path
import math


def flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, prefix + k + ("." if prefix or k else "")))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, prefix + str(i) + "."))
    else:
        out[prefix.rstrip('.')]=obj
    return out


def find_metric(candidates, prefer='val'):
    # prefer metrics that include 'val' and 'f1'
    pref = [k for k in candidates if 'f1' in k and prefer in k]
    if pref:
        return pref[0]
    pref2 = [k for k in candidates if 'f1' in k]
    if pref2:
        return pref2[0]
    # fallback to any numeric metric containing 'auc' or 'ap'
    pref3 = [k for k in candidates if 'auc' in k]
    if pref3:
        return pref3[0]
    pref4 = [k for k in candidates if 'ap' in k]
    if pref4:
        return pref4[0]
    return None


def find_checkpoint_for_metrics(metrics_path: Path):
    # Search for .pth files in the metrics directory, its ancestors (up to 3 levels),
    # and a common global checkpoints directory (`Models/checkpoints`). Return the
    # most-recent 'best' file if available, otherwise most-recent .pth.
    search_dirs = [metrics_path.parent]
    # add up to 3 ancestor dirs
    for i, anc in enumerate(metrics_path.parents):
        if i >= 3:
            break
        search_dirs.append(anc)

    for d in search_dirs:
        cands = list(d.glob('*.pth'))
        if cands:
            best_cands = [p for p in cands if 'best' in p.name.lower()]
            if best_cands:
                # pick most recently modified best candidate
                best_cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                return str(best_cands[0])
            # otherwise pick most recent .pth
            cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return str(cands[0])

    # Fallback: search a common checkpoints folder at repo root
    repo_root = Path('.').resolve()
    models_ckpt = repo_root / 'Models' / 'checkpoints'
    if models_ckpt.exists():
        cands = list(models_ckpt.rglob('*.pth'))
        if cands:
            best_cands = [p for p in cands if 'best' in p.name.lower()]
            if best_cands:
                best_cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                return str(best_cands[0])
            cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return str(cands[0])

    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--metrics-root', default='output', help='Root directory to search for metrics JSONs (default: output)')
    p.add_argument('--metric', default=None, help='Metric name to use (optional). If omitted the script will attempt to auto-detect a validation F1 metric')
    p.add_argument('--top', type=int, default=5, help='Show top N runs')
    args = p.parse_args()

    root = Path(args.metrics_root)
    if not root.exists():
        print(f"Metrics root {root} does not exist")
        raise SystemExit(2)

    json_files = list(root.rglob('*_metrics.json'))
    if not json_files:
        # try any json under root
        json_files = [p for p in root.rglob('*.json')]

    results = []
    for jf in json_files:
        try:
            txt = jf.read_text(encoding='utf-8')
            data = json.loads(txt)
        except Exception:
            continue
        flat = flatten(data)
        # select numeric candidates
        numeric = {k: v for k, v in flat.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        if not numeric:
            continue
        metric_name = args.metric
        if metric_name is None:
            metric_name = find_metric(list(numeric.keys()), prefer='val')
        if metric_name is None:
            # if nothing auto-detected, pick the key with 'best' or the max numeric key
            cand_keys = [k for k in numeric.keys() if 'best' in k or 'val' in k]
            if not cand_keys:
                # pick the largest numeric value key (heuristic)
                metric_name = max(numeric.keys(), key=lambda k: float(numeric[k]))
            else:
                metric_name = cand_keys[0]
        try:
            value = float(numeric[metric_name])
        except Exception:
            continue
        ckpt = find_checkpoint_for_metrics(jf)
        results.append((jf, metric_name, value, ckpt))

    if not results:
        print('No numeric metrics found under', root)
        raise SystemExit(1)

    # sort by value desc
    results.sort(key=lambda x: x[2], reverse=True)

    print(f"Found {len(results)} metrics files. Top {min(args.top, len(results))} by selected metric:")
    for i, (jf, mname, val, ckpt) in enumerate(results[:args.top], 1):
        print(f"{i:2d}. {jf}  -> {mname} = {val:.6f}  checkpoint: {ckpt}")

    best = results[0]
    best_ckpt = best[3]
    print('\nBest checkpoint:', best_ckpt)
    # write best path file for automation
    out_path = root / 'best_checkpoint.txt'
    # If no checkpoint was found, write an empty file instead of raising
    if best_ckpt is None:
        print('No checkpoint file found for best metrics; writing empty best_checkpoint.txt')
        out_path.write_text('')
    else:
        out_path.write_text(str(best_ckpt))
    print('Wrote', out_path)

if __name__ == '__main__':
    main()
