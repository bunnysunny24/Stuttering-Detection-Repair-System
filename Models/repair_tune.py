import json
import sys
from pathlib import Path
from statistics import mean

from repair_advanced import AdvancedStutterRepair


def load_analysis(path):
    with open(path, 'r') as f:
        return json.load(f)


def overlap(a, b):
    # a and b are (s,e)
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    return max(0.0, e - s)


def evaluate_predictions(gt_regions, pred_regions, iou_thresh=0.3):
    # Simple precision/recall using overlap ratio relative to union
    tp = 0
    matched_gt = set()

    for p in pred_regions:
        for j, g in enumerate(gt_regions):
            if j in matched_gt:
                continue
            inter = overlap(p, g)
            if inter <= 0:
                continue
            union = (p[1] - p[0]) + (g[1] - g[0]) - inter
            iou = inter / union if union > 0 else 0
            if iou >= iou_thresh:
                tp += 1
                matched_gt.add(j)
                break

    fp = max(0, len(pred_regions) - tp)
    fn = max(0, len(gt_regions) - tp)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {'precision': prec, 'recall': rec, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}


def sweep_thresholds(audio_path, analysis_json, model_path=None):
    # Load existing thresholds if available
    th_path = Path('output/thresholds.json')
    base_thresholds = None
    if th_path.exists():
        data = json.loads(th_path.read_text())
        base_thresholds = data.get('thresholds')

    repair = AdvancedStutterRepair(model_path=model_path)

    gt = load_analysis(analysis_json)
    gt_regions = [(r['start_time'], r['end_time']) for r in gt.get('stutter_regions', [])]

    # Sweep multipliers
    multipliers = [0.6, 0.7, 0.8, 0.9, 1.0, 1.05, 1.1]
    results = []

    for m in multipliers:
        if base_thresholds:
            thr = [min(0.99, max(0.01, float(x) * m)) for x in base_thresholds]
            repair.thresholds = thr
        else:
            # No base thresholds: try scoring by varying min_duration_s
            repair.thresholds = None
            repair.min_duration_s = max(0.05, 0.2 * m)

        repaired, preds = repair.repair_audio(audio_path, output_path=None, return_regions=True)

        metrics = evaluate_predictions(gt_regions, preds)
        total_repaired_time = sum(e - s for s, e in preds)
        results.append((m, metrics, preds, total_repaired_time, repair.thresholds))
        print(f"m={m:.2f} -> f1={metrics['f1']:.3f} prec={metrics['precision']:.3f} rec={metrics['recall']:.3f} repaired_time={total_repaired_time:.2f}s")

    # Choose best by F1 (higher better), break ties with higher precision, then higher recall, then lower repaired_time
    # Sorting key: (-f1, -precision, -recall, repaired_time)
    results_sorted = sorted(results, key=lambda x: (-x[1]['f1'], -x[1]['precision'], -x[1]['recall'], x[3]))
    best = results_sorted[0]
    m_best, metrics_best, preds_best, repaired_time_best, best_thresholds = best

    print('\nBest multiplier: {:.2f}'.format(m_best))
    print('Best metrics:', metrics_best)

    # Save chosen thresholds if available
    if best_thresholds is not None:
        out = {'thresholds': [float(x) for x in best_thresholds], 'multiplier': float(m_best)}
        Path('output').mkdir(parents=True, exist_ok=True)
        Path('output/thresholds_tuned.json').write_text(json.dumps(out, indent=2))
        print('Saved tuned thresholds to output/thresholds_tuned.json')

    return best


def main():
    audio = sys.argv[1] if len(sys.argv) > 1 else 'Online_test/I Have a Stutter  60 Second Docs.mp3'
    analysis = sys.argv[2] if len(sys.argv) > 2 else 'output/analysis/I Have a Stutter  60 Second Docs_analysis.json'
    model = None
    if '--model' in sys.argv:
        i = sys.argv.index('--model')
        if i + 1 < len(sys.argv):
            model = sys.argv[i+1]

    print('Tuning thresholds on:', audio)
    best = sweep_thresholds(audio, analysis, model_path=model)
    print('Done')


if __name__ == '__main__':
    main()
