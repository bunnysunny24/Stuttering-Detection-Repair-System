import csv, ast, pathlib

in_csv = pathlib.Path('output/review_top_errors_training_20260220_005807/top_errors_review.csv')
out_csv = pathlib.Path('output/review_top_errors_training_20260220_005807/corrections_template.csv')
if not in_csv.exists():
    print('Input CSV not found:', in_csv)
    raise SystemExit(1)
rows = []
with in_csv.open('r', encoding='utf-8') as fh:
    reader = csv.DictReader(fh)
    for r in reader:
        file = r.get('file','')
        etype = r.get('error_type','')
        cls = r.get('class','')
        prob = r.get('prob','')
        labels = r.get('labels','')
        # parse labels
        try:
            lbls = ast.literal_eval(labels) if labels and labels.strip()!='' else None
        except Exception:
            lbls = None
        current_label = ''
        try:
            if lbls is not None:
                current_label = int(lbls[int(cls)])
        except Exception:
            current_label = ''
        suggested = ''
        try:
            p = float(prob)
            if etype == 'FP' and p >= 0.9:
                suggested = '1'
            if etype == 'FN' and p <= 0.1:
                suggested = '0'
        except Exception:
            pass
        rows.append({'file': file, 'class': cls, 'error_type': etype, 'prob': prob, 'labels': labels, 'current_label': current_label, 'suggested_label': suggested})
# write template
out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open('w', encoding='utf-8', newline='') as fh:
    fieldnames = ['file','class','error_type','prob','labels','current_label','suggested_label']
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
print('Wrote corrections template to', out_csv)
