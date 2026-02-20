import csv, pathlib
in_csv = pathlib.Path('output/review_top_errors_training_20260220_005807/corrections_template.csv')
out_csv = pathlib.Path('output/review_top_errors_training_20260220_005807/apply_corrections_auto.csv')
if not in_csv.exists():
    print('Input template not found:', in_csv)
    raise SystemExit(1)
rows = []
with in_csv.open('r', encoding='utf-8') as fh:
    reader = csv.DictReader(fh)
    for r in reader:
        sug = r.get('suggested_label','').strip()
        if sug not in ('0','1'):
            continue
        file = r.get('file','')
        cls = r.get('class','')
        try:
            label_idx = int(cls)
            new_value = int(sug)
        except Exception:
            continue
        rows.append({'file': file, 'label_idx': label_idx, 'new_value': new_value})
if not rows:
    print('No auto-suggestions found')
    raise SystemExit(0)
out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open('w', encoding='utf-8', newline='') as fh:
    fieldnames = ['file','label_idx','new_value']
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
print('Wrote', out_csv, 'with', len(rows), 'rows')
