from pathlib import Path
from Models.extract_features_90plus import ExtractionCheckpoint

cp = ExtractionCheckpoint(Path('datasets/features/extraction_checkpoint.json'))
print('processed type:', type(cp.data.get('processed')))
print('failed type:', type(cp.data.get('failed')))
print('processed len:', len(cp.data.get('processed')))
print('failed len:', len(cp.data.get('failed')))
