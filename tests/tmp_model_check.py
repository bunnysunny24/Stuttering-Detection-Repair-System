import sys
from pathlib import Path

# Ensure project root is on sys.path so `Models` imports work when running the script directly
proj_root = Path(__file__).resolve().parents[1]
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

try:
    import torch
except Exception as e:
    print('Missing dependency: torch not found.')
    print('Install with (CPU-only example):')
    print('  pip install torch --index-url https://download.pytorch.org/whl/cpu')
    raise

from Models.model_cnn_bilstm import CNNBiLSTM

print('Torch version:', torch.__version__)
model = CNNBiLSTM(in_channels=123, n_classes=5, dropout=0.3)
params = sum(p.numel() for p in model.parameters())
print('Model class:', model.__class__.__name__)
print('Parameter count:', params)
# Dummy input: batch=4, channels=123, time=160
x = torch.randn(4, 123, 160)
with torch.no_grad():
    out = model(x)
print('Output shape:', out.shape)
print('Output sample:', out[0])
