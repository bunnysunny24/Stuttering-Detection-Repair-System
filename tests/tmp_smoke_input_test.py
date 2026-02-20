import torch
from torch.utils.data import DataLoader
from pathlib import Path

# Import project components
from Models.train_90plus_final import AudioDataset, collate_variable_length
from Models.model_improved_90plus import ImprovedStutteringCNN
from Models.constants import TOTAL_CHANNELS, NUM_CLASSES


def run_test(batch_size=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    dataset = AudioDataset('datasets/features', split='train', augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_variable_length)

    batch = next(iter(loader))
    X, y = batch
    print('X type:', type(X), 'shape:', getattr(X, 'shape', None), 'dtype:', getattr(X, 'dtype', None))
    print('y type:', type(y), 'shape:', getattr(y, 'shape', None), 'dtype:', getattr(y, 'dtype', None))

    # Ensure channel count
    assert X.dim() == 3, 'Expected X to be (batch, channels, time)'
    assert X.shape[1] == TOTAL_CHANNELS, f'Expected {TOTAL_CHANNELS} channels, got {X.shape[1]}'
    assert y.shape[1] == NUM_CLASSES, f'Expected {NUM_CLASSES} labels, got {y.shape[1]}'

    # Instantiate model
    model = ImprovedStutteringCNN(n_channels=TOTAL_CHANNELS, n_classes=NUM_CLASSES, dropout=0.4).to(device)
    X = X.to(device)

    with torch.no_grad():
        logits = model(X)
    print('Logits shape:', logits.shape, 'dtype:', logits.dtype)

    print('SMOKE TEST OK')

if __name__ == '__main__':
    run_test()
