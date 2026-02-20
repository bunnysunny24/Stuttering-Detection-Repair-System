import torch
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from Models.train_90plus_final import AudioDataset, collate_variable_length, Trainer
from Models.model_improved_90plus import ImprovedStutteringCNN


def run_smoke(train_size=200, val_size=50, batch_size=16, epochs=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    train_ds = AudioDataset('datasets/features', split='train', augment=False)
    val_ds = AudioDataset('datasets/features', split='val', augment=False)

    train_idx = list(range(min(train_size, len(train_ds))))
    val_idx = list(range(min(val_size, len(val_ds))))

    train_sub = Subset(train_ds, train_idx)
    val_sub = Subset(val_ds, val_idx)

    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True, collate_fn=collate_variable_length, num_workers=0)
    val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False, collate_fn=collate_variable_length, num_workers=0)

    model = ImprovedStutteringCNN(n_channels=123, n_classes=5, dropout=0.4)
    trainer = Trainer(model, train_loader, val_loader, device, model_name='smoke_test', early_stop_patience=100)

    metrics = trainer.train(num_epochs=epochs)
    print('Finished smoke training')

if __name__ == '__main__':
    run_smoke()
