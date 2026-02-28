import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self, in_channels=123, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(1),
        )
        self.out_dim = out_dim

    def forward(self, x):
        # x: (batch, channels, time)
        h = self.net(x)
        h = h.squeeze(-1)  # (batch, 128)
        return h

class CNNBiLSTM(nn.Module):
    def __init__(self, in_channels=123, cnn_out=128, lstm_hidden=256, n_classes=5, num_layers=1, dropout=0.3):
        super().__init__()
        # A simple 1D conv encoder that produces a short embedding per frame
        self.conv1 = nn.Conv1d(in_channels, 128, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)

        # BiLSTM expects (seq_len, batch, feat)
        self.lstm = nn.LSTM(input_size=128, hidden_size=lstm_hidden, num_layers=num_layers, bidirectional=True, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        # x: (batch, channels, time)
        h = self.conv1(x)
        h = self.relu(h)
        h = self.bn1(h)
        h = self.conv2(h)
        h = self.relu(h)
        h = self.bn2(h)

        # h: (batch, feat, time) -> transpose
        h = h.permute(0, 2, 1)  # (batch, time, feat)

        # LSTM
        outputs, _ = self.lstm(h)  # (batch, time, hidden*2)

        # Pool across time (mean)
        pooled = outputs.mean(dim=1)

        logits = self.classifier(pooled)
        return logits
