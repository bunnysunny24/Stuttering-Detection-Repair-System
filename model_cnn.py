"""Simple CNN multi-label classifier for log-mel inputs.

This PyTorch model expects input shaped (batch, 1, n_mels, time_frames).
It produces logits for 5 event classes corresponding to SEP-28k event labels.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=5, bias_init=None):
        """A slightly deeper CNN than the original starter.

        bias_init: Optional 1-D tensor/array of shape (n_classes,) used to initialize
        the final linear layer bias to match dataset priors (logit space).
        """
        super().__init__()
        # conv blocks: (in -> out) — increased capacity
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(256, 320, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d((2, 2))
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(320)
        self.dropout = nn.Dropout(0.3)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # final feature dim is last conv channels
        self.fc = nn.Linear(320, n_classes)

        # initialize final bias if provided (helps with severe class imbalance)
        if bias_init is not None:
            try:
                b = torch.as_tensor(bias_init, dtype=torch.float32)
                if b.numel() == n_classes:
                    with torch.no_grad():
                        self.fc.bias.copy_(b.view_as(self.fc.bias))
            except Exception:
                # ignore invalid bias_init shapes
                pass

    def forward(self, x):
        # x: batch x 1 x n_mels x time
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
