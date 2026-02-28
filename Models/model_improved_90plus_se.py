"""
ImprovedStutteringCNN variant with Squeeze-and-Excitation (SE) blocks added to residual blocks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(8, channels // reduction))
        self.fc2 = nn.Linear(max(8, channels // reduction), channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, t = x.size()
        y = F.adaptive_avg_pool1d(x, 1).view(b, c)
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1)
        return x * y


class ResidualBlockSE(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.4):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.se = SEBlock(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # Squeeze-and-Excite
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(8, channels // 16))
        self.fc2 = nn.Linear(max(8, channels // 16), channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, t = x.size()
        y = F.adaptive_avg_pool1d(x, 1).view(b, c)
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y.unsqueeze(2)


class ImprovedStutteringCNNLargeSE(nn.Module):
    """Higher-capacity CNN variant with SE blocks in residuals."""

    def __init__(self, n_channels=123, n_classes=5, dropout=0.35):
        super().__init__()
        self.block1 = ResidualBlockSE(n_channels, 128, dropout=dropout, stride=1)
        self.block2 = ResidualBlockSE(128, 256, dropout=dropout, stride=2)
        self.block3 = ResidualBlockSE(256, 512, dropout=dropout, stride=1)
        self.block4 = ResidualBlockSE(512, 512, dropout=dropout, stride=2)
        self.block5 = ResidualBlockSE(512, 1024, dropout=dropout, stride=2)
        self.block6 = ResidualBlockSE(1024, 1024, dropout=dropout, stride=1)
        self.block7 = ResidualBlockSE(1024, 512, dropout=dropout, stride=1)
        self.block8 = ResidualBlockSE(512, 256, dropout=dropout, stride=1)

        self.attention = AttentionBlock(256)

        self.fc1 = nn.Linear(256 * 2, 128)
        self.dropout_fc1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.attention(x)
        avg = F.adaptive_avg_pool1d(x, 1)
        mx = F.adaptive_max_pool1d(x, 1)
        x = torch.cat([avg, mx], dim=1)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout_fc1(x)
        x = self.fc2(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = ImprovedStutteringCNNLargeSE()
    m = m.to(device)
    print('Parameters:', m.count_parameters())
