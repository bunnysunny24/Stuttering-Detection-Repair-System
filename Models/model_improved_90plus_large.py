"""
ImprovedStutteringCNNLarge (v2)

Upgrades over original:
- Multi-scale dilated convolution branches for different receptive fields
- Residual blocks with optional Squeeze-and-Excitation
- Lightweight temporal Transformer encoder after CNN stack
- Attention-based learnable pooling (attentive pooling)
- Per-class heads (specialized outputs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(8, channels // reduction))
        self.fc2 = nn.Linear(max(8, channels // reduction), channels)

    def forward(self, x):
        # x: (B, C, T)
        b, c, t = x.size()
        y = x.mean(dim=2)  # (B, C)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y.view(b, c, 1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.4, use_se=True, dilation=1):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.downsample = None

        self.use_se = use_se
        if use_se:
            self.se = SEBlock(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        if self.use_se:
            out = self.se(out)
        return out


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(8, channels // 16))
        self.fc2 = nn.Linear(max(8, channels // 16), channels)

    def forward(self, x):
        b, c, t = x.size()
        y = F.adaptive_avg_pool1d(x, 1).view(b, c)
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        y = torch.sigmoid(y)
        return x * y.unsqueeze(2)


class AttentionPool(nn.Module):
    def __init__(self, channels, attn_dim=128):
        super().__init__()
        self.key = nn.Linear(channels, attn_dim)
        self.query = nn.Parameter(torch.randn(attn_dim))

    def forward(self, x):
        # x: (B, C, T)
        b, c, t = x.size()
        # (B, T, C)
        xt = x.permute(0, 2, 1)
        k = torch.tanh(self.key(xt))  # (B, T, attn_dim)
        q = self.query.unsqueeze(0).unsqueeze(0)  # (1,1,attn_dim)
        scores = (k * q).sum(-1)  # (B, T)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        pooled = (xt * weights).sum(dim=1)  # (B, C)
        return pooled


class ImprovedStutteringCNNLarge(nn.Module):
    """Higher-capacity CNN with multi-scale convs, Transformer, and attentive pooling.

    Designed as an upgraded version of the previous large model while keeping the overall
    methodology the same (1D conv residual stack + pooling + FC), but adding temporal
    modeling and per-class heads for specialization.
    """

    def __init__(self, n_channels=123, n_classes=5, dropout=0.35, d_model=256):
        super().__init__()
        # Multi-scale dilated conv branches (kept small)
        ms_ch = 64
        self.ms_conv1 = nn.Sequential(
            nn.Conv1d(n_channels, ms_ch, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm1d(ms_ch),
            nn.ReLU(inplace=True)
        )
        self.ms_conv2 = nn.Sequential(
            nn.Conv1d(n_channels, ms_ch, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm1d(ms_ch),
            nn.ReLU(inplace=True)
        )
        self.ms_conv3 = nn.Sequential(
            nn.Conv1d(n_channels, ms_ch, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm1d(ms_ch),
            nn.ReLU(inplace=True)
        )

        # Fuse multi-scale outputs
        fused_ch = ms_ch * 3
        self.fuse_conv = nn.Sequential(
            nn.Conv1d(fused_ch, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )

        # Residual stack (kept depth similar to original, modestly wider)
        self.block1 = ResidualBlock(256, 160, dropout=dropout, stride=1, use_se=True)
        self.block2 = ResidualBlock(160, 320, dropout=dropout, stride=2, use_se=True)
        self.block3 = ResidualBlock(320, 640, dropout=dropout, stride=1, use_se=True)
        self.block4 = ResidualBlock(640, 640, dropout=dropout, stride=2, use_se=True)
        self.block5 = ResidualBlock(640, 1280, dropout=dropout, stride=2, use_se=True)
        self.block6 = ResidualBlock(1280, 1280, dropout=dropout, stride=1, use_se=True)
        self.block7 = ResidualBlock(1280, 640, dropout=dropout, stride=1, use_se=True)
        self.block8 = ResidualBlock(640, 320, dropout=dropout, stride=1, use_se=True)

        # Lightweight attention block
        self.attention = AttentionBlock(320)

        # Project to d_model for Transformer
        self.proj_to_dmodel = nn.Conv1d(320, d_model, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=d_model * 2, dropout=dropout, activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Attentive pooling
        self.pool = AttentionPool(d_model)

        # Per-class heads
        self.class_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, max(32, d_model // 4)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(max(32, d_model // 4), 1)
            ) for _ in range(n_classes)
        ])

    def forward(self, x):
        # x: (B, C, T)
        m1 = self.ms_conv1(x)
        m2 = self.ms_conv2(x)
        m3 = self.ms_conv3(x)
        m = torch.cat([m1, m2, m3], dim=1)
        x = self.fuse_conv(m)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)

        x = self.attention(x)

        # Transformer expects (T, B, D)
        x_t = self.proj_to_dmodel(x)  # (B, d_model, T)
        x_t = x_t.permute(2, 0, 1)  # (T, B, d_model)
        x_t = self.transformer(x_t)  # (T, B, d_model)
        x_t = x_t.permute(1, 2, 0)  # (B, d_model, T)

        pooled = self.pool(x_t)  # (B, d_model)

        # Per-class logits
        logits = [head(pooled) for head in self.class_heads]
        logits = torch.cat(logits, dim=1)
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = ImprovedStutteringCNNLarge()
    m = m.to(device)
    print('Parameters:', m.count_parameters())