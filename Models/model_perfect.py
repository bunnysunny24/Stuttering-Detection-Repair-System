"""
PERFECT 90+ MODEL ARCHITECTURE
Combines the best elements: Multi-scale CNN + BiLSTM + Transformer + SE + Attention Pooling.

Key improvements over previous architectures:
1. Multi-scale dilated CNN front-end (captures short + long patterns)
2. Squeeze-and-Excitation (SE) in residual blocks (channel recalibration)
3. BiLSTM temporal layer (bidirectional context)
4. Lightweight Transformer layer (global self-attention) 
5. Attentive statistics pooling (learnable weighted pooling + std)
6. Per-class specialized heads (separate classifier per stutter type)
7. Careful dropout schedule (lower in early layers, higher in heads)

Input:  (batch, 123, time_steps)
Output: (batch, 5) logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation: recalibrate channel importance."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(8, channels // reduction)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x):
        b, c, t = x.size()
        y = x.mean(dim=2)
        y = F.relu(self.fc1(y), inplace=True)
        y = torch.sigmoid(self.fc2(y))
        return x * y.view(b, c, 1)


class ResBlock(nn.Module):
    """Residual block with SE, dilated convolutions, and careful dropout."""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, dropout=0.15, dilation=1, use_se=True):
        super().__init__()
        pad = (kernel // 2) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=pad, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, stride=1, padding=pad, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.se = SEBlock(out_ch) if use_se else nn.Identity()
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = out + identity
        out = F.relu(out, inplace=True)
        out = self.se(out)
        return out


class AttentiveStatsPool(nn.Module):
    """Attentive statistics pooling: weighted mean + weighted std."""
    def __init__(self, in_dim, attn_dim=128):
        super().__init__()
        self.linear = nn.Linear(in_dim, attn_dim)
        self.query = nn.Parameter(torch.randn(attn_dim))

    def forward(self, x):
        # x: (B, C, T)
        xt = x.permute(0, 2, 1)  # (B, T, C)
        e = torch.tanh(self.linear(xt))  # (B, T, attn_dim)
        scores = (e * self.query).sum(-1)  # (B, T)
        alpha = F.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        mean = (xt * alpha).sum(dim=1)  # (B, C)
        var = ((xt - mean.unsqueeze(1)) ** 2 * alpha).sum(dim=1)  # (B, C)
        std = torch.sqrt(var.clamp(min=1e-8))
        return torch.cat([mean, std], dim=1)  # (B, 2*C)


class PerfectStutterModel(nn.Module):
    """
    Highest-capacity model for 90%+ stuttering detection.
    
    Architecture:
        Multi-scale CNN → ResBlock stack w/ SE → BiLSTM → Transformer → Attentive Stats Pool → Per-class heads
    """

    def __init__(self, n_channels=123, n_classes=5, dropout=0.2, d_model=256, lstm_hidden=128, lstm_layers=2):
        super().__init__()
        self.n_classes = n_classes

        # === Multi-scale front-end ===
        ms = 64
        self.ms1 = nn.Sequential(nn.Conv1d(n_channels, ms, 3, padding=1, dilation=1, bias=False), nn.BatchNorm1d(ms), nn.ReLU(True))
        self.ms2 = nn.Sequential(nn.Conv1d(n_channels, ms, 3, padding=2, dilation=2, bias=False), nn.BatchNorm1d(ms), nn.ReLU(True))
        self.ms3 = nn.Sequential(nn.Conv1d(n_channels, ms, 3, padding=4, dilation=4, bias=False), nn.BatchNorm1d(ms), nn.ReLU(True))
        self.ms4 = nn.Sequential(nn.Conv1d(n_channels, ms, 5, padding=4, dilation=2, bias=False), nn.BatchNorm1d(ms), nn.ReLU(True))
        fused = ms * 4
        self.fuse = nn.Sequential(nn.Conv1d(fused, 256, 1, bias=False), nn.BatchNorm1d(256), nn.ReLU(True))

        # === Residual stack ===
        self.res1 = ResBlock(256, 256, dropout=dropout * 0.5, stride=1, use_se=True)
        self.res2 = ResBlock(256, 256, dropout=dropout * 0.5, stride=2, use_se=True)
        self.res3 = ResBlock(256, 384, dropout=dropout * 0.75, stride=1, use_se=True)
        self.res4 = ResBlock(384, 384, dropout=dropout * 0.75, stride=2, use_se=True)
        self.res5 = ResBlock(384, 512, dropout=dropout, stride=1, use_se=True)
        self.res6 = ResBlock(512, 512, dropout=dropout, stride=2, use_se=True)
        self.res7 = ResBlock(512, d_model, dropout=dropout, stride=1, use_se=True)

        # === BiLSTM ===
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out = lstm_hidden * 2  # bidirectional

        # === Transformer layer ===
        self.proj = nn.Conv1d(lstm_out, d_model, 1)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)

        # === Attentive stats pooling ===
        self.pool = AttentiveStatsPool(d_model, attn_dim=128)
        pool_out = d_model * 2  # mean + std

        # === Per-class heads ===
        head_hidden = max(64, d_model // 2)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(pool_out, head_hidden),
                nn.BatchNorm1d(head_hidden),
                nn.ReLU(True),
                nn.Dropout(dropout * 1.5),
                nn.Linear(head_hidden, head_hidden // 2),
                nn.ReLU(True),
                nn.Dropout(dropout),
                nn.Linear(head_hidden // 2, 1),
            )
            for _ in range(n_classes)
        ])

    def forward(self, x):
        # x: (B, C, T)
        # Multi-scale
        m = torch.cat([self.ms1(x), self.ms2(x), self.ms3(x), self.ms4(x)], dim=1)
        x = self.fuse(m)

        # Residual
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)

        # BiLSTM: (B, C, T) -> (B, T, C) for LSTM
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)  # (B, T, lstm_out)

        # Transformer: project back to d_model
        x = x.permute(0, 2, 1)  # (B, lstm_out, T)
        x = self.proj(x)  # (B, d_model, T)
        x = x.permute(0, 2, 1)  # (B, T, d_model)
        x = self.transformer(x)  # (B, T, d_model)
        x = x.permute(0, 2, 1)  # (B, d_model, T)

        # Pool
        pooled = self.pool(x)  # (B, 2*d_model)

        # Per-class logits
        logits = torch.cat([head(pooled) for head in self.heads], dim=1)  # (B, n_classes)
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = PerfectStutterModel().to(device)
    print(f'PerfectStutterModel parameters: {m.count_parameters():,}')
    x = torch.randn(2, 123, 256).to(device)
    y = m(x)
    print(f'Input: {x.shape} -> Output: {y.shape}')
    print(f'Output: {y}')
