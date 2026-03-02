"""
BiLSTM + Dilated CNN temporal classifier for wav2vec2 frame features.

This is the strongest classifier architecture for frozen wav2vec2 features.
It combines TWO complementary temporal processing strategies:

1. BiLSTM: Captures long-range temporal dependencies across the entire clip.
   Critical for detecting REPETITIONS (which can repeat 2-10+ times across
   many frames) and BLOCKS (prolonged silences with context).

2. Dilated CNN: Captures local multi-scale patterns with fixed receptive fields.
   Critical for detecting PROLONGATIONS (sustained sounds) and local
   SOUND REPETITIONS (rapid repeated phonemes).

3. Multi-Head Attention Pooling: Learns to focus on the most informative
   frames for each class independently.

Architecture:
  Input (768, T)
    → 1×1 Projection (256)
    → BiLSTM (2 layers, 128 per direction = 256 total)
    → 3× Dilated ConvBlocks (dilation 1, 2, 4)
    → Residual fusion with wide-kernel branch
    → Multi-head attention pooling (5 heads, one per class)
    → Per-class classification heads

Parameters: ~3.5M (enough capacity without overfitting on 25K samples)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """1D convolution block with residual connection."""

    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.3):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        self.residual = (nn.Conv1d(in_ch, out_ch, 1, bias=False)
                         if in_ch != out_ch else nn.Identity())

    def forward(self, x):
        """x: (batch, channels, time)"""
        residual = self.residual(x)
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.drop(out)
        return out + residual


class MultiHeadAttentionPool(nn.Module):
    """Multi-head attention pooling — one attention head per class.

    Each stutter type can attend to different parts of the signal:
      - Prolongation head focuses on sustained activations
      - Block head focuses on silence gaps
      - Repetition heads focus on periodic patterns
    """

    def __init__(self, dim, n_heads=5):
        super().__init__()
        self.n_heads = n_heads
        self.queries = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.Tanh(),
                nn.Linear(dim // 4, 1, bias=False)
            )
            for _ in range(n_heads)
        ])

    def forward(self, x):
        """x: (batch, channels, time) → list of (batch, channels), one per head"""
        x_t = x.transpose(1, 2)  # (B, T, C)
        pooled = []
        for query in self.queries:
            attn_logits = query(x_t)                   # (B, T, 1)
            attn_weights = F.softmax(attn_logits, dim=1)  # (B, T, 1)
            p = (x_t * attn_weights).sum(dim=1)        # (B, C)
            pooled.append(p)
        return pooled  # list of n_heads × (B, C)


class TemporalBiLSTMClassifier(nn.Module):
    """
    BiLSTM + 1D CNN temporal classifier for wav2vec2 frame features.

    This combines recurrent (global context) and convolutional (local patterns)
    processing for maximum temporal modeling power.

    Input:  (batch, 768, T) — wav2vec2 frame embeddings
    Output: (batch, n_classes) — per-class logits
    """

    def __init__(self, input_dim=768, n_classes=5, hidden_dim=256,
                 lstm_hidden=128, lstm_layers=2, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes

        # 1×1 projection from 768 → hidden_dim
        self.proj = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )

        # BiLSTM for long-range temporal dependencies
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        self.lstm_norm = nn.LayerNorm(lstm_hidden * 2)  # 2× for bidirectional
        self.lstm_drop = nn.Dropout(dropout)

        # Dilated CNN stack on top of LSTM output
        lstm_out_dim = lstm_hidden * 2  # bidirectional
        self.temporal_blocks = nn.Sequential(
            ConvBlock(lstm_out_dim, hidden_dim, kernel_size=3, dilation=1, dropout=dropout),
            ConvBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2, dropout=dropout),
            ConvBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=4, dropout=dropout),
        )

        # Parallel wide-kernel branch
        self.wide_conv = nn.Sequential(
            nn.Conv1d(lstm_out_dim, hidden_dim // 2, kernel_size=7,
                      padding=3, bias=False),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Fusion: dilated output + wide output
        fusion_dim = hidden_dim + hidden_dim // 2  # 384
        self.fusion = nn.Sequential(
            nn.Conv1d(fusion_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Multi-head attention pooling (one head per class)
        self.attention_pool = MultiHeadAttentionPool(hidden_dim, n_heads=n_classes)

        # Per-class classification heads
        self.class_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(64, 1)
            )
            for _ in range(n_classes)
        ])

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Initialize LSTM with orthogonal weights (better for training)
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
                # Set forget gate bias to 1 (helps LSTM remember longer)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)

    def forward(self, x):
        """
        Args:
            x: (batch, 768, T) frame-level wav2vec2 features

        Returns:
            logits: (batch, n_classes) raw logits
        """
        # Project to hidden dimension
        h = self.proj(x)  # (B, hidden_dim, T)

        # BiLSTM: expects (B, T, C)
        h_lstm_in = h.transpose(1, 2)  # (B, T, hidden_dim)
        h_lstm_out, _ = self.lstm(h_lstm_in)  # (B, T, lstm_hidden*2)
        h_lstm_out = self.lstm_norm(h_lstm_out)
        h_lstm_out = self.lstm_drop(h_lstm_out)

        # Back to (B, C, T) for CNN
        h_lstm = h_lstm_out.transpose(1, 2)  # (B, lstm_hidden*2, T)

        # Dilated CNN on LSTM output
        h_temporal = self.temporal_blocks(h_lstm)  # (B, hidden_dim, T)

        # Wide convolution branch (parallel to dilated CNN)
        h_wide = self.wide_conv(h_lstm)  # (B, hidden_dim//2, T)

        # Concatenate multi-scale features
        h_cat = torch.cat([h_temporal, h_wide], dim=1)  # (B, fusion_dim, T)

        # Fuse
        h_fused = self.fusion(h_cat)  # (B, hidden_dim, T)

        # Multi-head attention pooling (one per class)
        pooled_list = self.attention_pool(h_fused)  # list of n_classes × (B, hidden_dim)

        # Per-class predictions using class-specific attention
        logits = []
        for i, (pooled, head) in enumerate(zip(pooled_list, self.class_heads)):
            logits.append(head(pooled))  # (B, 1)
        logits = torch.cat(logits, dim=1)  # (B, n_classes)

        return logits


def count_parameters(model):
    """Count trainable parameters."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total


if __name__ == '__main__':
    # Quick test
    model = TemporalBiLSTMClassifier(
        input_dim=768, n_classes=5, hidden_dim=256,
        lstm_hidden=128, lstm_layers=2, dropout=0.3
    )
    n_params = count_parameters(model)
    print(f"TemporalBiLSTMClassifier: {n_params:,} trainable parameters")

    # Test forward pass
    batch = torch.randn(4, 768, 100)
    logits = model(batch)
    print(f"Input shape:  {batch.shape}")
    print(f"Output shape: {logits.shape}")
    assert logits.shape == (4, 5)

    # Test variable length
    batch_short = torch.randn(4, 768, 30)
    logits_short = model(batch_short)
    print(f"Short input:  {batch_short.shape} -> {logits_short.shape}")

    # Test single sample
    batch_one = torch.randn(1, 768, 50)
    logits_one = model(batch_one)
    print(f"Single:       {batch_one.shape} -> {logits_one.shape}")

    print("\nAll tests passed!")
