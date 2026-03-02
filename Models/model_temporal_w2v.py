"""
Temporal 1D CNN classifier for frame-level wav2vec2 features.

Instead of collapsing an entire audio clip into a single vector,
this model processes the SEQUENCE of wav2vec2 frame embeddings (768, T)
using 1D convolutions with dilated kernels to capture temporal stuttering
patterns at multiple time scales:
  - Dilation 1: local phoneme-level patterns (~60ms)
  - Dilation 2: syllable-level patterns (~120ms)
  - Dilation 4: word-level patterns (~240ms)

Architecture:
  Input (768, T) → Projection (256) → 3× Dilated ConvBlock → Attention Pool → 5 class heads

Total parameters: ~870K (vs 4.5M for MLP — much less overfitting risk)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """1D convolution block with residual connection."""

    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.3):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2  # same-length padding
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        # Residual projection if dimensions change
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


class AttentionPool(nn.Module):
    """Learned attention pooling over the time dimension.

    Instead of mean-pooling which weights all frames equally, this learns
    to focus on the frames that matter most for classification (e.g.,
    the frames where stuttering actually occurs).
    """

    def __init__(self, dim):
        super().__init__()
        self.query = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1, bias=False)
        )

    def forward(self, x):
        """x: (batch, channels, time) → (batch, channels)"""
        # Transpose to (batch, time, channels) for attention computation
        x_t = x.transpose(1, 2)           # (B, T, C)
        attn_logits = self.query(x_t)      # (B, T, 1)
        attn_weights = F.softmax(attn_logits, dim=1)  # (B, T, 1)
        pooled = (x_t * attn_weights).sum(dim=1)      # (B, C)
        return pooled


class TemporalStutterClassifier(nn.Module):
    """
    1D CNN temporal classifier for wav2vec2 frame features.

    Input:  (batch, 768, T) — wav2vec2 frame embeddings
    Output: (batch, n_classes) — per-class logits

    Architecture:
        1. Project 768 → hidden_dim (256) via 1×1 conv
        2. 3 dilated ConvBlocks (dilation 1, 2, 4) for multi-scale temporal patterns
        3. Attention pooling over time → fixed-size vector
        4. Per-class classification heads

    Receptive field: 1 + 2*(1+2+4) = 15 frames = 300ms at 50fps
    This covers typical stutter event durations.
    """

    def __init__(self, input_dim=768, n_classes=5, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes

        # 1×1 projection from 768 → hidden_dim
        self.proj = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )

        # Dilated temporal convolution stack
        self.temporal_blocks = nn.Sequential(
            ConvBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=1, dropout=dropout),
            ConvBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2, dropout=dropout),
            ConvBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=4, dropout=dropout),
        )

        # Multi-scale feature aggregation:
        # Also add a parallel branch with larger kernel for longer patterns
        self.wide_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=7,
                      padding=3, bias=False),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # After concatenation: hidden_dim + hidden_dim//2
        fusion_dim = hidden_dim + hidden_dim // 2  # 384

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv1d(fusion_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Attention pooling
        self.attention_pool = AttentionPool(hidden_dim)

        # Per-class classification heads (independent binary classifiers)
        self.class_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),  # lighter dropout in heads
                nn.Linear(64, 1)
            )
            for _ in range(n_classes)
        ])

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Kaiming normal for conv, Xavier for linear."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: (batch, 768, T) frame-level wav2vec2 features

        Returns:
            logits: (batch, n_classes) raw logits (apply sigmoid for probabilities)
        """
        # Project to hidden dimension
        h = self.proj(x)  # (B, hidden_dim, T)

        # Dilated temporal convolutions (captures local → medium patterns)
        h_temporal = self.temporal_blocks(h)  # (B, hidden_dim, T)

        # Wide convolution (captures broader patterns)
        h_wide = self.wide_conv(h)  # (B, hidden_dim//2, T)

        # Concatenate multi-scale features
        h_cat = torch.cat([h_temporal, h_wide], dim=1)  # (B, fusion_dim, T)

        # Fuse
        h_fused = self.fusion(h_cat)  # (B, hidden_dim, T)

        # Attention pool over time
        pooled = self.attention_pool(h_fused)  # (B, hidden_dim)

        # Per-class predictions
        logits = torch.cat([head(pooled) for head in self.class_heads], dim=1)  # (B, n_classes)

        return logits


def count_parameters(model):
    """Count trainable parameters."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total


if __name__ == '__main__':
    # Quick test
    model = TemporalStutterClassifier(
        input_dim=768, n_classes=5, hidden_dim=256, dropout=0.3
    )
    n_params = count_parameters(model)
    print(f"TemporalStutterClassifier: {n_params:,} trainable parameters")

    # Test forward pass
    batch = torch.randn(4, 768, 100)  # 4 clips, 100 frames each
    logits = model(batch)
    print(f"Input shape:  {batch.shape}")
    print(f"Output shape: {logits.shape}")
    assert logits.shape == (4, 5), f"Expected (4, 5), got {logits.shape}"

    # Test variable length
    batch_short = torch.randn(4, 768, 30)  # shorter clips
    logits_short = model(batch_short)
    print(f"Short input:  {batch_short.shape} → {logits_short.shape}")

    print("\nAll tests passed!")
