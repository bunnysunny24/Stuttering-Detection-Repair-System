"""
End-to-end wav2vec2-large fine-tuning model for stuttering detection.

Instead of using frozen wav2vec2-base features, this model:
  1. Loads wav2vec2-large (24 layers, 1024-dim, ~315M params)
  2. Freezes the CNN feature extractor + first N transformer layers
  3. Fine-tunes the last K transformer layers (default: last 6)
  4. Feeds temporal features into a BiLSTM+CNN+Attention classifier head

This gives the backbone the ability to ADAPT its representations
specifically for stuttering patterns, which is the single biggest
improvement possible over frozen features.

Memory estimate (partial fine-tune, last 6 layers):
  Static: ~2.1 GB (params + grads + optimizer for tuned layers)
  Activations: ~0.1 GB per batch of 4
  Total: ~3-5 GB — fits easily in 40 GB RAM

Architecture:
  Raw audio (16kHz)
    → wav2vec2-large CNN feature extractor (frozen)
    → Transformer layers 1-18 (frozen, forward-only)
    → Transformer layers 19-24 (fine-tuned)
    → Multi-layer weighted average of layers 19-24
    → BiLSTM (2 layers, 128 per direction)
    → Dilated CNN (d=1,2,4) + wide conv
    → Multi-head attention pooling (5 heads)
    → 5 classification heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor


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
        residual = self.residual(x)
        out = self.drop(self.act(self.bn(self.conv(x))))
        return out + residual


class MultiHeadAttentionPool(nn.Module):
    """Multi-head attention pooling — one head per stutter class."""

    def __init__(self, dim, n_heads=5):
        super().__init__()
        self.queries = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.Tanh(),
                nn.Linear(dim // 4, 1, bias=False)
            ) for _ in range(n_heads)
        ])

    def forward(self, x):
        """x: (B, C, T) → list of n_heads × (B, C)"""
        x_t = x.transpose(1, 2)  # (B, T, C)
        pooled = []
        for query in self.queries:
            attn = F.softmax(query(x_t), dim=1)  # (B, T, 1)
            pooled.append((x_t * attn).sum(dim=1))  # (B, C)
        return pooled


class Wav2VecFineTuneClassifier(nn.Module):
    """
    End-to-end wav2vec2-large fine-tuning for stuttering detection.

    Args:
        model_name: HuggingFace model name (default: wav2vec2-large)
        n_classes: Number of stutter classes (default: 5)
        freeze_layers: Number of transformer layers to freeze (default: 18 of 24)
        hidden_dim: Classifier hidden dimension (default: 256)
        lstm_hidden: LSTM hidden size per direction (default: 128)
        lstm_layers: Number of LSTM layers (default: 2)
        dropout: Dropout rate (default: 0.3)
        use_gradient_checkpointing: Save memory with gradient checkpointing
    """

    def __init__(self, model_name='facebook/wav2vec2-large',
                 n_classes=5, freeze_layers=18, hidden_dim=256,
                 lstm_hidden=128, lstm_layers=2, dropout=0.3,
                 use_gradient_checkpointing=True):
        super().__init__()
        self.n_classes = n_classes
        self.freeze_layers = freeze_layers

        # ---- Load wav2vec2-large backbone ----
        print(f"Loading {model_name}...")
        self.backbone = Wav2Vec2Model.from_pretrained(model_name)
        self.backbone.config.output_hidden_states = True

        # Enable gradient checkpointing to save memory
        if use_gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            print("  Gradient checkpointing enabled (use_reentrant=False)")

        # ---- Freeze layers ----
        # Always freeze the CNN feature extractor
        for param in self.backbone.feature_extractor.parameters():
            param.requires_grad = False
        for param in self.backbone.feature_projection.parameters():
            param.requires_grad = False

        # Freeze first N transformer layers
        n_transformer_layers = len(self.backbone.encoder.layers)
        layers_to_freeze = min(freeze_layers, n_transformer_layers)
        for i in range(layers_to_freeze):
            for param in self.backbone.encoder.layers[i].parameters():
                param.requires_grad = False

        tuned_layers = n_transformer_layers - layers_to_freeze
        print(f"  Transformer layers: {n_transformer_layers} total, "
              f"{layers_to_freeze} frozen, {tuned_layers} fine-tuned")

        # Hidden dimension from backbone (1024 for large, 768 for base)
        backbone_dim = self.backbone.config.hidden_size

        # ---- Weighted layer combination (tuned layers only) ----
        # Learnable weights for combining the fine-tuned layers
        self.layer_indices = list(range(layers_to_freeze + 1,
                                       n_transformer_layers + 1))  # +1 for 1-indexed hidden_states
        n_combined = len(self.layer_indices)
        self.layer_weights = nn.Parameter(torch.ones(n_combined) / n_combined)

        # ---- Classifier head (same architecture as TemporalBiLSTMClassifier) ----
        # Project from backbone dim to hidden dim
        self.proj = nn.Sequential(
            nn.Conv1d(backbone_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        self.lstm_norm = nn.LayerNorm(lstm_hidden * 2)
        self.lstm_drop = nn.Dropout(dropout)

        # Dilated CNN stack
        lstm_out_dim = lstm_hidden * 2
        self.temporal_blocks = nn.Sequential(
            ConvBlock(lstm_out_dim, hidden_dim, kernel_size=3, dilation=1, dropout=dropout),
            ConvBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2, dropout=dropout),
            ConvBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=4, dropout=dropout),
        )

        # Wide conv branch
        self.wide_conv = nn.Sequential(
            nn.Conv1d(lstm_out_dim, hidden_dim // 2, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Fusion
        fusion_dim = hidden_dim + hidden_dim // 2
        self.fusion = nn.Sequential(
            nn.Conv1d(fusion_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Multi-head attention pooling
        self.attention_pool = MultiHeadAttentionPool(hidden_dim, n_heads=n_classes)

        # Per-class heads
        self.class_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(64, 1)
            ) for _ in range(n_classes)
        ])

        # Binary auxiliary head: "any stutter" — helps representation learning
        # Uses global average pooling over fused features
        self.binary_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1)
        )

        self._init_head_weights()
        self._print_param_summary()

    def _init_head_weights(self):
        """Initialize classifier head weights."""
        for m in self.modules():
            if m is self.backbone:
                continue  # Don't reinit backbone
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # LSTM init
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)

    def _print_param_summary(self):
        """Print parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        backbone_total = sum(p.numel() for p in self.backbone.parameters())
        backbone_trainable = sum(p.numel() for p in self.backbone.parameters()
                                 if p.requires_grad)
        head_params = trainable - backbone_trainable
        print(f"\n  Parameter Summary:")
        print(f"    Total:              {total:>12,}")
        print(f"    Trainable:          {trainable:>12,} ({trainable/total*100:.1f}%)")
        print(f"    Frozen:             {frozen:>12,} ({frozen/total*100:.1f}%)")
        print(f"    Backbone trainable: {backbone_trainable:>12,}")
        print(f"    Head trainable:     {head_params:>12,}")
        print(f"    Memory (params):    {total * 4 / 1024**2:.0f} MB")
        print(f"    Memory (trainable): {trainable * (4+4+8) / 1024**2:.0f} MB "
              f"(params+grads+adam)\n")

    def forward(self, input_values, attention_mask=None):
        """
        Args:
            input_values: (B, audio_samples) raw 16kHz waveform
            attention_mask: (B, audio_samples) optional mask

        Returns:
            logits: (B, n_classes)
        """
        # ---- Backbone forward ----
        outputs = self.backbone(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # ---- Weighted layer combination ----
        hidden_states = outputs.hidden_states  # tuple of (n_layers+1,) × (B, T, D)
        weights = F.softmax(self.layer_weights, dim=0)
        combined = None
        for i, layer_idx in enumerate(self.layer_indices):
            h = hidden_states[layer_idx]  # (B, T, D)
            if combined is None:
                combined = weights[i] * h
            else:
                combined = combined + weights[i] * h

        # combined: (B, T, D) → (B, D, T) for CNN
        h = combined.transpose(1, 2)  # (B, backbone_dim, T)

        # ---- Classifier head ----
        h = self.proj(h)  # (B, hidden_dim, T)

        # BiLSTM
        h_lstm_in = h.transpose(1, 2)  # (B, T, hidden_dim)
        h_lstm_out, _ = self.lstm(h_lstm_in)  # (B, T, lstm_hidden*2)
        h_lstm_out = self.lstm_norm(h_lstm_out)
        h_lstm_out = self.lstm_drop(h_lstm_out)
        h_lstm = h_lstm_out.transpose(1, 2)  # (B, lstm_hidden*2, T)

        # Dilated CNN
        h_temporal = self.temporal_blocks(h_lstm)
        h_wide = self.wide_conv(h_lstm)
        h_cat = torch.cat([h_temporal, h_wide], dim=1)
        h_fused = self.fusion(h_cat)

        # Attention pooling + classification
        pooled_list = self.attention_pool(h_fused)
        logits = []
        for pooled, head in zip(pooled_list, self.class_heads):
            logits.append(head(pooled))
        logits = torch.cat(logits, dim=1)

        # Binary auxiliary head: global avg pool → single logit
        h_global = h_fused.mean(dim=2)  # (B, hidden_dim)
        binary_logit = self.binary_head(h_global)  # (B, 1)

        return logits, binary_logit

    def get_param_groups(self, backbone_lr=1e-5, head_lr=1e-4):
        """Get parameter groups with discriminative learning rates.

        The backbone (fine-tuned layers) uses a LOWER learning rate
        to avoid catastrophic forgetting. The head uses a HIGHER rate
        since it's training from scratch.
        """
        backbone_params = []
        head_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('backbone.'):
                backbone_params.append(param)
            else:
                head_params.append(param)

        return [
            {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': 1e-2},
            {'params': head_params, 'lr': head_lr, 'weight_decay': 1e-4},
        ]


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    import time

    print("=" * 60)
    print("Testing Wav2VecFineTuneClassifier")
    print("=" * 60)

    model = Wav2VecFineTuneClassifier(
        model_name='facebook/wav2vec2-large',
        n_classes=5,
        freeze_layers=18,
        hidden_dim=256,
        lstm_hidden=128,
        lstm_layers=2,
        dropout=0.3,
    )

    # Test forward pass with fake audio
    print("Testing forward pass...")
    batch_size = 2
    audio_len = 48000  # 3 seconds at 16kHz
    fake_audio = torch.randn(batch_size, audio_len)

    model.eval()
    with torch.no_grad():
        t0 = time.time()
        logits, binary_logit = model(fake_audio)
        elapsed = time.time() - t0

    print(f"  Input:  ({batch_size}, {audio_len})")
    print(f"  Output: logits={logits.shape}, binary={binary_logit.shape}")
    print(f"  Time:   {elapsed:.2f}s")

    # Test param groups
    groups = model.get_param_groups(backbone_lr=1e-5, head_lr=1e-4)
    for i, g in enumerate(groups):
        n = sum(p.numel() for p in g['params'])
        print(f"  Group {i}: {n:,} params, lr={g['lr']}")

    print("\nAll tests passed!")
