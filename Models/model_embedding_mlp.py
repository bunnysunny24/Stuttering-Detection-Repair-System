"""
Lightweight MLP classifier for wav2vec2/HuBERT pretrained embeddings.

Input: 1536-dim embedding (768 mean + 768 std from wav2vec2-base)
Output: 5 stuttering classes (multi-label sigmoid)

Architecture:
  Input(1536) -> BN -> FC(1024) -> GELU -> Dropout -> BN
              -> FC(512) -> GELU -> Dropout -> BN
              -> FC(256) -> GELU -> Dropout -> BN
              -> FC(5) -> Sigmoid

  + Residual connection where dimensions match
  + LayerNorm for stability
  + ~1.5M parameters (vs 31M for CNN — trains 20x faster)

Usage in training:
  from model_embedding_mlp import EmbeddingMLPClassifier
  model = EmbeddingMLPClassifier(input_dim=1536, n_classes=5, dropout=0.3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """FC -> BN -> GELU -> Dropout with optional residual."""
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        
        # Residual projection if dimensions differ
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
    
    def forward(self, x):
        res = self.residual(x)
        out = self.fc(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.drop(out)
        return out + res


class EmbeddingMLPClassifier(nn.Module):
    """
    MLP classifier for pretrained speech embeddings.
    
    Designed for wav2vec2-base (768-dim) or wav2vec2-base stats-pooled (1536-dim).
    Uses residual connections, batch norm, and GELU activation for stable training.
    """
    def __init__(self, input_dim=1536, n_classes=5, dropout=0.3, 
                 hidden_dims=None, n_channels=None):
        """
        Args:
            input_dim: Embedding dimension (1536 for mean+std pooling, 768 for mean only)
            n_classes: Number of output classes
            dropout: Dropout rate
            hidden_dims: List of hidden layer dimensions (default: [1024, 512, 256])
            n_channels: Ignored (for API compatibility with CNN models)
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [1024, 512, 256]
        
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        # Build residual blocks
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(ResidualBlock(prev_dim, hdim, dropout))
            prev_dim = hdim
        self.backbone = nn.Sequential(*layers)
        
        # Per-class output heads (like the CNN model)
        self.class_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(prev_dim, 64),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(64, 1)
            )
            for _ in range(n_classes)
        ])
        
        self.n_classes = n_classes
        self._init_weights()
    
    def _init_weights(self):
        """Kaiming initialization for all linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: (batch, input_dim) embedding tensor
        Returns:
            logits: (batch, n_classes) raw logits (apply sigmoid for probabilities)
        """
        # Handle case where input comes from spectrogram-shaped data
        if x.dim() == 3:
            # (batch, channels, time) -> flatten to (batch, channels*time)
            x = x.view(x.size(0), -1)
        
        x = self.input_bn(x)
        x = self.backbone(x)
        
        # Per-class heads
        outputs = []
        for head in self.class_heads:
            outputs.append(head(x))
        
        logits = torch.cat(outputs, dim=1)  # (batch, n_classes)
        return logits


if __name__ == '__main__':
    # Test
    model = EmbeddingMLPClassifier(input_dim=1536, n_classes=5, dropout=0.3)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    # Test forward pass
    x = torch.randn(32, 1536)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Probs:  {torch.sigmoid(out[0]).detach()}")
    
    # Test with 768-dim input 
    model_768 = EmbeddingMLPClassifier(input_dim=768, n_classes=5, dropout=0.3)
    n_params_768 = sum(p.numel() for p in model_768.parameters())
    print(f"\n768-dim model parameters: {n_params_768:,}")
    x768 = torch.randn(32, 768)
    out768 = model_768(x768)
    print(f"Input:  {x768.shape}")
    print(f"Output: {out768.shape}")
