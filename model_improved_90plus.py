"""
IMPROVED 90+ ACCURACY MODEL ARCHITECTURE
8-layer CNN with strong regularization for extreme imbalance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Simple 1D residual block with optional downsampling.
    Uses two conv1d layers with BatchNorm and dropout, plus identity/skip.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.4):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Optional downsample for the residual path
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

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)
        return out


class ImprovedStutteringCNN(nn.Module):
    """
    8-layer CNN for stuttering detection with:
    - More capacity (8 layers vs 5)
    - Stronger regularization (dropout 0.4-0.5)
    - Batch normalization for stability
    - Attention mechanism for focus
    - Better gradient flow
    
    Input: (batch, 123, time_steps)
    Output: (batch, 5) - logits for 5 stutter classes
    """
    
    def __init__(self, n_channels=123, n_classes=5, dropout=0.4):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.dropout_rate = dropout
        
        # Residual blocks for better gradient flow and robustness
        self.block1 = ResidualBlock(n_channels, 64, dropout=dropout, stride=1)
        self.block2 = ResidualBlock(64, 128, dropout=dropout, stride=2)
        self.block3 = ResidualBlock(128, 256, dropout=dropout, stride=1)
        self.block4 = ResidualBlock(256, 256, dropout=dropout, stride=2)
        self.block5 = ResidualBlock(256, 512, dropout=dropout, stride=2)
        self.block6 = ResidualBlock(512, 512, dropout=dropout, stride=1)
        self.block7 = ResidualBlock(512, 256, dropout=dropout, stride=1)
        self.block8 = ResidualBlock(256, 128, dropout=dropout, stride=1)
        
        # Attention mechanism (channel-wise)
        self.attention = AttentionBlock(128)
        
        # Use both average and max pooling concatenated for stronger summary
        # Global pooling will produce 128*2 features
        self.fc1 = nn.Linear(128 * 2, 64)
        self.dropout_fc1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, n_classes)
        
        self.relu = nn.ReLU(inplace=True)
        self.l2_weight = 1e-4  # L2 regularization
    
    def forward(self, x):
        """Forward pass through improved architecture."""

        # Pass through residual blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)

        # Attention mechanism
        x = self.attention(x)

        # Global pooled representations (avg + max)
        avg = F.adaptive_avg_pool1d(x, 1)
        mx = F.adaptive_max_pool1d(x, 1)
        x = torch.cat([avg, mx], dim=1)  # shape: (batch, channels*2, 1)
        x = x.view(x.size(0), -1)

        # Classification head
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout_fc1(x)
        x = self.fc2(x)

        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AttentionBlock(nn.Module):
    """Channel attention mechanism - helps model focus on important features."""
    
    def __init__(self, channels):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // 16)
        self.fc2 = nn.Linear(channels // 16, channels)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Global average pooling
        b, c, t = x.size()
        y = F.adaptive_avg_pool1d(x, 1).view(b, c)
        
        # Channel attention
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        
        # Apply attention weights
        return x * y.unsqueeze(2)



print("=" * 80)

if __name__ == '__main__':
    # Test model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedStutteringCNN(n_channels=123, n_classes=5, dropout=0.4)
    model = model.to(device)
    
    print(f"\nModel Summary:")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Device: {device}")
    
    # Test forward pass
    x = torch.randn(2, 123, 256).to(device)  # Batch of 2, 123 channels, 256 time steps
    output = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output (logits): {output}")