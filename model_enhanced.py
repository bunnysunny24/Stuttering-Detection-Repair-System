"""
ENHANCED CNN Architecture for Stuttering Detection & Speech Repair

Features:
- Deeper network (5 conv layers instead of 4)
- Residual connections for better gradient flow
- Attention mechanism for key feature focus
- Better feature extraction for speech patterns
- Optimized for multi-label stuttering classification

This model is specifically designed for high-accuracy stuttering detection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """Channel Attention Mechanism - focuses on important features."""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch, channels, height, width = x.size()
        avg = self.avg_pool(x).view(batch, channels)
        weights = self.fc(avg).view(batch, channels, 1, 1)
        return x * weights


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + self.skip(identity)
        return F.relu(x)


class EnhancedStutteringCNN(nn.Module):
    """
    Enhanced CNN for Stuttering Detection
    
    Architecture:
    - Input: (batch, 1, 80_mels, time_frames)
    - 5 residual blocks with attention
    - Dense feature extraction
    - Output: (batch, 5) logits (5 stuttering types)
    """
    
    def __init__(self, n_mels=80, n_classes=5, bias_init=None):
        super().__init__()
        
        # Initial conv layer
        self.conv_init = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks with increasing channels
        self.layer1 = self._make_residual_layer(32, 64, num_blocks=2, stride=2)
        self.layer2 = self._make_residual_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_residual_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_residual_layer(256, 256, num_blocks=1, stride=1)
        
        # Attention mechanisms for each layer
        self.attention1 = AttentionLayer(64)
        self.attention2 = AttentionLayer(128)
        self.attention3 = AttentionLayer(256)
        self.attention4 = AttentionLayer(256)
        
        # Global pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dense layers for classification
        self.fc_dense = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Increased from 0.4
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),  # Increased from 0.3
        )
        
        # Output layer (5 classes for 5 stuttering types)
        self.fc_out = nn.Linear(64, n_classes)
        
        # Initialize bias if provided
        if bias_init is not None:
            try:
                b = torch.as_tensor(bias_init, dtype=torch.float32)
                if b.numel() == n_classes:
                    with torch.no_grad():
                        self.fc_out.bias.copy_(b.view_as(self.fc_out.bias))
            except Exception:
                pass
    
    def _make_residual_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass
        x: (batch, 1, 80, time)
        """
        # Initial convolution
        x = self.conv_init(x)  # (batch, 32, 80, time)
        
        # Layer 1: residual blocks + attention
        x = self.layer1(x)  # (batch, 64, 40, time/2)
        x = self.attention1(x)
        
        # Layer 2
        x = self.layer2(x)  # (batch, 128, 20, time/4)
        x = self.attention2(x)
        
        # Layer 3
        x = self.layer3(x)  # (batch, 256, 10, time/8)
        x = self.attention3(x)
        
        # Layer 4
        x = self.layer4(x)  # (batch, 256, 10, time/8)
        x = self.attention4(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)  # (batch, 256, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 256)
        
        # Dense layers
        x = self.fc_dense(x)  # (batch, 64)
        
        # Output
        logits = self.fc_out(x)  # (batch, 5)
        
        return logits


class SimpleCNN(nn.Module):
    """
    Original SimpleCNN - kept for backward compatibility.
    The EnhancedStutteringCNN is recommended for better accuracy.
    """
    
    def __init__(self, n_mels=80, n_classes=5, bias_init=None):
        super().__init__()
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
        self.fc = nn.Linear(320, n_classes)

        if bias_init is not None:
            try:
                b = torch.as_tensor(bias_init, dtype=torch.float32)
                if b.numel() == n_classes:
                    with torch.no_grad():
                        self.fc.bias.copy_(b.view_as(self.fc.bias))
            except Exception:
                pass

    def forward(self, x):
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
