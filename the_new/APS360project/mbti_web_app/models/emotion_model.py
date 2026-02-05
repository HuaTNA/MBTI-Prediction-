"""
Emotion Recognition CNN Model
PyTorch neural network architecture for facial emotion classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SpatialAttention(nn.Module):
    """Spatial Attention Module for focusing on important spatial regions"""

    def __init__(self, kernel_size=7):
        """
        Initialize spatial attention module

        Args:
            kernel_size: Convolution kernel size (must be 3 or 7)
        """
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'Kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            torch.Tensor: Attention-weighted feature map
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv1(x_cat)
        attention = self.sigmoid(x_out)
        return x * attention


class ChannelAttention(nn.Module):
    """Channel Attention Module for emphasizing important feature channels"""

    def __init__(self, channels, reduction=16):
        """
        Initialize channel attention module

        Args:
            channels: Number of input channels
            reduction: Channel reduction ratio for bottleneck
        """
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            torch.Tensor: Attention-weighted feature map
        """
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        attention = self.sigmoid(out)
        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    Combines channel and spatial attention
    """

    def __init__(self, channels, reduction=16, kernel_size=7):
        """
        Initialize CBAM module

        Args:
            channels: Number of input channels
            reduction: Channel reduction ratio
            kernel_size: Kernel size for spatial attention
        """
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """
        Forward pass - applies channel then spatial attention

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            torch.Tensor: Attention-enhanced feature map
        """
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class EnhancedEmotionModel(nn.Module):
    """
    Enhanced Emotion Recognition CNN Model
    Uses EfficientNet backbone with CBAM attention and multi-head classification
    """

    def __init__(self, num_classes=8, dropout_rates=[0.5, 0.4, 0.3], backbone='efficientnet_b3'):
        """
        Initialize emotion recognition model

        Args:
            num_classes: Number of emotion classes to predict
            dropout_rates: Dropout rates for classifier layers
            backbone: Pretrained backbone model name
        """
        super(EnhancedEmotionModel, self).__init__()

        # Select base model
        if backbone == 'efficientnet_b3':
            self.base_model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
            last_channel = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Add attention module
        self.cbam = CBAM(channels=last_channel, reduction=16, kernel_size=7)

        # Global pooling after feature extraction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Main classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rates[0]),
            nn.Linear(last_channel, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rates[1]),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rates[2]),
            nn.Linear(512, num_classes)
        )

        # Specialized classifier for difficult-to-distinguish classes
        self.specialized_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(last_channel, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 3)
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input image tensor [B, 3, H, W]

        Returns:
            tuple: (main_output, features)
                - main_output: Class predictions [B, num_classes]
                - features: Extracted feature vector [B, feature_dim]
        """
        # Feature extraction
        features = self.base_model.features(x)

        # Apply attention mechanism
        features = self.cbam(features)

        # Global pooling
        features = self.avg_pool(features)
        features = torch.flatten(features, 1)

        # Main classifier output
        main_output = self.classifier(features)

        # In inference we only need the main output
        return main_output, features
