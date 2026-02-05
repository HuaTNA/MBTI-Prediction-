import os
import json
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR

from torchvision import datasets, transforms, models
from PIL import Image

# ====================== 32-Layer Deep Emotion Recognition Network ======================
class DeepEmotionNet32(nn.Module):
    def __init__(self, num_classes=7, filters_multiplier=1.0, dropout_rate=0.5):
        """
        32-layer deep CNN for facial emotion recognition using residual connections.

        Args:
            num_classes (int): Number of emotion classes.
            filters_multiplier (float): Multiplier for convolutional channel width to adjust model capacity.
            dropout_rate (float): Dropout rate applied in fully connected layers.
        """
        super(DeepEmotionNet32, self).__init__()

        # Compute number of filters per layer
        f = filters_multiplier
        init_channels = max(16, int(32 * f))
        channels_1 = init_channels
        channels_2 = max(32, int(64 * f))
        channels_3 = max(64, int(128 * f))

        # Initial convolution block (2 layers)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=channels_1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels_1)

        # Residual block 1
        self.conv2_1 = nn.Conv2d(channels_1, channels_1, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(channels_1)
        self.conv2_2 = nn.Conv2d(channels_1, channels_1, 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(channels_1)

        # Downsample + Residual block 2
        self.conv3_0 = nn.Conv2d(channels_1, channels_2, 1, stride=2)
        self.bn3_0 = nn.BatchNorm2d(channels_2)
        self.conv3_1 = nn.Conv2d(channels_1, channels_2, 3, stride=2, padding=1)
        self.bn3_1 = nn.BatchNorm2d(channels_2)
        self.conv3_2 = nn.Conv2d(channels_2, channels_2, 3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(channels_2)

        # Residual block 3
        self.conv4_1 = nn.Conv2d(channels_2, channels_2, 3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(channels_2)
        self.conv4_2 = nn.Conv2d(channels_2, channels_2, 3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(channels_2)

        # Downsample + Residual block 4
        self.conv5_0 = nn.Conv2d(channels_2, channels_3, 1, stride=2)
        self.bn5_0 = nn.BatchNorm2d(channels_3)
        self.conv5_1 = nn.Conv2d(channels_2, channels_3, 3, stride=2, padding=1)
        self.bn5_1 = nn.BatchNorm2d(channels_3)
        self.conv5_2 = nn.Conv2d(channels_3, channels_3, 3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(channels_3)

        # Final layers
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = nn.Linear(channels_3 * 7 * 7, 1024)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(512, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """He initialization for convolutional and linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))

        # Residual block 1
        identity = x
        out = F.relu(self.bn2_1(self.conv2_1(x)))
        out = self.bn2_2(self.conv2_2(out))
        out += identity
        out = F.relu(out)

        # Downsample + Residual block 2
        identity = self.bn3_0(self.conv3_0(out))
        out = F.relu(self.bn3_1(self.conv3_1(out)))
        out = self.bn3_2(self.conv3_2(out))
        out += identity
        out = F.relu(out)

        # Residual block 3
        identity = out
        out = F.relu(self.bn4_1(self.conv4_1(out)))
        out = self.bn4_2(self.conv4_2(out))
        out += identity
        out = F.relu(out)

        # Downsample + Residual block 4
        identity = self.bn5_0(self.conv5_0(out))
        out = F.relu(self.bn5_1(self.conv5_1(out)))
        out = self.bn5_2(self.conv5_2(out))
        out += identity
        out = F.relu(out)

        # Final fully connected layers
        out = self.adaptive_pool(out)
        out = torch.flatten(out, 1)
        out = F.relu(self.fc1(out))
        out = self.dropout1(out)
        out = F.relu(self.fc2(out))
        out = self.dropout2(out)
        out = self.fc3(out)
        return out
