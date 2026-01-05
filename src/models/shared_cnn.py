"""Shared CNN feature extractor for CarRacing-v3 environment."""

import torch.nn as nn


class SharedCNN(nn.Module):
    """
    Shared CNN feature extractor for 96x96x3 RGB images.

    Based on Nature DQN architecture adapted for CarRacing-v3.
    Takes RGB images and outputs a feature vector that can be used
    by different agent architectures (DQN, PPO, SAC).

    Architecture:
        - Conv2d(3, 32, kernel=8, stride=4) -> ReLU -> (32, 23, 23)
        - Conv2d(32, 64, kernel=4, stride=2) -> ReLU -> (64, 10, 10)
        - Conv2d(64, 64, kernel=3, stride=1) -> ReLU -> (64, 8, 8)
        - Flatten -> 4096 features

    Args:
        input_channels: Number of input channels (3 for RGB, 12 for 4-frame stack)
    """

    def __init__(self, input_channels=3):
        super(SharedCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),  # -> (32, 23, 23)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # -> (64, 10, 10)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # -> (64, 8, 8)
            nn.ReLU(),
        )

        # Calculate feature dimension: 64 * 8 * 8 = 4096
        self.feature_dim = 64 * 8 * 8

    def forward(self, x):
        """
        Forward pass through CNN.

        Args:
            x: Input tensor of shape (batch, channels, 96, 96) with values in [0, 255]

        Returns:
            Feature tensor of shape (batch, 4096)
        """
        # Normalize from [0, 255] to [0, 1]
        x = x.float() / 255.0

        # Extract features
        features = self.conv_layers(x)

        # Flatten
        features = features.view(features.size(0), -1)

        return features
