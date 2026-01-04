"""DQN network for discrete action spaces."""

import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for discrete action space (5 actions).

    Uses a shared CNN feature extractor followed by fully connected
    layers to output Q-values for each discrete action.

    Args:
        shared_cnn: Shared CNN feature extractor
        n_actions: Number of discrete actions (default: 5)
    """

    def __init__(self, shared_cnn, n_actions=5):
        super(DQNNetwork, self).__init__()

        self.cnn = shared_cnn
        self.n_actions = n_actions

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(shared_cnn.feature_dim, 512), nn.ReLU(), nn.Linear(512, n_actions)
        )

    def forward(self, x):
        """
        Forward pass to compute Q-values.

        Args:
            x: Input tensor of shape (batch, channels, 96, 96)

        Returns:
            Q-values tensor of shape (batch, n_actions)
        """
        features = self.cnn(x)
        q_values = self.fc(features)
        return q_values
