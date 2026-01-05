"""SAC networks for continuous action spaces."""

import torch
import torch.nn as nn

class SACPolicy(nn.Module):
    """
    Stochastic policy network for SAC.

    Outputs mean and log_std for a Gaussian policy over continuous actions.
    Uses the reparameterization trick for backpropagation.

    Args:
        shared_cnn: Shared CNN feature extractor
        action_dim: Dimension of continuous action space (default: 3)
    """

    def __init__(self, shared_cnn, action_dim=3):
        super(SACPolicy, self).__init__()

        self.cnn = shared_cnn
        self.action_dim = action_dim

        # Policy network
        self.fc = nn.Sequential(
            nn.Linear(shared_cnn.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.mean_linear = nn.Linear(256, action_dim)
        self.log_std_linear = nn.Linear(256, action_dim)

        # Action rescaling constants (for tanh squashing)
        self.action_scale = 1.0
        self.action_bias = 0.0

    def forward(self, x):
        """
        Forward pass to get action distribution.

        Args:
            x: Input tensor of shape (batch, channels, 96, 96)

        Returns:
            Tuple of (mean, log_std)
        """
        features = self.cnn(x)
        fc_out = self.fc(features)

        mean = self.mean_linear(fc_out)
        log_std = self.log_std_linear(fc_out)
        log_std = torch.clamp(log_std, -20, 2)  # Prevent extreme values

        return mean, log_std

    def sample(self, x):
        """
        Sample action using reparameterization trick.

        Args:
            x: Input state

        Returns:
            Tuple of (action, log_prob, mean)
        """
        mean, log_std = self.forward(x)
        std = log_std.exp()

        # Reparameterization trick
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Sample with reparameterization
        y_t = torch.tanh(x_t)  # Squash to [-1, 1]

        action = y_t * self.action_scale + self.action_bias

        # Compute log probability with change of variables for tanh
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean


class SACQNetwork(nn.Module):
    """
    Q-network for SAC (Critic).

    Takes state and action as input and outputs Q-value.
    SAC uses twin Q-networks to mitigate overestimation bias.

    Args:
        shared_cnn: Shared CNN feature extractor
        action_dim: Dimension of continuous action space (default: 3)
    """

    def __init__(self, shared_cnn, action_dim=3):
        super(SACQNetwork, self).__init__()

        self.cnn = shared_cnn
        self.action_dim = action_dim

        # Q-network that takes concatenated state features and action
        self.fc = nn.Sequential(
            nn.Linear(shared_cnn.feature_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x, action):
        """
        Forward pass to compute Q-value.

        Args:
            x: State tensor of shape (batch, channels, 96, 96)
            action: Action tensor of shape (batch, action_dim)

        Returns:
            Q-value tensor of shape (batch, 1)
        """
        features = self.cnn(x)
        q_input = torch.cat([features, action], dim=1)
        q_value = self.fc(q_input)
        return q_value
