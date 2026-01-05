"""PPO Actor-Critic network for continuous action spaces."""

import torch
import torch.nn as nn


class PPOActorCritic(nn.Module):
    """
    Actor-Critic network for PPO with continuous actions.

    Uses a shared CNN feature extractor with separate heads for:
    - Actor (policy): outputs mean and log_std for Gaussian policy
    - Critic (value function): outputs state value V(s)

    Args:
        shared_cnn: Shared CNN feature extractor
        action_dim: Dimension of continuous action space (default: 3)
    """

    def __init__(self, shared_cnn, action_dim=3):
        super(PPOActorCritic, self).__init__()

        self.cnn = shared_cnn
        self.action_dim = action_dim

        # Actor network (policy)
        self.actor_fc = nn.Sequential(
            nn.Linear(shared_cnn.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.actor_mean = nn.Linear(256, action_dim)
        # Log std as learnable parameter
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))

        # Critic network (value function)
        self.critic_fc = nn.Sequential(
            nn.Linear(shared_cnn.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        """
        Forward pass through actor-critic.

        Args:
            x: Input tensor of shape (batch, channels, 96, 96)

        Returns:
            Tuple of (mean, std, value):
                - mean: Action means of shape (batch, action_dim) in [-1, 1]
                - std: Action standard deviations of shape (action_dim,)
                - value: State values of shape (batch, 1)
        """
        features = self.cnn(x)

        # Actor: compute mean and std
        actor_features = self.actor_fc(features)
        mean = torch.tanh(self.actor_mean(actor_features))  # Scale to [-1, 1]
        std = torch.exp(self.actor_logstd.clamp(-20, 2))  # Ensure positive std

        # Critic: compute value
        value = self.critic_fc(features)

        return mean, std, value

    def get_value(self, x):
        """Get state value only (for efficiency during rollout)."""
        features = self.cnn(x)
        value = self.critic_fc(features)
        return value

    def evaluate_actions(self, x, actions):
        """
        Evaluate actions for policy update.

        Args:
            x: State tensor
            actions: Actions taken

        Returns:
            Tuple of (log_probs, entropy, values)
        """
        mean, std, values = self.forward(x)

        # Create normal distribution
        dist = torch.distributions.Normal(mean, std)

        # Calculate log probabilities
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)

        # Calculate entropy
        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        return log_probs, entropy, values
