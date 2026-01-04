"""DQN Agent implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Deep Q-Network agent for discrete action spaces.

    Implements:
    - Experience replay
    - Target network with periodic updates
    - Epsilon-greedy exploration
    - Huber loss for stability

    Args:
        q_network: Q-network (DQNNetwork instance)
        state_shape: Shape of state observations
        n_actions: Number of discrete actions
        lr: Learning rate
        gamma: Discount factor
        epsilon_start: Initial exploration rate
        epsilon_end: Final exploration rate
        epsilon_decay: Number of steps over which to decay epsilon
        buffer_size: Replay buffer capacity
        batch_size: Mini-batch size for training
        target_update_freq: Frequency of target network updates (steps)
        device: PyTorch device
    """

    def __init__(
        self,
        q_network,
        state_shape,
        n_actions=5,
        lr=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=10000,
        buffer_size=100000,
        batch_size=32,
        target_update_freq=1000,
        device="cuda",
    ):
        self.device = device
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Networks
        self.q_network = q_network.to(device)
        self.target_network = type(q_network)(q_network.cnn, n_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, state_shape, 1, device)

        # Exploration
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps = 0

    def get_epsilon(self):
        """Compute current epsilon value with linear decay."""
        epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (
            self.steps / self.epsilon_decay
        )
        return max(epsilon, self.epsilon_end)

    def select_action(self, state, evaluate=False):
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state observation
            evaluate: If True, always select greedy action (no exploration)

        Returns:
            Selected action (int)
        """
        if not evaluate and np.random.rand() < self.get_epsilon():
            return np.random.randint(self.n_actions)

        with torch.no_grad():
            state_tensor = torch.as_tensor(state, device=self.device).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            action = q_values.argmax(dim=1).item()

        return action

    def update(self):
        """
        Perform one gradient descent step on a mini-batch.

        Returns:
            Training loss value
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample mini-batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Compute current Q-values
        q_values = self.q_network(states).gather(1, actions)

        # Compute target Q-values using target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1, keepdim=True)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss (Huber loss for stability)
        loss = F.smooth_l1_loss(q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def save(self, path):
        """Save agent state."""
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps": self.steps,
            },
            path,
        )

    def load(self, path):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps = checkpoint["steps"]
