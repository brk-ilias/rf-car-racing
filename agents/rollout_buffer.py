"""Rollout buffer for on-policy algorithms (PPO)."""

import numpy as np
import torch


class RolloutBuffer:
    """
    Rollout buffer for on-policy RL algorithms like PPO.

    Stores complete trajectories with advantages computed using GAE.
    Buffer is cleared after each policy update.

    Args:
        capacity: Maximum number of transitions per rollout
        state_shape: Shape of state observations
        action_shape: Shape of actions
        device: PyTorch device for tensor operations
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
    """

    def __init__(
        self, capacity, state_shape, action_shape, device, gamma=0.99, gae_lambda=0.95
    ):
        self.capacity = capacity
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.position = 0

        # Preallocate memory
        self.states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.values = np.zeros((capacity,), dtype=np.float32)
        self.log_probs = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

        # Computed during finalize
        self.advantages = np.zeros((capacity,), dtype=np.float32)
        self.returns = np.zeros((capacity,), dtype=np.float32)

    def add(self, state, action, reward, value, log_prob, done):
        """Add a transition to the buffer."""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        # Handle both scalar and array/tensor values
        self.values[self.position] = (
            value.item() if hasattr(value, "item") else float(value)
        )
        self.log_probs[self.position] = (
            log_prob.item() if hasattr(log_prob, "item") else float(log_prob)
        )
        self.dones[self.position] = done

        self.position += 1

    def compute_returns_and_advantages(self, last_value):
        """
        Compute returns and advantages using GAE.

        Args:
            last_value: Value estimate for the state after the last transition
        """
        last_gae_lambda = 0

        for step in reversed(range(self.position)):
            if step == self.position - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = self.values[step + 1]

            delta = (
                self.rewards[step]
                + self.gamma * next_value * next_non_terminal
                - self.values[step]
            )
            last_gae_lambda = (
                delta
                + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lambda
            )
            self.advantages[step] = last_gae_lambda

        self.returns = self.advantages + self.values[: self.position]

    def get(self):
        """
        Get all data in the buffer as PyTorch tensors.

        Returns:
            Tuple of (states, actions, old_log_probs, returns, advantages)
        """
        # Normalize advantages
        advantages = self.advantages[: self.position]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = torch.as_tensor(self.states[: self.position], device=self.device)
        actions = torch.as_tensor(self.actions[: self.position], device=self.device)
        old_log_probs = torch.as_tensor(
            self.log_probs[: self.position], device=self.device
        )
        returns = torch.as_tensor(self.returns[: self.position], device=self.device)
        advantages = torch.as_tensor(advantages, device=self.device)

        return states, actions, old_log_probs, returns, advantages

    def reset(self):
        """Clear the buffer."""
        self.position = 0

    def __len__(self):
        """Return current size of the buffer."""
        return self.position
