"""Replay buffer for off-policy algorithms DQN."""

import numpy as np
import torch


class ReplayBuffer:
    """
    Experience replay buffer for off-policy RL algorithms.

    Stores transitions (state, action, reward, next_state, done) and
    provides random sampling for training.

    Args:
        capacity: Maximum number of transitions to store
        state_shape: Shape of state observations
        action_shape: Shape of actions (scalar for discrete, tuple for continuous)
        device: PyTorch device for tensor operations
    """

    def __init__(self, capacity, state_shape, action_shape, device):
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0

        # Preallocate memory
        self.states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        # Action shape depends on discrete vs continuous
        if isinstance(action_shape, int):
            self.actions = np.zeros((capacity, 1), dtype=np.int64)
        else:
            self.actions = np.zeros((capacity, *action_shape), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        """Add a transition to the buffer."""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """
        Sample a batch of transitions randomly.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as PyTorch tensors
        """
        indices = np.random.randint(0, self.size, size=batch_size)

        states = torch.as_tensor(self.states[indices], device=self.device)
        actions = torch.as_tensor(self.actions[indices], device=self.device)
        rewards = torch.as_tensor(self.rewards[indices], device=self.device)
        next_states = torch.as_tensor(self.next_states[indices], device=self.device)
        dones = torch.as_tensor(self.dones[indices], device=self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return current size of the buffer."""
        return self.size
