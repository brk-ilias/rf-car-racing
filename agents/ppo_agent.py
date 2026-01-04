"""PPO Agent implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .rollout_buffer import RolloutBuffer


class PPOAgent:
    """
    Proximal Policy Optimization agent for continuous action spaces.

    Implements:
    - Clipped surrogate objective
    - Generalized Advantage Estimation (GAE)
    - Multiple epochs of mini-batch updates
    - Value function clipping
    - Entropy regularization

    Args:
        actor_critic: Actor-Critic network (PPOActorCritic instance)
        state_shape: Shape of state observations
        action_dim: Dimension of continuous action space
        lr: Learning rate
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_epsilon: PPO clipping parameter
        value_coef: Value loss coefficient
        entropy_coef: Entropy bonus coefficient
        n_steps: Number of steps per rollout
        n_epochs: Number of epochs per update
        batch_size: Mini-batch size
        max_grad_norm: Max gradient norm for clipping
        device: PyTorch device
    """

    def __init__(
        self,
        actor_critic,
        state_shape,
        action_dim=3,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        n_steps=2048,
        n_epochs=10,
        batch_size=64,
        max_grad_norm=0.5,
        device="cuda",
    ):
        self.device = device
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm

        # Network
        self.actor_critic = actor_critic.to(device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr)

        # Rollout buffer
        self.rollout_buffer = RolloutBuffer(
            n_steps, state_shape, (action_dim,), device, gamma, gae_lambda
        )

        self.steps = 0

    def select_action(self, state, evaluate=False):
        """
        Select action from policy.

        Args:
            state: Current state observation
            evaluate: If True, use mean action (no sampling)

        Returns:
            Tuple of (action, value, log_prob)
        """
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, device=self.device).unsqueeze(0)
            mean, std, value = self.actor_critic(state_tensor)

            if evaluate:
                action = mean
            else:
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()

            log_prob = (
                torch.distributions.Normal(mean, std).log_prob(action).sum(dim=-1)
            )

        return (
            action.cpu().numpy()[0],
            value.cpu().numpy()[0],
            log_prob.cpu().numpy()[0],
        )

    def update(self):
        """
        Perform PPO update using collected rollout data.

        Returns:
            Dictionary of training metrics
        """
        # Compute returns and advantages
        with torch.no_grad():
            last_state = torch.as_tensor(
                self.rollout_buffer.states[self.rollout_buffer.position - 1],
                device=self.device,
            ).unsqueeze(0)
            last_value = self.actor_critic.get_value(last_state).cpu().numpy()[0, 0]

        self.rollout_buffer.compute_returns_and_advantages(last_value)

        # Get rollout data
        states, actions, old_log_probs, returns, advantages = self.rollout_buffer.get()

        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        # Multiple epochs of updates
        for epoch in range(self.n_epochs):
            # Create random indices for mini-batches
            indices = np.arange(len(states))
            np.random.shuffle(indices)

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Evaluate actions
                log_probs, entropy, values = self.actor_critic.evaluate_actions(
                    batch_states, batch_actions
                )

                # Policy loss with clipping
                ratio = torch.exp(log_probs - batch_old_log_probs.unsqueeze(1))
                surr1 = ratio * batch_advantages.unsqueeze(1)
                surr2 = torch.clamp(
                    ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                ) * batch_advantages.unsqueeze(1)
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, batch_returns.unsqueeze(1))

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        # Clear rollout buffer
        self.rollout_buffer.reset()
        self.steps += 1

        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }

    def save(self, path):
        """Save agent state."""
        torch.save(
            {
                "actor_critic": self.actor_critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps": self.steps,
            },
            path,
        )

    def load(self, path):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint["actor_critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps = checkpoint["steps"]
