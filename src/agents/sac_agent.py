"""SAC Agent implementation."""

import torch
import torch.nn.functional as F
from .replay_buffer import ReplayBuffer


class SACAgent:
    """
    Soft Actor-Critic agent for continuous action spaces.

    Implements:
    - Twin Q-networks to mitigate overestimation
    - Stochastic policy with reparameterization trick
    - Automatic entropy temperature tuning
    - Soft target updates

    Args:
        policy: Policy network (SACPolicy instance)
        q_network1: First Q-network (SACQNetwork instance)
        q_network2: Second Q-network (SACQNetwork instance)
        state_shape: Shape of state observations
        action_dim: Dimension of continuous action space
        lr: Learning rate
        gamma: Discount factor
        tau: Soft update coefficient
        alpha: Entropy temperature (if None, use automatic tuning)
        target_entropy: Target entropy for auto-tuning (if None, use -action_dim)
        buffer_size: Replay buffer capacity
        batch_size: Mini-batch size
        device: PyTorch device
    """

    def __init__(
        self,
        policy,
        q_network1,
        q_network2,
        state_shape,
        action_dim=3,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=None,
        target_entropy=None,
        buffer_size=100000,
        batch_size=256,
        device="cuda",
    ):
        self.device = device
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Policy network
        self.policy = policy.to(device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # Twin Q-networks
        self.q_network1 = q_network1.to(device)
        self.q_network2 = q_network2.to(device)
        self.q_optimizer = torch.optim.Adam(
            list(self.q_network1.parameters()) + list(self.q_network2.parameters()),
            lr=lr,
        )

        # Target Q-networks
        self.target_q_network1 = type(q_network1)(q_network1.cnn, action_dim).to(device)
        self.target_q_network2 = type(q_network2)(q_network2.cnn, action_dim).to(device)
        self.target_q_network1.load_state_dict(self.q_network1.state_dict())
        self.target_q_network2.load_state_dict(self.q_network2.state_dict())
        self.target_q_network1.eval()
        self.target_q_network2.eval()

        # Automatic entropy tuning
        self.use_automatic_entropy_tuning = alpha is None
        if self.use_automatic_entropy_tuning:
            self.target_entropy = (
                -action_dim if target_entropy is None else target_entropy
            )
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = alpha

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            buffer_size, state_shape, (action_dim,), device
        )

        self.steps = 0

    def select_action(self, state, evaluate=False):
        """
        Select action from policy.

        Args:
            state: Current state observation
            evaluate: If True, use mean action (deterministic)

        Returns:
            Selected action
        """
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, device=self.device).unsqueeze(0)

            if evaluate:
                _, _, action = self.policy.sample(state_tensor)
            else:
                action, _, _ = self.policy.sample(state_tensor)

        return action.cpu().numpy()[0]

    def update(self):
        """
        Perform one SAC update step.

        Returns:
            Dictionary of training metrics
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample mini-batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Update Q-functions
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.policy.sample(next_states)
            target_q1 = self.target_q_network1(next_states, next_actions)
            target_q2 = self.target_q_network2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + self.gamma * (1 - dones) * target_q

        q1 = self.q_network1(states, actions)
        q2 = self.q_network2(states, actions)
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)
        q_loss = q1_loss + q2_loss

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update policy
        new_actions, log_probs, _ = self.policy.sample(states)
        q1_new = self.q_network1(states, new_actions)
        q2_new = self.q_network2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)

        policy_loss = (self.alpha * log_probs - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update temperature (alpha)
        alpha_loss = None
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_probs + self.target_entropy).detach()
            ).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()

        # Soft update target networks
        self._soft_update(self.q_network1, self.target_q_network1)
        self._soft_update(self.q_network2, self.target_q_network2)

        self.steps += 1

        metrics = {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha": (
                self.alpha.item()
                if isinstance(self.alpha, torch.Tensor)
                else self.alpha
            ),
        }

        if alpha_loss is not None:
            metrics["alpha_loss"] = alpha_loss.item()

        return metrics

    def _soft_update(self, source, target):
        """Soft update of target network parameters."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def save(self, path):
        """Save agent state."""
        checkpoint = {
            "policy": self.policy.state_dict(),
            "q_network1": self.q_network1.state_dict(),
            "q_network2": self.q_network2.state_dict(),
            "target_q_network1": self.target_q_network1.state_dict(),
            "target_q_network2": self.target_q_network2.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "steps": self.steps,
        }

        if self.use_automatic_entropy_tuning:
            checkpoint["log_alpha"] = self.log_alpha
            checkpoint["alpha_optimizer"] = self.alpha_optimizer.state_dict()

        torch.save(checkpoint, path)

    def load(self, path):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.q_network1.load_state_dict(checkpoint["q_network1"])
        self.q_network2.load_state_dict(checkpoint["q_network2"])
        self.target_q_network1.load_state_dict(checkpoint["target_q_network1"])
        self.target_q_network2.load_state_dict(checkpoint["target_q_network2"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
        self.steps = checkpoint["steps"]

        if self.use_automatic_entropy_tuning and "log_alpha" in checkpoint:
            self.log_alpha = checkpoint["log_alpha"]
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
            self.alpha = self.log_alpha.exp()
