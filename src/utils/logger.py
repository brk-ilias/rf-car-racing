"""Logging utilities for training metrics."""

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import json


class Logger:
    """
    Logger for training metrics with TensorBoard support.

    Args:
        log_dir: Directory for logs
        agent_name: Name of the agent
        use_tensorboard: Whether to use TensorBoard
    """

    def __init__(self, log_dir, agent_name, use_tensorboard=True):
        self.log_dir = Path(log_dir) / agent_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.agent_name = agent_name

        # TensorBoard writer
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # Metrics storage
        self.metrics = {"episodes": [], "scores": [], "lengths": [], "losses": []}

    def log_episode(self, episode, score, length, **kwargs):
        """
        Log episode metrics.

        Args:
            episode: Episode number
            score: Episode total reward
            length: Episode length (steps)
            **kwargs: Additional metrics to log
        """
        self.metrics["episodes"].append(episode)
        self.metrics["scores"].append(score)
        self.metrics["lengths"].append(length)

        if self.use_tensorboard:
            self.writer.add_scalar("Episode/Score", score, episode)
            self.writer.add_scalar("Episode/Length", length, episode)

            for key, value in kwargs.items():
                if value is not None:
                    self.writer.add_scalar(f"Episode/{key}", value, episode)

    def log_training(self, step, **kwargs):
        """
        Log training metrics.

        Args:
            step: Training step number
            **kwargs: Metrics to log (e.g., loss, q_value, etc.)
        """
        if self.use_tensorboard:
            for key, value in kwargs.items():
                if value is not None:
                    self.writer.add_scalar(f"Training/{key}", value, step)

    def log_eval(self, episode, eval_mean, eval_std):
        """
        Log evaluation metrics.

        Args:
            episode: Episode number when evaluation was performed
            eval_mean: Mean score across evaluation episodes
            eval_std: Standard deviation of scores across evaluation episodes
        """
        if self.use_tensorboard:
            self.writer.add_scalar("Evaluation/Mean_Score", eval_mean, episode)
            self.writer.add_scalar("Evaluation/Std_Score", eval_std, episode)

    def save_config(self, config):
        """
        Save configuration to JSON file.

        Args:
            config: Dictionary of configuration parameters
        """
        config_path = self.log_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

    def save_results(self, results):
        """
        Save final results to JSON file.

        Args:
            results: Dictionary of final results
        """
        results_path = self.log_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)

    def close(self):
        """Close the logger."""
        if self.use_tensorboard:
            self.writer.close()
