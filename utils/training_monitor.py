"""Training monitor with early stopping logic."""

import numpy as np


class TrainingMonitor:
    """
    Monitor training progress and implement early stopping.

    Tracks episode scores and checks for convergence based on:
    - Maximum episode budget
    - Score threshold achievement
    - Score stability (low variance)

    Args:
        max_episodes: Maximum number of training episodes
        target_score: Score threshold for convergence consideration
        window_size: Moving average window size
        convergence_episodes: Number of recent episodes to check for convergence
        convergence_threshold: Maximum std deviation to consider converged
    """

    def __init__(
        self,
        max_episodes=1000,
        target_score=700,
        window_size=100,
        convergence_episodes=100,
        convergence_threshold=10,
    ):
        self.max_episodes = max_episodes
        self.target_score = target_score
        self.window_size = window_size
        self.convergence_episodes = convergence_episodes
        self.convergence_threshold = convergence_threshold

        self.episode_scores = []
        self.moving_averages = []
        self.converged_episode = None

    def update(self, score):
        """
        Update monitor with new episode score.

        Args:
            score: Episode reward/score
        """
        self.episode_scores.append(score)

        # Calculate moving average
        if len(self.episode_scores) >= self.window_size:
            ma = np.mean(self.episode_scores[-self.window_size :])
            self.moving_averages.append(ma)

    def check_convergence(self):
        """
        Check if training has converged.

        Returns:
            Tuple of (converged: bool, reason: str, metrics: dict)
        """
        if len(self.episode_scores) < self.convergence_episodes:
            return False, "insufficient_episodes", {}

        # Get recent scores
        recent_scores = self.episode_scores[-self.convergence_episodes :]
        recent_mean = np.mean(recent_scores)
        recent_std = np.std(recent_scores)

        metrics = {
            "recent_mean": recent_mean,
            "recent_std": recent_std,
            "episodes": len(self.episode_scores),
        }

        # Check convergence criteria
        if (
            recent_mean >= self.target_score
            and recent_std <= self.convergence_threshold
        ):
            if self.converged_episode is None:
                self.converged_episode = len(self.episode_scores)
            return (
                True,
                f"converged (mean={recent_mean:.1f}, std={recent_std:.1f})",
                metrics,
            )

        return (
            False,
            f"not_converged (mean={recent_mean:.1f}, std={recent_std:.1f})",
            metrics,
        )

    def should_stop(self, episode):
        """
        Check if training should stop.

        Args:
            episode: Current episode number (1-indexed)

        Returns:
            Tuple of (should_stop: bool, reason: str)
        """
        # Check max episodes
        if episode >= self.max_episodes:
            return True, "max_episodes_reached"

        # Check convergence
        converged, reason, _ = self.check_convergence()
        if converged:
            return True, reason

        return False, "continue"

    def get_stats(self):
        """
        Get training statistics.

        Returns:
            Dictionary of statistics
        """
        if not self.episode_scores:
            return {}

        stats = {
            "total_episodes": len(self.episode_scores),
            "mean_score": np.mean(self.episode_scores),
            "std_score": np.std(self.episode_scores),
            "min_score": np.min(self.episode_scores),
            "max_score": np.max(self.episode_scores),
            "converged_episode": self.converged_episode,
        }

        if len(self.episode_scores) >= self.window_size:
            stats["recent_mean"] = np.mean(self.episode_scores[-self.window_size :])
            stats["recent_std"] = np.std(self.episode_scores[-self.window_size :])

        return stats
