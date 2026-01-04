"""Checkpoint manager for saving and loading models."""

import os
from pathlib import Path
import torch


class CheckpointManager:
    """
    Manage model checkpoints during training.

    Saves checkpoints at different stages:
    - Latest: Most recent model
    - Best: Model with highest score
    - Periodic: Every N episodes
    - Final: At training completion

    Args:
        save_dir: Directory to save checkpoints
        agent_name: Name of the agent (for subdirectory)
        save_frequency: Episode frequency for periodic saves
    """

    def __init__(self, save_dir, agent_name, save_frequency=100):
        self.save_dir = Path(save_dir) / agent_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_frequency = save_frequency
        self.best_score = float("-inf")
        self.best_episode = 0

    def save_checkpoint(self, agent, episode, score, is_final=False):
        """
        Save model checkpoint.

        Args:
            agent: Agent object with save() method
            episode: Current episode number
            score: Episode score
            is_final: Whether this is the final checkpoint
        """
        # Always save latest
        latest_path = self.save_dir / "latest.pth"
        agent.save(latest_path)

        # Save best model
        if score > self.best_score:
            self.best_score = score
            self.best_episode = episode
            best_path = self.save_dir / "best.pth"
            agent.save(best_path)

            # Save metadata
            metadata_path = self.save_dir / "best_metadata.txt"
            with open(metadata_path, "w") as f:
                f.write(f"Episode: {episode}\n")
                f.write(f"Score: {score:.2f}\n")

        # Save periodic checkpoints
        if episode % self.save_frequency == 0:
            periodic_path = self.save_dir / f"episode_{episode}.pth"
            agent.save(periodic_path)

        # Save final model
        if is_final:
            final_path = self.save_dir / "final.pth"
            agent.save(final_path)

    def load_checkpoint(self, agent, checkpoint_name="best"):
        """
        Load model checkpoint.

        Args:
            agent: Agent object with load() method
            checkpoint_name: Name of checkpoint to load ('best', 'latest', 'final')

        Returns:
            True if successful, False otherwise
        """
        checkpoint_path = self.save_dir / f"{checkpoint_name}.pth"

        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            return False

        try:
            agent.load(checkpoint_path)
            print(f"Loaded checkpoint: {checkpoint_path}")
            return True
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False

    def get_best_score(self):
        """Get the best score achieved during training."""
        return self.best_score, self.best_episode

    def list_checkpoints(self):
        """List all available checkpoints."""
        checkpoints = list(self.save_dir.glob("*.pth"))
        return [cp.name for cp in checkpoints]
