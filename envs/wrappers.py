"""Environment wrappers for CarRacing-v3."""

import gymnasium as gym
import numpy as np
from collections import deque


class CarRacingWrapper(gym.Wrapper):
    """
    Wrapper for CarRacing-v3 environment with preprocessing.

    Features:
    - Dual action space support (continuous or discrete)
    - Frame stacking for temporal information
    - Frame skipping for efficiency
    - Reward shaping

    Args:
        env: CarRacing-v3 environment
        continuous: If True, use continuous actions [steering, gas, brake].
                   If False, use discrete actions (5 actions)
        frame_stack: Number of frames to stack (default: 4)
        skip_frames: Number of frames to skip/repeat action (default: 2)
    """

    def __init__(self, env, continuous=True, frame_stack=4, skip_frames=2):
        super(CarRacingWrapper, self).__init__(env)

        self.continuous = continuous
        self.frame_stack = frame_stack
        self.skip_frames = skip_frames

        # Frame buffer for stacking
        self.frames = deque(maxlen=frame_stack)

        # Action space configuration
        if not continuous:
            # Override action space for discrete mode
            self.action_space = gym.spaces.Discrete(5)
            # Discrete action mapping:
            # 0: do nothing, 1: steer right, 2: steer left, 3: gas, 4: brake
            self.action_map = {
                0: np.array([0.0, 0.0, 0.0]),
                1: np.array([0.6, 0.0, 0.0]),  # Moderate right turn
                2: np.array([-0.6, 0.0, 0.0]),  # Moderate left turn
                3: np.array([0.0, 1.0, 0.0]),  # Full gas
                4: np.array([0.0, 0.0, 0.8]),  # Strong brake
            }

        # Observation space: stacked frames
        original_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(
                frame_stack * original_shape[2],
                original_shape[0],
                original_shape[1],
            ),
            dtype=np.uint8,
        )

    def reset(self, **kwargs):
        """Reset environment and initialize frame stack."""
        obs, info = self.env.reset(**kwargs)

        # Initialize frame stack with first observation
        for _ in range(self.frame_stack):
            self.frames.append(obs)

        return self._get_observation(), info

    def step(self, action):
        """
        Execute action with frame skipping and return stacked observation.

        Args:
            action: Discrete action index (if discrete mode) or continuous action array

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Convert discrete action to continuous if needed
        if not self.continuous:
            action = self.action_map[action]

        # Execute action for skip_frames steps
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for _ in range(self.skip_frames):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            self.frames.append(obs)

            if terminated or truncated:
                break

        return self._get_observation(), total_reward, terminated, truncated, info

    def _get_observation(self):
        """Stack frames along channel dimension."""
        # Stack frames: (4, 96, 96, 3) -> (12, 96, 96)
        stacked = np.concatenate(list(self.frames), axis=-1)  # (96, 96, 12)
        stacked = np.transpose(stacked, (2, 0, 1))  # (12, 96, 96)
        return stacked


def make_carracing_env(continuous=True, frame_stack=4, skip_frames=2, render_mode=None):
    """
    Factory function to create wrapped CarRacing environment.

    Args:
        continuous: Use continuous (True) or discrete (False) actions
        frame_stack: Number of frames to stack
        skip_frames: Number of frames to skip per action
        render_mode: Rendering mode for the environment

    Returns:
        Wrapped CarRacing environment
    """
    env = gym.make("CarRacing-v3", render_mode=render_mode, continuous=continuous)
    env = CarRacingWrapper(
        env, continuous=continuous, frame_stack=frame_stack, skip_frames=skip_frames
    )
    return env
