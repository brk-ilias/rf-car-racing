"""Utility functions for evaluating reinforcement learning agents."""

import numpy as np
from src.envs.wrappers import make_carracing_env


def evaluate_agent(agent, env_config, n_episodes=5):
    """Evaluate agent with deterministic policy."""
    eval_env = make_carracing_env(
        continuous=env_config["continuous"],
        frame_stack=env_config["frame_stack"],
        skip_frames=env_config["skip_frames"],
    )

    eval_scores = []
    for _ in range(n_episodes):
        state, _ = eval_env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward

        eval_scores.append(episode_reward)

    eval_env.close()
    return np.mean(eval_scores), np.std(eval_scores)
