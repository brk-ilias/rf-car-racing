"""Training script for PPO agent on CarRacing-v3."""

import torch
import numpy as np

from src.envs.wrappers import make_carracing_env
from src.models.shared_cnn import SharedCNN
from src.models.ppo import PPOActorCritic
from src.agents.ppo_agent import PPOAgent
from src.utils.training_monitor import TrainingMonitor
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.logger import Logger
from src.utils.seed import set_seed
from src.utils.load_config import load_config
from src.utils.evaluation import evaluate_agent


def train_ppo(config_path="configs/ppo_config.yaml"):
    """Train PPO agent on CarRacing-v3."""

    # Load configuration
    config = load_config(config_path)

    # Set random seed
    set_seed(config["seed"])

    # Setup device
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create environment
    env = make_carracing_env(
        continuous=config["agent"]["continuous"],
        frame_stack=config["env"]["frame_stack"],
        skip_frames=config["env"]["skip_frames"],
    )

    # Get environment dimensions
    state_shape = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    print(f"State shape: {state_shape}")
    print(f"Action dimension: {action_dim}")

    # Create networks
    shared_cnn = SharedCNN(input_channels=12)
    actor_critic = PPOActorCritic(shared_cnn, action_dim=action_dim)

    # Create agent
    agent = PPOAgent(
        actor_critic=actor_critic,
        state_shape=state_shape,
        action_dim=action_dim,
        lr=config["agent"]["lr"],
        gamma=config["agent"]["gamma"],
        gae_lambda=config["agent"]["gae_lambda"],
        clip_epsilon=config["agent"]["clip_epsilon"],
        value_coef=config["agent"]["value_coef"],
        entropy_coef=config["agent"]["entropy_coef"],
        n_steps=config["agent"]["n_steps"],
        n_epochs=config["agent"]["n_epochs"],
        batch_size=config["agent"]["batch_size"],
        max_grad_norm=config["agent"]["max_grad_norm"],
        device=device,
    )

    # Create training utilities
    monitor = TrainingMonitor(
        max_episodes=config["training"]["max_episodes"],
        target_score=config["training"]["target_score"],
        convergence_episodes=config["training"]["convergence_episodes"],
        convergence_threshold=config["training"]["convergence_threshold"],
    )

    checkpoint_manager = CheckpointManager(
        save_dir=config["checkpoints"]["save_dir"],
        agent_name="ppo",
        save_frequency=config["training"]["save_frequency"],
    )

    logger = Logger(
        log_dir=config["logging"]["log_dir"],
        agent_name="ppo",
        use_tensorboard=config["logging"]["use_tensorboard"],
    )

    # Save configuration
    logger.save_config(config)

    print("\n" + "=" * 50)
    print("Starting PPO Training on CarRacing-v3")
    print("=" * 50 + "\n")

    # Training loop
    episode = 0
    total_steps = 0

    while episode < config["training"]["max_episodes"]:
        episode += 1
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        # Collect rollout
        while not done and len(agent.rollout_buffer) < config["agent"]["n_steps"]:
            # Select action
            action, value, log_prob = agent.select_action(state)

            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition
            agent.rollout_buffer.add(
                state, action, reward, value, log_prob, float(done)
            )

            state = next_state
            episode_reward += reward
            episode_length += 1
            total_steps += 1

            # Update if rollout buffer is full
            if len(agent.rollout_buffer) >= config["agent"]["n_steps"]:
                metrics = agent.update()
                logger.log_training(agent.steps, **metrics)

        # Update monitor
        monitor.update(episode_reward)

        # Log episode
        logger.log_episode(episode=episode, score=episode_reward, length=episode_length)

        # Print progress
        if episode % 10 == 0:
            stats = monitor.get_stats()
            print(
                f"Episode {episode}/{config['training']['max_episodes']} | "
                f"Score: {episode_reward:.2f} | "
                f"Avg Score: {stats.get('recent_mean', 0):.2f} | "
                f"Steps: {total_steps}"
            )

        # Evaluate agent periodically
        if episode % config["training"]["eval_frequency"] == 0:
            eval_mean, eval_std = evaluate_agent(
                agent, {
                    "continuous": config["agent"]["continuous"],
                    "frame_stack": config["env"]["frame_stack"],
                    "skip_frames": config["env"]["skip_frames"]
                }, config["training"]["eval_episodes"]
            )
            logger.log_eval(
                episode,
                eval_mean=eval_mean,
                eval_std=eval_std,
            )
            print(
                f"  Evaluation ({config['training']['eval_episodes']} episodes): "
                f"Mean={eval_mean:.2f}, Std={eval_std:.2f}"
            )

        # Save checkpoint
        checkpoint_manager.save_checkpoint(agent, episode, episode_reward)

        # Check stopping conditions
        should_stop, reason = monitor.should_stop(episode)
        if should_stop:
            print(f"\nTraining stopped: {reason}")
            checkpoint_manager.save_checkpoint(
                agent, episode, episode_reward, is_final=True
            )
            break

    # Save final results
    final_stats = monitor.get_stats()
    logger.save_results(final_stats)
    logger.close()

    print("\n" + "=" * 50)
    print("Training Complete!")
    print(f"Total Episodes: {final_stats['total_episodes']}")
    print(f"Mean Score: {final_stats['mean_score']:.2f}")
    print(f"Best Score: {final_stats['max_score']:.2f}")
    if final_stats["converged_episode"]:
        print(f"Converged at Episode: {final_stats['converged_episode']}")
    print("=" * 50 + "\n")

    env.close()


if __name__ == "__main__":
    train_ppo()
