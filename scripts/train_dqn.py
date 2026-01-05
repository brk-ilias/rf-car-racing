"""Training script for DQN agent on CarRacing-v3."""

import torch
import numpy as np

from src.envs.wrappers import make_carracing_env
from src.models.shared_cnn import SharedCNN
from src.models.dqn import DQNNetwork
from src.agents.dqn_agent import DQNAgent
from src.utils.training_monitor import TrainingMonitor
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.logger import Logger
from src.utils.seed import set_seed
from src.utils.load_config import load_config


def train_dqn(config_path="configs/dqn_config.yaml"):
    """Train DQN agent on CarRacing-v3."""

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
    n_actions = env.action_space.n
    print(f"State shape: {state_shape}")
    print(f"Number of actions: {n_actions}")

    # Create networks
    shared_cnn = SharedCNN(input_channels=12)
    q_network = DQNNetwork(shared_cnn, n_actions=n_actions)

    # Create agent
    agent = DQNAgent(
        q_network=q_network,
        state_shape=state_shape,
        n_actions=n_actions,
        lr=config["agent"]["lr"],
        gamma=config["agent"]["gamma"],
        epsilon_start=config["agent"]["epsilon_start"],
        epsilon_end=config["agent"]["epsilon_end"],
        epsilon_decay=config["agent"]["epsilon_decay"],
        buffer_size=config["agent"]["buffer_size"],
        batch_size=config["agent"]["batch_size"],
        target_update_freq=config["agent"]["target_update_freq"],
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
        agent_name="dqn",
        save_frequency=config["training"]["save_frequency"],
    )

    logger = Logger(
        log_dir=config["logging"]["log_dir"],
        agent_name="dqn",
        use_tensorboard=config["logging"]["use_tensorboard"],
    )

    # Save configuration
    logger.save_config(config)

    print("\n" + "=" * 50)
    print("Starting DQN Training on CarRacing-v3")
    print("=" * 50 + "\n")

    # Training loop
    episode = 0
    total_steps = 0

    while episode < config["training"]["max_episodes"]:
        episode += 1
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_losses = []
        done = False

        while not done:
            # Select action
            action = agent.select_action(state)

            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition
            agent.replay_buffer.add(state, action, reward, next_state, float(done))

            # Update agent
            if len(agent.replay_buffer) >= config["agent"]["learning_starts"]:
                loss = agent.update()
                if loss is not None:
                    episode_losses.append(loss)

            state = next_state
            episode_reward += reward
            episode_length += 1
            total_steps += 1

        # Update monitor
        monitor.update(episode_reward)

        # Log episode
        avg_loss = np.mean(episode_losses) if episode_losses else None
        logger.log_episode(
            episode=episode,
            score=episode_reward,
            length=episode_length,
            epsilon=agent.get_epsilon(),
            loss=avg_loss,
        )

        # Print progress
        if episode % 10 == 0:
            stats = monitor.get_stats()
            print(
                f"Episode {episode}/{config['training']['max_episodes']} | "
                f"Score: {episode_reward:.2f} | "
                f"Avg Score: {stats.get('recent_mean', 0):.2f} | "
                f"Epsilon: {agent.get_epsilon():.3f} | "
                f"Steps: {total_steps}"
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
    train_dqn()
