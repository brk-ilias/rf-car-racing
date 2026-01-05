"""Training script for SAC agent on CarRacing-v3."""

import torch

from envs.wrappers import make_carracing_env
from models.shared_cnn import SharedCNN
from models.sac import SACPolicy, SACQNetwork
from agents.sac_agent import SACAgent
from utils.training_monitor import TrainingMonitor
from utils.checkpoint_manager import CheckpointManager
from utils.logger import Logger
from utils.seed import set_seed
from utils.load_config import load_config


def train_sac(config_path="configs/sac_config.yaml"):
    """Train SAC agent on CarRacing-v3."""

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

    # Create networks (separate CNNs for each network)
    policy_cnn = SharedCNN(input_channels=12)
    q1_cnn = SharedCNN(input_channels=12)
    q2_cnn = SharedCNN(input_channels=12)

    policy = SACPolicy(policy_cnn, action_dim=action_dim)
    q_network1 = SACQNetwork(q1_cnn, action_dim=action_dim)
    q_network2 = SACQNetwork(q2_cnn, action_dim=action_dim)

    # Create agent
    agent = SACAgent(
        policy=policy,
        q_network1=q_network1,
        q_network2=q_network2,
        state_shape=state_shape,
        action_dim=action_dim,
        lr=config["agent"]["lr"],
        gamma=config["agent"]["gamma"],
        tau=config["agent"]["tau"],
        alpha=config["agent"]["alpha"],
        target_entropy=config["agent"]["target_entropy"],
        buffer_size=config["agent"]["buffer_size"],
        batch_size=config["agent"]["batch_size"],
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
        agent_name="sac",
        save_frequency=config["training"]["save_frequency"],
    )

    logger = Logger(
        log_dir=config["logging"]["log_dir"],
        agent_name="sac",
        use_tensorboard=config["logging"]["use_tensorboard"],
    )

    # Save configuration
    logger.save_config(config)

    print("\n" + "=" * 50)
    print("Starting SAC Training on CarRacing-v3")
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
                metrics = agent.update()
                if metrics:
                    logger.log_training(total_steps, **metrics)

            state = next_state
            episode_reward += reward
            episode_length += 1
            total_steps += 1

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
    train_sac()
