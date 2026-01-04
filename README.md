# CarRacing-v3 RL Agent Training

A modular reinforcement learning project for training agents on Gymnasium's CarRacing-v3 environment. Implements three algorithms with shared CNN architecture: DQN (discrete actions), PPO (continuous actions), and SAC (continuous actions).

## Project Structure

```
reinforcement-learning/
├── agents/                    # Agent implementations
│   ├── dqn_agent.py          # DQN with experience replay
│   ├── ppo_agent.py          # PPO with GAE
│   ├── sac_agent.py          # SAC with twin Q-networks
│   ├── replay_buffer.py      # Off-policy replay buffer
│   └── rollout_buffer.py     # On-policy rollout buffer
├── models/                    # Neural network architectures
│   ├── shared_cnn.py         # Shared CNN feature extractor
│   ├── dqn.py               # DQN Q-network
│   ├── ppo.py               # PPO Actor-Critic
│   └── sac.py               # SAC Policy and Q-networks
├── envs/                      # Environment wrappers
│   └── wrappers.py           # CarRacing preprocessing
├── utils/                     # Training utilities
│   ├── training_monitor.py   # Early stopping logic
│   ├── checkpoint_manager.py # Model checkpointing
│   ├── logger.py            # TensorBoard logging
│   └── seed.py              # Reproducibility
├── configs/                   # YAML configurations
│   ├── dqn_config.yaml
│   ├── ppo_config.yaml
│   └── sac_config.yaml
├── scripts/                   # Training scripts
│   ├── train_dqn.py
│   ├── train_ppo.py
│   └── train_sac.py
├── notebooks/                 # Analysis notebooks
│   └── comparison.ipynb      # Agent comparison
├── checkpoints/              # Saved models (created at runtime)
├── logs/                     # Training logs (created at runtime)
└── requirements.txt
```

## Features

- **Modular Design**: Clean separation of agents, models, environments, and utilities
- **Shared CNN Architecture**: Nature DQN-style CNN (32→64→64 filters, 4096 features) used across all agents
- **Dual Action Space Support**: DQN uses 5 discrete actions, PPO/SAC use continuous [steering, gas, brake]
- **Frame Preprocessing**: 4-frame stacking, frame skipping, and normalization
- **Hybrid Training Strategy**: 1000-episode budget + early stopping at 700+ score threshold
- **Convergence Tracking**: Monitors last 100 episodes (mean ≥ 700 AND std ≤ 10)
- **Comprehensive Logging**: TensorBoard integration with episode metrics
- **Checkpoint Management**: Saves latest/best/periodic/final models

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd reinforcement-learning
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training Individual Agents

Train DQN (discrete actions):
```bash
python scripts/train_dqn.py
```

Train PPO (continuous actions):
```bash
python scripts/train_ppo.py
```

Train SAC (continuous actions):
```bash
python scripts/train_sac.py
```

### Monitoring Training

View training progress with TensorBoard:
```bash
tensorboard --logdir logs
```

### Comparing Agents

Open and run the comparison notebook:
```bash
jupyter notebook notebooks/comparison.ipynb
```

## Configuration

Each agent has a YAML configuration file in `configs/` with tunable hyperparameters:

### Exposed Hyperparameters (Actively Studied)
- `learning_rate`: Learning rate for optimizer
- `frame_stack`: Number of frames to stack (default: 4)
- `skip_frames`: Action repeat frequency (default: 2)

### Fixed Hyperparameters (Widely Accepted Defaults)

**DQN:**
- `gamma = 0.99`: Discount factor
- `buffer_size = 100000`: Replay buffer capacity
- `batch_size = 32`: Mini-batch size
- `epsilon_decay = 10000`: Exploration decay steps
- `target_update_freq = 1000`: Target network update frequency

**PPO:**
- `gamma = 0.99`: Discount factor
- `gae_lambda = 0.95`: GAE parameter
- `clip_epsilon = 0.2`: PPO clipping parameter
- `n_steps = 2048`: Rollout length
- `n_epochs = 10`: Update epochs per rollout
- `batch_size = 64`: Mini-batch size

**SAC:**
- `gamma = 0.99`: Discount factor
- `tau = 0.005`: Soft update coefficient
- `alpha = auto`: Automatic entropy tuning
- `buffer_size = 100000`: Replay buffer capacity
- `batch_size = 256`: Mini-batch size

## Algorithm Details

### DQN (Deep Q-Network)
- **Action Space**: Discrete (5 actions: nothing, steer right, steer left, gas, brake)
- **Key Features**: Experience replay, target network, epsilon-greedy exploration
- **Expected Performance**: 700-850 score
- **Training Time**: Slower convergence, typically 600-800 episodes

### PPO (Proximal Policy Optimization)
- **Action Space**: Continuous [steering, gas, brake]
- **Key Features**: Clipped surrogate objective, GAE, on-policy updates
- **Expected Performance**: 800-900 score
- **Training Time**: Good balance, typically 400-600 episodes

### SAC (Soft Actor-Critic)
- **Action Space**: Continuous [steering, gas, brake]
- **Key Features**: Twin Q-networks, entropy regularization, off-policy
- **Expected Performance**: 850-950 score
- **Training Time**: Most sample-efficient, typically 300-500 episodes

## Early Stopping Criteria

Training stops when either condition is met:
1. **Max Episodes**: 1000 episodes reached
2. **Convergence**: Last 100 episodes have mean score ≥ 700 AND standard deviation ≤ 10

The episode number at convergence is logged for comparison.

## Checkpoints

Models are saved at multiple stages:
- `latest.pth`: Most recent model (every episode)
- `best.pth`: Model with highest score
- `episode_N.pth`: Periodic saves every 100 episodes
- `final.pth`: Model at training completion

## Results

After training, results are saved in:
- `logs/<agent>/results.json`: Final statistics
- `logs/<agent>/config.json`: Training configuration
- `checkpoints/<agent>/`: Model checkpoints
- TensorBoard logs in `logs/<agent>/`

Use the comparison notebook to visualize:
- Learning curves
- Sample efficiency (episodes to convergence)
- Final performance statistics
- Score distributions

## Environment Details

**CarRacing-v3:**
- Observation: 96×96×3 RGB images
- Actions: Continuous [steering, gas, brake] or discrete (5 actions)
- Reward: -0.1 per frame + 1000/N per tile visited
- Episode Length: Variable, max ~1000 steps
- Termination: All tiles visited OR car goes off-track (-100 penalty)

## Hardware Requirements

- **GPU**: Recommended (CUDA-compatible)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 2GB for checkpoints and logs
- **Training Time**: 4-12 hours per agent on modern GPU

## Troubleshooting

**Issue: Out of CUDA memory**
- Reduce `batch_size` in config files
- Use smaller CNN architecture
- Switch to CPU: Set `device: "cpu"` in configs

**Issue: Environment rendering fails**
- Install Box2D dependencies: `pip install gymnasium[box2d]`
- On Linux: `sudo apt-get install swig python3-dev`

**Issue: Slow training**
- Enable GPU: Check `torch.cuda.is_available()`
- Reduce `frame_stack` or increase `skip_frames`
- Use smaller replay buffer

## Citation

If you use this project, please cite:

```bibtex
@misc{carracing_rl_2026,
  title={CarRacing-v3 RL Agent Training},
  author={Your Name},
  year={2026},
  howpublished={\url{https://github.com/your-repo}}
}
```

## License

MIT License

## References

1. Mnih et al. (2015). Human-level control through deep reinforcement learning. Nature.
2. Schulman et al. (2017). Proximal Policy Optimization Algorithms. arXiv.
3. Haarnoja et al. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL. arXiv.
4. Gymnasium Documentation: https://gymnasium.farama.org/

## Contact

For questions or issues, please open a GitHub issue or contact [your-email].
