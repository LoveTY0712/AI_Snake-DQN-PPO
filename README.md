# Snake-DQN-PPO: Use DQN and PPO to Train Snake Game Agents

## Project Introduction
This project aims to train AI agents to play the classic Snake game using two major reinforcement learning algorithms: DQN (Deep Q-Network) and PPO (Proximal Policy Optimization). By implementing and comparing these algorithms, this project provides insights into the application of reinforcement learning in game agent training.

## Features
- Implements **DQN** and **PPO** algorithms to train agents separately.
- Provides detailed analysis and comparison of the performance of both algorithms.
- Supports visualization of the training process, including agent scores and survival steps.
- Includes complete code implementations for environment setup, model training, and result visualization.

## Project Structure
Snake-DQN-PPO/
├── README.md                   # Project documentation
├── config.py                   # Configuration file with hyperparameters
├── snake_game.py               # Implementation of the Snake game environment
├── my_dqn.py                   # DQN model and agent implementation
├── dqn_train.py                # DQN model training script
├── Env4PPO.py                  # Snake environment class adapted for PPO
├── ppo_train.py                # PPO model training script
├── models/                     # Trained model files (optional)
├── results/                    # Training results and visualizations
└── Report.pdf                  # Experimental report

## Environment Requirements
- **Python**: Version 3.8 or higher
- **Dependencies**:
  - PyTorch: For building and training deep learning models.
  - Stable Baselines3: For implementing the PPO algorithm.
  - Gymnasium: For defining and interacting with the environment.
  - Pygame: For visualizing the Snake game.
  - WandB: For visualizing the training process and logging (optional).

To install dependencies:
```bash
pip install torch stable-baselines3 gymnasium pygame wandb
Usage
1. DQN Model Training
To start training the DQN model, run the following command:
```bash
python dqn_train.py
During training, the agent's performance (scores, survival steps, etc.) will be visualized via WandB (if enabled). Trained model files will be saved to the models/ directory.
