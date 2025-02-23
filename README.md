# Snake-DQN-PPO: Training Snake Game Agents with DQN and PPO

## Project Introduction
This project aims to train AI agents to play the classic Snake game using two major reinforcement learning algorithms: DQN (Deep Q-Network) and PPO (Proximal Policy Optimization). The project implements the modeling and training of both algorithms and compares their performance in the game. Through this project, you can gain a deep understanding of the application of reinforcement learning algorithms in training game agents.

## Features
- **DQN (Deep Q-Network)** and **PPO (Proximal Policy Optimization)** algorithms are used to train agents separately.
- Detailed analysis and comparison of the performance of both algorithms are provided.
- Visualization of the training process is supported, including recording the agent's scores and survival steps.
- Complete code implementations are provided, including environment setup, model training, and result visualization.

## Project Structure
```
Snake-DQN-PPO/
├── README.md                   # Project documentation
├── src/                        # Source code folder
│   ├── config.py               # Configuration file with hyperparameters
│   ├── snake_game.py           # Implementation of the Snake game environment
│   ├── my_dqn.py               # DQN model and agent implementation
│   ├── dqn_train.py            # DQN model training script
│   ├── Env4PPO.py              # Snake environment class adapted for PPO
│   └── ppo_train.py            # PPO model training script
├── models/                     # Trained model files (optional)
├── reference/                  # References
└── Report.pdf                  # Experimental report
```

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
```

## Usage
### 1. DQN Model Training
To start training the DQN model, run the following command:
```bash
python dqn_train.py
```
During training, the agent's performance (scores, survival steps, etc.) will be visualized via WandB (if enabled). Trained model files will be saved to the `models/` directory.

### 2. PPO Model Training
To start training the PPO model, run the following command:
```bash
python ppo_train.py
```
Similarly, training metrics will be visualized via WandB (if enabled).

### 3. Viewing Training Results
Training results (scores, survival steps, etc.) will be saved in the `results/` directory. You can also view real-time training logs via WandB.

## Comparison of Algorithms
- **DQN**:
  - The learning curve is relatively smooth but converges slowly.
  - Performs poorly in the early stages of training and requires more training time to reach an ideal level.
- **PPO**:
  - The learning curve grows rapidly and converges quickly.
  - Achieves high scores in the early stages of training and is suitable for quick training scenarios.

## Experimental Results
The project includes a detailed experimental report (`Report.pdf`), which compares the performance of DQN and PPO algorithms in the Snake game. The report covers algorithm principles, experimental design, and result analysis.

## Contribution Guidelines
Contributions to this project are welcome! If you have new ideas or suggestions for improvement, please participate by:
1. Submitting **Issues**: Report problems or suggest improvements.
2. Submitting **Pull Requests**: Contribute code or improve documentation.
