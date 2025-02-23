# Snake-DQN-PPO: 使用 DQN 和 PPO 训练贪吃蛇智能体

## 项目简介
本项目旨在通过深度强化学习算法（DQN 和 PPO）训练智能体玩经典的贪吃蛇游戏。项目实现了两种算法的建模与训练，并对比了它们在游戏中的表现。通过此项目，可以深入了解强化学习算法在游戏智能体训练中的应用。

## 功能特点
- 使用 **DQN（深度 Q 网络）** 和 **PPO（近端策略优化）** 算法分别训练智能体。
- 提供详细的实验结果分析，对比两种算法的性能。
- 支持可视化训练过程，记录智能体的得分、存活步数等指标。
- 提供完整的代码实现，包括环境搭建、模型训练和结果展示。

## 项目结构
```
Snake-DQN-PPO/
├── README.md                   # 项目说明文档
├── config.py                   # 配置文件，包含超参数设置
├── snake_game.py               # 贪吃蛇游戏环境实现
├── my_dqn.py                   # DQN 模型和智能体实现
├── dqn_train.py                # DQN 模型训练脚本
├── Env4PPO.py                  # 适配 PPO 的贪吃蛇环境类
├── ppo_train.py                # PPO 模型训练脚本
├── models/                     # 训练好的模型文件（可选）
├── results/                    # 训练结果和可视化图像
└── Report.pdf                  # 实验报告
```

## 运行环境
- **Python**：3.8 或更高版本
- **依赖库**：
  - PyTorch：用于深度学习模型的构建和训练。
  - Stable Baselines3：用于 PPO 算法的实现。
  - Gymnasium：用于环境接口的定义和交互。
  - Pygame：用于贪吃蛇游戏的可视化。
  - WandB：用于训练过程的可视化和日志记录（可选）。

安装依赖：
```bash
pip install torch stable-baselines3 gymnasium pygame wandb
```

## 使用方法
### 1. DQN 模型训练
运行以下命令启动 DQN 模型的训练：
```bash
python dqn_train.py
```
训练过程中，智能体的表现（得分、存活步数等）将通过 WandB 可视化（如果启用）。训练完成后，模型文件将保存到 `models/` 目录。

### 2. PPO 模型训练
运行以下命令启动 PPO 模型的训练：
```bash
python ppo_train.py
```
同样，训练过程中的指标将通过 WandB 可视化（如果启用）。

### 3. 查看训练结果
训练结果（得分、存活步数等）将保存在 `results/` 目录中。你也可以通过 WandB 查看实时训练日志。

## 项目对比
- **DQN**：
  - 学习曲线较为平稳，但收敛速度较慢。
  - 在训练初期表现不佳，需要较多的训练时间才能达到理想水平。
- **PPO**：
  - 学习曲线增长迅速，收敛速度较快。
  - 在训练初期即可达到较高分数，适合快速训练场景。

## 实验结果
项目中提供了详细的实验报告（`Report.pdf`），对比了 DQN 和 PPO 算法在贪吃蛇游戏中的表现。报告中包括算法原理、实验设计、结果分析等内容。

## 贡献指南
欢迎对项目进行改进和扩展！如果你有新的想法或优化建议，请通过以下方式参与：
1. 提交 **Issues**：报告问题或提出建议。
2. 提交 **Pull Requests**：贡献代码或改进文档。
