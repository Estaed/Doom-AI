# Doom AI: Comparing DQN and PPO Algorithms

## Overview
This project was developed as part of my Bachelor's degree at Firat University.
This project compares Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) reinforcement learning algorithms in the Doom game environment. The goal is to analyze the performance of both approaches in terms of training efficiency, reward accumulation, and overall gameplay effectiveness.

## Repository Contents
- **Jupyter Notebook**: Contains the full implementation. Run all cells in order to execute the experiments.
- **Training Logs**: Collected data from multiple training sessions.
- **Results & Analysis**: Performance comparison including graphs and insights.
- **Paper**: A detailed report explaining the methodology, results, and conclusions.

## Requirements
To run this project, install the necessary dependencies:
```bash
pip install gym[all] stable-baselines3 vizdoom numpy matplotlib
```

## Running the Code
1. Clone the repository:
```bash
git clone https://github.com/Estaed/Doom-AI.git
cd Doom-AI
```
2. Open the Jupyter Notebook and run all cells in order.

## Results
- **DQN**: Performs well but requires more training time.
- **PPO**: More stable and generalizes better across different Doom environments.

For detailed findings, refer to the included research paper.

## Future Work
- Fine-tuning hyperparameters for better performance.
- Testing additional reinforcement learning algorithms.
- Expanding to more complex Doom scenarios.

## Author
Tarık Bulut

For any questions or suggestions, feel free to open an issue or contact me!

