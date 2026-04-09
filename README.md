## dev/creator = tubakhxn

# Reinforcement Learning Trading Bot

This project is a Python-based trading agent that learns to make buy/sell/hold decisions using reinforcement learning. It simulates trading on price data, tracks portfolio value, and visualizes trading performance.

## What is this project about?
- Implements a simple Q-learning agent for trading
- Uses price window, moving average, and optionally RSI as state
- Actions: hold, buy, sell
- Rewards: profit/loss after each action
- Visualizes price, buy/sell points, and equity curve
- Outputs final profit and Sharpe ratio

## How to fork this project
1. Click the "Fork" button at the top right of this repository on GitHub.
2. Clone your forked repo:
   ```
   git clone https://github.com/YOUR_USERNAME/REPO_NAME.git
   ```
3. Install requirements:
   ```
   pip install -r requirements.txt
   ```
4. Run the project:
   ```
   python main.py
   ```

## Relevant Wikipedia Links
- [Reinforcement Learning](https://en.wikipedia.org/wiki/Reinforcement_learning)
- [Q-learning](https://en.wikipedia.org/wiki/Q-learning)
- [Sharpe Ratio](https://en.wikipedia.org/wiki/Sharpe_ratio)
- [Moving Average](https://en.wikipedia.org/wiki/Moving_average)
- [Relative Strength Index (RSI)](https://en.wikipedia.org/wiki/Relative_strength_index)
