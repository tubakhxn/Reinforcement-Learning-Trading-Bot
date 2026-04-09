import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rl_trader.environment import TradingEnvironment
from rl_trader.agent import QLearningAgent

def load_data():
    # For demo, generate synthetic price data
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(500)) + 100
    return prices

def plot_results(prices, env, title='Trading Performance'):
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 1, 1)
    plt.plot(prices, label='Price')
    buys = [i for i, a in env.actions if a == 'buy']
    sells = [i for i, a in env.actions if a == 'sell']
    plt.scatter(buys, prices[buys], marker='^', color='g', label='Buy', s=80)
    plt.scatter(sells, prices[sells], marker='v', color='r', label='Sell', s=80)
    plt.title(title)
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(env.equity_curve, label='Equity Curve')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    prices = load_data()
    window_size = 10
    env = TradingEnvironment(prices, window_size=window_size, rsi=False)
    state_size = window_size + 1  # window + moving average
    agent = QLearningAgent(state_size, action_size=3)
    episodes = 50
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        while not env.done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        print(f"Episode {ep+1}/{episodes} - Final Equity: {env.equity:.2f}  Total Reward: {total_reward:.2f}")
    plot_results(prices, env)
    print(f"Final Profit: {env.equity - 10000:.2f}")
    returns = np.diff(env.equity_curve)
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252) if np.std(returns) > 0 else 0
    print(f"Sharpe Ratio: {sharpe:.2f}")

if __name__ == "__main__":
    main()
