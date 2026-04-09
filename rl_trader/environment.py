import numpy as np
import pandas as pd

class TradingEnvironment:
    def __init__(self, prices, window_size=10, rsi=False):
        self.prices = prices
        self.window_size = window_size
        self.rsi = rsi
        self.reset()

    def reset(self):
        self.current_step = self.window_size
        self.position = 0  # 0: neutral, 1: long, -1: short
        self.entry_price = 0
        self.done = False
        self.equity = 10000  # starting cash
        self.equity_curve = [self.equity]
        self.actions = []
        return self._get_state()

    def _get_state(self):
        window = self.prices[self.current_step - self.window_size:self.current_step]
        ma = np.mean(window)
        state = [*window, ma]
        if self.rsi:
            state.append(self._rsi(window))
        return np.array(state, dtype=np.float32)

    def _rsi(self, window, period=14):
        delta = np.diff(window)
        up = delta[delta > 0].sum() / period if (delta > 0).any() else 0
        down = -delta[delta < 0].sum() / period if (delta < 0).any() else 0
        rs = up / down if down != 0 else 0
        return 100 - (100 / (1 + rs))

    def step(self, action):
        reward = 0
        price = self.prices[self.current_step]
        if action == 1:  # buy
            if self.position == 0:
                self.position = 1
                self.entry_price = price
                self.actions.append((self.current_step, 'buy'))
        elif action == 2:  # sell
            if self.position == 1:
                reward = price - self.entry_price
                self.equity += reward
                self.position = 0
                self.actions.append((self.current_step, 'sell'))
        self.current_step += 1
        self.equity_curve.append(self.equity)
        if self.current_step >= len(self.prices):
            self.done = True
        return self._get_state(), reward, self.done, {}
