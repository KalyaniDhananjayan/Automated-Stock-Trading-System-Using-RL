import numpy as np
import gym
from gym import spaces

class EnhancedStockTradingEnv(gym.Env):
    def __init__(self, df, window_size=10, initial_balance=10000, transaction_cost=0.001):
        super(EnhancedStockTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost

        self.num_features = len([
            'Close', 'High', 'Low', 'Open', 'Volume', 'SMA_10', 'SMA_50', 'RSI', 'MACD', 'MACD_signal',
            'EMA_20', 'WILLIAMS_R', 'STOCH_K', 'STOCH_D', 'ADX', 'CCI', 'OBV'
        ])

        self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([2, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size, self.num_features + 3), dtype=np.float32)

        self._initialize_metrics()

    def _initialize_metrics(self):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.current_step = self.window_size
        self.equity_curve = []
        self.done = False

    def reset(self):
        self._initialize_metrics()
        return self._next_observation()

    def _next_observation(self):
        frame = self.df.iloc[self.current_step - self.window_size:self.current_step].copy()
        frame = frame.replace([np.inf, -np.inf], np.nan).fillna(0)
        norm_frame = frame / (np.abs(frame).max() + 1e-10)
        obs = norm_frame.to_numpy()

        portfolio = np.array([
            self.balance / (self.net_worth + 1e-10),
            self.shares_held / 1000,
            self.cost_basis / (self.df['Close'].iloc[self.current_step] + 1e-10)
        ])

        portfolio = np.nan_to_num(portfolio)
        obs = np.hstack((obs, np.tile(portfolio, (self.window_size, 1))))
        return obs.astype(np.float32)

    def step(self, action):
        action_type = int(np.clip(np.round(action[0]), 0, 2))
        action_strength = float(np.clip(action[1], 0, 1))

        if self.current_step >= len(self.df) - 1:
            self.done = True
            return self._next_observation(), 0.0, self.done, self._get_info()

        current_price = self.df.loc[self.current_step, "Close"]
        if not np.isfinite(current_price) or current_price <= 0:
            current_price = 1.0

        if action_type == 1:  # BUY
            if self.balance > 0:
                amount = (self.balance * action_strength) / current_price
                if amount > 0.01:
                    self.shares_held += amount
                    self.balance -= amount * current_price
                    self.cost_basis = current_price
        elif action_type == 2:  # SELL
            if self.shares_held > 0:
                amount = self.shares_held * action_strength
                if amount > 0.01:
                    self.shares_held -= amount
                    self.balance += amount * current_price

        self.net_worth = self.balance + self.shares_held * current_price
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        self.equity_curve.append(self.net_worth)

        reward = (self.net_worth -  self.initial_balance)/ self.initial_balance
        
# Penalty for large losses
        if reward < -0.02:
            reward -= abs(reward) * 2  # harsher penalty for sharp drops

        # Small bonus for stable growth
        elif 0 < reward < 0.01:
            reward += 0.001  # encourage small steady growth

        # Optional: bonus for reaching new high net worth (uncomment if needed)
        if self.net_worth > self.max_net_worth:
            reward += 0.01

        self.current_step += 1
        return self._next_observation(), reward, self.done, self._get_info()

    def _get_info(self):
        return {
            "step": self.current_step,
            "net_worth": self.net_worth,
            "balance": self.balance,
            "shares_held": self.shares_held
        }
