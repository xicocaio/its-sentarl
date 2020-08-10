from enum import Enum

import numpy as np
from gym import spaces
import numpy as np

from .exchange import ExchangeEnv


class Actions(Enum):
    Short = -1
    Neutral = 0
    Long = 1


# TODO: add tuple arg for determining feature and its window size
class StockExchangeEnv(ExchangeEnv):
    def __init__(self, df, window_size, frame_bound, discrete_actions):
        # assert df.ndim == 2
        assert len(frame_bound) == 2

        self._df = df
        self._frame_bound = frame_bound
        self._window_size = window_size
        self._discrete_actions = discrete_actions
        self._prices, self._signal_features = self._process_data()
        self._shape = (window_size, self._signal_features.shape[1])

        # action and state spaces
        np_type = np.int64 if self._discrete_actions else np.float32
        action_space = spaces.Box(low=Actions.Short.value, high=Actions.Long.value, shape=(1,), dtype=np_type)
        # shape = np.array([2, 2])  # REMOVE after _process_data is ready
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self._shape, dtype=np.float32)

        start_t = window_size
        end_t = len(self._prices) - 1

        super().__init__(action_space, observation_space, start_t, end_t)

        # market characteristics
        self._phi = 1  # number of shares
        self._c = 0.001  # trading cost (TC)

    def _process_data(self):
        # TODO start and end should validated and assgned in init, based on frame_bound
        start = self._frame_bound[0] - self._window_size
        end = self._frame_bound[1]
        prices = self._df.loc[:, 'Close'].to_numpy()[start:end]

        diff = np.insert(np.diff(prices), 0, 0)

        print(prices)

        in_features = self._df.loc[:, ['Close']].to_numpy()[start:end]
        # in_features = self._df.loc[:, ['Close', 'avg-sent']].to_numpy()[start:end]

        signal_features = np.column_stack((in_features, diff))

        return prices, signal_features

    def _calculate_reward(self, action):
        current_price = self._prices[self._current_t]
        last_trade_price = self._prices[self._last_trade_tick]
        price_diff = current_price - last_trade_price

        price_diff = self._prices[self._current_t] - self._prices[self._last_trade_tick]

        # converting None actions to Neutral = 0
        action = action if action else Actions.Neutral.value
        past_action = self._action_history[-1] if self._action_history[-1] else Actions.Neutral.value

        # phi * [At-1*zt -c*|At - At-1|]
        if self._discrete_actions:
            step_reward = self._phi * (past_action * price_diff - self._c * abs(action - past_action))
        else:
            NotImplementedError

        return step_reward

    def _get_observation(self):
        return self._signal_features[(self._current_t - self._window_size):self._current_t]

    # in our case the last action is equivalent to a noop = None
    def _process_last_action(self, last_action):
        return None
