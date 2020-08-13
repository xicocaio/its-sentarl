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
    def __init__(self, df, frame_bound, pivot_window_size, pivot_price_feature, features, use_discrete_actions):
        assert df.ndim == 2
        assert len(frame_bound) == 2  # checking if the tuple is size 2

        self._df = df
        self._frame_bound = frame_bound
        self._pivot_window_size = pivot_window_size
        self._use_discrete_actions = use_discrete_actions
        self._prices, self._state_features, self._window_sizes = self._process_data(pivot_price_feature, features)
        self._shape = (np.sum(self._window_sizes),)

        # action and state spaces
        np_type = np.int64 if self._use_discrete_actions else np.float32
        action_space = spaces.Box(low=Actions.Short.value, high=Actions.Long.value, shape=(1,), dtype=np_type)
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self._shape, dtype=np.float32)
        # observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=[10,], dtype=np.float32)

        start_t = self._pivot_window_size
        end_t = len(self._prices) - 1

        super().__init__(action_space, observation_space, start_t, end_t)

        # market characteristics
        self._phi = 1  # number of shares
        self._c = 0.001  # trading cost (TC)

    def _process_data(self, pivot_price_feature, features):
        # TODO start and end should validated and assgned in init, based on frame_bound
        start = self._frame_bound[0] - self._pivot_window_size
        end = self._frame_bound[1]
        prices = self._df.loc[:, pivot_price_feature].to_numpy()[start:end]

        diff = np.insert(np.diff(prices), 0, 0)
        in_features = self._df.loc[:, [feat for feat, _ in features]].to_numpy()[start:end]
        state_features = np.column_stack((diff, in_features))

        window_sizes = np.column_stack((self._pivot_window_size, [size for _, size in features])) if features else [
            self._pivot_window_size]

        return prices, state_features, window_sizes

    def _calculate_reward(self, action):
        price_diff = self._prices[self._current_t] - self._prices[self._last_trade_tick]

        # converting None actions to Neutral = 0
        action = action if action else Actions.Neutral.value
        past_action = self._action_history[-1] if self._action_history[-1] else Actions.Neutral.value

        # phi * [At-1*zt -c*|At - At-1|]
        if self._use_discrete_actions:
            step_reward = self._phi * (past_action * price_diff - self._c * abs(action - past_action))
        else:
            NotImplementedError

        return step_reward

    def _get_observation(self):
        # selecting current complete look-back window of features and inverting it for processing
        inv_state_features = self._state_features[(self._current_t - self._pivot_window_size):self._current_t][::-1]

        # mask for selecting most recent data according for the look-back window size for each feature
        mask = np.arange(len(inv_state_features))[:, None] < self._window_sizes

        return inv_state_features.T[mask.T]

    # in our case the last action is equivalent to a noop = None
    def _process_last_action(self, last_action):
        return None
