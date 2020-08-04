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

        self._frame_bound = frame_bound
        self._discrete_actions = discrete_actions

        # action and state spaces
        np_type = np.int64 if self._discrete_actions else np.float32
        action_space = spaces.Box(low=Actions.Short.value, high=Actions.Long.value, shape=(1,), dtype=np_type)
        shape = np.array([2, 2])  # REMOVE after _process_data is ready
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)

        self._window_size = window_size
        self._prices, self.signal_features = self._process_data()

        start_t = window_size
        end_t = len(self.prices) - 1

        super().__init__(action_space, observation_space, start_t, end_t)
