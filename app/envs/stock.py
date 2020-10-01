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
    def __init__(self, df, frame_bound, pivot_window_size, pivot_price_feature, features, action_type,
                 reward_type, reward_function, initial_wealth, transaction_cost):
        assert df.ndim == 2
        assert len(frame_bound) == 2  # checking if the tuple is size 2

        self._df = df
        self._dates = df.index
        self._frame_bound = frame_bound
        self._pivot_window_size = pivot_window_size
        self._prices, self._state_features, self._window_sizes = self._process_data(pivot_price_feature, features)
        self._shape = (np.sum(self._window_sizes),)

        self._reward_type = reward_type

        # market characteristics #
        self._tc = transaction_cost
        if self._reward_type == 'additive':
            self._shares_amount = initial_wealth
        elif self._reward_type == 'multiplicative':
            self._initial_wealth = initial_wealth
        else:
            raise ValueError('reward_type: {} not supported'.format(self._reward_type))

        ### action and state spaces ###

        # if reward type if multiplicative system forces continuous actions
        self._use_discrete_actions = action_type == 'discrete' and self._reward_type != 'multiplicative'

        action_space = spaces.Discrete(len(Actions)) if self._use_discrete_actions else spaces.Box(
            low=Actions.Short.value, high=Actions.Long.value, shape=(1,), dtype=np.float64)
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self._shape, dtype=np.float64)

        start_t = self._pivot_window_size
        end_t = len(self._prices) - 1

        super().__init__(action_space, observation_space, reward_function, start_t, end_t)

    def _process_data(self, pivot_price_feature, features):
        # TODO start and end should be validated and assigned in init, based on frame_bound
        start = self._frame_bound[0] - self._pivot_window_size
        end = self._frame_bound[1]
        prices = self._df.loc[:, pivot_price_feature].to_numpy()[start:end]

        diff = np.insert(np.diff(prices), 0, 0)
        in_features = self._df.loc[:, [feat for feat, _ in features]].to_numpy()[start:end]
        state_features = np.column_stack((diff, in_features))

        window_sizes = np.column_stack((self._pivot_window_size, [size for _, size in features])) if features else [
            self._pivot_window_size]

        return prices, state_features, window_sizes

    def _calculate_return(self, action_value):
        # converting None actions to Neutral = 0
        action = action_value if action_value else Actions.Neutral.value
        past_action = self._action_value_history[-1] if self._action_value_history[-1] else Actions.Neutral.value

        if self._reward_type == 'additive':
            # reward = shares * [At-1*zt - tc*|At - At-1|], where zt = pt - pt-1
            price_diff = self._prices[self._current_t] - self._prices[self._last_trade_tick]
            step_return = self._shares_amount * (past_action * price_diff - self._tc * abs(action - past_action))
        else:
            # reward = wealth * (zt * At-1) * (1 - c*|At - At-1|) - wealth, where zt = (pt/pt-1) - 1
            wealth = self._initial_wealth + self._total_profit
            price_ratio = self._prices[self._current_t] / self._prices[self._last_trade_tick] - 1
            new_wealth = wealth * (1 + price_ratio * past_action) * (1 - self._tc * abs(action - past_action))

            step_return = new_wealth - wealth

        return step_return

    def _get_observation(self):
        # selecting current complete look-back window of features and inverting it for processing
        inv_state_features = self._state_features[(self._current_t - self._pivot_window_size):self._current_t][::-1]

        # mask for selecting most recent data according for the look-back window size for each feature
        mask = np.arange(len(inv_state_features))[:, None] < self._window_sizes

        return inv_state_features.T[mask.T]

    # in our case the last action is equivalent to a noop = None
    def _get_action_value(self, action, last_step):
        if last_step:
            return None
        else:
            if self._use_discrete_actions:
                return {
                    0: Actions.Short.value,
                    1: Actions.Neutral.value,
                    2: Actions.Long.value
                }[action]
            else:
                # the used formulation of multiplicative reward requires values in range [0,1]
                # however, stables baseline recommends using simmetric actions, and that why we adjust it here
                # and not on initialization
                if self._reward_type == 'multiplicative':
                    return (action[0] - Actions.Short.value) / (Actions.Long.value - Actions.Short.value)

                return action[0]  # extracting scalar (np.float64) from single item np array

    def _update_profit(self, action_value):
        self._total_profit += self._calculate_return(action_value)

    def _get_current_date(self):
        return self._dates[self._current_t]