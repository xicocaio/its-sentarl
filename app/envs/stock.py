# Built-in imports
from enum import Enum
import numpy as np
from gym import spaces
import logging

# Internal imports
from .exchange import ExchangeEnv


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("escrita_dados")
# logger = logging.getLogger(__name__)


class Actions(Enum):
    Short = -1
    Neutral = 0
    Long = 1


# TODO: add tuple arg for determining feature and its window size
class StockExchangeEnv(ExchangeEnv):
    def __init__(
        self,
        df,
        frame_bound,
        pivot_window_size,
        pivot_price_feature,
        features,
        action_type,
        reward_type,
        reward_function,
        initial_wealth,
        transaction_cost,
        action_window_size,
    ):
        assert df.ndim == 2
        assert len(frame_bound) == 2  # checking if the tuple is size 2

        self._df = df
        self._dates = df.index
        self._frame_bound = frame_bound
        self._pivot_window_size = pivot_window_size
        (
            self._prices,
            self._state_features,
            self._window_sizes,
        ) = self._process_data(pivot_price_feature, features)
        self._action_window_size = action_window_size
        self._shape = (np.sum(self._window_sizes) + self._action_window_size,)

        self._reward_type = reward_type
        self._reward_function = reward_function

        # market characteristics #
        self._tc = transaction_cost
        if self._reward_type == "additive":
            self._shares_amount = initial_wealth
        elif self._reward_type == "multiplicative":
            self._initial_wealth = initial_wealth
        else:
            raise ValueError(
                "reward_type: {} not supported".format(self._reward_type)
            )

        # --- action and state spaces ---

        # if reward type if multiplicative system forces continuous actions
        self._use_discrete_actions = (
            action_type == "discrete" and self._reward_type != "multiplicative"
        )

        action_space = (
            spaces.Discrete(len(Actions))
            if self._use_discrete_actions
            else spaces.Box(
                low=Actions.Short.value,
                high=Actions.Long.value,
                shape=(1,),
                dtype=np.float64,
            )
        )
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self._shape, dtype=np.float64
        )

        start_t = self._pivot_window_size
        end_t = len(self._prices) - 1

        super().__init__(action_space, observation_space, start_t, end_t)

        # other env params
        self.start_date, self.end_date = (
            self._dates[self._start_t],
            self._dates[self._end_t],
        )

    def _process_data(self, pivot_price_feature, features):
        # TODO start and end should be validated and assigned in init,
        # based on frame_bound
        start = self._frame_bound[0] - self._pivot_window_size
        end = self._frame_bound[1]

        # if the end value is higher than the size of the df,
        # the returned value get up to the last data point
        prices = self._df.loc[:, pivot_price_feature].to_numpy()[start:end]

        state_features = self._df.loc[
            :, [feat for feat, _ in features]
        ].to_numpy()[start:end]

        window_sizes = np.expand_dims([size for _, size in features], axis=0)

        return prices, state_features, window_sizes

    def _calculate_sharpe_ratio(self, step_return):
        reward_mean = np.nanmean(
            np.array(self._return_history).astype("float64")
        )
        reward_std = np.nanstd(
            np.array(self._return_history).astype("float64")
        )

        if reward_std != 0:
            sr = (reward_mean / reward_std).item()
        else:
            sr = step_return

        return sr

    def _calculate_return(self, action_value):
        # converting None actions to Neutral = 0
        action = action_value if action_value else Actions.Neutral.value
        past_action = (
            self._action_value_history[-1]
            if self._action_value_history[-1]
            else Actions.Neutral.value
        )

        if self._reward_type == "additive":
            # reward = shares * [At-1*zt - tc*|At - At-1|],
            # where zt = pt - pt-1
            price_diff = (
                self._prices[self._current_t]
                - self._prices[self._last_trade_tick]
            )
            step_return = self._shares_amount * (
                past_action * price_diff - self._tc * abs(action - past_action)
            )
        else:
            # reward = wealth * (zt * At-1) * (1 - c*|At - At-1|) - wealth,
            # where zt = (pt/pt-1) - 1
            wealth = self._initial_wealth + self._total_return
            price_ratio = (
                self._prices[self._current_t]
                / self._prices[self._last_trade_tick]
                - 1
            )
            new_wealth = (
                wealth
                * (1 + price_ratio * past_action)
                * (1 - self._tc * abs(action - past_action))
            )

            step_return = new_wealth - wealth

        return step_return

    def _calculate_reward(self):
        # TODO: maybe move this and other metrics (SR, MDD, Sortino)
        # calculations to utils or other utility module
        step_return = self._calculate_return(self._action_value)
        self._return_history.append(step_return)
        self._total_return += step_return

        sr = self._calculate_sharpe_ratio(step_return)
        self._sr_history.append(sr)

        if self._reward_function == "return":
            step_reward = step_return
            self._total_reward += step_reward
        elif self._reward_function == "sharpe_ratio":
            step_reward = sr
            self._total_reward = sr

        self._reward_history.append(step_reward)

        return step_reward, step_return, sr

    def _get_observation(self):
        # selecting current complete look-back window of features
        # and inverting it for processing
        start_idx = self._current_t - self._pivot_window_size
        end_idx = self._current_t
        inv_state_features = self._state_features[start_idx:end_idx][::-1]

        # mask for selecting most recent data accordingly
        # for the look-back window size for each feature
        mask = np.arange(len(inv_state_features))[:, None] < self._window_sizes
        lookback_window = inv_state_features.T[mask.T]

        past_actions = []
        if self._action_window_size > 0 and self._action_window_size <= len(
            self._action_value_history
        ):
            start_idx = -self._action_window_size
            past_actions = self._action_value_history[start_idx:]
            past_actions = [
                Actions.Neutral.value if action is None else action
                for action in past_actions
            ]
        else:
            raise ValueError(
                "action_window_size: {} not supported".format(
                    self.action_window_size
                )
            )

        return np.append(lookback_window, past_actions)

    # in our case the last action is equivalent to a noop = None
    def _get_action_value(self, action, last_step):
        if last_step:
            return None
        else:
            if self._use_discrete_actions:
                return {
                    0: Actions.Short.value,
                    1: Actions.Neutral.value,
                    2: Actions.Long.value,
                }[action]
            else:
                # the used formulation of multiplicative reward
                # requires values in range [0,1]
                # however, stables baseline recommends using simmetric actions,
                # and that why we adjust it here and not on initialization
                if self._reward_type == "multiplicative":
                    return (action[0] - Actions.Short.value) / (
                        Actions.Long.value - Actions.Short.value
                    )

                return action[
                    0
                ]  # extracting scalar (np.float64) from single item np array

    def _get_current_date(self):
        return self._dates[self._current_t]
