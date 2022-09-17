import gym
from gym.utils import seeding

import numpy as np
import matplotlib.pyplot as plt


# import matplotlib.pyplot as plt


class ExchangeEnv(gym.Env):
    """Base stock trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(self, action_space, observation_space, start_t, end_t):
        super(ExchangeEnv, self).__init__()

        self.seed()

        # openAi gym env params
        self.action_space = action_space
        self.observation_space = observation_space

        # historic data
        self.history = {}
        self._action_value_history = start_t * [None]

        # agent's performance market metrics
        self._total_return = 0.0
        self._total_reward = 0.0

        # simulation params
        self._start_t = start_t
        self._end_t = end_t

        # additional env params
        self.max_episode_steps = self._end_t - (self._start_t - 1)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        err_msg = f"{action} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg

        last_step = True if self._current_t == self._end_t else False

        self._action_value = self._get_action_value(action, last_step)

        step_reward, step_return, sr = self._calculate_reward()

        observation = self._get_observation()

        self._last_trade_tick = self._current_t

        self._action_value_history.append(self._action_value)

        info = dict(
            date=self._get_current_date(),
            step_return=step_return,
            sharpe_ratio=sr,
            total_return=self._total_return,
            total_reward=self._total_reward,
            action_value=self._action_value,
        )

        self._update_history(info)

        if last_step:
            self._episode_over = True
        else:
            self._episode_over = False
            self._current_t += 1

        return observation, step_reward, self._episode_over, info

    def reset(self):
        self._episode_over = False
        self._current_t = self._start_t
        self._last_trade_tick = self._current_t - 1
        self._action_value = None
        self._total_return = 0.0
        self._total_reward = 0.0
        self._action_value_history = self._start_t * [None]
        self._reward_history = self._start_t * [None]
        self._return_history = self._start_t * [None]
        self._sr_history = self._start_t * [None]
        self._first_rendering = True

        self.history = {}

        return self._get_observation()

    # This render was not designed to run step by step,
    # it should run only at the end of episode
    def render(self, mode="human"):
        self._plot_history()

    def close(self):
        return

    def _process_data(self):
        raise NotImplementedError

    def _calculate_reward(self):
        return NotImplementedError

    def _get_observation(self):
        return NotImplementedError

    def _get_action_value(self, action, last_step):
        return NotImplementedError

    def _plot_history(self):
        window_ticks = np.arange(len(self._return_history))

        fig, ax1 = plt.subplots()

        ax1.plot(self._prices)

        ax2 = ax1.twinx()

        short_ticks = []
        long_ticks = []
        neg_rewards = []
        pos_rewards = []
        neutral_ticks = []
        for i, tick in enumerate(window_ticks):
            # actions history plotting
            if self._action_value_history[i]:
                if self._action_value_history[i] < 0:
                    short_ticks.append(tick)
                elif self._action_value_history[i] > 0:
                    long_ticks.append(tick)
                else:
                    neutral_ticks.append(tick)

            # reward history plotting
            if self._return_history[i]:
                if self._return_history[i] < 0:
                    neg_rewards.append(tick)
                elif self._return_history[i] >= 0:
                    pos_rewards.append(tick)

        ax1.plot(short_ticks, self._prices[short_ticks], "bv")
        ax1.plot(long_ticks, self._prices[long_ticks], "b^")
        ax1.plot(neutral_ticks, self._prices[neutral_ticks], "b_")

        reward_array = np.asarray(self._return_history)

        ax2.bar(neg_rewards, reward_array[neg_rewards], color="r")
        ax2.bar(pos_rewards, reward_array[pos_rewards], color="g")

        fig.suptitle("Total Profit: %.6f" % self._total_reward)

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def _convert_action(self, action):
        return NotImplementedError

    def _get_current_date(self):
        raise NotImplementedError
