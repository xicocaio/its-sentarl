import gym
from gym import spaces
from gym.utils import seeding

from enum import Enum
import numpy as np


# import matplotlib.pyplot as plt

class ExchangeEnv(gym.Env):
    """Base stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, action_space, observation_space, start_t, end_t):
        super(ExchangeEnv, self).__init__()

        self.seed()

        self.action_space = action_space
        self.observation_space = observation_space

        # episode
        self._start_t = start_t
        self._end_t = end_t
        self._episode_over = None
        self._current_tick = None
        self._last_trade_tick = None
        self._action = None
        self._action_history = None
        self._total_reward = None
        self._reward_history = None
        self._first_rendering = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        err_msg = "%r (%s) invalid".format(action, type(action))
        assert self.action_space.contains(action), err_msg

        self._action = action[0]

        self._episode_over = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._episode_over = True

        step_reward = self._calculate_reward(self._action)
        self._reward_history.append(step_reward)
        self._total_reward += step_reward

        self._action_history.append(self._action)
        observation = self._get_observation()

        self._last_trade_tick = self._current_tick

        info = dict(
            total_reward=self._total_reward,
            action=self._action.value
        )

        return observation, step_reward, self._episode_over, info

    def reset(self):
        self._episode_over = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._action = Actions.Neutral.value
        self._action_history = []
        self._reward_history = []
        self._total_reward = 0.
        self._first_rendering = True
        return self._get_observation()

    def render(self, mode='human'):
        window_ticks = np.arange(len(self._action_history))
        plt.plot(self._reward_history)

        short_ticks = []
        long_ticks = []
        neutral_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._action_history[i] == Actions.Short.value:
                short_ticks.append(tick)
            elif self._action_history[i] == Actions.Long.value:
                long_ticks.append(tick)
            else:
                neutral_ticks.append(tick)

        plt.plot(short_ticks, self.prices[short_ticks], 'ro')
        plt.plot(long_ticks, self.prices[long_ticks], 'go')
        # plt.plot(neutral_ticks, self.prices[neutral_ticks], 'yx')

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward
            # + ' ~ ' + "Total Profit: %.6f" % self._total_profit
        )

    def close(self):
        plt.close()

    def _process_data(self):
        raise NotImplementedError

    def _calculate_reward(self, action):
        raise NotImplementedError
