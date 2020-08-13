import gym
from gym import spaces
from gym.utils import seeding

from enum import Enum
import numpy as np
import matplotlib.pyplot as plt


# import matplotlib.pyplot as plt

class ExchangeEnv(gym.Env):
    """Base stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, action_space, observation_space, start_t, end_t):
        super(ExchangeEnv, self).__init__()

        self.seed()

        # env params
        self.action_space = action_space
        self.observation_space = observation_space

        # simulation params
        self._start_t = start_t
        self._end_t = end_t

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        err_msg = "%r (%s) invalid".format(action, type(action))
        assert self.action_space.contains(action), err_msg

        last_step = True if self._current_t == self._end_t else False

        self._action = self._process_last_action(action) if last_step else action[0]

        step_reward = self._calculate_reward(self._action)

        observation = self._get_observation()

        self._last_trade_tick = self._current_t

        self._action_history.append(self._action)
        self._reward_history.append(step_reward)

        self._total_reward += step_reward

        if last_step:
            self._episode_over = True
        else:
            self._episode_over = False
            self._current_t += 1

        info = dict(
            total_reward=self._total_reward,
            action=self._action
        )

        return observation, step_reward, self._episode_over, info

    def reset(self):
        self._episode_over = False
        self._current_t = self._start_t
        self._last_trade_tick = self._current_t - 1
        self._action = None
        self._total_reward = 0.
        self._action_history = self._start_t * [None]
        self._reward_history = self._start_t * [None]
        self._first_rendering = True

        return self._get_observation()

    # This render was not designed to run step by step, it should run only at the end of episode
    def render(self, mode='human'):
        self._plot_history()

    def close(self):
        fig.close()

    def _process_data(self):
        raise NotImplementedError

    def _calculate_reward(self, action):
        raise NotImplementedError

    def _get_observation(self):
        return NotImplementedError

    def _process_last_action(self, last_action):
        return NotImplementedError

    def _plot_history(self):
        window_ticks = np.arange(len(self._reward_history))

        # print(self._reward_history)
        # print(self._action_history)

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
            if self._action_history[i] is not None:
                if self._action_history[i] < 0:
                    short_ticks.append(tick)
                elif self._action_history[i] > 0:
                    long_ticks.append(tick)
                else:
                    neutral_ticks.append(tick)

            # reward history plotting
            if self._reward_history[i] is not None:
                if self._reward_history[i] < 0:
                    neg_rewards.append(tick)
                elif self._reward_history[i] >= 0:
                    pos_rewards.append(tick)

        ax1.plot(short_ticks, self._prices[short_ticks], 'bv')
        ax1.plot(long_ticks, self._prices[long_ticks], 'b^')
        ax1.plot(neutral_ticks, self._prices[neutral_ticks], 'b_')

        reward_array = np.asarray(self._reward_history)

        ax2.bar(neg_rewards, reward_array[neg_rewards], color='r')
        ax2.bar(pos_rewards, reward_array[pos_rewards], color='g')

        fig.suptitle(
            "Total Profit: %.6f" % self._total_reward
        )
