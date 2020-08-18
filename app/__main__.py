import sys

import gym
from envs import StockExchangeEnv

import pandas as pd
import numpy as np

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C, SAC, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.cmd_util import make_vec_env


def main(**kwargs):
    test_module = kwargs.get('test') if 'env' in kwargs else 'all'  # allow: all, gen_csv

    env_maker = lambda: gym.make(
        # 'stock_exchange-v0',
        'stock_exchange_sent-v0',
        frame_bound=(50, 100),
        pivot_window_size=10,
        pivot_price_feature='Close',
        # features=[],
        features=[('avg-sent', 5)],
        use_discrete_actions=True,
        reward_type='additive',
        initial_wealth=1,
        transaction_cost=0.001)

    if test_module == 'env' or test_module == 'all':
        # when debugging values do not forget that check_env runs env methods
        check_env(env_maker())

    if test_module == 'random':
        env = env_maker()

        observation = env.reset()
        while True:
            action = env.action_space.sample()
            # action = 1
            # obs, reward, done, info = env.step(action)
            obs, reward, done, info = env.step(action)

            # env.render()
            if done:
                print("info:", info)
                break

        # plt.cla()
        env.render()
        plt.show()

    if test_module == 'model' or test_module == 'all':
        env = DummyVecEnv([env_maker])

        # device = 'cpu'
        device = 'cuda'

        ### Training ###
        # policy_kwargs = dict(net_arch=[64, 'lstm', dict(vf=[128, 128, 128], pi=[64, 64])])
        model = A2C('MlpPolicy', env, device=device, verbose=1)
        model.learn(total_timesteps=1000)

        # model = DQN('MlpPolicy', env, device=device, verbose=1)
        # model.learn(total_timesteps=1000, log_interval=4)

        ### TESTING ###
        env = env_maker()
        observation = env.reset()

        while True:
            observation = observation[np.newaxis, ...]

            action, _states = model.predict(observation)
            # action = [np.squeeze(action]
            observation, reward, done, info = env.step(action[0])
            print(info)

            # env.render()
            if done:
                print("info:", info)
                break

        # plt.cla()
        env.render()
        plt.show()


if __name__ == '__main__':
    main(**dict(arg.replace('-', '').split('=') for arg in sys.argv[1:]))  # kwargs
