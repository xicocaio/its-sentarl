import sys

import gym
from envs import StockExchangeEnv

import pandas as pd

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# from envs import ExchangeEnv, StockExchangeEnv


def main(**kwargs):
    test_module = kwargs.get('test') if 'env' in kwargs else 'all'  # allow: all, gen_csv

    if test_module == 'env' or test_module == 'all':
        env = gym.make('stock_exchange-v0', frame_bound=(50, 100), window_size=10, discrete_actions=True)

        observation = env.reset()

        while True:
            action = env.action_space.sample()
            # action = 1
            # obs, reward, done, info = env.step(action)
            obs, reward, done, info = env.step(action)
            # print("action: {} \n obs [price diff]: {}".format(action[0], obs))
            # env.render()
            if done:
                print("info:", info)
                break

        # plt.cla()
        env.render()
        plt.show()

    else:
        print('Test --{}: {}'.format(test_module, 'Nothing implemented'))


if __name__ == '__main__':
    main(**dict(arg.replace('-', '').split('=') for arg in sys.argv[1:]))  # kwargs
