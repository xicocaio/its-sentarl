import sys

import gym
from envs import StockExchangeEnv

# from envs import ExchangeEnv, StockExchangeEnv


def main(**kwargs):
    test_module = kwargs.get('test') if 'env' in kwargs else 'all'  # allow: all, gen_csv

    if test_module == 'env' or test_module == 'all':
        env = gym.make('stock_exchange-v0', frame_bound=(50, 100), window_size=10, discrete_actions=True)
        # env = gym.make('stock_exchange-v0', window_size=10, discrete_actions=True)
    else:
        print('Test --{}: {}'.format(test_module, 'Nothing implemented'))


if __name__ == '__main__':
    main(**dict(arg.replace('-', '').split('=') for arg in sys.argv[1:]))  # kwargs