import sys

import gym

# internal imports
from envs import StockExchangeEnv
from common import CSVOutput, load_dataset, split_data, get_standard_result_filename
import settings

# generic imports
import pandas as pd
import numpy as np

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C, SAC, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

import quantstats as qs

import os


def prepare_base_result_values(asset, action_type, reward_type, stg, algo, frequency, seed):
    return {'asset': asset,
            'company_name': settings.AVAILABLE_DATA[asset]['name'],
            'action_type': action_type,
            'reward_type': reward_type,
            'stg': stg,
            'algo': algo,
            'initial_wealth': settings.DEFAULT_ENV['initial_wealth'],
            'transaction_cost': settings.DEFAULT_ENV['transaction_cost'],
            'frequency': frequency,
            'reward_function': settings.DEFAULT_ENV['reward_function']}


def train_model(algo, env, action_type, device, total_timesteps, eval_env, eval_freq, seed):
    if algo == 'A2C':
        # policy_kwargs = dict(net_arch=[64, 'lstm', dict(vf=[128, 128, 128], pi=[64, 64])])
        model = A2C('MlpPolicy', env, device=device, verbose=1, seed=seed)
        model.learn(total_timesteps=total_timesteps, eval_env=eval_env, eval_freq=eval_freq,
                    n_eval_episodes=1)
    if algo == 'DQN':
        model = DQN('MlpPolicy', env, device=device, verbose=1)
        model.learn(total_timesteps=total_timesteps)

    return model


def get_stg_action(stg, env, action_type, observation, model):
    if stg == 'random':
        action = env.action_space.sample()
    elif stg == 'bh':
        action = env.action_space.n - 1 if action_type == 'discrete' else env.action_space.high
    else:
        observation[np.newaxis, ...]

        deterministic = False
        action, _states = model.predict(observation, deterministic=deterministic)

    return action


def prepare_env(df, frame_bound, pivot_window_size, features, action_type, reward_type, initial_wealth,
                transaction_cost):
    return lambda: gym.make('stock_exchange-v0',
                            df=df,
                            frame_bound=frame_bound,
                            pivot_window_size=pivot_window_size,
                            features=features,
                            action_type=action_type,
                            reward_type=reward_type,
                            initial_wealth=initial_wealth,
                            transaction_cost=transaction_cost)


def main(**kwargs):
    stg = kwargs.get('stg', settings.AVAILABLE_STGS_BASE[0])
    asset = kwargs.get('asset', settings.DEFAULT_ENV['asset']).upper()
    max_episodes = int(kwargs.get('episodes', settings.DEFAULT_EPISODES))
    action_type = kwargs.get('actions', settings.DEFAULT_ENV['action_type'])
    test_data_ratio = kwargs.get('test_ratio', settings.TEST_DATA_RATIO)
    val_data_ratio = kwargs.get('val_ratio', settings.VAL_DATA_RATIO)
    reward_type = kwargs.get('reward_type', settings.DEFAULT_ENV['reward_type'])
    initial_wealth = kwargs.get('initial_wealth', settings.DEFAULT_ENV['initial_wealth'])
    transaction_cost = kwargs.get('transaction_cost', settings.DEFAULT_ENV['transaction_cost'])
    k_rolls = kwargs.get('k_rolls', 10)

    if stg not in settings.AVAILABLE_STGS_BASE + settings.AVAILABLE_STGS_ALGO:
        raise ValueError('Strategy entry not found: {}'.format(stg))

    if stg in ['bh', 'random']:
        algo = None
    else:
        algo = kwargs.get('algo', settings.AVAILABLE_ALGOS[0]).upper()
        if algo not in settings.AVAILABLE_ALGOS:
            raise ValueError('Algorithm entry not found: {}'.format(algo))

    features = [('avg-sent', 10)]
    if stg == 'relesa':
        features = features if kwargs.get('use_sent') == 'yes' else []

    if action_type not in settings.ACTION_TYPES:
        raise ValueError('Asset entry not found: {}'.format(action_type))

    if asset not in settings.AVAILABLE_DATA and asset != 'all':
        raise ValueError('Asset entry not found: {}'.format(asset))

    seed = 42

    pivot_window_size = 20
    df = load_dataset(settings.AVAILABLE_DATA[asset]['fname'], settings.AVAILABLE_DATA[asset]['index_col'])
    frequency = settings.AVAILABLE_DATA[asset]['frequency']

    df_train, df_test, df_eval = split_data(df, test_data_ratio, pivot_window_size, val_data_ratio)

    frame_bound = (pivot_window_size, len(df_train.index))
    total_timesteps = max_episodes * (frame_bound[1] - frame_bound[0])
    print('Total training timesteps: {}'.format(total_timesteps))

    env_maker_train = prepare_env(df_train, frame_bound, pivot_window_size, features, action_type, reward_type,
                                  initial_wealth, transaction_cost)

    env_maker_eval = prepare_env(df_eval, frame_bound, pivot_window_size, features, action_type, reward_type,
                                 initial_wealth, transaction_cost)

    # when debugging values do not forget that check_env runs env methods
    check_env(env_maker_train())
    env_train = env_maker_train()
    env_eval = env_maker_eval()
    model = None

    if stg in settings.AVAILABLE_STGS_ALGO:
        filename_train = os.path.join(settings.RESULTS_DIR,
                                      get_standard_result_filename(asset, stg, algo, seed, max_episodes, 'train'))
        filename_eval = os.path.join(settings.RESULTS_DIR,
                                     get_standard_result_filename(asset, stg, algo, seed, max_episodes, 'eval'))

        env_monitored_train = Monitor(env_train, filename=filename_train, info_keywords=("total_return",))
        env_monitored_eval = Monitor(env_eval, filename=filename_eval, info_keywords=("total_return",))
        device = 'cpu' if not settings.USE_GPU else 'cuda'

        eval_freq = (len(df_eval.index) - pivot_window_size)
        eval_freq = frame_bound[1] - frame_bound[0]

        model = train_model(algo, env_monitored_train, action_type, device, total_timesteps, env_monitored_eval,
                            eval_freq, seed)

    frame_bound = (pivot_window_size, len(df_test.index))
    env_maker_test = prepare_env(df_test, frame_bound, pivot_window_size, features, action_type, reward_type,
                                 initial_wealth, transaction_cost)
    env_test = env_maker_test()

    filename_test = get_standard_result_filename(asset, stg, algo, seed, max_episodes, 'test')
    csv_output = CSVOutput(filename_test, overwrite_file=True, delimiter=';')

    base_result_values = prepare_base_result_values(asset, action_type, reward_type, stg, algo, frequency, seed)

    for k in range(k_rolls):
        observation = env_test.reset()

        while True:
            action = get_stg_action(stg, env_test, action_type, observation, model)
            observation, reward, done, info = env_test.step(action)

            base_result_values['train_episodes'] = max_episodes
            base_result_values['window_step'] = k
            base_result_values['reward'] = reward

            csv_output.write({**base_result_values, **info})

            # env.render()
            if done:
                print("info:", info)
                break

        # plt.cla()
        # env_test.render()
        # plt.show()

    csv_output.close()


if __name__ == '__main__':
    main(**dict(arg.replace('-', '').split('=') for arg in sys.argv[1:]))  # kwargs
