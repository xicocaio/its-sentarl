import os
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(Path(BASE_DIR).parent, 'app', 'data')
RESULTS_DIR = os.path.join(Path(BASE_DIR).parent, 'app', 'results')

AVAILABLE_DATA = {
    'AAPL': {
        'fname': 'ready_aapl_hour_duska_2018-01-02_2019-12-31',
        'index_col': 'datetime',
        'name': 'Apple',
        'type': 'stocks',
        'frequency': 'hour'
    },
    'AMZN': {
        'fname': 'ready_amzn_hour_duska_2018-01-02_2019-12-31',
        'index_col': 'datetime',
        'name': 'Amazon',
        'type': 'stocks',
        'frequency': 'hour'
    },
    'BA': {
        'fname': 'ready_ba_hour_duska_2018-01-02_2019-12-31',
        'index_col': 'datetime',
        'name': 'Boeing',
        'type': 'stocks',
        'frequency': 'hour'
    },
    'BTCUSD': {
        'fname': 'ready_btcusd_hour_duska_2018-01-01_2019-12-31',
        'index_col': 'datetime',
        'name': 'Bitcoin',
        'type': 'stocks',
        'frequency': 'hour'
    },
    'FB': {
        'fname': 'ready_fb_hour_duska_2018-01-02_2019-12-31',
        'index_col': 'datetime',
        'name': 'Facebook',
        'type': 'stocks',
        'frequency': 'hour'
    },
    'GOOGL': {
        'fname': 'ready_googl_hour_duska_2018-01-02_2019-12-31',
        'index_col': 'datetime',
        'name': 'Google',
        'type': 'stocks',
        'frequency': 'hour'
    },
    'MSFT': {
        'fname': 'ready_msft_hour_duska_2018-01-02_2019-12-31',
        'index_col': 'datetime',
        'name': 'Microsoft',
        'type': 'stocks',
        'frequency': 'hour'
    },
    'NFLX': {
        'fname': 'ready_nflx_hour_duska_2018-01-02_2019-12-31',
        'index_col': 'datetime',
        'name': 'Netflix',
        'type': 'stocks',
        'frequency': 'hour'
    },
    'SPY': {
        'fname': 'ready_spy_hour_duska_2018-01-02_2019-12-31',
        'index_col': 'datetime',
        'name': 'SPY',
        'type': 'stocks',
        'frequency': 'hour'
    },
    # 'STOCKS_AAPL': {
    #     'fname': 'AAPL.USUSD_Candlestick_1_Hour_BID_19.06.2017-13.03.2020',
    #     'index_col': 'datetime',
    # }
}

AVAILABLE_STGS_BASE = ['bh', 'random']

AVAILABLE_STGS_ALGO = ['vanilla', 'relesa']

AVAILABLE_ALGOS = ['A2C', 'DQN']

ACTION_TYPES = ['discrete', 'continuous']

REWARD_FUNCTIONS = ['return', 'sharpe_ratio']

RESULTS_HEADER = ['window_step', 'current_step', 'date', 'asset', 'company_name', 'frequency', 'initial_wealth',
                  'transaction_cost', 'stg', 'algo',
                  'action_type', 'reward_type', 'reward_function', 'seed', 'train_episodes', 'action_value',
                  'total_return', 'reward',
                  'total_reward',
                  'total_profit']

DEFAULT_ASSETS = ['AAPL']

DEFAULT_EPISODES = 10

VAL_DATA_RATIO = 0.1
TEST_DATA_RATIO = 0.1

USE_GPU = False

# Default Env Settings
DEFAULT_ENV = {'asset': 'AAPL',
               'pivot_price_feature': 'Close',
               'pivot_window_size': 50,
               'features': [('avg-sent', 5)],
               'action_type': 'discrete',
               'reward_type': 'additive',
               'reward_function': 'return',
               'initial_wealth': 1,
               'transaction_cost': 0.025}
