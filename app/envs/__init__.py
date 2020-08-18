### name space declaration ###
from .stock import StockExchangeEnv

### other code necessary for proper initialization of env
from gym.envs.registration import register
from copy import deepcopy
import os
import pandas as pd
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(Path(BASE_DIR).parent, 'data')


def load_dataset(name, index_name):
    path = os.path.join(DATA_DIR, name + '.csv')
    return pd.read_csv(path, index_col=index_name)


STOCKS_AAPL = load_dataset('AAPL.USUSD_Candlestick_1_Hour_BID_19.06.2017-13.03.2020', 'datetime')
STOCKS_AAPL_SENT = load_dataset('ready_aapl_hour_duska_2018-01-02_2019-12-31', 'datetime')

register(
    id='stock_exchange-v0',
    entry_point='envs:StockExchangeEnv',
    kwargs={
        'df': STOCKS_AAPL,
        'frame_bound': (50, 100),
        'pivot_window_size': 10,
        'pivot_price_feature': 'Close',
        'features': [('avg-sent', 5)],
        'use_discrete_actions': True,
        'reward_type': 'additive',
        'initial_wealth': 1,
        'transaction_cost': 0.001
    }
)

register(
    id='stock_exchange_sent-v0',
    entry_point='envs:StockExchangeEnv',
    kwargs={
        'df': STOCKS_AAPL_SENT,
        'frame_bound': (50, 100),
        'pivot_window_size': 10,
        'pivot_price_feature': 'Close',
        'features': [('avg-sent', 5)],
        'use_discrete_actions': True,
        'reward_type': 'additive',
        'initial_wealth': 1,
        'transaction_cost': 0.001
    }
)
