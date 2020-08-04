### name space declaration ###
from .stock import StockExchangeEnv

### other code necessary for proper initialization of env
from gym.envs.registration import register
from copy import deepcopy

# from . import datasets

# register(
#     id='stock_exchange-v0',
#     entry_point='StockExchangeEnv',
#     kwargs={
#         'df': deepcopy(datasets.STOCKS_AAPL_SENT),
#         'window_size': 50,
#         'frame_bound': (50, len(datasets.STOCKS_AAPL_SENT)),
#         'discrete_actions': True
#     }
# )

register(
    id='stock_exchange-v0',
    entry_point='envs:StockExchangeEnv',
    kwargs={
        'df': 'test',
        'window_size': 50,
        'frame_bound': (50, 100),
        'discrete_actions': True
    }
)
