### name space declaration ###
from .stock import StockExchangeEnv

# internal imports
import settings
from common import load_dataset

### other code necessary for proper initialization of env
from gym.envs.registration import register
from copy import deepcopy
import os
import pandas as pd
from pathlib import Path


def _load_default_env_settings():
    default_kwargs = settings.DEFAULT_ENV.copy()
    asset = default_kwargs.pop('asset')

    df = load_dataset(settings.AVAILABLE_DATA[asset]['fname'], settings.AVAILABLE_DATA[asset]['index_col'])
    default_kwargs['df'] = df
    default_kwargs['frame_bound'] = (default_kwargs['pivot_window_size'], len(df.index))

    return default_kwargs


### Register of default envs, override of args occurs if other args are passed during initialization ###

register(
    id='stock_exchange-v0',
    entry_point='envs:StockExchangeEnv',
    kwargs=_load_default_env_settings()
)
