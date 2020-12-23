import os
from pathlib import Path
import data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(Path(BASE_DIR).parent, 'app', 'data')

AVAILABLE_DATA = data.AVAILABLE_DATA

STGS_BASE = ['bh', 'random']
STGS_ALGO = ['vanilla', 'relesa']  # options: relesa, vanilla

FREQUENCIES = ['hour']
REWARD_FUNCTIONS = ['return' ]  # options: return, sharpe_ratio
TRANSACTION_COSTS = [0.0025]  # options: 0.0, 0.0005, 0.0015, 0.0025

ROUTINES = ['default']

ALLOWED_INPUT_PARAMS = {'mode': ['single', 'routine', 'consolidation'],
                        'frequency': FREQUENCIES,
                        'stg': STGS_BASE + STGS_ALGO,
                        'algo': ['a2c', 'dqn'],
                        'action_type': ['discrete', 'continuous'],
                        'reward_function': REWARD_FUNCTIONS,
                        'setup': ['static', 'rolling'],
                        'transaction_cost': TRANSACTION_COSTS,
                        'routine_name': ROUTINES
                        }

DEFAULT_ASSETS = ['aapl']

USE_GPU = False
