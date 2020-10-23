import os
from pathlib import Path
import data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(Path(BASE_DIR).parent, 'app', 'data')

AVAILABLE_DATA = data.AVAILABLE_DATA

STGS_BASE = ['bh', 'random']

STGS_ALGO = ['vanilla', 'relesa']

ALLOWED_INPUT_PARAMS = {'mode': ['single', 'routine', 'consolidation'],
                        'frequency': ['hour'],
                        'stg': STGS_BASE + STGS_ALGO,
                        'algo': ['a2c', 'dqn'],
                        'action_type': ['discrete', 'continuous'],
                        'reward_function': ['return', 'sharpe_ratio'],
                        'setup': ['static', 'rolling']
                        }

DEFAULT_ASSETS = ['aapl']

USE_GPU = False
