import csv
from typing import Any, Dict, Tuple, Union

import os
from pathlib import Path

import settings

from stable_baselines3.common.monitor import Monitor

BASE_FILENAME_FIELDS = ['asset', 'stg']
FOLDER_LEVELS = ('frequency', 'setup', 'asset', 'transaction_cost', 'stg')
ABS_RESULT_PATH = os.path.dirname(os.path.abspath(__file__))
RESULTS_HEADER = ['window_step', 'current_step', 'date', 'asset', 'company_name', 'frequency', 'set', 'initial_wealth',
                  'transaction_cost', 'stg', 'algo', 'action_type', 'reward_type', 'reward_function', 'seed',
                  'train_episodes', 'action_value', 'reward', 'total_reward', 'total_return']


def get_filename(config, window):
    config_dict = vars(config)
    filename_fields = BASE_FILENAME_FIELDS.copy()

    if config_dict['stg'] in settings.STGS_ALGO:
        filename_fields.extend(['algo', 'episodes'])
    if config_dict['stg'] not in settings.STGS_BASE:
        filename_fields.extend(['seed'])

    # preparing filename
    fields = [[field, str(config_dict[field])] for field in filename_fields]
    fields.append([window])

    return '-'.join('_'.join(field) for field in fields)


def get_dest_path(config):
    config_dict = vars(config)
    if not all(k in config_dict for k in FOLDER_LEVELS):
        raise ValueError('Some of the following entries not found in config: ', FOLDER_LEVELS)

    return os.path.join(ABS_RESULT_PATH, *[str(config_dict[k]) for k in FOLDER_LEVELS])


def prepare_folder_structure(folder_path):
    Path(folder_path).mkdir(parents=True, exist_ok=True)


class CSVOutput(object):
    """
    log to a file, in a CSV format
    """

    def __init__(self,
                 config: Dict[Any, Any],
                 window: str,
                 overwrite_file: bool = True,
                 delimiter: str = ';'):
        dest_path = get_dest_path(config)
        prepare_folder_structure(dest_path)
        filename = get_filename(config, window)

        abs_path = os.path.join(dest_path, filename + '.csv')

        mode = 'w' if overwrite_file else 'a'  # use 'w+' or 'a+' if also required to read file

        self.file = open(abs_path, mode)

        self.csv_writer = csv.DictWriter(self.file, delimiter=delimiter, fieldnames=RESULTS_HEADER)
        self.csv_writer.writeheader()
        # self.keys = []

    def write(self, result: Dict[str, Union[str, Tuple[str, ...]]]) -> None:
        """
        writes to  file
        """
        self.csv_writer.writerow(result)

    def close(self) -> None:
        """
        closes the file
        """
        self.file.close()


class ResultsMonitor(Monitor):

    def __init__(self, env, info_keywords, config, window):
        rel_dest_path = get_dest_path(config)
        prepare_folder_structure(rel_dest_path)
        filename = get_filename(config, window)

        abs_path = os.path.join(ABS_RESULT_PATH, rel_dest_path, filename)

        super(ResultsMonitor, self).__init__(env=env, filename=abs_path, info_keywords=info_keywords)
