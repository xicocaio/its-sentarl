import os
from pathlib import Path
from typing import Dict, Tuple, Union, List
import csv
import inspect

import pandas as pd

# internal imports
import settings
from .config import Config

_color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


def generate_filename(config: Config, window: str, filetype: str = 'result'):
    config_dict = dict(inspect.getmembers(config))
    filename_fields = settings.BASE_FILENAME_FIELDS.copy()

    if config_dict['stg'] in settings.STGS_ALGO:
        config_dict['config'] = config_dict['name']  # adjusting config name field for filename
        ending_fields = ['algo', 'episodes'] if filetype == 'result' else ['algo']
        filename_fields = ['config'] + filename_fields + ending_fields  # adjusting order of fields
    if config_dict['stg'] not in settings.STGS_BASE:
        filename_fields.extend(['reward_function', 'seed'])

    # preparing filename
    fields = [[field, str(config_dict[field])] for field in filename_fields]
    fields.append([window])

    return '-'.join('_'.join(field) for field in fields)


def prepare_folder_structure(abs_origin_path, config: Config = None, foldertype: str = 'result', seed: int = 0,
                             window_roll: int = 0):
    folder_path = abs_origin_path

    if config:
        folder_list = dict(inspect.getmembers(config))
        folder_levels = settings.FOLDER_LEVELS_RESULTS
        if foldertype == 'model':
            folder_levels = settings.FOLDER_LEVELS_MODELS
            folder_list['window_roll'] = 'roll_' + str(window_roll)
            folder_list['seed'] = 'seed_' + str(seed)

        if not all(k in folder_list for k in folder_levels):
            raise ValueError('Some of the following entries not found in config: ', folder_levels)

        folder_path = os.path.join(abs_origin_path, *[str(folder_list[k]) for k in folder_levels])

    Path(folder_path).mkdir(parents=True, exist_ok=True)

    return folder_path


def load_dataset(name, index_name):
    path = os.path.join(settings.DATA_DIR, name + '.csv')
    return pd.read_csv(path, index_col=index_name)


def split_data(df, window_size, test_size, val_size=0, use_ratio=True):
    """
    Function for spliting data
    @param df: data to split
    @param window_size: lookback window size
    @param test_size: can be either a specific size or a ratio
    @param val_size: can be either a specific size or a ratio
    @param use_ratio: determines whether to treat test and val sizes as ratios
    @return: In sequence returns dfs for train, test and validation
    """
    expected_type = float if use_ratio else int
    if not isinstance(test_size, expected_type) or (val_size and not isinstance(val_size, expected_type)):
        raise ValueError('use_ratio={} while size args are of type {}'.format(use_ratio, expected_type))

    df_size = len(df.index)

    train_size = 1 - test_size if use_ratio else df_size - test_size
    idx_mark = int(train_size * df_size) if use_ratio else train_size
    df_train, df_test = df.iloc[:idx_mark], df.iloc[idx_mark - window_size:]  # adjust to window size

    # if no val_size is passed, df_val should be equal to df_test
    df_val = df_test
    if val_size > 0:
        if use_ratio:
            val_factor = val_size / train_size  # val_factor x train_ratio = val_ratio
            train_factor = 1 - val_factor
        idx_mark = int(train_factor * idx_mark) if use_ratio else idx_mark - val_size
        df_train, df_val = df_train.iloc[:idx_mark], df_train.iloc[idx_mark - window_size:]  # adjust to window size

    return df_train, df_test, df_val


# From open ai gym source code
def colorize(string, color, bold=False, highlight=False):
    """Return string surrounded by appropriate terminal color codes to
    print colorized text.  Valid colors: gray, red, green, yellow,
    blue, magenta, cyan, white, crimson
    """

    attr = []
    num = _color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))

    if bold:
        attr.append('1')
    attrs = ';'.join(attr)
    return '\x1b[%sm%s\x1b[0m' % (attrs, string)


class CSVOutput(object):
    """
    log to a file, in a CSV format
    """

    def __init__(self,
                 config: Config,
                 fieldnames: List,
                 abs_filename: str,
                 overwrite_file: bool = True,
                 delimiter: str = ';'):
        self.config = config

        mode = 'w' if overwrite_file else 'a'  # use 'w+' or 'a+' if also required to read file
        self.file_handler = open(os.path.join(abs_filename + '.csv'), mode)

        self.csv_writer = csv.DictWriter(self.file_handler, delimiter=delimiter, fieldnames=fieldnames,
                                         extrasaction='ignore')

        # only write header if file is being created
        if overwrite_file:
            self.csv_writer.writeheader()
            self.file_handler.flush()

    def write(self, row: Dict[str, Union[str, Tuple[str, ...]]]) -> None:
        """
        writes to  file
        """
        self.csv_writer.writerow(row)
        self.file_handler.flush()

    def close(self) -> None:
        """
        closes the file
        """
        self.file_handler.close()
