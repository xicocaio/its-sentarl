import os
from pathlib import Path
from typing import Dict, Tuple, Union, List
import csv

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


def load_dataset(name, index_name):
    path = os.path.join(settings.DATA_DIR, name + '.csv')
    return pd.read_csv(path, index_col=index_name)


def split_data(df, test_ratio, window_size, val_ratio=None):
    train_ratio = 1 - test_ratio
    idx_mark = int(train_ratio * len(df.index))
    df_train, df_test = df.iloc[:idx_mark], df.iloc[idx_mark - window_size:]  # adjust to window size

    df_val = df_test
    if val_ratio:
        val_factor = val_ratio / train_ratio  # val_factor x train_ratio = val_ratio
        train_factor = 1 - val_factor
        idx_mark = int(train_factor * idx_mark)
        df_train, df_val = df_train.iloc[:idx_mark], df_train.iloc[idx_mark - window_size:]  # adjust to window size

    return df_train, df_test, df_val


def prepare_folder_structure(folder_path):
    Path(folder_path).mkdir(parents=True, exist_ok=True)


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

        self.file = open(os.path.join(abs_filename + '.csv'), mode)

        self.csv_writer = csv.DictWriter(self.file, delimiter=delimiter, fieldnames=fieldnames, extrasaction='ignore')
        self.csv_writer.writeheader()

    def write(self, row: Dict[str, Union[str, Tuple[str, ...]]]) -> None:
        """
        writes to  file
        """
        self.csv_writer.writerow(row)

    def close(self) -> None:
        """
        closes the file
        """
        self.file.close()
