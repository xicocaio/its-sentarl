import os
import pandas as pd

import settings

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
