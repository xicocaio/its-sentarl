import os
import pandas as pd

import settings


def get_standard_result_filename(*args) -> str:
    return '_'.join(str(arg) for arg in args if arg)


def load_dataset(name, index_name):
    path = os.path.join(settings.DATA_DIR, name + '.csv')
    return pd.read_csv(path, index_col=index_name)


def split_data(df, test_ratio, window_size, val_ratio=None):
    train_ratio = 1 - test_ratio
    idx_mark = int(train_ratio * len(df.index))
    df_train, df_test = df.iloc[:idx_mark], df.iloc[idx_mark - window_size:]  # ajust to window size

    df_val = df_test
    if val_ratio:
        val_factor = val_ratio / train_ratio  # val_factor x train_ratio = val_ratio
        train_factor = 1 - val_factor
        idx_mark = int(train_factor * idx_mark)
        df_train, df_val = df_train.iloc[:idx_mark], df_train.iloc[idx_mark - window_size:]  # ajust to window size

    return df_train, df_test, df_val
