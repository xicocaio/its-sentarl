import os
import pandas as pd
from common import prepare_folder_structure

ABS_RESULT_PATH = os.path.dirname(os.path.abspath(__file__))
FREQUENCIES = ["hour"]
SETUP_TYPES = ["rolling"]
ASSETS = ["aapl"]
TCS = [str(0.0025)]
STGS = ["bh", "vanilla", "sentarl"]
WINDOW_TYPES = ["train", "val", "test"]


# TODO check for duplicates if using appending and not starting from zero


def _get_src_path(over_test=True):
    src_path = os.path.join(ABS_RESULT_PATH, FREQUENCIES[0], SETUP_TYPES[0])

    return src_path


def singlefile_consolidation(exclude=[]):
    dest_path = os.path.join(ABS_RESULT_PATH, "consolidated")
    prepare_folder_structure(dest_path)

    abs_src_filepath = _get_src_path()

    dfs_train_val = []
    dfs_test = []
    for root, dirs, files in os.walk(abs_src_filepath):
        for file in files:
            abs_file = os.path.join(root, file)

            if file not in exclude and file.endswith(("train.csv", "val.csv")):
                dfs_train_val.append(pd.read_csv(abs_file, sep=";"))
            elif file.endswith("test.csv"):
                dfs_test.append(pd.read_csv(abs_file, sep=";"))

    if dfs_train_val:
        dest_fname = "final_experiments_1_asset_train_val"
        pd.concat(dfs_train_val).to_csv(
            os.path.join(dest_path, dest_fname + ".csv"), index=False, sep=";"
        )

    if dfs_test:
        dest_fname = "final_experiments_1_asset_test"
        pd.concat(dfs_test).to_csv(
            os.path.join(dest_path, dest_fname + ".csv"), index=False, sep=";"
        )
