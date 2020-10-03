import os

import numpy as np

from stable_baselines3.common.callbacks import BaseCallback


class CSVCallback(BaseCallback):
    """
    Custom callback for plotting additional values to a CSV.

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, verbose=0):
        super(CSVCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value = np.random.random()
        self.logger.record('random_value', value)
        return True
