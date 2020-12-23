# internal imports
import settings
from .base_setup import BaseSetup
from common import CSVOutput, load_dataset, split_data
from models import train_model
import itertools


class RollingWindowSetup(BaseSetup):
    def __init__(self, config):
        super(RollingWindowSetup, self).__init__(config=config)

        self.k_rolls = 5

        self.train_size, self.test_size, self.val_size = self._get_dataset_sizes()
        self.dfs = self._get_dfs()

        if self.stg in settings.STGS_ALGO:
            self.total_timesteps = self.episodes * (self.train_size - self.pivot_window_size)
            print('Total training timesteps: {}'.format(self.total_timesteps))

    def run(self):
        for roll, df_item in enumerate(self.dfs):
            additional_info = dict({'window_roll': roll,
                                    'seed': self.seed if self.stg in settings.STGS_ALGO else None,
                                    'config': self.config.name if self.stg in settings.STGS_ALGO else None,
                                    'train_episodes': self.episodes})
            overwrite_file = True if roll == 0 else False

            envs = {}
            for window in self.window_types:
                frame_bound = (self.pivot_window_size, len(df_item[window].index))
                envs[window] = self._prepare_env(df_item[window], frame_bound, window, additional_info=additional_info,
                                                 overwrite_file=overwrite_file)

            if self.stg in settings.STGS_BASE:
                for window in self.window_types:
                    self._run_window(window, envs[window])
            else:
                model = train_model(self.algo, envs['train'], self.device, self.total_timesteps, envs['val'],
                                    self.episodes, self.seed, self.sb_verbose)

                self._run_window('test', envs['test'], model)

    def _run_window(self, window, env, model=None):
        # test_runs = self.test_runs if self.stg != 'bh' and window == 'test' and not self.deterministic_test else 1

        test_runs = 1

        for k in range(test_runs):
            observation = env.reset()

            while True:
                action = self._get_stg_action(env, observation, model)
                observation, reward, done, info = env.step(action)

                if done:
                    if self.ep_verbose:
                        print("info:", info)
                    break

        env.close()

    def _get_dataset_sizes(self, min_train_size=2000, min_train_ratio=0.8):
        total_size = len(self.df.index)

        for i in itertools.count(start=min_train_size):
            # adjusting for min_train_size despite window size
            train_size = i
            train_size_adjusted = train_size + self.pivot_window_size

            total_size_available = total_size - train_size_adjusted

            # simple offset to k_rolls to take into account the validation set as it is the same size as test
            k_rolls_offset = self.k_rolls + 1

            test_size = total_size_available / k_rolls_offset
            val_size = test_size
            train_ratio = train_size / (train_size + val_size + test_size)

            # test and val sizes must be an exact whole number and
            # train_ratio (and not train_size_adjusted) must satisfy min_train_ratio
            if test_size.is_integer() and train_ratio >= min_train_ratio:
                break

        return train_size_adjusted, int(test_size), int(val_size)

    def _get_dfs(self):
        dfs = []
        df_remaining = self.df.copy()

        adopted_size = self.train_size + 2 * self.test_size
        roll_size = self.test_size
        for k in range(self.k_rolls):
            df_roll = df_remaining.iloc[:adopted_size]
            df_remaining = df_remaining.iloc[roll_size:]

            df_train, df_test, df_val = split_data(df_roll, self.pivot_window_size, self.test_size, self.val_size,
                                                   False)

            dfs.append({'train': df_train, 'val': df_val, 'test': df_test})

            # roll_size

        return dfs