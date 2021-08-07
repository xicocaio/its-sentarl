# internal imports
import settings
from .base_setup import BaseSetup
from common import CSVOutput, load_dataset, split_data
from models import train_model


class StaticSetup(BaseSetup):
    def __init__(self, config):
        super(StaticSetup, self).__init__(config=config)

        df_train, df_test, df_val = split_data(self.df, self.pivot_window_size, self.test_data_ratio,
                                               self.val_data_ratio, True)

        frame_bound_train = (self.pivot_window_size, len(df_train.index))
        frame_bound_val = (self.pivot_window_size, len(df_val.index))
        frame_bound_test = (self.pivot_window_size, len(df_test.index))

        if self.stg in settings.STGS_ALGO:
            self.total_timesteps = self.episodes * (frame_bound_train[1] - frame_bound_train[0])
            print('Total training timesteps: {}'.format(self.total_timesteps))

        additional_info = dict({'window_roll': 0,
                                'seed': self.seed if self.stg in settings.STGS_ALGO else None,
                                'config': self.config.name if self.stg in settings.STGS_ALGO else None,
                                'train_episodes': self.episodes})

        self.env_train = self._prepare_env(df_train, frame_bound_train, 'train', additional_info=additional_info)
        self.env_val = self._prepare_env(df_val, frame_bound_val, 'val', additional_info=additional_info)
        self.env_test = self._prepare_env(df_test, frame_bound_test, 'test', additional_info=additional_info)

    def _run_window(self, window, model=None):
        test_runs = self.test_runs if self.stg != 'bh' and window == 'test' and not self.deterministic_test else 1

        if window == 'train':
            env = self.env_train
        elif window == 'val':
            env = self.env_val
        else:
            env = self.env_test

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

    def run(self):
        if self.stg in settings.STGS_BASE:
            for window in self.window_types:
                self._run_window(window)
        else:
            model = train_model(self.algo, self.env_train, self.device, self.total_timesteps, self.env_val,
                                self.episodes, self.seed, self.sb_verbose)

            self._run_window('test', model)
