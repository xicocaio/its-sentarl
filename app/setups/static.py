# internal imports
import settings
from .base_setup import BaseSetup
from results import CSVOutput
from common import load_dataset, split_data
from models import train_model


class StaticSetup(BaseSetup):
    def __init__(self, config):
        super(StaticSetup, self).__init__(config=config)

        df_train, df_test, df_val = split_data(self.df, self.test_data_ratio, self.pivot_window_size,
                                               self.val_data_ratio)

        frame_bound_train = (self.pivot_window_size, len(df_train.index))
        frame_bound_val = (self.pivot_window_size, len(df_val.index))
        frame_bound_test = (self.pivot_window_size, len(df_test.index))

        monitored = False
        if self.stg in settings.STGS_ALGO:
            self.total_timesteps = self.episodes * (frame_bound_train[1] - frame_bound_train[0])
            print('Total training timesteps: {}'.format(self.total_timesteps))
            monitored = True

        self.env_train = self._prepare_env(df_train, frame_bound_train, 'train', monitored=monitored)
        self.env_val = self._prepare_env(df_val, frame_bound_val, 'val', monitored=monitored)
        self.env_test = self._prepare_env(df_test, frame_bound_test, 'test', monitored=False)

    def _run_window(self, window, model=None):
        test_runs = self.test_runs if self.stg != 'bh' and window == 'test' and not self.deterministic_test else 1

        if window == 'train':
            env = self.env_train
        elif window == 'val':
            env = self.env_val
        elif window == 'test':
            env = self.env_test

        csv_output = CSVOutput(config=self.config, window=window)
        base_result_values = self._prepare_base_result_values(window)

        for k in range(test_runs):
            observation = env.reset()

            while True:
                action = self._get_stg_action(env, observation, model)
                observation, reward, done, info = env.step(action)

                base_result_values['train_episodes'] = self.episodes
                base_result_values['window_step'] = k
                base_result_values['reward'] = reward

                csv_output.write({**base_result_values, **info})

                if done:
                    print("info:", info)
                    break

        csv_output.close()

    def run(self):
        if self.stg in settings.STGS_BASE:
            for window in ['train', 'val', 'test']:
                self._run_window(window)
        else:
            model = train_model(self.algo, self.env_train, self.device, self.total_timesteps, self.env_val,
                                self.episodes, self.seed)

            self._run_window('test', model)
