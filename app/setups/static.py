# Built-in imports
import logging

# internal imports
import settings
from .base_setup import BaseSetup
from common import split_data
from models import train_model
from common import Config


class StaticSetup(BaseSetup):
    def __init__(self, config: Config):
        super(StaticSetup, self).__init__(config=config)
        self.logger = logging.getLogger(__name__)

        df_train, df_test, df_val = split_data(
            self.df,
            self.pivot_window_size,
            self.config.test_data_ratio,
            self.config.val_data_ratio,
            True,
        )

        frame_bound_train = (self.pivot_window_size, len(df_train.index))
        frame_bound_val = (self.pivot_window_size, len(df_val.index))
        frame_bound_test = (self.pivot_window_size, len(df_test.index))

        if self.config.stg in settings.STGS_ALGO:
            self.total_timesteps = self.config.episodes * (
                frame_bound_train[1] - frame_bound_train[0]
            )
            self.logger.info(
                f"Total training timesteps: {self.total_timesteps}"
            )

        additional_info = dict(
            {
                "window_roll": 0,
                "seed": self.config.seed
                if self.config.stg in settings.STGS_ALGO
                else None,
                "config": self.config.name
                if self.config.stg in settings.STGS_ALGO
                else None,
                "train_episodes": self.config.episodes,
            }
        )

        self.env_train = self._prepare_env(
            df_train,
            frame_bound_train,
            "train",
            additional_info=additional_info,
        )
        self.env_val = self._prepare_env(
            df_val, frame_bound_val, "val", additional_info=additional_info
        )
        self.env_test = self._prepare_env(
            df_test, frame_bound_test, "test", additional_info=additional_info
        )

    def _run_window(self, window: str, model: object = None):
        """run each type of window appropriately
        Parameters
        ----------
        window: str
            Type of window being tested, this impacts the output file
        model: object
            Model if applicable, to use for running
        Returns:
        ----------
        """
        test_runs = (
            self.config.test_runs
            if self.config.stg != "bh"
            and window == "test"
            and not self.config.deterministic_test
            else 1
        )

        if window == "train":
            env = self.env_train
        elif window == "val":
            env = self.env_val
        else:
            env = self.env_test

        for _ in range(test_runs):
            observation = env.reset()

            while True:
                action = self._get_stg_action(env, observation, model)
                observation, _, done, info = env.step(action)

                if done:
                    if self.config.ep_verbose:
                        self.logger.info(f"{info}")
                    break

        env.close()

    def run(self):
        """Checks if a base strategies that do not use a model
            or a algo strategy and runs accordingly
        Parameters
        ----------
        Returns:
        ----------
        """
        if self.config.stg in settings.STGS_BASE:
            for window in self.window_types:
                self._run_window(window)
        else:
            model = train_model(
                self.env_train,
                self.total_timesteps,
                self.env_val,
                self.config,
                save_model=False,
            )

            self._run_window("test", model)
