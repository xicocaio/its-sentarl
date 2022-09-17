# Built-in imports
import logging
import logging.config

# Data related imports
import numpy as np

# RL imports
import gym

# internal imports
import settings
from common import load_dataset
from results import ResultsMonitor
from stable_baselines3.common.env_checker import check_env
from envs import (  # noqa F401
    StockExchangeEnv,
)  # do not remove this import or env make will break
from common import Config


logging.config.dictConfig(settings.LOGGING_CONFIG)


class BaseSetup(object):
    def __init__(self, config: Config):
        self.config = config

        # time window of 5 for sent seems better at first than 10
        self.pivot_window_size = 20

        self.features = [
            ("diff", self.pivot_window_size),
            ("hour_of_day_relative", self.pivot_window_size),
        ]

        if self.config.stg == "sentarl":
            # self.features.extend([('avg-sent', 3),
            #                       ('max-sent', 3)])
            # self.features.extend([('min-sent', 5), ('news_count_div', 5)])
            self.features.extend([("min-sent", 5)])

        # self.features = [
        #     ("hour_of_day", self.pivot_window_size),
        #     ("day_of_week", self.pivot_window_size),
        #     ("Open", self.pivot_window_size),
        #     ("High", self.pivot_window_size),
        #     ("Low", self.pivot_window_size),
        #     ("Close", self.pivot_window_size),
        # ]

        self.df = load_dataset(
            settings.AVAILABLE_DATA[self.config.asset]["fname"],
            settings.AVAILABLE_DATA[self.config.asset]["index_col"],
        )

        self.window_types = ["train", "val", "test"]

    def run(self):
        return NotImplementedError

    def _get_stg_action(self, env, observation, model=None):
        if self.config.stg == "random":
            action = env.action_space.sample()
        elif self.config.stg == "bh":
            action = (
                env.action_space.n - 1
                if self.config.action_type == "discrete"
                else env.action_space.high
            )
        else:
            action, _states = model.predict(
                observation, deterministic=self.config.deterministic_test
            )

        return action

    def _env_maker(self, df, frame_bound):
        return lambda: gym.make(
            "stock_exchange-v0",
            df=df,
            frame_bound=frame_bound,
            pivot_window_size=self.pivot_window_size,
            features=self.features,
            action_type=self.config.action_type,
            reward_type=self.config.reward_type,
            reward_function=self.config.reward_function,
            initial_wealth=self.config.initial_wealth,
            transaction_cost=self.config.transaction_cost,
        )

    def prep_data(self, df):
        df["diff"] = np.insert(np.diff(df["Close"].to_numpy()), 0, 0)
        df["hour_of_day_relative"] = df["hour_of_day"] / 24
        df["day_of_week_relative"] = df["day_of_week"] / 7
        df["close_norm"] = (
            2
            * (df["Close"] - df["Close"].min())
            / (df["Close"].max() - df["Close"].min())
            - 1
        )
        df["news_count_div"] = df["news-count"] / 10

    def _prepare_env(
        self, df, frame_bound, window, additional_info={}, overwrite_file=None
    ):
        self.prep_data(df)
        env_maker = self._env_maker(df, frame_bound)
        check_env(env_maker())

        return ResultsMonitor(
            env_maker(),
            config=self.config,
            window=window,
            overwrite_file=overwrite_file,
            additional_info=additional_info,
        )
