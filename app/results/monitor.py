import os
import time
from typing import Any, Dict, Tuple

# RL imports
import gym
import numpy as np

# internal imports
import settings
from common import (
    Config,
    CSVOutput,
    prepare_folder_structure,
    generate_filename,
)

BASE_FILENAME_FIELDS = settings.BASE_FILENAME_FIELDS
ABS_RESULT_PATH = os.path.dirname(os.path.abspath(__file__))

BASE_HEADER = [
    "config",
    "asset",
    "company_name",
    "frequency",
    "set",
    "initial_wealth",
    "transaction_cost",
    "stg",
    "algo",
    "action_type",
    "reward_type",
    "reward_function",
    "seed",
    "total_reward",
    "total_return",
    "sharpe_ratio",
]

TRAIN_RESULTS_HEADER = [
    "window_roll",
    "current_episode",
    "total_steps",
    "start_date",
    "end_date",
] + BASE_HEADER

TEST_RESULTS_HEADER = (
    ["window_roll", "test_run", "current_step", "date", "train_episodes"]
    + BASE_HEADER
    + ["action_value", "reward", "step_return"]
)


class ResultsMonitor(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        config: Config,
        window: str,
        overwrite_file: bool = True,
        additional_info: Dict = {},
    ):
        super(ResultsMonitor, self).__init__(env=env)
        self.config = config
        self.window = window

        filename = generate_filename(config, window)
        dest_path = prepare_folder_structure(ABS_RESULT_PATH, config)
        abs_filename = os.path.join(dest_path, filename)

        self.t_start = time.time()

        additional_info["set"] = self.window

        fieldnames = TEST_RESULTS_HEADER
        if self.window in ["train", "val"]:
            fieldnames = TRAIN_RESULTS_HEADER
            additional_info["total_steps"] = self.max_episode_steps
            additional_info["start_date"] = self.start_date
            additional_info["end_date"] = self.end_date

        self.base_result_values = self._get_base_result_values(additional_info)

        self.csv_output = CSVOutput(
            config=self.config,
            fieldnames=fieldnames,
            abs_filename=abs_filename,
            overwrite_file=overwrite_file,
        )

        self.rewards = None
        self.needs_reset = True
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0
        self.current_episode = 0

    def reset(self, **kwargs) -> np.ndarray:
        """
        Calls the Gym environment reset.

        :return: (np.ndarray) the first observation of the environment
        """
        self.rewards = []
        self.needs_reset = False

        return self.env.reset(**kwargs)

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        """
        Step the environment with the given action

        :param action: (np.ndarray) the action
        :return: (Tuple[np.ndarray, float, bool, Dict[Any, Any]]) observation,
            reward, done, information
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")

        observation, reward, done, info = self.env.step(action)
        self.rewards.append(reward)
        self.total_steps += 1

        result_values = dict()

        if self.window == "test":
            result_values["current_step"] = self.total_steps
            result_values["reward"] = reward
            result_values["test_run"] = self.current_episode
            self.csv_output.write(
                {**self.base_result_values, **result_values, **info}
            )

        if done:
            self.current_episode += 1
            if self.window in ["train", "val"]:
                self.needs_reset = True
                ep_len = len(self.rewards)
                ep_info = {
                    "ep_length": ep_len,
                    "time_elapsed": round(time.time() - self.t_start, 6),
                }
                for key in info:
                    ep_info[key] = info[key]
                self.episode_lengths.append(ep_len)
                self.episode_times.append(time.time() - self.t_start)

                if self.csv_output:
                    result_values = dict(
                        {"current_episode": self.current_episode}
                    )

                    self.csv_output.write(
                        {**self.base_result_values, **ep_info, **result_values}
                    )

        return observation, reward, done, info

    def close(self):
        """
        Closes the environment
        """
        super(ResultsMonitor, self).close()
        if self.csv_output is not None:
            self.csv_output.close()

    def _get_base_result_values(self, additional_info: Dict = {}):
        base_result_values = {
            "asset": self.config.asset,
            "company_name": settings.AVAILABLE_DATA[self.config.asset]["name"],
            "action_type": self.config.action_type,
            "reward_type": self.config.reward_type,
            "stg": self.config.stg,
            "algo": self.config.algo,
            "initial_wealth": self.config.initial_wealth,
            "transaction_cost": self.config.transaction_cost,
            "frequency": self.config.frequency,
            "reward_function": self.config.reward_function,
        }

        return {**base_result_values, **additional_info}
