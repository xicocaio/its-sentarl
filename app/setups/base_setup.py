# RL imports
import gym

from stable_baselines3.common.env_checker import check_env

# internal imports
import settings
from envs import StockExchangeEnv
from results import CSVOutput, ResultsMonitor
from common import load_dataset


class BaseSetup:
    def __init__(self, config):
        self.config = config

        # TODO: consider removing this and just use config directly
        for key, value in vars(self.config).items():
            setattr(self, key, value)

        self.features = [('avg-sent', 10)] if self.stg == 'relesa' else []

        self.pivot_window_size = 20
        self.df = load_dataset(settings.AVAILABLE_DATA[self.asset]['fname'],
                               settings.AVAILABLE_DATA[self.asset]['index_col'])

    def _get_stg_action(self, env, observation, model=None):
        if self.stg == 'random':
            action = env.action_space.sample()
        elif self.stg == 'bh':
            action = env.action_space.n - 1 if self.action_type == 'discrete' else env.action_space.high
        else:
            action, _states = model.predict(observation, deterministic=self.deterministic_test)

        return action

    def _prepare_base_result_values(self, window):
        return {'asset': self.asset,
                'company_name': settings.AVAILABLE_DATA[self.asset]['name'],
                'action_type': self.action_type,
                'reward_type': self.reward_type,
                'stg': self.stg,
                'algo': self.algo,
                'initial_wealth': self.config.initial_wealth,
                'transaction_cost': self.config.transaction_cost,
                'frequency': self.frequency,
                'reward_function': self.config.reward_function,
                'set': window}

    def _env_maker(self, df, frame_bound):
        return lambda: gym.make('stock_exchange-v0',
                                df=df,
                                frame_bound=frame_bound,
                                pivot_window_size=self.pivot_window_size,
                                features=self.features,
                                action_type=self.action_type,
                                reward_type=self.reward_type,
                                initial_wealth=self.initial_wealth,
                                transaction_cost=self.transaction_cost)

    def _prepare_env(self, df, frame_bound, window, monitored=False):
        env_maker = self._env_maker(df, frame_bound)
        check_env(env_maker())

        if monitored:
            return ResultsMonitor(env_maker(), info_keywords=("total_return",), config=self.config, window=window)

        return env_maker()
