# RL imports
import gym

# internal imports
import settings
from common import load_dataset
from results import ResultsMonitor
from stable_baselines3.common.env_checker import check_env
from envs import StockExchangeEnv  # do not remove this import or env make will break


class BaseSetup(object):
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

    def _prepare_env(self, df, frame_bound, window, additional_info={}):
        env_maker = self._env_maker(df, frame_bound)
        check_env(env_maker())

        return ResultsMonitor(env_maker(), config=self.config, window=window, additional_info=additional_info)
