# internal modules
import settings


class Config:
    def __init__(self,
                 name='default',
                 stg='relesa',
                 algo='a2c',
                 asset='aapl',
                 episodes=100,
                 action_type='discrete',
                 reward_type='additive',
                 initial_wealth=1.0,
                 transaction_cost=0.0025,
                 frequency='hour',
                 reward_function='return',
                 test_data_ratio=0.1,
                 val_data_ratio=0.1,
                 setup='static',
                 deterministic_test=False,
                 test_runs=10,
                 sb_verbose=True,
                 ep_verbose=True):
        self.name = name
        self.stg = stg
        self.algo = algo if self.stg not in settings.STGS_BASE else None
        self.asset = asset
        self.episodes = episodes if self.stg not in settings.STGS_BASE else 1
        self.action_type = action_type
        self.reward_type = reward_type
        self.initial_wealth = initial_wealth
        self.transaction_cost = transaction_cost
        self.frequency = frequency
        self.reward_function = reward_function if self.stg not in settings.STGS_BASE else settings.REWARD_FUNCTIONS[0]
        self.test_data_ratio = test_data_ratio
        self.val_data_ratio = val_data_ratio
        self.setup = setup
        self.test_runs = test_runs
        self.sb_verbose = sb_verbose
        self.ep_verbose = ep_verbose

        self.device = 'cpu' if not settings.USE_GPU else 'cuda'
        self.seed = 42

        # forcing to deterministic_test to True if setup type is rolling window
        self.deterministic_test = True if self.setup == 'rolling' else deterministic_test