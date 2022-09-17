# internal modules
import settings


class Config:
    def __init__(
        self,
        name="default",
        stg="sentarl",
        algo="a2c",
        asset="aapl",
        episodes=1,
        action_type="discrete",
        reward_type="additive",
        seed=42,
        initial_wealth=1.0,
        transaction_cost=0.0025,
        frequency="hour",
        reward_function="return",
        test_data_ratio=0.1,
        val_data_ratio=0.1,
        setup="static",
        deterministic_test=False,
        test_runs=10,
        sb_verbose=True,
        ep_verbose=True,
    ):
        self._name = name
        self._stg = stg
        self._algo = algo if self.stg not in settings.STGS_BASE else None
        self._asset = asset
        self._episodes = episodes if self.stg not in settings.STGS_BASE else 1
        self._action_type = action_type
        self._reward_type = reward_type
        self._seed = seed
        self._initial_wealth = initial_wealth
        self._transaction_cost = transaction_cost
        self._frequency = frequency
        self._reward_function = (
            reward_function
            if self.stg not in settings.STGS_BASE
            else settings.REWARD_FUNCTIONS[0]
        )
        self._test_data_ratio = test_data_ratio
        self._val_data_ratio = val_data_ratio
        self._setup = setup
        self._test_runs = test_runs
        self._sb_verbose = sb_verbose
        self._ep_verbose = ep_verbose

        self._device = "cpu" if not settings.USE_GPU else "cuda"

        # forcing to deterministic_test to True if setup type is rolling window
        self._deterministic_test = (
            True if self.setup == "rolling" else deterministic_test
        )

    @property
    def name(self):
        return self._name

    @property
    def stg(self):
        return self._stg

    @property
    def algo(self):
        return self._algo

    @property
    def asset(self):
        return self._asset

    @property
    def episodes(self):
        return self._episodes

    @property
    def action_type(self):
        return self._action_type

    @property
    def reward_type(self):
        return self._reward_type

    @property
    def seed(self):
        return self._seed

    @property
    def initial_wealth(self):
        return self._initial_wealth

    @property
    def transaction_cost(self):
        return self._transaction_cost

    @property
    def frequency(self):
        return self._frequency

    @property
    def reward_function(self):
        return self._reward_function

    @property
    def test_data_ratio(self):
        return self._test_data_ratio

    @property
    def val_data_ratio(self):
        return self._val_data_ratio

    @property
    def setup(self):
        return self._setup

    @property
    def test_runs(self):
        return self._test_runs

    @property
    def sb_verbose(self):
        return self._sb_verbose

    @property
    def ep_verbose(self):
        return self._ep_verbose

    @property
    def device(self):
        return self._device

    @property
    def deterministic_test(self):
        return self._deterministic_test
