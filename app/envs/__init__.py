# Internal imports
from .stock import StockExchangeEnv  # noqa F401
import settings
from common import load_dataset

# required imports for proper initialization of env
from gym.envs.registration import register

# Default Env Settings
DEFAULT_ENV = {
    "asset": "aapl",
    "pivot_price_feature": "Close",
    "pivot_window_size": 20,
    "features": [("avg-sent", 5)],
    "action_window_size": 1,
    "action_type": "discrete",
    "reward_type": "additive",
    "reward_function": "return",
    "initial_wealth": 1.0,
    "transaction_cost": 0.0025,
}


def _load_default_env_settings():
    default_kwargs = DEFAULT_ENV.copy()
    asset = default_kwargs.pop("asset")

    df = load_dataset(
        settings.AVAILABLE_DATA[asset]["fname"],
        settings.AVAILABLE_DATA[asset]["index_col"],
    )
    default_kwargs["df"] = df
    default_kwargs["frame_bound"] = (
        default_kwargs["pivot_window_size"],
        len(df.index),
    )

    return default_kwargs


# Register of default envs, override of args occurs
# if other args are passed during initialization

register(
    id="stock_exchange-v0",
    entry_point="envs:StockExchangeEnv",
    kwargs=_load_default_env_settings(),
)
