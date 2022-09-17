# Built-in imports
import sys
import ast
import logging
import logging.config

# internal imports
import settings
from common import Config
from setups import StaticSetup, RollingWindowSetup
from results import singlefile_consolidation
from routines import Routine

logging.dictConfig(settings.LOGGING_CONFIG)

NUMERICAL_INPUT_KEYS = [
    "episodes",
    "initial_wealth",
    "transaction_cost",
    "test_data_ratio",
    "val_data_ratio",
]


def verify_allowed_input(inputs):
    for key, value in inputs.items():
        if (
            key in settings.ALLOWED_INPUT_PARAMS
            and value not in settings.ALLOWED_INPUT_PARAMS[key]
        ):
            raise ValueError(
                "`{}` entry not found for key `{}`".format(value, key)
            )


def main(**kwargs):
    logger = logging.getLogger(__name__)

    adjusted_kwargs = kwargs.copy()
    for k, v in kwargs.items():
        adjusted_kwargs[k] = (
            ast.literal_eval(v) if k in NUMERICAL_INPUT_KEYS else v.lower()
        )

    verify_allowed_input(adjusted_kwargs)

    # if no mode is specified the default is single mode
    mode = "single" if "mode" not in kwargs else adjusted_kwargs.pop("mode")

    # if mode other than single, no simulation runs
    if mode == "single":
        config = Config(**adjusted_kwargs)
        setup = (
            StaticSetup(config)
            if config.setup == "static"
            else RollingWindowSetup(config)
        )
        setup.run()

    elif mode == "routine":
        routine_name = (
            "default"
            if "routine_name" not in kwargs
            else adjusted_kwargs.pop("routine_name")
        )
        Routine(routine_name).run()

    # consolidation always run
    logger.info("--- Starting consolidation ---")
    singlefile_consolidation()


if __name__ == "__main__":
    main(
        **dict(arg.replace("-", "").split("=") for arg in sys.argv[1:])
    )  # kwargs
