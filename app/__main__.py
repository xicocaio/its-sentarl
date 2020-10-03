import sys
import ast

# internal imports
import settings
from common import Config
from setups import StaticSetup

NUMERICAL_INPUT_KEYS = ['episodes', 'initial_wealth', 'transaction_cost', 'test_data_ratio', 'val_data_ratio']


def verify_allowed_input(inputs):
    for key, value in inputs.items():
        if key in settings.ALLOWED_INPUT_PARAMS and value not in settings.ALLOWED_INPUT_PARAMS[key]:
            raise ValueError('{} entry not found: {}'.format(key, value))


def main(**kwargs):
    adjusted_kwargs = kwargs.copy()
    for k, v in kwargs.items():
        adjusted_kwargs[k] = ast.literal_eval(v) if k in NUMERICAL_INPUT_KEYS else v.lower()

    verify_allowed_input(adjusted_kwargs)

    config = Config(**adjusted_kwargs)

    if config.setup == 'static':
        setup = StaticSetup(config)

        setup.run()


if __name__ == '__main__':
    main(**dict(arg.replace('-', '').split('=') for arg in sys.argv[1:]))  # kwargs
