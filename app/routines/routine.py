import time
import itertools
import datetime

import settings
from common import Config
from setups import StaticSetup, RollingWindowSetup

ROUTINE_DEFAULT = {'setup': 'rolling',
                   'assets': settings.AVAILABLE_DATA.keys(),
                   'tcs': settings.TRANSACTION_COSTS,
                   'frequencies': settings.FREQUENCIES,
                   'reward_functions': settings.REWARD_FUNCTIONS}


class Routine(object):
    def __init__(self, routine_name):
        self.routine_name = routine_name

        if self.routine_name == 'default':
            self.config_name = 'min-sent-5'
            self.setup = 'rolling'
            self.assets = settings.AVAILABLE_DATA.keys()
            self.tcs = settings.TRANSACTION_COSTS
            self.frequencies = settings.FREQUENCIES
            self.reward_functions = settings.REWARD_FUNCTIONS
            self.stgs = settings.STGS_ALGO
            # self.stgs = settings.STGS_BASE

    # def run(kwargs):
    #     # if no mode is specified the default is single mode
    #     mode = 'single' if 'mode' not in kwargs else kwargs.pop('mode')
    #
    #     config = Config(**kwargs)
    #
    #     # if mode other than single, no simulation runs
    #     if mode == 'single':
    #         setup = StaticSetup(config) if config.setup == 'static' else RollingWindowSetup(config)
    #
    #         setup.run()

    def run(self):
        print('\n### Routine option selected ###')
        print('------- Routine Summary -------')
        print('Assets: ', self.assets)
        print('Transaction Costs: ', self.tcs)
        print('Frequencies: ', self.frequencies)
        print('Reward Functions: ', self.reward_functions)
        print('-------------------------------')

        current_time = datetime.datetime.now()
        print('\n>>> Starting routine `{}` at {} <<<'.format(self.routine_name, current_time))

        specs = list(itertools.product(*[self.stgs, self.assets, self.tcs, self.frequencies, self.reward_functions]))
        total_runs = len(specs)

        initial_p_time = time.process_time()
        run_count = 1
        elapsed_t = []
        for stg, asset, tc, frequency, reward_function in specs:
            cfg = Config(name=self.config_name, stg=stg, asset=asset, setup=self.setup, frequency=frequency,
                         reward_function=reward_function, transaction_cost=tc, sb_verbose=False, ep_verbose=False)

            initial_run_t = time.process_time()
            print('\n--- Starting run #{} for: {}, {}, tc {}, {}, {} ---'.format(stg, run_count, asset, tc, frequency,
                                                                                 reward_function))

            setup = StaticSetup(cfg) if cfg.setup == 'static' else RollingWindowSetup(cfg)

            setup.run()

            t = time.process_time()
            # t = datetime.datetime.now()
            elapsed_t.append(t - initial_run_t)
            print(
                '--- Finishing run #{} for: {}, {}, tc {}, {}, {} with elapsed time {:.2f}s ---'.format(stg, run_count,
                                                                                                        asset,
                                                                                                        tc,
                                                                                                        frequency,
                                                                                                        reward_function,
                                                                                                        elapsed_t[0]))

            run_count += 1

            # Consider storing run time to take the average to calculate ETA
            total_runs -= 1
            mean_elapsed_t = sum(elapsed_t) / len(elapsed_t)
            remaining_time = datetime.timedelta(seconds=total_runs * mean_elapsed_t)
            current_time = datetime.datetime.now()
            estimated_end_time = current_time + remaining_time
            if total_runs > 0:
                print('\nEstimated time of finish: {}; remaining ({})'.format(estimated_end_time, remaining_time))

        current_time = datetime.datetime.now()
        elapsed_t = t - initial_p_time
        print('\n>>> Ending routine {} at {} with a total runtime of {:.2f}s <<<'.format(self.routine_name,
                                                                                         current_time,
                                                                                         elapsed_t))
