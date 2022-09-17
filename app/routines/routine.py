import time
import itertools
import datetime

import settings
from common import Config
from setups import StaticSetup, RollingWindowSetup

ROUTINE_DEFAULT = {
    "setup": "rolling",
    "assets": settings.AVAILABLE_DATA.keys(),
    "tcs": settings.TRANSACTION_COSTS,
    "frequencies": settings.FREQUENCIES,
    "reward_functions": settings.REWARD_FUNCTIONS,
}


class Routine(object):
    def __init__(self, routine_name, target_episodes=[50]):
        self.routine_name = routine_name

        self.config_name = None
        self.setup = "rolling"
        self.assets = settings.AVAILABLE_DATA.keys()
        self.tcs = settings.TRANSACTION_COSTS
        self.frequencies = settings.FREQUENCIES
        self.reward_functions = settings.REWARD_FUNCTIONS
        self.stgs = settings.STGS_ALGO
        # self.stgs = settings.STGS_BASE
        self.seeds = [42, 1, 6888, 13, 17]
        self.episodes = target_episodes
        self.load_model = False

        if self.routine_name == "load_model":
            self.episodes = target_episodes
            self.load_model = True

    def run(self):
        print("\n### Routine option selected ###")
        print("------- Routine Summary -------")
        print("Assets: ", self.assets)
        print("Transaction Costs: ", self.tcs)
        print("Frequencies: ", self.frequencies)
        print("Reward Functions: ", self.reward_functions)
        print("Seed: ", self.seeds)
        print("Episodes: ", self.episodes)
        print("-------------------------------")

        current_time = datetime.datetime.now()
        print(
            "\n>>> Starting routine `{}` at {} <<<".format(
                self.routine_name, current_time
            )
        )

        specs = list(
            itertools.product(
                *[
                    self.frequencies,
                    self.tcs,
                    self.assets,
                    self.reward_functions,
                    self.seeds,
                    self.stgs,
                    self.episodes,
                ]
            )
        )
        total_runs = remaining_runs = len(specs)

        initial_p_time = time.process_time()
        run_count = 1
        elapsed_t = []
        for frequency, tc, asset, reward_function, seed, stg, episode in specs:
            self.config_name = "min-sent-5" if stg == "sentarl" else "default"
            cfg = Config(
                name=self.config_name,
                seed=seed,
                stg=stg,
                asset=asset,
                episodes=episode,
                setup=self.setup,
                frequency=frequency,
                reward_function=reward_function,
                transaction_cost=tc,
                sb_verbose=False,
                ep_verbose=False,
            )

            initial_run_t = time.process_time()
            print(
                f"\n--- Starting run {run_count}/{total_runs} for: {stg}, \
                    {asset}, tc {tc}, {frequency}, {reward_function}, \
                    seed {seed}, episode {episode} ---"
            )

            setup = (
                StaticSetup(cfg)
                if cfg.setup == "static"
                else RollingWindowSetup(cfg, self.load_model)
            )

            setup.run()

            t = time.process_time()
            # t = datetime.datetime.now()
            elapsed_t.append(t - initial_run_t)
            print(
                f"--- Finishing run #{run_count} for: {stg}, {asset}, \
                    tc {tc}, {frequency}, {reward_function} \
                    with elapsed time {elapsed_t[0]:.2f}s ---"
            )

            run_count += 1

            # Consider storing run time to take the average to calculate ETA
            remaining_runs -= 1
            mean_elapsed_t = sum(elapsed_t) / len(elapsed_t)
            remaining_time = datetime.timedelta(
                seconds=remaining_runs * mean_elapsed_t
            )
            current_time = datetime.datetime.now()
            estimated_end_time = current_time + remaining_time
            if remaining_runs > 0:
                print(
                    "\nEstimated time of finish: {}; remaining ({})".format(
                        estimated_end_time, remaining_time
                    )
                )

        current_time = datetime.datetime.now()
        elapsed_t = t - initial_p_time
        print(
            f"\n>>> Ending routine {self.routine_name} at {current_time} \
                with a total runtime of {elapsed_t:.2f}s <<<"
        )
