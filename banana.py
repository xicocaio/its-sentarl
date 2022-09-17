# import numpy as np
# from numpy.random import RandomState

# import gym

# import torch

# from stable_baselines3 import PPO

# env = gym.make('CartPole-v1')

# model = PPO('MlpPolicy', env, verbose=1)
# model.learn(total_timesteps=10000)

# obs = env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)

#     env.render()
#     if done:
#         obs = env.reset()

# env.close()

# from stable_baselines3 import A2C
import logging
import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "default_formatter": {
            "format": "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "stream_handler": {
            "class": "logging.StreamHandler",
            "formatter": "default_formatter",
            "level": "INFO",
        },
    },
    "root": {
        "handlers": ["stream_handler"],
        "level": "INFO",
    },
}

logging.config.dictConfig(LOGGING_CONFIG)

logger = logging.getLogger(__name__)

assets = ["aapl", "msft"]
tcs = [0.0, 0.0025]
logger.info("### Routine option selected ###")
logger.info(f"------- Routine Summary ------- {assets}")
logger.info(f"Assets: {assets}, Transaction Costs: {tcs}")
logger.info(f"Transaction Costs: {tcs}")


# Set the seed, you can also pass `seed=None` to have different results
# at every run
# model = A2C("MlpPolicy", "Pendulum-v0", seed=1)
# obs = model.get_env().reset()


# model = A2C("MlpPolicy", "Pendulum-v0", seed=None)
# obs = model.get_env().reset()

# for _ in range(3):
#     print(model.predict(obs, deterministic=False)[0])
