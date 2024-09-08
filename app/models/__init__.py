# System imports
import os

# Gymnasium imports
import gymnasium as gym

# RL imports
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.monitor import Monitor as SB3Monitor
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from common import Config, generate_filename, prepare_folder_structure

ABS_MODEL_PATH = os.path.dirname(os.path.abspath(__file__))


def load_model(
    total_timesteps, config: Config, window_roll: int = 0
) -> object:
    """Load trained SB3 model according to the specified config
    Parameters
    ----------
    total_timesteps: int
        The gym env that is ready for simulation
    config: Configuration Object
        The default project config the user specified
    window_roll: int
        The number of window rolls to execute
    Returns: SB3 model
        Lods a ready to use SB3 model
    ----------
    """
    load_path = prepare_folder_structure(
        ABS_MODEL_PATH, config, "model", window_roll
    )
    load_name = generate_filename(config, filetype="model")

    abs_filename = os.path.join(
        load_path, load_name + "_{}_steps.zip".format(total_timesteps)
    )

    if not os.path.isfile(abs_filename):
        raise OSError("Model not found: {}".format(abs_filename))

    if config.algo == "a2c":
        return A2C.load(abs_filename)
    if config.algo == "ppo":
        return PPO.load(abs_filename)
    if config.algo == "dqn":
        return DQN.load(abs_filename)


def train_model(
    env: gym.Env,
    total_timesteps: int,
    env_val: gym.Env,
    config: Config,
    window_roll: int = 0,
    overwrite_file: bool = True,
    save_model: bool = True,
):
    """Calls SB3 train method for the selected model
    Parameters
    ----------
    env: Gymnasium Environment
        The gym env that is ready for training
    total_timesteps: int
        The gym env that is ready for simulation
    env_val: Gymnasium Environment
        The gym env that is ready for evaluation
    config: Configuration Object
        The default project config the user specified
    window_roll: int
        The number of window rolls to execute
    overwrite_file: bool
        Wheter to overwrite model files
    save_model: bool
        Wheter to save models at each episode
    Returns:
    ----------
    """
    eval_freq = total_timesteps // config.episodes

    save_model_callback = _prepare_save_callback(
        eval_freq,
        save_model,
        config,
        window_roll,
        overwrite_file,
    )

    # There are some differences between SB3 and gymnasium environments,
    # and given that most present code adopts gymnasium standards,
    # where there is a connection between SB3 and gym, SB3 env should be
    # wrapped into SB3 wrappers to avoid issues with reward syncronization
    # for more information:
    # https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api
    sb3monitor = SB3Monitor(env_val)

    eval_callback = EvalCallback(
        sb3monitor, n_eval_episodes=1, eval_freq=eval_freq
    )

    callback_list = [eval_callback]

    if save_model:
        callback_list.append(save_model_callback)

    callbacks = CallbackList(callback_list)

    if config.algo == "a2c":
        # policy_kwargs = dict(
        #     net_arch=[64,
        #               "lstm",
        #               dict(vf=[128, 128, 128],
        #               pi=[128, 128, 128])]
        # )
        model = A2C(
            "MlpPolicy",
            env,
            device=config.device,
            verbose=config.sb_verbose,
            seed=config.seed,
        )
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
        )
    if config.algo == "ppo":
        # policy_kwargs = dict(
        #     net_arch=[64,
        #               "lstm",
        #               dict(vf=[128, 128, 128],
        #               pi=[128, 128, 128])]
        # )
        model = PPO(
            "MlpPolicy",
            env,
            device=config.device,
            verbose=config.sb_verbose,
            seed=config.seed,
        )
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
        )
    if config.algo == "dqn":
        model = DQN(
            "MlpPolicy", env, device=config.device, verbose=config.sb_verbose
        )
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
        )

    return model


def _prepare_save_callback(
    eval_freq,
    save_model,
    config: Config,
    window_roll: int,
    overwrite_file: bool,
):
    save_path = prepare_folder_structure(
        ABS_MODEL_PATH, config, "model", window_roll
    )
    save_name = generate_filename(config, filetype="model")

    # if overwrite is turned on or file does not exists we save the model
    if overwrite_file or not os.path.isfile(
        os.path.join(save_path, save_name + "_{}_steps.zip".format(eval_freq))
    ):
        return CheckpointCallback(
            save_freq=eval_freq, save_path=save_path, name_prefix=save_name
        )
