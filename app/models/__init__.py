# System imports
import os

# RL imports
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from common import Config, generate_filename, prepare_folder_structure

ABS_MODEL_PATH = os.path.dirname(os.path.abspath(__file__))


def load_model(
    total_timesteps, config: Config, window_roll: int = 0
) -> object:
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
    env,
    total_timesteps,
    val_env,
    config: Config,
    window_roll: int = 0,
    overwrite_file: bool = True,
    save_model=True,
):
    eval_freq = total_timesteps // config.episodes

    save_model_callback = _prepare_save_callback(
        eval_freq,
        save_model,
        config,
        window_roll,
        overwrite_file,
    )

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
            eval_env=val_env,
            eval_freq=eval_freq,
            n_eval_episodes=1,
            callback=save_model_callback,
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
            eval_env=val_env,
            eval_freq=eval_freq,
            n_eval_episodes=1,
            callback=save_model_callback,
        )
    if config.algo == "dqn":
        model = DQN(
            "MlpPolicy", env, device=config.device, verbose=config.sb_verbose
        )
        model.learn(
            total_timesteps=total_timesteps,
            eval_env=val_env,
            eval_freq=eval_freq,
            n_eval_episodes=1,
            callback=save_model_callback,
        )

    return model


def _prepare_save_callback(
    eval_freq,
    save_model,
    config: Config,
    window_roll: int,
    overwrite_file: bool,
):
    if save_model:
        save_path = prepare_folder_structure(
            ABS_MODEL_PATH, config, "model", window_roll
        )
        save_name = generate_filename(config, filetype="model")

        # if overwrite is turned on or file does not exists we save the model
        if overwrite_file or not os.path.isfile(
            os.path.join(
                save_path, save_name + "_{}_steps.zip".format(eval_freq)
            )
        ):
            return CheckpointCallback(
                save_freq=eval_freq, save_path=save_path, name_prefix=save_name
            )

    return None
