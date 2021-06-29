# RL imports
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import CheckpointCallback

def train_model(algo, env, device, total_timesteps, val_env, episodes, seed, sb_verbose):
    eval_freq = total_timesteps // episodes

    # TODO: refactor results organizer, so it works here too and use the same structure for results and model
    # save_model_callback = save_model_callback(save_freq=eval_freq, save_path='./logs/', name_prefix=)

    if algo == 'a2c':
        # policy_kwargs = dict(net_arch=[64, 'lstm', dict(vf=[128, 128, 128], pi=[128, 128, 128])])
        model = A2C('MlpPolicy', env, device=device, verbose=sb_verbose, seed=seed)
        model.learn(total_timesteps=total_timesteps, eval_env=val_env, eval_freq=eval_freq, n_eval_episodes=1)
        # model.learn(total_timesteps=total_timesteps, eval_env=val_env, eval_freq=eval_freq, n_eval_episodes=1, callback=save_model_callback)
    if algo == 'ppo':
        # policy_kwargs = dict(net_arch=[64, 'lstm', dict(vf=[128, 128, 128], pi=[128, 128, 128])])
        model = PPO('MlpPolicy', env, device=device, verbose=sb_verbose, seed=seed)
        model.learn(total_timesteps=total_timesteps, eval_env=val_env, eval_freq=eval_freq, n_eval_episodes=1)
        # model.learn(total_timesteps=total_timesteps, eval_env=val_env, eval_freq=eval_freq, n_eval_episodes=1, callback=save_model_callback)
    if algo == 'dqn':
        model = DQN('MlpPolicy', env, device=device, verbose=sb_verbose)
        model.learn(total_timesteps=total_timesteps)

    return model
