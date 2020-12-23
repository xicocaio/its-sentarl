# RL imports
from stable_baselines3 import A2C, DQN


def train_model(algo, env, device, total_timesteps, val_env, episodes, seed, sb_verbose):
    eval_freq = total_timesteps // episodes

    if algo == 'a2c':
        # policy_kwargs = dict(net_arch=[64, 'lstm', dict(vf=[128, 128, 128], pi=[128, 128, 128])])
        model = A2C('MlpPolicy', env, device=device, verbose=sb_verbose, seed=seed)
        model.learn(total_timesteps=total_timesteps, eval_env=val_env, eval_freq=eval_freq, n_eval_episodes=1)
    if algo == 'dqn':
        model = DQN('MlpPolicy', env, device=device, verbose=sb_verbose)
        model.learn(total_timesteps=total_timesteps)

    return model
