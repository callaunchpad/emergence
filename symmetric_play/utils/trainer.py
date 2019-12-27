import os
from stable_baselines.bench import Monitor
from stable_baselines import logger
import numpy as np
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.common.misc_util import mpi_rank_or_zero
from stable_baselines.results_plotter import load_results, ts2xy
from symmetric_play.utils.loader import get_paths, get_env, get_alg, get_policy

best_mean_reward, n_steps = -np.inf, 0

def callback(_locals, _globals, data_dir, freq=None):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    if not freq:
        freq = 10000
    global n_steps, best_mean_reward
    # Print stats every freq calls
    if (n_steps + 1) % freq == 0:
        # Evaluate policy training performance
        x, y = ts2xy(load_results(data_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-200:])
            print(x[-1], 'timesteps')
            print("Best 200 mean reward: {:.2f} - Last 200 mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))
            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['self'].save(data_dir + '/best_model')
        # TODO: Perhaps augment to save a video?
    n_steps += 1
    return True

def create_training_callback(data_dir, freq=None):
    return lambda _locals, _globals: callback(_locals, _globals, data_dir, freq=freq)

def train(params, model=None, env=None): 
    print("Training Parameters: ", params)

    data_dir, tb_path = get_paths(params)
    os.makedirs(data_dir, exist_ok=True)
    # Save parameters immediatly
    params.save(data_dir)

    rank = mpi_rank_or_zero()
    if rank != 0:
        logger.set_level(logger.DISABLED)

    # Create the environment if not given
    if env is None:  
        def make_env(i):
            env = get_env(params)
            env = Monitor(env, data_dir + '/' + str(i), allow_early_resets=params['early_reset'])
            return env

        env = DummyVecEnv([(lambda n: lambda: make_env(n))(i) for i in range(params['num_proc'])])

        if params['normalize']:
            env = VecNormalize(env)
    # Set the seeds
    if params['seed']:
        seed = params['seed'] + 100000 * rank
        set_global_seeds(seed)

    if 'noise' in params and params['noise']:
        from stable_baselines.ddpg import OrnsteinUhlenbeckActionNoise
        n_actions = env.action_space.shape[-1]
        params['alg_args']['action_noise'] = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(params['noise'])*np.ones(n_actions))
    
    alg = get_alg(params)
    if model is None:
        policy = get_policy(params)
        model = alg(policy,  env, verbose=1, tensorboard_log=tb_path, policy_kwargs=params['policy_args'], **params['alg_args'])
    else:
        model.set_env(env)

    callback_freq = {
        'PPO2': int(1000 / params['num_proc']),
        'SAC' : 10000,
        'DSAC' : 10000
    }[params['alg']]

    model.learn(total_timesteps=params['timesteps'], log_interval=params['log_interval'], callback=create_training_callback(data_dir, freq=callback_freq))
    
    model.save(data_dir +'/final_model')

    if params['normalize']:
        env.save(data_dir + '/environment.pkl')
        
    env.close()
