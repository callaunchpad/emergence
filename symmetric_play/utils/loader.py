import os
import symmetric_play
import json
from datetime import date
import stable_baselines

BASE = os.path.dirname(os.path.dirname(symmetric_play.__file__)) + '/output'
LOGS = os.path.dirname(os.path.dirname(symmetric_play.__file__)) + '/tb_logs'

class ModelParams(dict):

    def __init__(self, env : str, alg : str):
        super(ModelParams, self).__init__()
        # Construction Specification
        self['alg'] = alg
        self['env'] = env
        self['policy'] = 'MlpPolicy' # TODO: Support different types of policies!
        # Arg Dicts
        self['env_args'] = dict()
        self['env_wrapper_args'] = dict()
        self['alg_args'] = dict()
        self['policy_args'] = dict()
        # Env Wrapper Arguments
        self['early_reset'] = True
        self['normalize'] = False
        self['time_limit'] = None
        # Training Args
        self['seed'] = None
        self['timesteps'] = 250000
        # Logistical Args
        self['log_interval'] = 20
        self['name'] = None
        self['tensorboard'] = None
        self['num_proc'] = 1 # Default to single process
        self['eval_freq'] = 100000
        self['checkpoint_freq'] = None

    def get_save_name(self) -> str:
        if self['name']:
            name =  self['name']
        else:
            name = self['env'] + '_' + self['alg']
        if not self['seed'] is None:
            name += '_s' + str(self['seed'])
        return name

    def save(self, path : str):
        with open(os.path.join(path, 'params.json'), 'w') as fp:
            json.dump(self, fp, indent=4)

    @classmethod
    def load(cls, path):
        if not path.startswith('/'):
            path = os.path.join(BASE, path)
        if os.path.isdir(path) and 'params.json' in os.listdir(path):
            path = os.path.join(path, 'params.json')
        elif os.path.exists(path):
            pass
        else:
            raise ValueError("Params file not found in specified save directory.")
        with open(path, 'r') as fp:
            data = json.load(fp)
        params = cls(data['env'], data['alg'])
        params.update(data)
        return params

def get_alg(params: ModelParams):
    alg_name = params['alg']
    try:
        alg = vars(symmetric_play.algs)[alg_name]
    except:
        alg = vars(stable_baselines)[alg_name]
    return alg

def get_env(params: ModelParams):
    env_names = params['env'].split('_')
    try:
        env_cls = vars(symmetric_play.envs)[env_names[0]]
        env = env_cls(**params['env_args'])
        if len(env_names) == 2:
            env = vars(symmetric_play.envs)[env_names[1]](env, **params['env_wrapper_args'])
        if params['time_limit']:
            from gym.wrappers import TimeLimit
            env = TimeLimit(env, params['time_limit'])
    except:
        # If we don't get the env, then we assume its a gym environment
        import gym
        env = gym.make(params['env'])
    return env    

def get_policy(params: ModelParams):
    policy_name = params['policy']
    try:
        policy = vars(symmetric_play.policies)[policy_name]
        return policy
    except:
        alg_name = params['alg']
        if alg_name == 'SAC':
            search_location = stable_baselines.sac.policies
        elif alg_name == 'DDPG':
            search_location = stable_baselines.ddpg.policies
        elif alg_name == 'DQN':
            search_location = stable_baselines.deepq.policies
        elif alg_name == 'TD3':
            search_location = stable_baselines.td3.policies
        else:
            search_location = stable_baselines.common.policies
        policy = vars(search_location)[policy_name]
        return policy
    
def get_paths(params: ModelParams):
    date_prefix = date.today().strftime('%m_%d_%y')
    date_dir = os.path.join(BASE, date_prefix)
    save_name = params.get_save_name()
    if os.path.isdir(date_dir):
        candidates = [f_name for f_name in os.listdir(date_dir) if '_'.join(f_name.split('_')[:-1]) == save_name]
        if len(candidates) == 0:
            save_name += '_0'
        else:
            num = max([int(dirname[-1]) for dirname in candidates]) + 1
            save_name += '_' + str(num)
    else:
        save_name += '_0'
    
    save_path = os.path.join(date_dir, save_name)
    tb_path = os.path.join(LOGS, date_prefix, save_name) if params['tensorboard'] else None
    return save_path, tb_path

def load_from_name(path, best=True, load_env=True):
    if not path.startswith('/'):
        path = os.path.join(BASE, path)
    params = ModelParams.load(path)
    return load(path, params, best=best, load_env=load_env)

def load(path: str, params : ModelParams, best=True, load_env=True):
    if not path.startswith('/'):
        path = os.path.join(BASE, path)
    files = os.listdir(path)
    if not 'final_model.zip' in files and 'best_model.zip' in files:
        model_path = path + '/best_model.zip'
    elif 'best_model.zip' in files and best:
        model_path = path + '/best_model.zip'
    elif 'final_model.zip' in files:
        model_path = path + '/final_model.zip'
    else:
        raise ValueError("Cannot find a model for name: " + path)
    # get model
    alg = get_alg(params)
    model = alg.load(model_path)
    if load_env:
        env = get_env(params)
    else:
        env = None
    return model, env
