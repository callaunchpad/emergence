import argparse
from symmetric_play.utils.loader import ModelParams

def boolean(item):
    if item == 'true' or item == 'True':
        return True
    elif item == 'false' or item == 'False':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


BASE_ARGS = {
    # Exclude Env and Alg as these are required for creating the params object.
    'timesteps' : int,
    'policy' : str,
    'early_reset' : boolean,
    'normalize' : boolean,
    'time_limit' : int,
    'seed' : int,
    'log_interval' : int,
    'tensorboard' : str,
    'name' : str,
    'num_proc' : str,
    'eval_freq' : int,
    'checkpoint_freq' : int,
}

ENV_ARGS = {
    
}

ENV_WRAPPER_ARGS = {

}

ALG_ARGS = {
    'layers' : (int, '+')
}

POLICY_ARGS = {

}

def add_args_from_dict(parser, arg_dict):
    for arg_name, arg_type in arg_dict.items():
        arg_name = "--" + arg_name.replace('_', '-')
        if isinstance(arg_type, tuple) and len(arg_type) == 2:
            parser.add_argument(arg_name, type=arg_type[0], nargs=arg_type[1], default=None)
        else:
            parser.add_argument(arg_name, type=arg_type, default=None)

def base_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", "-e", type=str)
    parser.add_argument("--alg", "-a", type=str)
    add_args_from_dict(parser, BASE_ARGS)
    return parser

def train_parser():
    parser = base_parser()
    add_args_from_dict(parser, ENV_ARGS)
    add_args_from_dict(parser, ENV_WRAPPER_ARGS)
    add_args_from_dict(parser, ALG_ARGS)
    add_args_from_dict(parser, POLICY_ARGS)
    return parser

def args_to_params(args):
    params = ModelParams(args.env, args.alg)
    for arg_name, arg_value in vars(args).items():
        if not arg_value is None:
            if arg_name in BASE_ARGS or arg_name == "env" or arg_name == "alg":
                params[arg_name] = arg_value
            elif arg_name in ENV_ARGS:
                params['env_args'][arg_name] = arg_value
            elif arg_name in ENV_WRAPPER_ARGS:
                params['env_wrapper_args'][arg_name] = arg_value
            elif arg_name in ALG_ARGS:
                params['alg_args'][arg_name] = arg.value
            elif arg_name in POLICY_ARGS:
                params['policy_args'][arg_name] = arg.value
            else:
                raise ValueError("Provided argument does not fit into categories")
    return params