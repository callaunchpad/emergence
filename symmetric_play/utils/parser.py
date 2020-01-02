import argparse
from symmetric_play.utils.loader import ModelParams

def base_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", "-e", type=str)
    parser.add_argument("--alg", "-a", type=str)
    return parser

def train_parser():
    parser = base_parser()
    parser.add_argument("--timesteps", "-t", type=int)

def args_to_params(args):
    params = ModelParams(args.env, args.alg)
    if args.timesteps:
        params['timesteps'] = args.timesteps
