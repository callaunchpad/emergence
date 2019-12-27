import argparse
from symmetric_play.utils.trainer import train
from symmetric_play.utils.loader import ModelParams

parser = argparse.ArgumentParser()

parser.add_argument("--env", "-e", type=str)
parser.add_argument("--alg", "-a", type=str)
parser.add_argument("--timesteps", "-t", type=int)

args = parser.parse_args()

params = ModelParams(args.env, args.alg)

if args.timesteps:
    params['timesteps'] = args.timesteps

train(params)