import argparse
import os

from symmetric_play.utils.tester import test

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', help='name of checkpoint model', default='./log')
    parser.add_argument('--num_timesteps', type=int, default=int(50000))
    parser.add_argument("--gif", "-g", action='store_true', default=False)
    args = parser.parse_args()
    test(args.name, args.num_timesteps, gif=args.gif)

if __name__ == '__main__':
    main()