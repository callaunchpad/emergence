from stable_baselines import results_plotter

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['svg.fonttype'] = 'none'

X_TIMESTEPS = 'timesteps'
X_EPISODES = 'episodes'
X_WALLTIME = 'walltime_hrs'
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
EPISODES_WINDOW = 100
COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
          'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
          'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue']

def main():
    """
    Example usage in jupyter-notebook

    .. code-block:: python

        from stable_baselines import results_plotter
        %matplotlib inline
        results_plotter.plot_results(["./log"], 10e6, results_plotter.X_TIMESTEPS, "Breakout")

    Here ./log is a directory containing the monitor.csv files
    """
    import argparse
    import os
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dirs', help='List of log directories', nargs='*', default=['./log'])
    parser.add_argument('--num_timesteps', type=int, default=int(10e6))
    parser.add_argument('--xaxis', help='Varible on X-axis', default=X_TIMESTEPS)
    parser.add_argument('--task_name', help='Title of plot', default='Title')
    args = parser.parse_args()
    args.dirs = [os.path.abspath(folder) for folder in args.dirs]
    results_plotter.plot_results(args.dirs, args.num_timesteps, args.xaxis, args.task_name)
    plt.show()


if __name__ == '__main__':
    main()