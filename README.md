# Emergence

Framework for running symmetric play experiments. 

Framework code taken from [here](https://github.com/jhejna/hierarchical_morphology_transfer)

## Setup
This repo has been tested with Ubuntu 19 and python 3.7.5. Other linux based systems and python versions >3.5 should function fine. Windows users should use the linux subsystem. 

First, clone this repository to where you want to store the project by running `git clone https://github.com/jhejna/symmetric_play`.
Then, navigate to the directory.

I *highly* recommend using a virutal environment for installation as there are very specific verisons of packages that need to be used for our algorithms. This can be done through virtualenv or conda, though I prefer the former. Here are the steps for doing so.
1. Make sure you have python > 3.5 installed on your computer. 
2. Make sure you have `pip` installed for `python3`. As I have both `python` and `python3` installed on my computer, for me this is `pip3`.
3. Then, install the virtualenv package with `pip3 install virtualenv`. Ideally, this is the only package you should have in your system level python.
4. While in the project directory, create a virtual environment called `venv` by running `python3 -m virtualenv venv`. If you want to name your environment something other than venv, please add the name to the `.gitignore`. DO NOT PUSH YOUR ENVIRONMENT TO THE REPO.
5. To activate the environment anytime you want to work on the project, run `source venv/bin/activate`.

After you have a virtual environment, you need to setup all of the packages for our environment. Run the following command from the base directory of the project repo.
If using CPU:
```
pip install -e .[cpu]
```
If using GPU:
```
pip install -e .[gpu]
```

## Training with Stable Baselines
The script *scripts/train.py* can be used to train Stable Baseline algorithms in both gym and custom environments.
```
usage: train.py [-h] [--env ENV] [--alg ALG] [--timesteps TIMESTEPS]
                [--policy POLICY] [--early-reset EARLY_RESET]
                [--normalize NORMALIZE] [--time-limit TIME_LIMIT]
                [--seed SEED] [--log-interval LOG_INTERVAL]
                [--tensorboard TENSORBOARD] [--name NAME]
                [--num-proc NUM_PROC] [--eval-freq EVAL_FREQ]
                [--checkpoint-freq CHECKPOINT_FREQ]
                [--layers LAYERS [LAYERS ...]]
```
Example: train PPO2 algorithm in CartPole-v1 environment for 25000 timesteps saving a checkpoint model every 100 training iterations.
```
python scripts/train.py --env CartPole-v1 --alg PPO2 --timesteps 25000 --checkpoint-freq 100
```
To view training progress in TensorBoard we can run...
```
python scripts/train.py --env CartPole-v1 --alg PPO2 --timesteps 25000 --checkpoint-freq 100 --tensorboard TENSORBOARD
```
In a separate terminal run... (REST_OF_PATH can be found in pre-logs before training starts)
```
tensorboard --logdir ./tb_logs/REST_OF_PATH
```

## Contributors
* Joey Hejna
* Sean Kim
* Reina Wang
* Nareg Megan
* Sumer Kohli
