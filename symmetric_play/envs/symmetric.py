from abc import ABC, abstractmethod
import gym

class SymmetricGame(ABC, gym.Env):

    def __init__(self):
        pass

    @abstractmethod
    def step(self, actions):
        '''
        Moves all players in the game. actions should be a matrix of size (N, A, D)
        N = batch, A = number of agents, D = dimension of action space.
        Returns:
			state - state of the environment
            reward - rewards the players gets from their actions.
            done - T/F boolean representing if the agents are still in the game.
            info - a dict object containing other information about step.
        '''
        return NotImplemented

    @abstractmethod
    def reset(self, *args, **kwargs):
        return NotImplemented

    @abstractmethod
    def render(self, mode='human', close=False):
        return NotImplemented