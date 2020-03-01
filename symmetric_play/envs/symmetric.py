from abc import ABC, abstractmethod
import gym

class SymmetricGame(ABC, gym.Env):

    def __init__(self):
        pass

    @abstractmethod
    def step(self, actions):
        '''
<<<<<<< HEAD
        Moves the players in the game according to list of "actions".
=======
        Moves all players in the game. actions should be a matrix of size (N, A, D)
        N = batch, A = number of agents, D = dimension of action space.
>>>>>>> 56ad40bc9763bf9c409c58d4fbb5d46697054eab
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