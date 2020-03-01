from abc import ABC, abstractmethod
import gym

class SymmetricGame(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def step(self, actions):
        '''
        Moves the players in the game according to list of "actions".
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

class SimultaneousGame(gym.Env):
    '''
    Must implement the gym interface.s
    '''
