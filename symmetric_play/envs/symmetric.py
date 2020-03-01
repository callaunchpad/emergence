from abc import ABC, abstractmethod
import gym

class SymmetricGame(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def step(self, actions):
        '''
        Moves the player "player" in the game.
        Returns:
            obs - observation from the perspective of player "player" after moving
            reward - reward the player "player" gets from his action.
            done - T/F boolean representing if the agent is still in the game.
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
