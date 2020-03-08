import gym
from gym import spaces
import numpy as np
import random

class PongObject:
    def __init__(self, pos, vel):
        self.pos = np.array(pos)
        self.vel = np.array(vel)
        self.score = 0

class Pong(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['console']}

    def __init__(self):
        super(Pong, self).__init__()
        self.viewer = None
        self.screenWidth = 400
        self.screenHeight = 600
        self.ballHeight = 5
        self.ballWidth = 5, 
        self.paddleWidth = 5
        self.paddleHeight = 100
        self.n_actions = 6
        self.paddleSpeed = 10
        self.maxscore = 10
        self.reward = np.zeros(2)
        self.info = None
        self.done = False
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low = 0, high = 255, shape = (self.screenHeight, self.screenWidth, 3), dtype=np.uint8)

    def reset(self):
        self.ball = PongObject([self.screenWidth/2, self.screenHeight/2], [random.choice([1, -1]), 0])
        self.paddle0 = PongObject([self.paddleWidth, self.screenHeight/2], [0,0])
        self.paddle1 = PongObject([self.screenWidth - self.paddleWidth, self.screenHeight/2], [0,0])
        return np.array([self.ball.pos, self.ball.vel, self.paddle0.pos, self.paddle0.vel, self.paddle1.pos, self.paddle1.vel])
        
    def step(self, action):

        # Update velocities
        self.paddle0.vel[1] = self.paddleSpeed*(action[0]-2)
        self.paddle1.vel[1] = self.paddleSpeed*(action[1]-2)

        # Update positions but keep paddles in the screen
        self.paddle0.pos = min(max(self.paddle0.vel+self.paddle1.pos, 0), self.screenHeight - self.paddleHeight)
        self.paddle1.pos = min(max(self.paddle1.vel+self.paddle1.pos, 0), self.screenHeight - self.paddleHeight)

        # Update ball position, don't let it overlap with paddle or leave the screen
        meetpaddle0 = False
        meetpaddle1 = False
        meetscreen = False
        newBallX = self.ball.pos[0] + self.ball.vel[0]
        newBallY = self.ball.pos[0] + self.ball.vel[0]
        if (newBallY < 0): 
            newBallY = 0
            meetscreen = True
        elif (newBallY > self.screenHeight - self.ballHeight*0.5):
            newBallY = self.screenHeight - self.ballHeight*0.5
            meetscreen = True
        if (self.paddle0.pos - self.paddleHeight*0.5 < newBallY and self.paddle0.pos + self.paddleHeight*0.5 > newBallY
                and newBallX - self.ballWidth*0.5 <= self.paddleWidth): 
            newBallX = self.paddleWidth
            meetpaddle0 = True
        elif (self.paddle1.pos + self.paddleHeight*0.5 < newBallY and self.paddle1.pos + self.paddleHeight*0.5 > newBallY
                and newBallX + self.ballWidth*0.5 >= self.screenWidth - self.paddleWidth):
            newBallX = self.screenWidth - self.paddleWidth
            meetpaddle1 = True
        self.ball.pos = np.array([newBallX, newBallY])

        # Update ball velocity: hitting the edge of the screen and the paddles will cause it to bounce back
        # Hitting the paddle while the paddle is moving will cause the ball to accelerate in that direction as well
        if meetpaddle0: 
            self.ball.vel[0] *= -1
            self.ball.vel[1] += self.paddle0.vel[1]
        elif meetpaddle1: 
            self.ball.vel[0] *= -1
            self.ball.vel[1] += self.paddle1.vel[1]
        if meetscreen:
            self.ball.vel[1] *= -1

        # Player 1 scores a goal
        if (self.ball.pos[0] < 0): 
            self.reward[1] += 1
            self.reward[0] -= 1
            self.paddle1.score += 1
        # Player 0 scores a goal
        elif (self.ball.pos[0] > self.screenWidth):
            self.reward[1] += 1
            self.reward[0] -= 1
            self.paddle0.score += 1
        
        # If any player scores above maxscore, the game ends
        if (self.paddle1.score >= self.maxscore or self.paddle0.score >= self.maxscore):
            self.done = True
        else:
            self.done = False
        self.info = np.array([meetpaddle0, meetpaddle1])

        return np.array([self.ball.pos, self.ball.vel, self.paddle0.pos, self.paddle0.vel, self.paddle1.pos, self.paddle1.vel]), self.reward, self.done, self.info

    def render(self, mode='human'):
        pass #TODO

    def close (self):
        #TODO
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None