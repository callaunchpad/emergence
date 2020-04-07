import gym
from gym import spaces
import numpy as np
import random
import pygame, sys
from pygame.locals import *
import pygame.surfarray as sarray
pygame.init()

class PongObject:
    def __init__(self, pos, vel):
        self.pos = np.array(pos)
        self.vel = np.array(vel)
        self.score = 0

class Pong(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['console']}

    def __init__(self, num_agents=1):
        super(Pong, self).__init__()
        self.viewer = None
        self.numAgents = num_agents
        self.screenWidth = 400
        self.screenHeight = 600
        self.ballHeight = 5
        self.ballWidth = 5
        self.paddleWidth = 5
        self.paddleHeight = 100
        self.n_actions = 5
        self.paddleSpeed = 10
        self.maxscore = 10
        self.observation_space = []
        self.info = {}
        self.done = False
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(low = 0, high = 255, shape = (12,), dtype=np.uint8)

    def generateObs(self, ballPos, paddle0Pos, paddle1Pos, ballVel, paddle0Vel, paddle1Vel):
        obs0 = np.array([ballPos, paddle0Pos, paddle1Pos, ballVel, paddle0Vel, paddle1Vel]).flatten()

        ballPos[0] = self.screenWidth - ballPos[0]
        ballVel[0] = self.screenWidth - ballVel[0]
        paddle0Pos[0] = self.screenWidth - paddle0Pos[0]
        paddle1Pos[0] = self.screenWidth - paddle1Pos[0]
        paddle0Vel[0] = self.screenWidth - paddle0Vel[0]
        paddle1Vel[0] = self.screenWidth - paddle1Vel[0]

        obs1 = np.array([ballPos, paddle0Pos, paddle1Pos, ballVel, paddle0Vel, paddle1Vel]).flatten()
        return np.array([obs0, obs1])

    def reset(self):
        self.ball = PongObject([self.screenWidth/2, self.screenHeight/2], [random.choice([1, -1]), 0])
        self.paddle0 = PongObject([self.paddleWidth, self.screenHeight/2], [0,0])
        self.paddle1 = PongObject([self.screenWidth - self.paddleWidth, self.screenHeight/2], [0,0])
        if (self.numAgents == 2):
            self.observation = self.generateObs(self.ball.pos, self.paddle0.pos, self.paddle1.pos, self.ball.vel, self.paddle0.vel, self.paddle1.vel)
        elif (self.numAgents == 1):
            self.observation = np.array([self.ball.pos, self.paddle0.pos, self.paddle1.pos, self.ball.vel, self.paddle0.vel, self.paddle1.vel]).flatten()
            self.observation = np.expand_dims(self.observation, axis=0)
        return self.observation

    def step(self, action):
        # Update velocities
        if (self.numAgents == 2): 
            self.paddle0.vel[1] = self.paddleSpeed*(action[0]-2)
            self.paddle1.vel[1] = self.paddleSpeed*(action[1]-2)
        elif (self.numAgents == 1):
            self.paddle0.vel[1] = self.paddleSpeed*(action[0]-2)
            self.paddle1.vel[1] = self.paddleSpeed*(random.choice([0,1,2,3,4])-2)

        # Update positions but keep paddles in the screen
        self.paddle0.pos[1] = min(max(self.paddle0.vel[0] + self.paddle1.pos[0], 0), self.screenHeight - self.paddleHeight)
        self.paddle1.pos[1] = min(max(self.paddle1.vel[1] + self.paddle1.pos[1], 0), self.screenHeight - self.paddleHeight)

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
        if (self.paddle0.pos[1] - self.paddleHeight*0.5 < newBallY and self.paddle0.pos[1] + self.paddleHeight*0.5 > newBallY
                and newBallX - self.ballWidth*0.5 <= self.paddleWidth): 
            newBallX = self.paddleWidth
            meetpaddle0 = True
        elif (self.paddle1.pos[1] + self.paddleHeight*0.5 < newBallY and self.paddle1.pos[1] + self.paddleHeight*0.5 > newBallY
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

        reward = np.zeros(2)
        # Player 1 scores a goal
        if (self.ball.pos[0] < 0):
            reward[1] = 1
            reward[0] = 0
            self.paddle1.score += 1
        # Player 0 scores a goal
        elif (self.ball.pos[0] > self.screenWidth):
            reward[1] = 1
            reward[0] = 0
            self.paddle0.score += 1

        # If any player scores above maxscore, the game ends
        if (self.paddle1.score >= self.maxscore or self.paddle0.score >= self.maxscore):
            self.done = np.array([True, True])
        else:
            self.done = np.array([False, False])

        if (self.numAgents == 2):
            self.observation = self.generateObs(self.ball.pos, self.paddle0.pos, self.paddle1.pos, self.ball.vel, self.paddle0.vel, self.paddle1.vel)
        elif (self.numAgents == 1):
            self.observation = np.array([self.ball.pos, self.paddle0.pos, self.paddle1.pos, self.ball.vel, self.paddle0.vel, self.paddle1.vel]).flatten()
            self.observation = np.expand_dims(self.observation, axis=0)
        return self.observation, reward, self.done, self.info

    def render(self, mode='human'):
        # Not sure if this works
        WHITE = (255,255,255)
        RED = (255, 0, 0)
        GREEN = (0, 255, 0)
        BLACK = (0, 0, 0)

        self.canvas = pygame.Surface((self.screenWidth, self.screenHeight))
        self.screen = sarray.array3d(self.canvas)

		#set up canvas
        self.canvas.fill(BLACK)
        pygame.draw.line(self.canvas, WHITE, [self.screenWidth / 2, 0],[self.screenWidth / 2, self.screenHeight], 1)
        pygame.draw.line(self.canvas, WHITE, [self.paddleWidth, 0],[self.paddleWidth, self.screenheight], 1)
        pygame.draw.line(self.canvas, WHITE, [self.screenWidth - self.paddleWidth, 0],[self.screenWidth - self.paddleWidth, self.screenHeight], 1)
        pygame.draw.circle(self.canvas, WHITE, [self.screenWidth//2, self.screenHeight//2], 70, 1)

		#draw paddles and ball
        pygame.draw.circle(self.canvas, RED, [*map(int,self.ball.pos)], (self.ballHeight / 2), 0)
        pygame.draw.polygon(self.canvas, GREEN, [[self.paddle1.pos[0] - (self.paddleWidth / 2), self.paddle1.pos[1] - (self.paddleHeight / 2)], 
												[self.paddle1.pos[0] - (self.paddleWidth / 2), self.paddle1.pos[1] + (self.paddleHeight / 2)], 
												[self.paddle1.pos[0] + (self.paddleWidth / 2), self.paddle1.pos[1] + (self.paddleHeight / 2)], 
												[self.paddle1.pos[0] + (self.paddleWidth / 2), self.paddle1.pos[1] - (self.paddleHeight / 2)]], 0)
        pygame.draw.polygon(self.canvas, GREEN, [[self.paddle2.pos[0] - (self.paddleWidth / 2), self.paddle2.pos[1] - (self.paddleHeight / 2)], 
												[self.paddle2.pos[0] - (self.paddleWidth / 2), self.paddle2.pos[1] + (self.paddleHeight / 2)], 
												[self.paddle2.pos[0] + (self.paddleWidth / 2), self.paddle2.pos[1] + (self.paddleHeight / 2)], 
												[self.paddle2.pos[0] + (self.paddleWidth / 2), self.paddle2.pos[1] - (self.paddleHeight / 2)]], 0)

        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        if mode == 'rgb_array':
            return self.screen
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
                self.viewer.imshow(self.screen)

    def close (self):
        #TODO
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
