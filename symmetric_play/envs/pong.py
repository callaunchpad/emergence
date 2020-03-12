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

    def __init__(self):
        super(Pong, self).__init__()
        self.viewer = None
        self.numAgents = 1
        self.screenWidth = 400
        self.screenHeight = 600
        self.ballHeight = 5
        self.ballWidth = 5
        self.paddleWidth = 5
        self.paddleHeight = 100
        self.n_actions = 5
        self.paddleSpeed = 10
        self.maxscore = 10
        self.reward = np.zeros(2)
        self.info = None
        self.done = False
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(low = 0, high = 255, shape = (12,), dtype=np.uint8)

    def reset(self):
        self.ball = PongObject([self.screenWidth/2, self.screenHeight/2], [random.choice([1, -1]), 0])
        self.paddle0 = PongObject([self.paddleWidth, self.screenHeight/2], [0,0])
        self.paddle1 = PongObject([self.screenWidth - self.paddleWidth, self.screenHeight/2], [0,0])
        return np.array([self.ball.pos, self.ball.vel, self.paddle0.pos, self.paddle0.vel, self.paddle1.pos, self.paddle1.vel]).flatten()
        
    def step(self, action):

        # Update velocities
        if (self.numAgents == 2): 
            self.paddle0.vel[1] = self.paddleSpeed*(action[0]-2)
            self.paddle1.vel[1] = self.paddleSpeed*(action[1]-2)
        elif (self.numAgents == 1):
            self.paddle0.vel[1] = self.paddleSpeed*(action-2)
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

        return np.array([self.ball.pos, self.ball.vel, self.paddle0.pos, self.paddle0.vel, self.paddle1.pos, self.paddle1.vel]).flatten(), self.reward, self.done, self.info
        
    def render(self, mode='human'):
		self.canvas = pygame.Surface((WIDTH, HEIGHT))
		self.screen = sarray.array3d(self.canvas)

		#set up canvas
		self.canvas.fill(BLACK)
		pygame.draw.line(self.canvas, WHITE, [WIDTH / 2, 0],[WIDTH / 2, HEIGHT], 1)
		pygame.draw.line(self.canvas, WHITE, [PAD_WIDTH, 0],[PAD_WIDTH, HEIGHT], 1)
		pygame.draw.line(self.canvas, WHITE, [WIDTH - PAD_WIDTH, 0],[WIDTH - PAD_WIDTH, HEIGHT], 1)
		pygame.draw.circle(self.canvas, WHITE, [WIDTH//2, HEIGHT//2], 70, 1)

		#draw paddles and ball
		pygame.draw.circle(self.canvas, RED, [*map(int,self.ball.pos)], BALL_RADIUS, 0)
		pygame.draw.polygon(self.canvas, GREEN, [[self.paddle1.pos[0] - HALF_PAD_WIDTH, self.paddle1.pos[1] - HALF_PAD_HEIGHT], 
												[self.paddle1.pos[0] - HALF_PAD_WIDTH, self.paddle1.pos[1] + HALF_PAD_HEIGHT], 
												[self.paddle1.pos[0] + HALF_PAD_WIDTH, self.paddle1.pos[1] + HALF_PAD_HEIGHT], 
												[self.paddle1.pos[0] + HALF_PAD_WIDTH, self.paddle1.pos[1] - HALF_PAD_HEIGHT]], 0)
		pygame.draw.polygon(self.canvas, GREEN, [[self.paddle2.pos[0] - HALF_PAD_WIDTH, self.paddle2.pos[1] - HALF_PAD_HEIGHT], 
												[self.paddle2.pos[0] - HALF_PAD_WIDTH, self.paddle2.pos[1] + HALF_PAD_HEIGHT], 
												[self.paddle2.pos[0] + HALF_PAD_WIDTH, self.paddle2.pos[1] + HALF_PAD_HEIGHT], 
												[self.paddle2.pos[0] + HALF_PAD_WIDTH, self.paddle2.pos[1] - HALF_PAD_HEIGHT]], 0)

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