from gym import spaces
import numpy as np
import random

class PongObject:
	def __init__(self, pos, vel):
        self.pos = pos
		self.vel = vel
        self.score = 0

class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['console']}

    # Game Parameters
    screenWidth = 400
    screenHeight = 600
    ballHeight = 5
    ballWidth = 5, 
    paddleWidth = 5
    paddleHeight = 100
    n_actions = 6
    paddleSpeed = 10
    maxscore = 10

    def __init__(self):
        super(CustomEnv, self).__init__()
        self.viewer = None
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=255,shape=(screenHeight, screenWidth, 3), dtype=np.uint8)

    def reset(self):
        self.ball = PongObject((screenWidth/2, screenHeight/2), (random.choice(1,-1), 0))
        self.paddle0 = PongObject((paddleWidth, screenHeight/2), (0,0))
        self.paddle1 = PongObject((screenWidth - paddleWidth, screenHeight/2), (0,0))
        return (self.ball, self.paddle0, self.paddle1)
        
    def step(self, action):

        # Update velocities
        self.paddle0.vel[1] = paddleSpeed*(action[0]-2)
        self.paddle1.vel[1] = paddleSpeed*(action[1]-2)

        # Update positions but keep paddles in the screen
        self.paddle0.pos = min(max(self.paddle0.vel+self.paddle1.pos, 0), screenHeight - paddleHeight)
        self.paddle1.pos = min(max(self.paddle1.vel+self.paddle1.pos, 0), screenHeight - paddleHeight)

        # Update ball position, don't let it overlap with paddle or leave the screen
        meetpaddle0 = False
        meetpaddle1 = False
        meetscreen = False
        newBallX = self.ball.pos[0] + self.ball.vel[0]
        newBallY = self.ball.pos[0] + self.ball.vel[0]
        if (newBallY < 0): 
            newBallY = 0
            meetscreen = True
        else if (newBallY > screenHeight - ballHeight*0.5):
            newBallY = screenHeight - ballHeight*0.5
            meetscreen = True
        if (self.paddle0.pos - paddleHeight*0.5 < newBallY and self.paddle0.pos + paddleHeight*0.5 > newBallY
                and newBallX - ballWidth*0.5 <= paddleWidth): 
            newBallX = paddleWidth
            meetpaddle0 = True
        else if (self.paddle1.pos + paddleHeight*0.5 < newBallY and self.paddle1.pos + paddleHeight*0.5 > newBallY
                and newBallX + ballWidth*0.5 >= screenWidth - paddleWidth):
            newBallX = screenWidth - paddleWidth
            meetpaddle1 = True
        self.ball.pos = (newBallX, newBallY)

        # Update ball velocity: hitting the edge of the screen and the paddles will cause it to bounce back
        # Hitting the paddle while the paddle is moving will cause the ball to accelerate in that direction as well
        if meetpaddle0: 
            self.ball.vel[0] *= -1
            self.ball.vel[1] += self.paddle0.vel[1]
        else if meetpaddle1: 
            self.ball.vel[0] *= -1
            self.ball.vel[1] += self.paddle1.vel[1]
        if meetscreen:
            self.ball.vel[1] *= -1

        # Player 1 scores a goal
        if (self.ball.pos[0] < 0): 
            reward[1] += 1
            reward[0] -= 1
            self.paddle1.score += 1
        # Player 0 scores a goal
        else if (self.ball.pos[0] > screenWidth):
            reward[1] += 1
            reward[0] -= 1
            self.paddle0.score += 1
        
        # If any player scores above maxscore, the game ends
        if (self.paddle1.score >= maxscore or self.paddle0.score >= maxscore):
            done = True
        else:
            done = False
        info = {meetpaddle0, meetpaddle1}

    return (self.ball, self.paddle0, self.paddle1), reward, done, info

    def render(self, mode='human'):
        pass #TODO

    def close (self):
        #TODO
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None