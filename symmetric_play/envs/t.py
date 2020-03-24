import gym

env = gym.make('Pong-v0')
print("Observation space:", env.observation_space)
print("Shape:", env.observation_space.shape)
print("Action space:", env.action_space)
env.reset()

for _ in range(5):
    env.render()
    action = env.action_space.sample()
    print("Sampled action:", action)
    obs, reward, done, info = env.step(action)
    print("Step returns: ", obs, reward, done, info)
    
env.close()

# import pong
# import gym
# from gym import error, spaces
# from gym import utils
# from gym.utils import seeding

# env = AtariEnv(gym.Env, utils.EzPickle):
# env.reset()

# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample())
    
# env.close()
