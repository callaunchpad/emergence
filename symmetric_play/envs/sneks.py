'''
    Multi-player snake environment.

    Actions: [0,3] representing the directions, see world class.
'''
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import random

# copy and pasted a bunch of stuff so don't have to pip install -e . for second repo
'''
    Rendering object of Sneks. Receives a map from gridworld and
    transform it into a visible image (applies colors and zoom)
'''

class SnekColor:

    def __init__(self, body_color, head_color):
        self.body_color = body_color
        self.head_color = head_color

'''
    This class translates the world state with block ids into an RGB image, with
    a selected zoom factor. This can be used to return an RGB observation or
    to render the world.
'''
class RGBifier:

    def __init__(self, size, zoom_factor=1, players_colors = {}):
        # Setting default colors
        self.pcolors = {
            0: SnekColor((0, 204, 0), (0, 77, 0)),
            1: SnekColor((0, 0, 204), (0, 0, 77)),
        }
        self.zoom_factor = zoom_factor
        self.size = size
        self.height = size[0]
        self.width = size[1]

    def get_color(self, state):
        # Void => BLACK
        if state == 0:
            return (0,0,0)
        # Wall => WHITE
        elif state == 255:
            return (255, 255, 255)
        # Food => RED
        elif state == 64:
            return (255, 0, 0)
        else:
            # Get player ID
            pid = (state) // 2
            is_head = (state) % 2
            # Checking that default color exists
            if pid not in self.pcolors.keys():
                pid = 0
            # Assign color (default or given)
            if is_head == 0:
                return self.pcolors[pid].body_color
            else:
                return self.pcolors[pid].head_color

    def get_image(self, state):
        # Transform to RGB image with 3 channels
        color_lu = np.vectorize(lambda x: self.get_color(x), otypes=[np.uint8, np.uint8, np.uint8])
        _img = np.array(color_lu(state))
        # Zoom every channel
        _img_zoomed = np.zeros((3, self.height * self.zoom_factor, self.width * self.zoom_factor), dtype=np.uint8)
        for c in range(3):
            for i in range(_img.shape[1]):
                for j in range(_img.shape[2]):
                        _img_zoomed[c, i*self.zoom_factor:i*self.zoom_factor+self.zoom_factor,
                                    j*self.zoom_factor:j*self.zoom_factor+self.zoom_factor] = np.full((self.zoom_factor, self.zoom_factor), _img[c,i,j])
        # Transpose to get channels as last
        _img_zoomed = np.transpose(_img_zoomed, [1,2,0])
        return _img_zoomed

'''
    This class specifically handles the renderer for the environment.
'''
class Renderer:

    def __init__(self, size, zoom_factor=1, players_colors={}):
        self.rgb = RGBifier(size, zoom_factor, players_colors)
        self.viewer = None

    def _render(self, state, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        img = self.rgb.get_image(state)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

class Snek:

    '''
        DIRECTIONS:
        0: UP (North)
        1: RIGHT (East)
        2: DOWN (South)
        3: LEFT (West)

        ACTIONS:
        0: UP
        1: RIGHT
        2: DOWN
        3: LEFT
    '''
    DIRECTIONS = [np.array([-1,0]), np.array([0,1]), np.array([1,0]), np.array([0,-1])]

    def __init__(self, snek_id, start_position, start_direction_index, start_length):
        self.snek_id = snek_id
        self.current_direction_index = start_direction_index
        self.alive = True
        # Place the snek
        start_position = start_position
        self.my_blocks = [start_position]
        current_positon = np.array(start_position)
        for i in range(1, start_length):
            # Direction inverse of moving
            current_positon = current_positon - self.DIRECTIONS[self.current_direction_index]
            self.my_blocks.append(tuple(current_positon))

    def step(self, action):
        # Check if action can be performed (do nothing if in the same direction or opposite)
        if (action != self.current_direction_index) and (action != (self.current_direction_index+2)%len(self.DIRECTIONS)):
            self.current_direction_index = action
        # Remove tail
        tail = self.my_blocks[-1]
        self.my_blocks = self.my_blocks[:-1]
        # Check new head
        new_head = tuple(np.array(self.my_blocks[0]) + self.DIRECTIONS[self.current_direction_index])
        # Add new head
        self.my_blocks = [new_head] + self.my_blocks
        return new_head, tail

class World:

    def __init__(self, size, n_sneks=1, n_food=1, add_walls=False):
        self.DEAD_REWARD = -1.0
        self.MOVE_REWARD = 0.1
        self.EAT_REWARD = 5.0
        self.FOOD = 64
        self.WALL = 255
        self.DIRECTIONS = Snek.DIRECTIONS
        self.add_walls = add_walls
        # Init a numpy matrix with zeros of predefined size
        self.size = size
        self.world = np.zeros(size)
        # Add walls if requested
        if add_walls:
            self.world[0, :] = self.WALL
            self.world[-1, :] = self.WALL
            self.world[:, 0] = self.WALL
            self.world[:, -1] = self.WALL
        # Compute all available_positions for food
        self.base_available_position = set(zip(*np.where(self.world == 0)))
        # Init sneks
        self.sneks = []
        for _ in range(n_sneks):
            snek = self.register_snek()
        # Set N foods
        self.place_food(n_food = n_food)

    def register_snek(self):
        # Choose position (between [4 and SIZE-4])
        # TODO better choice, no overlap
        SNEK_SIZE = 4
        p = (random.randint(SNEK_SIZE, self.size[0]-SNEK_SIZE), random.randint(SNEK_SIZE, self.size[1]-SNEK_SIZE))
        start_direction_index = random.randrange(len(Snek.DIRECTIONS))
        # Create snek and append
        new_snek = Snek(100 + 2*len(self.sneks), p, start_direction_index, SNEK_SIZE) # 100 +
        self.sneks.append(new_snek)
        return new_snek

    def get_alive_sneks(self):
        return [snek for snek in self.sneks if snek.alive]

    def place_food(self, n_food=1):
        # Update the available_positions from sneks
        available_positions = self.base_available_position
        for snek in self.get_alive_sneks():
            available_positions = available_positions - set(snek.my_blocks)
        # Place food objects
        for _ in range(n_food):
            # Choose a place
            choosen_position = random.choice(list(available_positions))
            self.world[choosen_position[0], choosen_position[1]] = self.FOOD
            # Remove the current choice for next steps
            available_positions.remove(choosen_position)

    def get_observation(self):
        obs = self.world.copy()
        # Draw snek over the world
        for snek in self.get_alive_sneks():
            for block in snek.my_blocks:
                obs[block[0], block[1]] = snek.snek_id
            # Highlight head
            obs[snek.my_blocks[0][0], snek.my_blocks[0][1]] = snek.snek_id + 1
        return obs

    # Move the selected snek
    # Returns reward and done flag
    def move_snek(self, actions):
        rewards = [0] * len(self.sneks)
        dones = []
        new_food_needed = 0 #Will be used for the food update
        for i, (snek, action) in enumerate(zip(self.sneks, actions)):
            if not snek.alive:
                continue
            new_snek_head, old_snek_tail = snek.step(action)
            # Check if snek is outside bounds
            if not (0 <= new_snek_head[0] < self.size[0]) or not(0 <= new_snek_head[1] < self.size[1]) or self.world[new_snek_head[0], new_snek_head[1]] == self.WALL:
                snek.my_blocks = snek.my_blocks[1:]
                snek.alive = False
            # Check if snek eats himself
            elif new_snek_head in snek.my_blocks[1:]:
                snek.alive = False
            # Check if snek is eating another snek
            for j, other_snek in enumerate(self.sneks):
                if i != j and other_snek.alive:
                    # Check if heads collided
                    if new_snek_head == other_snek.my_blocks[0]:
                        snek.alive = False
                        other_snek.alive = False
                    # Check head collided with another snek body
                    elif new_snek_head in other_snek.my_blocks[1:]:
                        snek.alive = False
            # Check if snek eats something
            if snek.alive and self.world[new_snek_head[0], new_snek_head[1]] == self.FOOD:
                # Remove old food
                self.world[new_snek_head[0], new_snek_head[1]] = -1
                # Add tail again
                snek.my_blocks.append(old_snek_tail)
                # Request to place new food. New food creation cannot be called here directly, need to update all sneks before
                new_food_needed = new_food_needed + 1
                rewards[i] = self.EAT_REWARD
            elif snek.alive:
                # Didn't eat anything, move reward
                rewards[i] = self.MOVE_REWARD
        # Compute done flags and assign dead rewards
        dones = [not snek.alive for snek in self.sneks]
        rewards = [r if snek.alive else self.DEAD_REWARD for r, snek in zip(rewards, self.sneks)]
		#Adding new food.
        if new_food_needed > 0:
            self.place_food(n_food = new_food_needed)
        return rewards, dones

'''
    Configurable single snek environment.
    Parameters:
        - SIZE: size of the world (default: 16x16)
        - FOOD: number of foods in the world at a given time (default: 1)
        - OBSERVATION_MODE: return a raw observation (block ids) or RGB observation
            - Layered observation: each channel of the state represent different entities: food, snek, enemies, obstacles
        - OBS_ZOOM: zoom the observation (only for RGB mode, FIXME)
        - STEP_LIMIT: hard step limit of the environment
        - DYNAMIC_STEP_LIMIT: step limit from the last eaten food (HUNGER)
        - DIE_ON_EAT: set a low difficulty, episode ends after eating the first piece
'''
class MultiSneks(gym.Env):

    metadata = {
        'render.modes': ['human','rgb_array'],
        'observation.types': ['raw', 'rgb', 'layered']
    }

    def __init__(self, size=(12,12), num_agents=2, step_limit=1000, dynamic_step_limit=1000, obs_type='raw', obs_zoom=1, n_food=1, render_zoom=20, add_walls=False):
        # Set size of the game world
        self.SIZE = size
        self.N_SNEKS = num_agents
        self.alive = None
        # Set step limit
        self.STEP_LIMIT = step_limit
        # Set dynamic step limit (hunger)
        self.DYNAMIC_STEP_LIMIT = dynamic_step_limit
        self.hunger = 0
        # Walls flag
        self.add_walls = add_walls
        # Create world
        self.n_food = n_food
        self.world = World(self.SIZE, n_sneks=self.N_SNEKS, n_food=self.n_food, add_walls=self.add_walls)
        # Set observation type and space
        self.obs_type = obs_type
        if self.obs_type == 'raw':
            self.observation_space = spaces.Box(low=0, high=255, shape=(num_agents, self.SIZE[0]*obs_zoom, self.SIZE[1]*obs_zoom), dtype=np.uint8)
        elif self.obs_type == 'rgb':
            self.observation_space = spaces.Box(low=0, high=255, shape=(num_agents, self.SIZE[0]*obs_zoom, self.SIZE[1]*obs_zoom, 3), dtype=np.uint8)
            self.RGBify = RGBifier(self.SIZE, zoom_factor = obs_zoom, players_colors={})
        elif self.obs_type == 'layered':
            # Only 2 layers here, food and snek
            self.observation_space = spaces.Box(low=0, high=255, shape=(num_agents, self.SIZE[0]*obs_zoom, self.SIZE[1]*obs_zoom, 2), dtype=np.uint8)
        else:
            raise(Exception('Unrecognized observation mode.'))
        # Action space
        self.action_space = spaces.Discrete(len(self.world.DIRECTIONS))
        # Set renderer
        self.RENDER_ZOOM = render_zoom
        self.renderer = None

    def step(self, actions):
        # Check if game is ended (raise exception otherwise)
        if self.alive is None or not any(self.alive):
            raise Exception('Need to reset env now.')
        # Check hard and dynamic step limit before performing the action
        self.current_step += 1
        '''
        if (self.current_step >= self.STEP_LIMIT) or (self.hunger > self.DYNAMIC_STEP_LIMIT):
            self.alive = False
            return self._get_state(), 0, True, {}
        '''
        # Perform the action
        rewards, dones = self.world.move_snek(actions)
        # Update and check hunger
        self.hunger = [h+1 if r <= 0 else 0 for h, r in zip(self.hunger, rewards)]
        # Disable interactions if snek has died
        self.alive = [not done for done in dones]
        return self._get_state(), rewards, dones, {}

    # def flip(a, b, c):
    #     ret = []
    #     for i in range(len(a)):
    #         temp = []
    #         temp.append(a[i])
    #         temp.append(b[i])
    #         temp.append(c[i])
    #         ret.append(temp)
    #     return ret

    def reset(self):
        # Reset step counters
        self.current_step = 0
        self.alive = [True] * self.N_SNEKS
        self.hunger = [0] * self.N_SNEKS
        # Create world
        self.world = World(self.SIZE, n_sneks=self.N_SNEKS, n_food=self.n_food, add_walls=self.add_walls)
        return self._get_state()

    def seed(self, seed):
        random.seed(seed)

    def _get_state(self):
        _state = self.world.get_observation()
        if self.obs_type == 'rgb':
            return self.RGBify.get_image(_state)
        elif self.obs_type == 'layered':
            s = np.array([(_state == self.world.FOOD).astype(int), ((_state == self.world.sneks[0].snek_id) or (_state == self.world.sneks[0].snek_id+1)).astype(int)])
            s = np.transpose(s, [1, 2, 0])
            return s
        else:
            return np.array([_state, _state])

    # def get_multi_state(self):
    #     _state = self.world.get_observation()
    #     if self.obs_type == 'rgb':
    #         return self.RGBify.get_image(_state)
    #     elif self.obs_type == 'layered':
    #         s = np.array([(_state == self.world.FOOD).astype(int), ((_state == self.world.sneks[0].snek_id) or (_state == self.world.sneks[0].snek_id+1)).astype(int)])
    #         s = np.transpose(s, [1, 2, 0])
    #         return s
    #     else:
    #         return self.convert(_state)
    def simplify(self, state, id):
        copy = state.copy()
        for i in range(len(state)):
            for j in range(len(state[i])):
                if state[i][j] == 64:
                    copy[i][j] = -10
                elif state[i][j] == id:
                    copy[i][j] = 10
                elif state[i][j] >= 100:
                    copy[i][j] = 1
                else:
                    copy[i][j] = 0
        return copy
    
    def convert(self, state):
        allsneks = []
        #allsneks.append(state)
        state = self.world.get_observation()
        allsneks.append(self.simplify(state, self.world.sneks[0]))
        for i in range(1, self.N_SNEKS):
            temp_2d = []
            for row in state:
                temp_row = []
                for col in row:
                    if col >= 100: # 100
                        col -= (100 + 2 * i) # 100 +
                        col %= (self.N_SNEKS * 2) 
                        col += 100 
                    temp_row.append(col)
                temp_2d.append(temp_row)
            allsneks.append(np.array(self.simplify(temp_2d, self.world.sneks[i])))
        return np.array(allsneks)


    def render(self, mode='human', close=False):
        if not close:
            # Renderer lazy loading
            if self.renderer is None:
                self.renderer = Renderer(self.SIZE, zoom_factor = self.RENDER_ZOOM, players_colors={})
            return self.renderer._render(self.world.get_observation(), mode=mode, close=False)

    def close(self):
        if self.renderer:
            self.renderer.close()
            self.renderer = None

# import time

# if __name__ == "__main__":
#     env = MultiSneks()
#     obs = env.reset()
#     for i in range(1000):
#         time.sleep(1)
#         env.render(mode='human')
#         action = env.action_space.sample()
#         obs, reward, done, _ = env.step([action])
#         print("Reward: ", reward)
#         print(env._get_state())
#         if done:
#             obs = env.reset()
