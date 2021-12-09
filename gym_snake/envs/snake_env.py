import os, subprocess, time, signal
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_snake.envs.snake import Controller, Discrete
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: see matplotlib documentation for installation https://matplotlib.org/faq/installing_faq.html#installation".format(e))


class Direction:
    '''
        0: up
        1: right
        2: down
        3: left
        '''
    U = 0
    R = 1
    D = 2
    L = 3

class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        print('---------- initilize snake setting -----------')
        self.set_parameters()
        self.set_reward()
        print('------------- End snake setting --------------\n')
        
    
    def set_parameters(self, grid_size=[15,15], unit_size=1, unit_gap=0, snake_size=3, n_snakes=1, n_foods=1, random_init=True):
        self.grid_size = grid_size
        self.unit_size = unit_size
        self.unit_gap = unit_gap
        self.snake_size = snake_size
        self.n_snakes = n_snakes
        self.n_foods = n_foods
        self.viewer = None
        self.action_space = Discrete(4)
        self.random_init = random_init
        print('Set parameters, grid_size: {}, snake_size: {}, n_snakes: {}, n_foods: {}'.format(grid_size, snake_size, n_snakes, n_foods))

    def set_reward(self, dead=-1, food=1, idle=0, dist=0):
        self.dead_reward = dead
        self.food_reward = food
        self.idle_reward = idle
        self.dist_reward = dist
        print('Set reward, dead: {}, food: {}, idle: {}, dist: {}'.format(dead, food, idle, dist))

    def step(self, action):

        de_rotated_action = self.de_rotate_action(action)

        self.last_obs, rewards, done, info = self.controller.step(de_rotated_action)

        a = de_rotated_action # update new heading
        if a == 0:
            self.head_direction = Direction.U
        elif a == 1:
            self.head_direction = Direction.R
        elif a == 2:
            self.head_direction = Direction.D
        elif a == 3:
            self.head_direction = Direction.L

        return self.obs_wapper(), rewards, done, info

    def reset(self):
        self.controller = Controller(self.grid_size, self.unit_size, self.unit_gap, self.snake_size, self.n_snakes, self.n_foods, random_init=self.random_init,
                                    dead_reward=self.dead_reward, food_reward=self.food_reward, idle_reward=self.idle_reward, dist_reward=self.dist_reward)
        self.last_obs = self.controller.grid.grid.copy()

        self.head_direction = Direction.D
        return self.obs_wapper()

    def render(self, mode='human', close=False, frame_speed=.1):
        if self.viewer is None:
            self.fig = plt.figure()
            self.viewer = self.fig.add_subplot(111)
            plt.ion()
            self.fig.show()
        else:
            self.viewer.clear()
            self.viewer.imshow(self.last_obs)
            plt.pause(frame_speed)
        self.fig.canvas.draw()
        return self.last_obs

    def seed(self, x):
        pass

    def obs_wapper(self):
        # only fetch one layer of image
        raw_obs = self.last_obs

        rotated_obs = self.rotate_obs(raw_obs)

        wrapped_obs = np.ones(rotated_obs.shape, dtype=np.uint8) * 255 - rotated_obs
        wrapped_obs = np.ndarray.flatten(wrapped_obs) 
        wrapped_obs.dtype = np.uint8

        idx = np.where(wrapped_obs==1)[0]
        wrapped_obs[idx] = 0
        wrapped_obs = np.clip(wrapped_obs, None, 1)

        return wrapped_obs

    def get_color(self):
        color_map = {'snake_head':[], 'snake_body':[], 'food': None, 'space': None}
        color_map['food'] = np.array(255, dtype=np.uint8) - self.controller.grid.FOOD_COLOR
        color_map['space'] = np.array(255, dtype=np.uint8) - self.controller.grid.SPACE_COLOR

        for snake in self.controller.snakes:
            color_map['snake_head'].append( np.array(255, dtype=np.uint8) - snake.head_color)
            color_map['snake_body'].append( np.array(255, dtype=np.uint8) - snake.body_color)

        # print(color_map)
        return color_map

    def rotate_obs(self, raw_obs):
        '''
        0: up
        1: right
        2: down
        3: left
        '''


        if self.head_direction == Direction.U:
            rotated_obs = raw_obs

        elif self.head_direction == Direction.R:
            rotated_obs = [np.rot90(d1_obs, 1) for d1_obs in raw_obs] # 逆时针90

        elif self.head_direction == Direction.D:
            rotated_obs = [np.rot90(d1_obs, 2) for d1_obs in raw_obs] # 逆时针180

        elif self.head_direction == Direction.L:
            rotated_obs = [np.rot90(d1_obs, 3) for d1_obs in raw_obs] # 逆时针270

        return np.array(rotated_obs)

    def de_rotate_action(self, action):

        
        if self.head_direction == Direction.U:
            de_rotated_action = action + 0

        elif self.head_direction == Direction.R:
            de_rotated_action = action + 1

        elif self.head_direction == Direction.D:
            de_rotated_action = action + 2

        elif self.head_direction == Direction.L:
            de_rotated_action = action + 3

        if de_rotated_action > 3:
            de_rotated_action -= 4

        # print('\n===\ninitial head direction ', self.head_direction)
        # print('original action is ', action)
        # print('de_rotated actin is ', de_rotated_action)

        return de_rotated_action