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

    def set_reward(self, dead=-1, fruit=1, idle=0):
        self.dead_reward = dead
        self.fruit_reward = fruit
        self.idle_reward = idle
        print('Set reward, dead: {}, fruit: {}, idle: {}'.format(dead, fruit, idle))

    def step(self, action):
        self.last_obs, rewards, done, info = self.controller.step(action)
        return self.last_obs, rewards, done, info

    def reset(self):
        self.controller = Controller(self.grid_size, self.unit_size, self.unit_gap, self.snake_size, self.n_snakes, self.n_foods, random_init=self.random_init,
                                    dead_reward=self.dead_reward, fruit_reward=self.fruit_reward, idle_reward=self.idle_reward)
        self.last_obs = self.controller.grid.grid.copy()
        return self.last_obs

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

    def seed(self, x):
        pass

    def obs_wapper(self):
        # last_obs -> new obs

        wrapped_obs = np.zeros(1)

        # fruit 

