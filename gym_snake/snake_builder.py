import gym
import gym_snake

import time


class SnakeBuilder:
    def __init__(self, grid_size=[15, 15], snake_size=3, n_snakes=1, n_foods=1):
        self.env = gym.make('snake-v0')
        self.env.grid_size=[30,30] 
        self.env.snake_size = 3
        self.env.n_snakes = 2
        self.env.n_foods = 20


    def set_reward(self, dead, fruit, idle):
        self.env.set_reward(dead, fruit, idle)


    def get_env(self):
        return self.env


    