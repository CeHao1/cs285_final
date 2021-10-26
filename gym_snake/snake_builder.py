import gym
import gym_snake

import time


class SnakeBuilder:
    def __init__(self, grid_size=[15, 15], snake_size=3, n_snakes=1, n_foods=1):
        self.env = gym.make('snake-v0')
        self.env.set_parameters(grid_size=grid_size, snake_size=snake_size, n_snakes=n_snakes, n_foods=n_foods)

    def set_reward(self, dead, food, idle, dist=0.2):
        self.env.set_reward(dead, food, idle, dist)

    def test_env(self, max_iteration=1000):
        observation = self.env.reset()

        for iter in range(max_iteration):
            actions = [self.env.action_space.sample() for i in range(self.env.n_snakes)]
            obs, rewards, done, info = self.env.step(actions)
            self.env.render()
            print('rewards at iter {}: {}'.format(iter, rewards))

            if done:
                print('!! game is done')
                break

    def get_env(self):
        return self.env


    