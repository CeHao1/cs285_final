import gym
import gym_snake

import time


class SnakeBuilder:
    def __init__(self, grid_size=[10, 10], snake_size=3, n_snakes=1, n_foods=1):
        self.env = gym.make('snake-v0')
        self.env.grid_size=grid_size 
        self.env.snake_size = snake_size
        self.env.n_snakes = n_snakes
        self.env.n_foods = n_foods


    def set_reward(self, dead, fruit, idle):
        self.env.set_reward(dead, fruit, idle)

    def test_env(self, max_iteration=1000):
        observation = self.env.reset()

        for iter in range(max_iteration):
            actions = [self.env.action_space.sample() for i in range(self.env.n_snakes)]
            obs, rewards, done, info = self.env.step(actions)
            self.env.render()
            print('rewards at iter {}: {}'.format(iter, rewards))
            time.sleep(0.01)

            if done:
                print('!! game is done')
                break

    def get_env(self):
        return self.env


    