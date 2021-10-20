import gym
import gym_snake

import time
import matplotlib.pyplot as plt


def main():
    # Construct Environment
    env = gym.make('snake-v0')
    env.grid_size=[30,30] 
    env.snake_size = 3
    env.n_snakes = 2
    env.n_foods = 20


    observation = env.reset() # Constructs an instance of the game
    env.render()
    

    for i in range(1000):
        actions = [env.action_space.sample() for i in range(env.n_snakes) ]
        obs, rewards, done, info = env.step(actions)
        env.render()
        time.sleep(0.01)
        print(rewards)

        

    # Controller
    game_controller = env.controller

    # Grid
    grid_object = game_controller.grid
    grid_pixels = grid_object.grid

    # Snake(s)
    snakes_array = game_controller.snakes
    snake_object1 = snakes_array[0]



if __name__ == '__main__':
    main()