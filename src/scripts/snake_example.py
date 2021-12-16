import matplotlib.pyplot as plt
from gym_snake.snake_builder import SnakeBuilder

def main():

    builder = SnakeBuilder(grid_size=[15, 15], snake_size=3, n_snakes=1, n_foods=1)
    builder.set_reward(dead=-10, food=1, idle=-0.1)
    builder.test_env()    
    env = builder.get_env()


    # after reset the env, we can obtain the color of each snake, and also action space
    env.reset()

    # color of observation
    color_map = env.get_color()
    print(color_map)

    # agent action space
    action_space = env.action_space
    print(action_space)
    '''
    0: up
    1: right
    2: down
    3: left
    '''

if __name__ == '__main__':
    main()