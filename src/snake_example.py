import matplotlib.pyplot as plt
from gym_snake.snake_builder import SnakeBuilder

def main():

    builder = SnakeBuilder(grid_size=[40, 40], snake_size=3, n_snakes=2, n_foods=2)
    builder.set_reward(dead=-10, food=1, idle=-0.1)
    builder.test_env()    
    env = builder.get_env()

if __name__ == '__main__':
    main()