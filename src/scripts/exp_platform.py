from gym_snake.snake_builder import SnakeBuilder


class ExpPlatform:
    def __init__(self):
        
        builder = SnakeBuilder(grid_size=[15, 15], snake_size=3, n_snakes=1, n_foods=1)
        builder.set_reward(dead=-10, food=1, idle=-0.1)
        builder.test_env()    
        self.env = builder.get_env()
        self.color_map = env.get_color()
        self.action_space = env.action_space