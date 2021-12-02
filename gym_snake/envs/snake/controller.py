from gym_snake.envs.snake import Snake
from gym_snake.envs.snake import Grid
import numpy as np

class Controller():
    """
    This class combines the Snake, Food, and Grid classes to handle the game logic.
    """

    def __init__(self, grid_size=[30,30], unit_size=10, unit_gap=1, snake_size=3, n_snakes=1, n_foods=1, random_init=True,
                    dead_reward=-1, food_reward=1, idle_reward=0, dist_reward=0):

        # assert n_snakes < grid_size[0]//3
        assert n_snakes < 25
        # assert snake_size < grid_size[1]//2
        assert unit_gap >= 0 and unit_gap < unit_size

        self.snakes_remaining = n_snakes
        self.grid = Grid(grid_size, unit_size, unit_gap)
        self.dead_reward = dead_reward
        self.food_reward = food_reward
        self.idle_reward = idle_reward
        self.dist_reward = dist_reward

        self.snakes = []
        self.dead_snakes = []
        for i in range(1,n_snakes+1):
            start_coord = [i*grid_size[0]//(n_snakes+1), snake_size+1]
            head_color = [self.grid.HEAD_COLOR[0], self.grid.HEAD_COLOR[1] - i*10, 0]
            body_color = [self.grid.BODY_COLOR[0], self.grid.BODY_COLOR[1] - i*10, 0]
            self.snakes.append(Snake(start_coord, snake_size, head_color, body_color))

            self.grid.draw_snake(self.snakes[-1], self.snakes[-1].head_color, self.snakes[-1].body_color)
            self.dead_snakes.append(None)

        if not random_init:
            for i in range(2,n_foods+2):
                start_coord = [i*grid_size[0]//(n_foods+3), grid_size[1]-5]
                self.grid.place_food(start_coord)
        else:
            for i in range(n_foods):
                self.grid.new_food()



    def move_snake(self, direction, snake_idx):
        """
        Moves the specified snake according to the game's rules dependent on the direction.
        Does not draw head and does not check for reward scenarios. See move_result for these
        functionalities.
        """

        snake = self.snakes[snake_idx]
        if type(snake) == type(None):
            return

        # Cover old head position with body
        self.grid.cover(snake.head, self.grid.BODY_COLOR)
        # Erase tail without popping so as to redraw if food eaten
        self.grid.erase(snake.body[0])
        # Find and set next head position conditioned on direction
        snake.action(direction)

    def move_result(self, direction, snake_idx=0):
        """
        Checks for food and death collisions after moving snake. Draws head of snake if
        no death scenarios.
        """

        snake = self.snakes[snake_idx]
        if type(snake) == type(None):
            return 0, 0

        # Check for death of snake
        if self.grid.check_death(snake.head): # dead
            self.dead_snakes[snake_idx] = self.snakes[snake_idx]
            self.snakes[snake_idx] = None
            self.grid.cover(snake.head, snake.head_color) # Avoid miscount of grid.open_space
            self.grid.connect(snake.body.popleft(), snake.body[0], self.grid.SPACE_COLOR)
            reward = self.dead_reward
            ate_food = 0
        # Check for reward
        elif self.grid.food_space(snake.head): # eat food
            self.grid.draw(snake.body[0], self.grid.BODY_COLOR) # Redraw tail
            self.grid.connect(snake.body[0], snake.body[1], self.grid.BODY_COLOR)
            self.grid.cover(snake.head, snake.head_color) # Avoid miscount of grid.open_space
            reward = self.food_reward
            ate_food = 1
            self.grid.new_food()
        else:
            reward = self.idle_reward # idle
            ate_food = 0
            empty_coord = snake.body.popleft()
            self.grid.connect(empty_coord, snake.body[0], self.grid.SPACE_COLOR)
            self.grid.draw(snake.head, snake.head_color)

        self.grid.connect(snake.body[-1], snake.head, self.grid.BODY_COLOR)

        if self.dist_reward !=0: # dist reward
            food_coord = self.grid.food_coords()
            reward_dist = self.distance_reward(snake.head, food_coord)
            reward += reward_dist

        return reward, ate_food

    def kill_snake(self, snake_idx):
        """
        Deletes snake from game and subtracts from the snake_count 
        """
        
        assert self.dead_snakes[snake_idx] is not None
        self.grid.erase(self.dead_snakes[snake_idx].head)
        self.grid.erase_snake_body(self.dead_snakes[snake_idx])
        self.dead_snakes[snake_idx] = None
        self.snakes_remaining -= 1

    def step(self, directions):
        """
        Takes an action for each snake in the specified direction and collects their rewards
        and dones.

        directions - tuple, list, or ndarray of directions corresponding to each snake.
        """

        # Ensure no more play until reset
        if self.snakes_remaining < 1 or self.grid.open_space < 1:

            info = {"snakes_remaining":self.snakes_remaining, 
                "ate_foods": [0]}

            if type(directions) == type(int()) or len(directions) is 1:
                return self.grid.grid.copy(), 0, True, info
            else:
                return self.grid.grid.copy(), [0]*len(directions), True, info

        rewards = []
        ate_foods = []

        # if type(directions) == type(int()):
        #     directions = [directions]
        if not isinstance(directions, list):
            directions = [directions]

        for i, direction in enumerate(directions):
            if self.snakes[i] is None and self.dead_snakes[i] is not None:
                self.kill_snake(i)
            self.move_snake(direction,i)
            reward, ate_food = self.move_result(direction, i)
            rewards.append(reward)
            ate_foods.append(ate_food)

        done = self.snakes_remaining < 1 or self.grid.open_space < 1

        if len(rewards) is 1:
            info = {"snakes_remaining":self.snakes_remaining, 
                "ate_foods": ate_foods[0]}
            # print('info1, ', info)    
            return self.grid.grid.copy(), rewards[0], done, info

        else:
            info = {"snakes_remaining":self.snakes_remaining, 
                "ate_foods": ate_foods}
            # print('info2, ', info)
            return self.grid.grid.copy(), rewards, done, info


    # extra rewards

    def distance_reward(self, head, food_coords):
        dist = np.sqrt( (head[0] - food_coords[:,0])**2 + (head[1] - food_coords[:,1])**2)
        min_dist = np.min(dist)
        dist_reward = 1/min_dist * self.dist_reward
        # print('min dist ', min_dist)
        return dist_reward