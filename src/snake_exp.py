import gym
import gym_snake


def main():
    # Construct Environment
    env = gym.make('snake-v0')
    observation = env.reset() # Constructs an instance of the game
    env.render()

    # Controller
    game_controller = env.controller

    # Grid
    grid_object = game_controller.grid
    grid_pixels = grid_object.grid

    # Snake(s)
    snakes_array = game_controller.snakes
    snake_object1 = snakes_array[0]
    print('yes')


if __name__ == '__main__':
    main()