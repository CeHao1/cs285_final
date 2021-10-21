# cs285_final

## install env
python 3.6 -3.8 anaconda env  
cd to the repo, and run    
``` pip install -e .  ```


## how to build snake
See example in src/snake_example.py

```
from gym_snake.snake_builder import SnakeBuilder

builder = SnakeBuilder(grid_size=[10, 10], snake_size=3, n_snakes=2, n_foods=2)
builder.set_reward(dead=-10, food=1, idle=-0.1)
builder.test_env()    
env = builder.get_env()
```

directly run:   
python src/snake_example.py 
