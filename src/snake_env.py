import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(self, render_mode=None, grid_size=20, block_size=20, reward_shaping=False):
        super().__init__()
        self.grid_size = grid_size
        self.block_size = block_size
        self.window_size = self.grid_size * self.block_size
        self.reward_shaping = reward_shaping
        
        # Actions: 0: Straight, 1: Right turn, 2: Left turn
        self.action_space = spaces.Discrete(3)
        
        # State: 11 boolean values -> Box(0, 1, shape=(11,), dtype=np.int8)
        # [danger_straight, danger_right, danger_left,
        #  dir_l, dir_r, dir_u, dir_d,
        #  food_l, food_r, food_u, food_d]
        self.observation_space = spaces.Box(low=0, high=1, shape=(11,), dtype=np.int8)
        
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initial snake: center of grid
        self.head = [self.grid_size // 2, self.grid_size // 2]
        self.snake = [
            self.head.copy(),
            [self.head[0] - 1, self.head[1]],
            [self.head[0] - 2, self.head[1]]
        ]
        
        # Direction: [x, y]
        # Right: [1, 0], Left: [-1, 0], Up: [0, -1], Down: [0, 1]
        self.direction = [1, 0]
        
        self.food = self._place_food()
        self.score = 0
        self.frame_iteration = 0
        
        if self.render_mode == "human":
            self._render_frame()
            
        return self._get_state(), {}
        
    def _place_food(self):
        while True:
            food = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
            if food not in self.snake:
                return food

    def step(self, action):
        self.frame_iteration += 1
        
        # Calculate new direction based on action
        # Actions: 0: Straight, 1: Right turn, 2: Left turn
        clock_wise = [[1, 0], [0, 1], [-1, 0], [0, -1]]
        idx = clock_wise.index(self.direction)
        
        if action == 0:
            new_dir = clock_wise[idx]
        elif action == 1:
            new_dir = clock_wise[(idx + 1) % 4]
        else: # action == 2
            new_dir = clock_wise[(idx - 1) % 4]
            
        self.direction = new_dir
        
        # Move head
        self.head = [self.head[0] + self.direction[0], self.head[1] + self.direction[1]]
        self.snake.insert(0, self.head.copy())
        
        reward = 0
        terminated = False
        
        # Check game over (collision with wall or self)
        if self._is_collision() or self.frame_iteration > 100 * len(self.snake):
            terminated = True
            reward = -10
            return self._get_state(), reward, terminated, False, {}
            
        # Check if food eaten
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.food = self._place_food()
        else:
            self.snake.pop()
            if self.reward_shaping:
                # Closer to food is good, further is bad
                # Using Manhattan distance
                old_dist = abs(self.snake[1][0] - self.food[0]) + abs(self.snake[1][1] - self.food[1])
                new_dist = abs(self.head[0] - self.food[0]) + abs(self.head[1] - self.food[1])
                if new_dist < old_dist:
                    reward = 0.1
                else:
                    reward = -0.1
            
        if self.render_mode == "human":
            self._render_frame()
            
        return self._get_state(), reward, terminated, False, {}

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Wall collision
        if pt[0] < 0 or pt[0] >= self.grid_size or pt[1] < 0 or pt[1] >= self.grid_size:
            return True
        # Self collision
        if pt in self.snake[1:]:
            return True
        return False
        
    def _get_state(self):
        head_x, head_y = self.head
        
        # point straight, right, left
        clock_wise = [[1, 0], [0, 1], [-1, 0], [0, -1]] # R, D, L, U
        idx = clock_wise.index(self.direction)
        dir_s = clock_wise[idx]
        dir_r = clock_wise[(idx + 1) % 4]
        dir_l = clock_wise[(idx - 1) % 4]
        
        pt_s = [head_x + dir_s[0], head_y + dir_s[1]]
        pt_r = [head_x + dir_r[0], head_y + dir_r[1]]
        pt_l = [head_x + dir_l[0], head_y + dir_l[1]]
        
        dir_l_bool = self.direction == [-1, 0]
        dir_r_bool = self.direction == [1, 0]
        dir_u_bool = self.direction == [0, -1]
        dir_d_bool = self.direction == [0, 1]
        
        state = [
            # Danger
            int(self._is_collision(pt_s)), # Danger Straight
            int(self._is_collision(pt_r)), # Danger Right
            int(self._is_collision(pt_l)), # Danger Left
            
            # Move direction
            int(dir_l_bool),
            int(dir_r_bool),
            int(dir_u_bool),
            int(dir_d_bool),
            
            # Food location 
            int(self.food[0] < self.head[0]),  # food left
            int(self.food[0] > self.head[0]),  # food right
            int(self.food[1] < self.head[1]),  # food up
            int(self.food[1] > self.head[1])   # food down
        ]
        
        return np.array(state, dtype=np.int8)
        
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Snake RL")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
            
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((0, 0, 0)) # black background
        
        # Draw snake
        for i, pt in enumerate(self.snake):
            color = (0, 255, 0) if i == 0 else (0, 200, 0)
            pygame.draw.rect(
                canvas, color,
                pygame.Rect(pt[0] * self.block_size, pt[1] * self.block_size, self.block_size, self.block_size)
            )
            
        # Draw food
        pygame.draw.rect(
            canvas, (255, 0, 0),
            pygame.Rect(self.food[0] * self.block_size, self.food[1] * self.block_size, self.block_size, self.block_size)
        )
        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
