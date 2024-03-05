import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class OthelloEnv(gym.Env):
    metadata = {'render_modes': ['pygame'], 'render_fps': 4}

    def __init__(self, render_mode=None):
        self.observation_space = spaces.Box(low=0, high=2, shape=(8, 8), dtype=np.int8)
        self.action_space = spaces.Box(low=0, high=1, shape=(8, 8), dtype=np.float32)  # How much the model wants to place at each space
        self.state = np.zeros((8, 8), dtype=np.int8)
        self.reset()

        self.render_mode = render_mode
        self.window_size = 800
        self.window = None
        self.clock = None

    def get_valid_moves(self):
        valid_moves = np.zeros((8, 8), dtype=bool)
        for i in range(8):
            for j in range(8):
                if self.state[i, j] != 0:
                    continue
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        if x == 0 and y == 0:
                            continue
                        if self.is_valid(i, j, x, y):
                            valid_moves[i, j] = True
        return valid_moves

    def is_valid(self, x, y, dx, dy):
        x += dx
        y += dy
        if x < 0 or x >= 8 or y < 0 or y >= 8 or self.state[x, y] != 3 - self.turn:
            return False
        x += dx
        y += dy
        while x >= 0 and x < 8 and y >= 0 and y < 8:
            if self.state[x, y] == 0:
                return False
            if self.state[x, y] == self.turn:
                return True
            x += dx
            y += dy
        return False

    def flip(self, x, y):
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                if self.is_valid(x, y, dx, dy):
                    x0, y0 = x, y
                    x0 += dx
                    y0 += dy
                    while self.state[x0, y0] == 3 - self.turn:
                        self.state[x0, y0] = self.turn
                        x0 += dx
                        y0 += dy

    def calculate_reward(self):
        # Example weights for each factor
        win_loss_weight = 1.0
        piece_diff_weight = 0.5
        mobility_weight = 0.2

        # Determine the winner of the game
        if np.all(self.state != 0):
            winner = 1 if self.score[0] > self.score[1] else 2
        else:
            winner = 0

        # 100 if I won, -100 if I lost, 0 if game not over yet
        win_loss_reward = 0
        if winner == 0:
            win_loss_reward = 0
        else:
            win_loss_reward = 100 if winner == self.turn else -100

        # Calculate piece differentials
        piece_diff_reward = self.score[self.turn - 1] - self.score[2 - self.turn]

        # Calculate mobility differential
        my_move_count = np.sum(self.valid_moves)
        self.turn = 3 - self.turn
        their_move_count = np.sum(self.get_valid_moves())
        self.turn = 3 - self.turn
        mobility_reward = my_move_count - their_move_count

        # Calculate reward based on combination of factors
        reward = win_loss_weight * win_loss_reward + \
                piece_diff_weight * piece_diff_reward + \
                mobility_weight * mobility_reward

        return reward

    def step(self, action):
        # If no valid moves, return negative reward
        valid_moves_mask = self.get_valid_moves()
        if not np.any(valid_moves_mask):
            reward = self.calculate_reward()
            return self.state, reward, False, False, {}

        # Add a small constant to action to avoid zero values
        action += np.finfo(float).eps

        # Mask action with valid moves mask & normalize
        masked_action_space = action * valid_moves_mask

        # Normalize the masked action space
        normalized_action_space = masked_action_space / (np.sum(masked_action_space) + np.finfo(float).eps)

        # Set the probabilities of invalid moves to zero
        normalized_action_space = normalized_action_space * valid_moves_mask

        # Flatten the array and create an array of indices
        flat_normalized_action_space = normalized_action_space.flatten()
        indices = np.arange(len(flat_normalized_action_space))

        # Ensure the probabilities sum to 1
        flat_normalized_action_space = flat_normalized_action_space / np.sum(flat_normalized_action_space)

        # Sample from the flattened array based on the values
        sampled_index = np.random.choice(indices, p=flat_normalized_action_space)

        # Convert the sampled index back to 2D coordinates
        move = np.unravel_index(sampled_index, shape=(8, 8))

        # Take move
        x, y = move
        self.state[x, y] = self.turn
        self.flip(x, y)
        self.turn = 3 - self.turn
        self.valid_moves = self.get_valid_moves()
        self.score = [np.sum(self.state == 1), np.sum(self.state == 2)]
        self.done = not np.any(self.valid_moves)

        # Reward and info
        observation = self.state
        reward = self.calculate_reward()
        terminated = False
        truncated = False
        info = {}

        if self.done:
            self.winner = 1 if self.score[0] > self.score[1] else 2
            return observation, reward, terminated, truncated, info
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        super().reset(seed=seed)

        self.state = np.zeros((8, 8), dtype=np.int8)
        self.state[3, 3] = 1
        self.state[3, 4] = 2
        self.state[4, 3] = 2
        self.state[4, 4] = 1
        self.turn = 1
        self.done = False
        self.winner = 0
        self.valid_moves = self.get_valid_moves()
        self.score = [2, 2]

        return self.state, None

    def render(self):
        if self.window is None and self.render_mode == 'pygame':
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == 'pygame':
            self.clock = pygame.time.Clock()

        if self.render_mode == 'pygame':
            self.window.fill((0, 0, 0))
            for i in range(8):
                for j in range(8):
                    if self.state[i, j] == 1:
                        pygame.draw.circle(self.window, (0, 0, 0), ( int((j + 0.5) * self.window_size / 8), int((i + 0.5) * self.window_size / 8)), int(self.window_size / 16))
                    elif self.state[i, j] == 2:
                        pygame.draw.circle(self.window, (255, 255, 255), (int((j + 0.5) * self.window_size / 8), int((i + 0.5) * self.window_size / 8)), int(self.window_size / 16))
                    if self.valid_moves[i, j]:
                        pygame.draw.circle(self.window, (0, 255, 0), (int((j + 0.5) * self.window_size / 8), int((i + 0.5) * self.window_size / 8)), int(self.window_size / 32))
            pygame.display.flip()
            self.clock.tick(self.metadata['render_fps'])

    def close(self):
        if self.render_mode == 'pygame':
            pygame.quit()

