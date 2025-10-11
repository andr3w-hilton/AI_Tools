"""
Shared module for maze solver - contains neural network architecture,
maze environment with walls, and utility functions.
"""
import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque


class MazeSolver(nn.Module):
    """
    Convolutional Neural Network for solving mazes using Q-learning.
    Takes a 3-channel maze state (walls, player, goal) and outputs Q-values for 4 actions.
    """
    def __init__(self, maze_size, hidden_size=256, output_size=4):
        super(MazeSolver, self).__init__()
        self.maze_size = maze_size

        # Convolutional layers - now using 3 input channels (walls, player, goal)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Calculate flattened size after convolutions
        self.flatten_size = 64 * maze_size * maze_size

        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, hidden_size)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        # Input shape: (batch, channels, height, width) or (channels, height, width)
        if x.dim() == 3:
            x = x.unsqueeze(0)

        # Convolutional layers with batch normalization
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x


class Maze:
    """
    Maze environment with walls generated using depth-first search algorithm.
    State representation uses 3 channels: walls, player position, goal position.
    """
    def __init__(self, maze_size):
        self.maze_size = maze_size
        self.walls = self._generate_maze()
        self.start_pos = (1, 1)
        self.goal_pos = (maze_size - 2, maze_size - 2)
        self.reset()

    def _generate_maze(self):
        """
        Generate a maze using depth-first search algorithm.
        Returns a 2D numpy array where 1 = wall, 0 = path.
        """
        # Start with all walls
        maze = np.ones((self.maze_size, self.maze_size), dtype=np.float32)

        # DFS maze generation
        def carve_path(x, y):
            maze[x, y] = 0
            directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
            random.shuffle(directions)

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 1 <= nx < self.maze_size - 1 and 1 <= ny < self.maze_size - 1 and maze[nx, ny] == 1:
                    maze[x + dx // 2, y + dy // 2] = 0
                    carve_path(nx, ny)

        # Start carving from (1, 1)
        carve_path(1, 1)

        # Ensure start and goal are not walls
        maze[1, 1] = 0
        maze[self.maze_size - 2, self.maze_size - 2] = 0

        return maze

    def reset(self):
        """Reset the maze to initial state."""
        self.player_pos = self.start_pos
        self.steps_taken = 0
        return self._get_state()

    def _get_state(self):
        """
        Get the current state as a 3-channel tensor:
        Channel 0: Walls (1 = wall, 0 = path)
        Channel 1: Player position (1 at player location, 0 elsewhere)
        Channel 2: Goal position (1 at goal location, 0 elsewhere)
        """
        state = np.zeros((3, self.maze_size, self.maze_size), dtype=np.float32)

        # Channel 0: Walls
        state[0] = self.walls

        # Channel 1: Player position
        state[1, self.player_pos[0], self.player_pos[1]] = 1

        # Channel 2: Goal position
        state[2, self.goal_pos[0], self.goal_pos[1]] = 1

        return state

    def step(self, action):
        """
        Execute an action and return next state, reward, and done flag.
        Actions: 0=Up, 1=Down, 2=Left, 3=Right
        """
        row, col = self.player_pos

        # Calculate new position based on action
        if action == 0:  # Up
            new_pos = (row - 1, col)
        elif action == 1:  # Down
            new_pos = (row + 1, col)
        elif action == 2:  # Left
            new_pos = (row, col - 1)
        elif action == 3:  # Right
            new_pos = (row, col + 1)
        else:
            new_pos = (row, col)

        # Check if new position is valid (not a wall and within bounds)
        if (0 <= new_pos[0] < self.maze_size and
            0 <= new_pos[1] < self.maze_size and
            self.walls[new_pos[0], new_pos[1]] == 0):
            self.player_pos = new_pos
            moved = True
        else:
            moved = False

        self.steps_taken += 1

        # Calculate reward
        if self.player_pos == self.goal_pos:
            reward = 100.0  # Large reward for reaching goal
            done = True
        elif not moved:
            reward = -1.0  # Penalty for hitting wall
            done = False
        else:
            # Small reward based on getting closer to goal
            distance = np.sqrt((self.player_pos[0] - self.goal_pos[0]) ** 2 +
                             (self.player_pos[1] - self.goal_pos[1]) ** 2)
            reward = -0.1 + 0.5 / (distance + 1.0)
            done = False

        # Also end if taking too many steps
        if self.steps_taken > self.maze_size * self.maze_size * 2:
            done = True

        return self._get_state(), reward, done

    def render(self):
        """Print a simple text representation of the maze."""
        for i in range(self.maze_size):
            for j in range(self.maze_size):
                if (i, j) == self.player_pos:
                    print('P', end='')
                elif (i, j) == self.goal_pos:
                    print('G', end='')
                elif self.walls[i, j] == 1:
                    print('#', end='')
                else:
                    print(' ', end='')
            print()


class ReplayBuffer:
    """
    Experience replay buffer for stable Q-learning.
    Stores transitions and samples random minibatches for training.
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a random batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


def select_action(model, state, epsilon):
    """
    Epsilon-greedy action selection.
    With probability epsilon, choose random action.
    Otherwise, choose action with highest Q-value.
    """
    if random.random() < epsilon:
        return random.randint(0, 3)
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = model(state_tensor)
            return q_values.argmax(1).item()