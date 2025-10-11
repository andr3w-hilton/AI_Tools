# Maze Solver - Deep Q-Learning Implementation

A neural network-based maze solver using Deep Q-Learning with experience replay. The agent learns to navigate randomly generated mazes from start to goal.

## Features

### Improvements Over Original Version

1. **Real Maze Environment with Walls**
   - Mazes generated using depth-first search algorithm
   - Actual obstacles to navigate around
   - Random maze generation for better generalization

2. **Improved State Representation**
   - 3-channel input: walls, player position, goal position
   - Better spatial awareness for the neural network
   - More informative than single-channel encoding

3. **Experience Replay Buffer**
   - Stores 10,000 transitions
   - Samples random mini-batches for training
   - Breaks correlation between consecutive samples
   - Leads to more stable learning

4. **Proper Epsilon-Greedy Exploration**
   - Starts at 100% exploration (epsilon=1.0)
   - Decays to 1% over time (epsilon=0.01)
   - Balances exploration vs exploitation

5. **Target Network**
   - Separate target network for stable Q-value estimation
   - Updated every 10 episodes
   - Reduces oscillations during training

6. **Enhanced Neural Network Architecture**
   - 3 convolutional layers with batch normalization
   - Dropout for regularization
   - Better suited for spatial reasoning

7. **Comprehensive Training Metrics**
   - Success rate tracking (rolling 50-episode window)
   - Episode rewards and steps
   - Training progress visualization
   - Periodic checkpoint saving

8. **Optimized Visualization**
   - Only visualizes every 50 episodes (not every step!)
   - Saves training progress plots
   - Much faster training

9. **Better Reward Structure**
   - Large reward (+100) for reaching goal
   - Penalty (-1) for hitting walls
   - Small distance-based reward to guide learning
   - Episode timeout to prevent infinite loops

## Files

- `maze_env.py` - Shared module containing:
  - `MazeSolver` - Neural network class
  - `Maze` - Environment with wall generation
  - `ReplayBuffer` - Experience replay implementation
  - `select_action()` - Epsilon-greedy action selection

- `maze_trainer.py` - Training script
  - Trains the model for 500 episodes
  - Uses Deep Q-Learning with experience replay
  - Saves trained model and training plots

- `maze_solver.py` - Testing/inference script
  - Loads trained model
  - Tests on 5 different randomly generated mazes
  - Visualizes solutions
  - Reports success rate and statistics

- `requirements.txt` - Python dependencies

## Installation

**Important**: PyTorch currently supports Python 3.8-3.12. If you have Python 3.13+, you'll need to create a virtual environment with Python 3.12 or earlier.

### Option 1: Using Python 3.12 or earlier

```bash
# Install dependencies
pip install -r requirements.txt
```

### Option 2: Create a Python 3.12 Environment

Using conda:
```bash
conda create -n maze-solver python=3.12
conda activate maze-solver
pip install torch numpy matplotlib
```

Using pyenv:
```bash
pyenv install 3.12.0
pyenv virtualenv 3.12.0 maze-solver
pyenv activate maze-solver
pip install torch numpy matplotlib
```

## Usage

### Training

Train a new model:

```bash
cd "Maze Solver"
python maze_trainer.py
```

Training will:
- Run for 500 episodes (takes 10-30 minutes depending on hardware)
- Print progress every 10 episodes
- Save visualizations every 50 episodes
- Save the final trained model to `trained_model.pth`
- Generate training summary plots

### Testing

Test the trained model on new mazes:

```bash
cd "Maze Solver"
python maze_solver.py
```

This will:
- Load the trained model
- Generate 5 random mazes
- Solve each maze
- Display success rate and statistics
- Save solution visualizations

## Hyperparameters

Key hyperparameters in `maze_trainer.py`:

```python
MAZE_SIZE = 10              # 10x10 grid
HIDDEN_SIZE = 256           # Neural network hidden layer size
LEARNING_RATE = 0.0005      # Adam optimizer learning rate
NUM_EPISODES = 500          # Training episodes
DISCOUNT_FACTOR = 0.99      # Future reward discount (gamma)
EPSILON_START = 1.0         # Initial exploration rate
EPSILON_END = 0.01          # Minimum exploration rate
EPSILON_DECAY = 0.995       # Epsilon decay per episode
BATCH_SIZE = 64             # Mini-batch size for training
REPLAY_BUFFER_SIZE = 10000  # Experience replay capacity
TARGET_UPDATE_FREQ = 10     # Target network update frequency
```

## How It Works

### 1. Maze Generation
Uses depth-first search to create a perfect maze (no loops, single solution path possible).

### 2. State Representation
The agent observes a 3-channel image:
- Channel 0: Wall locations (1=wall, 0=empty)
- Channel 1: Player position (1 at player, 0 elsewhere)
- Channel 2: Goal position (1 at goal, 0 elsewhere)

### 3. Actions
Four discrete actions: Up (0), Down (1), Left (2), Right (3)

### 4. Rewards
- +100: Reach goal
- -1: Hit wall
- -0.1 + 0.5/(distance+1): Moving through maze (encourages progress toward goal)

### 5. Training Algorithm (DQN)
1. Agent explores maze using epsilon-greedy policy
2. Transitions stored in replay buffer
3. Random mini-batch sampled from buffer
4. Q-values updated using Bellman equation
5. Target network provides stable Q-value estimates
6. Epsilon decays over time (more exploitation, less exploration)

### 6. Neural Network
- Input: 3x10x10 state tensor
- 3 convolutional layers (extract spatial features)
- 3 fully connected layers (decision making)
- Output: 4 Q-values (one per action)

## Expected Results

After training:
- **Success rate**: 60-90% (depends on maze complexity)
- **Average steps**: 20-50 steps (optimal is ~18 for 10x10 maze)
- **Training time**: 10-30 minutes on CPU

The model learns to:
- Avoid walls
- Find paths to the goal
- Take reasonably efficient routes
- Generalize to new unseen mazes

## Troubleshooting

### PyTorch Installation Issues
If you see "No matching distribution found for torch", you're likely using Python 3.13+ which isn't supported yet. Use Python 3.12 or earlier.

### Training Not Converging
- Increase `NUM_EPISODES` (try 1000)
- Adjust `LEARNING_RATE` (try 0.001 or 0.0001)
- Increase `HIDDEN_SIZE` (try 512)
- Check that mazes are solvable (they should be with DFS generation)

### Model Not Finding Goal
- Model may need more training episodes
- Try training on smaller mazes first (MAZE_SIZE=5)
- Check that epsilon is decaying properly

### Memory Issues
- Reduce `REPLAY_BUFFER_SIZE` (try 5000)
- Reduce `BATCH_SIZE` (try 32)
- Reduce `HIDDEN_SIZE` (try 128)

## Future Improvements

1. **Larger/More Complex Mazes**: Test on 20x20 or 50x50 mazes
2. **A* Baseline**: Compare with traditional pathfinding algorithms
3. **Different Maze Types**: Add loops, multiple paths, dynamic obstacles
4. **Prioritized Experience Replay**: Sample important transitions more frequently
5. **Dueling DQN**: Separate value and advantage streams
6. **Double DQN**: Reduce overestimation bias
7. **Curriculum Learning**: Start with small mazes, gradually increase size
8. **Multi-Goal**: Train to reach different goal positions
9. **Partial Observability**: Agent only sees nearby cells (fog of war)
10. **3D Mazes**: Extend to three dimensions

## References

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) - Original DQN paper
- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)

## License

This is an educational project. Feel free to use and modify as needed.

## Author

Created with AI assistance as a learning exercise in deep reinforcement learning.