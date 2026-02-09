"""
Maze Solver - Loads a trained model and demonstrates solving a maze.
Can test on multiple randomly generated mazes.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from maze_env import MazeSolver, Maze


# Configuration
MAZE_SIZE = 10
HIDDEN_SIZE = 256
OUTPUT_SIZE = 4
MAX_STEPS = 1000
NUM_TEST_MAZES = 5  # Number of different mazes to test

# Define colors for visualization
empty_color = 'white'
wall_color = 'black'
path_color = 'lightblue'
start_color = 'green'
goal_color = 'red'

# Create colormap
cmap = mcolors.ListedColormap([empty_color, wall_color, path_color, start_color, goal_color])

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the trained model
print("Loading trained model...")
model = MazeSolver(MAZE_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)

try:
    checkpoint = torch.load('Maze Solver/trained_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded successfully!")
    print(f"Training episodes: {checkpoint['episode']}")
    print(f"Training success rate: {checkpoint['success_rate']*100:.1f}%")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please train the model first by running maze_trainer.py")
    exit(1)

model.eval()

# Test statistics
total_successes = 0
total_steps_list = []

# Test the model on multiple mazes
for test_num in range(NUM_TEST_MAZES):
    print(f"\n{'='*60}")
    print(f"Testing on Maze {test_num + 1}/{NUM_TEST_MAZES}")
    print('='*60)

    # Create a new maze
    maze = Maze(MAZE_SIZE)

    # Print text representation
    print("\nMaze layout:")
    maze.render()

    # Solve the maze using the trained model
    state = maze.reset()
    done = False
    path = []
    steps = 0

    print(f"\nSolving maze...")

    while not done and steps < MAX_STEPS:
        # Convert state to tensor and get Q-values
        state_tensor = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            q_values = model(state_tensor)

        # Select best action (greedy - no exploration)
        action = torch.argmax(q_values).item()

        # Execute action
        next_state, reward, done = maze.step(action)
        state = next_state
        path.append(maze.player_pos)
        steps += 1

        # Print progress periodically
        if steps % 100 == 0:
            print(f"  Step {steps}: Position {maze.player_pos}")

    # Report results
    success = done and maze.player_pos == maze.goal_pos
    total_successes += success
    total_steps_list.append(steps)

    print(f"\nResults:")
    print(f"  Status: {'SUCCESS - Goal reached!' if success else 'FAILED - Max steps exceeded'}")
    print(f"  Steps taken: {steps}")
    print(f"  Path length: {len(path)}")

    # Visualize the solution
    fig, ax = plt.subplots(figsize=(8, 8))

    # Create visualization: 0=empty, 1=wall, 2=path, 3=start, 4=goal
    maze_vis = maze.walls.copy()

    # Draw path
    for i, pos in enumerate(path):
        if pos != maze.start_pos and pos != maze.goal_pos:
            maze_vis[pos[0], pos[1]] = 2

    # Mark start and goal
    maze_vis[maze.start_pos[0], maze.start_pos[1]] = 3
    maze_vis[maze.goal_pos[0], maze.goal_pos[1]] = 4

    # Display
    ax.imshow(maze_vis, cmap=cmap, vmin=0, vmax=4)
    ax.set_xticks([])
    ax.set_yticks([])

    # Title with result
    status_text = "SUCCESS" if success else "FAILED"
    color = 'green' if success else 'red'
    ax.set_title(f"Maze {test_num + 1} - {status_text}\nSteps: {steps}",
                fontsize=14, fontweight='bold', color=color)

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', label='Open Path',
                  markerfacecolor=empty_color, markersize=12, markeredgecolor='black'),
        plt.Line2D([0], [0], marker='s', color='w', label='Wall',
                  markerfacecolor=wall_color, markersize=12),
        plt.Line2D([0], [0], marker='s', color='w', label='Agent Path',
                  markerfacecolor=path_color, markersize=12),
        plt.Line2D([0], [0], marker='s', color='w', label='Start',
                  markerfacecolor=start_color, markersize=12),
        plt.Line2D([0], [0], marker='s', color='w', label='Goal',
                  markerfacecolor=goal_color, markersize=12)
    ]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(f'Maze Solver/solution_maze_{test_num + 1}.png', dpi=150, bbox_inches='tight')
    print(f"  Visualization saved to: Maze Solver/solution_maze_{test_num + 1}.png")

    # Close the figure to free memory
    plt.close()

# Print overall statistics
print(f"\n{'='*60}")
print("OVERALL TEST RESULTS")
print('='*60)
print(f"Total mazes tested: {NUM_TEST_MAZES}")
print(f"Successes: {total_successes}")
print(f"Success rate: {total_successes/NUM_TEST_MAZES*100:.1f}%")
if total_steps_list:
    print(f"Average steps: {np.mean(total_steps_list):.1f}")
    print(f"Min steps: {min(total_steps_list)}")
    print(f"Max steps: {max(total_steps_list)}")
print('='*60)

# Create summary visualization
if NUM_TEST_MAZES > 1:
    fig, axes = plt.subplots(1, min(NUM_TEST_MAZES, 5), figsize=(4*min(NUM_TEST_MAZES, 5), 4))
    if NUM_TEST_MAZES == 1:
        axes = [axes]

    for i in range(min(NUM_TEST_MAZES, 5)):
        # Re-create each maze (they're random, so this will show different mazes)
        # In practice, we would save them, but this is just for demonstration
        test_maze = Maze(MAZE_SIZE)
        maze_vis = test_maze.walls.copy()
        maze_vis[test_maze.start_pos[0], test_maze.start_pos[1]] = 3
        maze_vis[test_maze.goal_pos[0], test_maze.goal_pos[1]] = 4

        axes[i].imshow(maze_vis, cmap=cmap, vmin=0, vmax=4)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_title(f"Test {i+1}", fontsize=10)

    plt.suptitle(f"Maze Solver Test Results: {total_successes}/{NUM_TEST_MAZES} Success",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('Maze Solver/test_summary.png', dpi=150, bbox_inches='tight')
    print(f"\nSummary visualization saved to: Maze Solver/test_summary.png")

print("\nAll tests completed!")
plt.show()