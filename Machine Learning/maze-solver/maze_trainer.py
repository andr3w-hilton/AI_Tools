"""
Maze Solver Trainer - Trains a neural network to solve mazes using Deep Q-Learning
with experience replay and improved training strategies.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from maze_env import MazeSolver, Maze, ReplayBuffer, select_action


# Hyperparameters
MAZE_SIZE = 10
HIDDEN_SIZE = 256
OUTPUT_SIZE = 4  # 4 actions: up, down, left, right
LEARNING_RATE = 0.0005
NUM_EPISODES = 500
DISCOUNT_FACTOR = 0.99
MAX_STEPS = 1000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 10000
TARGET_UPDATE_FREQ = 10  # Update target network every N episodes
VISUALIZE_FREQ = 50  # Visualize every N episodes

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize environment, models, and training components
maze = Maze(MAZE_SIZE)
policy_net = MazeSolver(MAZE_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
target_net = MazeSolver(MAZE_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
criterion = nn.SmoothL1Loss()  # Huber loss - more stable than MSE
replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

# Metrics tracking
episode_rewards = []
episode_steps = []
success_count = 0
success_rate_history = []

# Define colors for visualization
empty_color = 'white'
wall_color = 'black'
path_color = 'lightblue'
player_color = 'yellow'
start_color = 'green'
goal_color = 'red'

print(f"Starting training for {NUM_EPISODES} episodes...")
print(f"Maze size: {MAZE_SIZE}x{MAZE_SIZE}")
print("-" * 60)

# Training loop
epsilon = EPSILON_START

for episode in range(NUM_EPISODES):
    state = maze.reset()
    done = False
    path = []
    total_reward = 0
    steps_taken = 0

    # Episode loop
    while not done and steps_taken < MAX_STEPS:
        # Select action using epsilon-greedy strategy
        action = select_action(policy_net, state, epsilon)

        # Execute action
        next_state, reward, done = maze.step(action)
        total_reward += reward
        steps_taken += 1

        # Store transition in replay buffer
        replay_buffer.push(state, action, reward, next_state, done)

        # Store path for visualization
        path.append(maze.player_pos)

        state = next_state

        # Train the network if we have enough samples
        if len(replay_buffer) >= BATCH_SIZE:
            # Sample a batch from replay buffer
            states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

            # Convert to tensors
            states_t = torch.FloatTensor(states).to(device)
            actions_t = torch.LongTensor(actions).to(device)
            rewards_t = torch.FloatTensor(rewards).to(device)
            next_states_t = torch.FloatTensor(next_states).to(device)
            dones_t = torch.FloatTensor(dones).to(device)

            # Compute current Q values
            current_q_values = policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

            # Compute next Q values using target network
            with torch.no_grad():
                next_q_values = target_net(next_states_t).max(1)[0]
                target_q_values = rewards_t + (1 - dones_t) * DISCOUNT_FACTOR * next_q_values

            # Compute loss and update
            loss = criterion(current_q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)  # Gradient clipping
            optimizer.step()

    # Decay epsilon
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    # Track metrics
    episode_rewards.append(total_reward)
    episode_steps.append(steps_taken)
    if done and maze.player_pos == maze.goal_pos:
        success_count += 1

    # Calculate success rate over last 50 episodes
    if episode >= 49:
        recent_successes = sum(1 for i in range(episode - 49, episode + 1)
                              if episode_rewards[i] > 50)  # Arbitrary threshold
        success_rate = recent_successes / 50
        success_rate_history.append(success_rate)
    else:
        success_rate_history.append(success_count / (episode + 1))

    # Print progress
    if (episode + 1) % 10 == 0:
        avg_reward = np.mean(episode_rewards[-10:])
        avg_steps = np.mean(episode_steps[-10:])
        print(f"Episode {episode + 1:4d} | "
              f"Avg Reward: {avg_reward:7.2f} | "
              f"Avg Steps: {avg_steps:6.1f} | "
              f"Epsilon: {epsilon:.3f} | "
              f"Success Rate: {success_rate_history[-1]*100:.1f}%")

    # Update target network periodically
    if (episode + 1) % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Visualize progress periodically
    if (episode + 1) % VISUALIZE_FREQ == 0 or episode == NUM_EPISODES - 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot maze solution
        ax1 = axes[0]
        maze_vis = np.zeros((MAZE_SIZE, MAZE_SIZE))

        # Set colors: 0=empty, 1=wall, 2=path, 3=start, 4=goal
        maze_vis = maze.walls.copy()  # Start with walls (1)
        for pos in path:
            if pos != maze.start_pos and pos != maze.goal_pos:
                maze_vis[pos[0], pos[1]] = 2  # Path
        maze_vis[maze.start_pos[0], maze.start_pos[1]] = 3  # Start
        maze_vis[maze.goal_pos[0], maze.goal_pos[1]] = 4  # Goal

        cmap = mcolors.ListedColormap([empty_color, wall_color, path_color, start_color, goal_color])
        ax1.imshow(maze_vis, cmap=cmap, vmin=0, vmax=4)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title(f"Episode {episode + 1} - Steps: {steps_taken}, Reward: {total_reward:.1f}")

        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', label='Path',
                      markerfacecolor=empty_color, markersize=10, markeredgecolor='black'),
            plt.Line2D([0], [0], marker='s', color='w', label='Wall',
                      markerfacecolor=wall_color, markersize=10),
            plt.Line2D([0], [0], marker='s', color='w', label='Agent Path',
                      markerfacecolor=path_color, markersize=10),
            plt.Line2D([0], [0], marker='s', color='w', label='Start',
                      markerfacecolor=start_color, markersize=10),
            plt.Line2D([0], [0], marker='s', color='w', label='Goal',
                      markerfacecolor=goal_color, markersize=10)
        ]
        ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))

        # Plot training metrics
        ax2 = axes[1]
        ax2_twin = ax2.twinx()

        # Plot success rate
        episodes_range = range(len(success_rate_history))
        ax2.plot(episodes_range, success_rate_history, 'g-', label='Success Rate', linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Success Rate', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.set_ylim([0, 1.1])

        # Plot average reward
        window_size = 20
        if len(episode_rewards) >= window_size:
            smoothed_rewards = np.convolve(episode_rewards,
                                          np.ones(window_size)/window_size,
                                          mode='valid')
            ax2_twin.plot(range(window_size-1, len(episode_rewards)),
                         smoothed_rewards, 'b-', label='Avg Reward', alpha=0.7)
            ax2_twin.set_ylabel('Average Reward (20-episode window)', color='b')
            ax2_twin.tick_params(axis='y', labelcolor='b')

        ax2.set_title('Training Progress')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'Maze Solver/training_progress_ep{episode+1}.png', dpi=100, bbox_inches='tight')
        plt.close()

print("\n" + "=" * 60)
print("Training completed!")
print(f"Total episodes: {NUM_EPISODES}")
print(f"Final success rate: {success_rate_history[-1]*100:.1f}%")
print(f"Final epsilon: {epsilon:.4f}")
print("=" * 60)

# Save the trained model
torch.save({
    'episode': NUM_EPISODES,
    'model_state_dict': policy_net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'success_rate': success_rate_history[-1],
}, 'Maze Solver/trained_model.pth')

print("\nModel saved to: Maze Solver/trained_model.pth")

# Plot final training curves
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Episode rewards
axes[0, 0].plot(episode_rewards, alpha=0.3, color='blue')
if len(episode_rewards) >= 20:
    smoothed = np.convolve(episode_rewards, np.ones(20)/20, mode='valid')
    axes[0, 0].plot(range(19, len(episode_rewards)), smoothed, color='blue', linewidth=2)
axes[0, 0].set_xlabel('Episode')
axes[0, 0].set_ylabel('Total Reward')
axes[0, 0].set_title('Episode Rewards')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Episode steps
axes[0, 1].plot(episode_steps, alpha=0.3, color='green')
if len(episode_steps) >= 20:
    smoothed = np.convolve(episode_steps, np.ones(20)/20, mode='valid')
    axes[0, 1].plot(range(19, len(episode_steps)), smoothed, color='green', linewidth=2)
axes[0, 1].set_xlabel('Episode')
axes[0, 1].set_ylabel('Steps Taken')
axes[0, 1].set_title('Episode Steps')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Success rate
axes[1, 0].plot(success_rate_history, color='red', linewidth=2)
axes[1, 0].set_xlabel('Episode')
axes[1, 0].set_ylabel('Success Rate')
axes[1, 0].set_title('Success Rate (50-episode window)')
axes[1, 0].set_ylim([0, 1.1])
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Final maze solution
maze_vis = np.zeros((MAZE_SIZE, MAZE_SIZE))
maze_vis = maze.walls.copy()
for pos in path:
    if pos != maze.start_pos and pos != maze.goal_pos:
        maze_vis[pos[0], pos[1]] = 2
maze_vis[maze.start_pos[0], maze.start_pos[1]] = 3
maze_vis[maze.goal_pos[0], maze.goal_pos[1]] = 4

cmap = mcolors.ListedColormap([empty_color, wall_color, path_color, start_color, goal_color])
axes[1, 1].imshow(maze_vis, cmap=cmap, vmin=0, vmax=4)
axes[1, 1].set_xticks([])
axes[1, 1].set_yticks([])
axes[1, 1].set_title(f'Final Episode Solution')

plt.tight_layout()
plt.savefig('Maze Solver/training_summary.png', dpi=150, bbox_inches='tight')
print("Training summary plot saved to: Maze Solver/training_summary.png")
plt.show()