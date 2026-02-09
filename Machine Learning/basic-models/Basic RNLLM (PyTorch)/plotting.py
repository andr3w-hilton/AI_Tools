import matplotlib.pyplot as plt  # Import matplotlib for plotting
import numpy as np  # Import numpy for numerical operations
import json  # Import json to read/write json files

plt.ion()  # Turn interactive mode on. This means that figures and plots can be updated and redrawn whenever new data is available.

# Create a figure and an axis to plot on
# A figure in matplotlib is like a canvas on which we create our plots.
# An axis is a part of that figure where we plot our data.
fig, ax = plt.subplots()

# Set labels and title for the plot
ax.set_xlabel('Epochs')  # Set label for x-axis
ax.set_ylabel('Loss')  # Set label for y-axis
ax.set_title('Model Training Loss')  # Set title for the plot

# The script enters an infinite loop and keeps updating the plot in real-time
while True:
    # Open the 'losses.json' file in read mode
    with open('losses.json', 'r') as f:
        # Load the json data from the file into the 'losses' variable
        losses = json.load(f)

    ax.clear()  # Clear previous plot. This is done to prevent old data from appearing in the new plot.

    ax.plot(losses,
            label='Training loss')  # Plot the new loss data on the axis. This draws a line connecting each loss point.

    ax.legend()  # Display the legend on the plot. This helps to identify which line corresponds to which data series.

    # Calculate minimum and maximum loss for setting y-ticks.
    # y-ticks are the markers on the y-axis which help in reading the values on the plot.
    min_loss = min(losses)  # Find minimum loss value
    max_loss = max(losses)  # Find maximum loss value

    # Generate y-ticks.
    # We use numpy's linspace function which returns evenly spaced numbers over a specified range.
    # We specify that range to be from min_loss to max_loss and the number of ticks to be 5.
    y_ticks = np.linspace(min_loss, max_loss, num=5)

    # Format y-ticks to 6 decimal places.
    # This is done for better readability on the plot.
    y_ticks_labels = [f'{tick:.6f}' for tick in y_ticks]  # Create labels for the ticks
    ax.set_yticks(y_ticks)  # Set the positions of the ticks on the y-axis
    ax.set_yticklabels(y_ticks_labels)  # Set the labels of the ticks

    plt.pause(0.001)  # Pause the script for a very short period (0.001 seconds) to allow the plot to update.
