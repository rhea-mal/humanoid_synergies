import matplotlib.pyplot as plt
import numpy as np
import time

# Initial values for the two categories
values1 = [0.5, 0.8]  # Data for "Hannah"
values2 = [100, 200]   # Data for "Tracy"

# Define the number of bars and their width
bar_width = 0.35
x = np.arange(len(values1))  # The label locations

# Create a bar chart
fig, ax1 = plt.subplots()
fig.patch.set_facecolor('black')
ax1.set_facecolor('black')

# Initial plot
bars1 = ax1.bar(x - bar_width/2, values1, width=bar_width, label='Hannah', color='b')
bars2 = ax1.bar(x + bar_width/2, values2, width=bar_width, label='Tracy', color='r')
ax1.set_ylabel('Kinetic Energy Ratio (left scale)', color='white')
ax1.tick_params(axis='y', labelcolor='white')
ax1.set_yticks([0, 10, 20])

ax2 = ax1.twinx()
ax2.set_ylabel('Power Ratio (right scale)', color='white')
ax2.tick_params(axis='y', labelcolor='white')
ax2.set_yticks([0, 1, 2, 3, 4, 5])
ax2.set_ylim(0, 5)

ax1.set_xticks(x)
ax1.set_xticklabels(['Kinetic Energy Ratio', 'Power Ratio'])
ax1.set_title('Bar Chart with Kinetic Energy Ratio and Power Ratio', color='white')

# Combine legends for both datasets
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper left', facecolor='black', edgecolor='white')

# Function to update the values
def update_values():
    global values1, values2
    values1 = [np.random.rand() * 20]  # Random values for "Hannah"
    values2 = [np.random.randint(0, 250)]  # Random values for "Tracy"

# Iteratively update the values in a while loop
try:
    while True:
        # update_values()

        # Update bar heights
        for bar, new_height in zip(bars1, values1):
            bar.set_height(new_height)
        for bar, new_height in zip(bars2, values2):
            bar.set_height(new_height)

        # Redraw the canvas
        plt.draw()
        plt.pause(1)  # Pause for 1 second
except KeyboardInterrupt:
    print("Stopped by user.")
finally:
    plt.close(fig)  # Close the figure when done
