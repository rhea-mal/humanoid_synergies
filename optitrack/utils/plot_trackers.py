import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import redis
import time
import numpy as np

# Redis connection
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Set up matplotlib figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.ion()  # Enable interactive mode

# Function to retrieve and parse position data from Redis
def get_position_data():
    keys = redis_client.keys("0::*::pos")
    positions = []
    labels = []
    for key in keys:
        pos_str = redis_client.get(key)
        if pos_str:
            pos_list = [float(coord) for coord in pos_str.strip('[]').split(',')]
            tracker_id = key.split("::")[1]  # Extract tracker number from key
            positions.append(pos_list)
            labels.append(tracker_id)
    return positions, labels

# Live plotting loop
while True:
    ax.clear()
    positions, labels = get_position_data()
    if positions:
        # Convert to numpy array for easy slicing
        positions = np.array(positions)
        # Plot points
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='b', marker='o')
        # Add labels for each point
        for i, label in enumerate(labels):
            ax.text(positions[i, 0], positions[i, 1], positions[i, 2], label, color='red')

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Real-Time 3D Position Tracking with Labels")
    
    # Update plot
    plt.draw()
    plt.pause(0.1)  # Adjust pause for refresh rate

    # Break loop with a condition or use Ctrl+C to stop
