import redis
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# Constants for Redis keys
LAGRANGIAN = "sai2::sim::toro::lagrangian"
KINETIC = "sai2::sim::toro::kinetic"
POTENTIAL = "sai2::sim::toro::potential"

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)

# Initialize data
labels = ['Lagrangian', 'Kinetic', 'Potential', 'Human Energy Consumption']
values = [0, 0, 0, 0]  # Initial values for the bar graph

# Set up the plot style
plt.style.use('ggplot')

# Set up the bar plot
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(labels, values, color=['blue', 'green', 'red', 'purple'])

# Set constant y-axis limits
ax.set_ylim(-600, 1500)

# Function to update the bar graph
def update(frame):
    # Read values from Redis
    lagrangian_value = float(r.get(LAGRANGIAN) or 0)
    kinetic_value = float(r.get(KINETIC) or 0)
    # potential_value = float(r.get(POTENTIAL) or 0)
    potential_value = float(r.get(POTENTIAL))

    # Calculate human energy consumption as Lagrangian minus a random value between 0 and 50
    human_energy_value = random.uniform(50, 150)
    
    # Update the values in the bar graph
    values[0] = lagrangian_value
    values[1] = kinetic_value
    values[2] = potential_value
    values[3] = human_energy_value

    # Update the bar heights
    for bar, value in zip(bars, values):
        bar.set_height(value)

    return bars

# Create an animation object
ani = animation.FuncAnimation(fig, update, interval=100)

# Display the bar graph
plt.tight_layout()
plt.show()
