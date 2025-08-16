import redis
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)

# Define the keys to fetch data for
KINETIC_KEYS = {
    'Human #1': 'sai2::sim::hannah::human::kinetic',
    'Robot #1': 'sai2::sim::hannah::robot::kinetic',
    'Human #2': 'sai2::sim::tracy::human::kinetic',
    'Robot #2': 'sai2::sim::tracy::robot::kinetic'
}

EFFORT_KEYS = {
    'Human #1': 'sai2::sim::hannah::human::effort',
    'Robot #1': 'sai2::sim::hannah::robot::effort',
    'Human #2': 'sai2::sim::tracy::human::effort',
    'Robot #2': 'sai2::sim::tracy::robot::effort'
}

# Initialize data
labels = list(KINETIC_KEYS.keys())
data_kinetic = [0] * len(labels)
data_effort = [0] * len(labels)

# Setup log file
log_filename = "kinetic_power_log.txt"
if os.path.exists(log_filename):
    os.remove(log_filename)  # Clear previous logs
with open(log_filename, "w") as f:
    f.write("hannah_robot_kinetic,hannah_human_kinetic,tracy_robot_kinetic,tracy_human_kinetic," 
            "hannah_robot_effort,hannah_human_effort,tracy_robot_effort,tracy_human_effort\n")

# Create figure and subplots
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 10))
bar1 = ax1.bar(labels, data_kinetic, color=['red', 'blue', 'red', 'blue'])
bar2 = ax2.bar(labels, data_effort, color=['red', 'blue', 'red', 'blue'])

ax1.set_title('Kinetic Energy (Human vs. Robot)')
ax2.set_title('Power Output (Human vs. Robot)')

# Hide y-axis labels
ax1.get_yaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)

# Function to update the bars in the plot
def update(frame):
    try:
        # Fetch kinetic values
        kinetic_values = [float(r.get(key) or 0) for key in KINETIC_KEYS.values()]
        effort_values = [float(r.get(key) or 0) for key in EFFORT_KEYS.values()]

        # Tolerance check to avoid division by zero issues
        tol = 1e0

        for i in range(len(labels)):
            data_kinetic[i] = kinetic_values[i] if abs(kinetic_values[i]) >= tol else 1
            data_effort[i] = effort_values[i] if abs(effort_values[i]) >= tol else 1

        # Update bar heights
        for rect, height in zip(bar1, data_kinetic):
            rect.set_height(height)
        for rect, height in zip(bar2, data_effort):
            rect.set_height(height)

        # Log to file
        with open(log_filename, "a") as f:
            f.write(",".join(f"{v:.4f}" for v in kinetic_values + effort_values) + "\n")

    except Exception as e:
        print(f"Error during update: {e}")

# Animate
ani = FuncAnimation(fig, update, interval=10)

plt.tight_layout()
plt.show()
