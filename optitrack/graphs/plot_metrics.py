#!/usr/bin/env python3

import redis
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# Constants for Redis keys
# LAGRANGIAN = "sai2::sim::toro::lagrangian"
# KINETIC = "sai2::sim::toro::kinetic"
# POTENTIAL = "sai2::sim::toro::potential"

ROBOT_KINETIC_KEY = "sai2::sim::robot::kinetic"
HUMAN_KINETIC_KEY = "sai2::sim::human::kinetic"
ROBOT_EFFORT_KEY = "sai2::sim::robot::effort"
HUMAN_EFFORT_KEY = "sai2::sim::human::effort"

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)

# Initialize data
labels = ['Kinetic Energy Ratio', 'Effort Ratio']
values = [0, 0]  # Initial values for the bar graph

# Set up the plot style
# plt.style.use('ggplot')
plt.style.use('dark_background')

# Set up the bar plot
unit = 2
fig, ax = plt.subplots(figsize=(3 * unit, 3 * unit))
bars = ax.bar(labels, values, color=['blue', 'green'], width=0.5)
ax.legend(handletextpad=0.5, labelspacing=0.01, borderpad=0.5)

# Set constant y-axis limits
ax.set_ylim(0, 20)
ax.yticks([0, 10, 20])  # Adjust as needed

# Function to update the bar graph
def update(frame):
    # Read values from Redis
    # lagrangian_value = float(r.get(LAGRANGIAN) or 0)
    # kinetic_value = float(r.get(KINETIC) or 0)
    # potential_value = float(r.get(POTENTIAL) or 0)
    # potential_value = float(r.get(POTENTIAL))

    robot_kinetic_value = float(r.get(ROBOT_KINETIC_KEY) or 0)
    human_kinetic_value = float(r.get(HUMAN_KINETIC_KEY) or 0)
    robot_effort_value = float(r.get(ROBOT_EFFORT_KEY) or 0)
    human_effort_value = float(r.get(HUMAN_EFFORT_KEY) or 0)

    # Calculate human energy consumption as Lagrangian minus a random value between 0 and 50
    # human_energy_value = random.uniform(50, 150)
    
    # Update the values in the bar graph
    # values[0] = lagrangian_value
    # values[1] = kinetic_value
    # values[2] = potential_value
    # values[3] = human_energy_value

    if human_kinetic_value == 0:
        values[0] = 0
    else:
        values[0] = robot_kinetic_value / human_kinetic_value 

    if human_effort_value == 0:
        values[1] = 0
    else:
        values[1] = robot_effort_value / human_effort_value

    # Update the bar heights
    for bar, value in zip(bars, values):
        bar.set_height(value)

    return bars

# Create an animation object
ani = animation.FuncAnimation(fig, update, interval=100)

# Display the bar graph
plt.tight_layout()
plt.show()
