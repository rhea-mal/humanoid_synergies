import redis
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants for Redis keys
LAGRANGIAN = "sai2::sim::toro::lagrangian"
KINETIC = "sai2::sim::toro::kinetic"
POTENTIAL = "sai2::sim::toro::potential"

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)

# Initialize lists to store data
lagrangian_data = []
kinetic_data = []
potential_data = []

# Set up the plot style
plt.style.use('ggplot')  # Use 'ggplot' style as an alternative

# Set up the plots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

line_lagrangian, = ax1.plot([], [], label='Lagrangian', color='blue', linewidth=2.0)
ax1.legend()
ax1.set_xlim(0, 100)
ax1.set_ylim(-50, 120)
ax1.set_ylabel('Lagrangian')
ax1.grid(True)

line_kinetic, = ax2.plot([], [], label='Kinetic', color='green', linewidth=2.0)
ax2.legend()
ax2.set_xlim(0, 100)
ax2.set_ylim(-10, 60)
ax2.set_ylabel('Kinetic')
ax2.grid(True)

line_potential, = ax3.plot([], [], label='Potential', color='red', linewidth=2.0)
ax3.legend()
ax3.set_xlim(0, 100)
ax3.set_ylim(-100, 0)
ax3.set_xlabel('Time')
ax3.set_ylabel('Potential')
ax3.grid(True)

# Function to initialize the plot
def init():
    line_lagrangian.set_data([], [])
    line_kinetic.set_data([], [])
    line_potential.set_data([], [])
    return line_lagrangian, line_kinetic, line_potential

# Function to update the plot
def update(frame):
    # Read values from Redis
    lagrangian_value = float(r.get(LAGRANGIAN) or 0)
    kinetic_value = float(r.get(KINETIC) or 0)
    potential_value = float(r.get(POTENTIAL) or 0)
    
    # Append values to the lists
    lagrangian_data.append(lagrangian_value)
    kinetic_data.append(kinetic_value)
    potential_data.append(potential_value)

    # Update the data of the plot lines
    line_lagrangian.set_data(range(len(lagrangian_data)), lagrangian_data)
    line_kinetic.set_data(range(len(kinetic_data)), kinetic_data)
    line_potential.set_data(range(len(potential_data)), potential_data)

    # Adjust x-axis limit if necessary
    ax1.set_xlim(0, len(lagrangian_data))
    ax2.set_xlim(0, len(kinetic_data))
    ax3.set_xlim(0, len(potential_data))
    
    return line_lagrangian, line_kinetic, line_potential

# Create an animation object
ani = animation.FuncAnimation(fig, update, init_func=init, blit=True, interval=1000)

# Display the plots
plt.tight_layout()
plt.show()
