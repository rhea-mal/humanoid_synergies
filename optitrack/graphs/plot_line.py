import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Setup minimalistic seaborn style
sns.set_theme(style="white", context="paper", font_scale=1.5)

# Load all 3 data files
log_filename_1 = "walkinplace.txt"
log_filename_2 = "jumps.txt"
log_filename_3 = "squats.txt"

data1 = np.loadtxt(log_filename_1, delimiter=',', skiprows=1)
data2 = np.loadtxt(log_filename_2, delimiter=',', skiprows=1)
data3 = np.loadtxt(log_filename_3, delimiter=',', skiprows=1)

# Define a function to process each dataset
def process_data(data, scale=1000.0):
    hannah_robot_power = data[:, 4]
    hannah_human_power = data[:, 5]
    
    window_size = 10
    hannah_robot_power_smoothed = np.convolve(hannah_robot_power, np.ones(window_size)/window_size, mode='valid')
    hannah_human_power_smoothed = np.convolve(hannah_human_power, np.ones(window_size)/window_size, mode='valid')

    N = 5
    robot_power_final = hannah_robot_power_smoothed[::N]
    human_power_final = hannah_human_power_smoothed[::N]
    
    # Scale 
    robot_power_final /= scale
    human_power_final /= scale

    frame_indices = np.arange(len(human_power_final)) * N + (window_size // 2)
    time_axis = frame_indices / 20.0  # 20 frames per second
    
    return time_axis, human_power_final, robot_power_final

# Process each file
time1, human1, robot1 = process_data(data1, scale=1000.0)
time2, human2, robot2 = process_data(data2, scale=500.0)
time3, human3, robot3 = process_data(data3, scale=10000.0)

# Find maximums across human and robot for each dataset
max1 = max(np.max(human1), np.max(robot1))
max2 = max(np.max(human2), np.max(robot2))
max3 = max(np.max(human3), np.max(robot3))

# Normalize so that highest value becomes 1
robot1 *= 750/max1 
human1 *= 750/max1

human2 *= 1000/max2
robot2 *= 1000/max2

human3 *= 1000/max3
robot3 *= 1000/max3

# Set up figure with 3 vertical plots
fig, axs = plt.subplots(3, 1, figsize=(16, 10), sharex=False)

# Plot 1: Walk in Place
axs[0].plot(time1, human1, color="#E69F00", linewidth=1.8, label="Robot")
axs[0].plot(time1, robot1, color="royalblue", linewidth=1.8, label="Human")
# axs[0].set_ylabel('Power (kW)', fontsize=16)
axs[0].set_title('Walk in Place', fontsize=24, fontweight='bold')

# Plot 2: Jumps
axs[1].plot(time2, human2, color="#E69F00", linewidth=1.8)
axs[1].plot(time2, robot2, color="royalblue", linewidth=1.8)
axs[1].set_ylabel('Power (W)', fontsize=24)
axs[1].set_title('Jumps', fontsize=24, fontweight='bold')

# Plot 3: Clap Hands
axs[2].plot(time3, human3, color="#E69F00", linewidth=1.8)
axs[2].plot(time3, robot3, color="royalblue", linewidth=1.8)
axs[2].set_xlabel('Time (s)', fontsize=24)
# axs[2].set_ylabel('Power (kW)', fontsize=16)
axs[2].set_title('Squats', fontsize=24, fontweight='bold')

# Clean up all subplots
for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.grid(False)
    ax.tick_params(axis='both', which='major', labelsize=15)

# Add legend to the first subplot only
axs[0].legend(frameon=False, loc="upper left", fontsize=20)

plt.tight_layout()

# Save figure if you want
# plt.savefig('hannah_power_threeplots.png', dpi=600, bbox_inches='tight')

plt.show()