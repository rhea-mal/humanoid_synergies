import redis
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import numpy as np
import ast 

n_pts = 51

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialize a list to store the points and a counter
points = []
point_counter = 1

# Function to read values from Redis and parse the string into coordinates
def read_from_redis(index):
    # Assume the Redis key is 'coordinates' and it stores a string '[x, y, z]'
    data = r.get('1::' + str(index) + "::pos")  # Get and decode the string from Redis
    x, y, z = ast.literal_eval(data)  # Safely convert the string '[x, y, z]' to a list of numbers
    return float(x), float(y), float(z)

# Main loop to update the plot continuously
while True:
	points = []
	for i in range(n_pts):
		# Read new values from Redis
		x, y, z = read_from_redis(i+1)

		# Append the new point and label
		points.append((x, y, z))

	# Clear the previous points
	ax.cla()

	# Plot all points with labels
	for i, (px, py, pz) in enumerate(points):
		ax.scatter(px, py, pz, color='b')
		ax.text(px, py, pz, f'{i+1}', color='red')  # Label points sequentially

	# Set labels for axes
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')

	# Set plot limits (optional)
	# ax.set_xlim([-1, 1])
	# ax.set_ylim([-1, 1])
	# ax.set_zlim([-1, 1])

	# Draw the updated plot
	plt.draw()

	# Pause to allow the plot to update
	plt.pause(0.1)

	# Increment the point counter
	point_counter += 1

	# Delay before reading the next set of points
	time.sleep(0.1)

# Show the final plot (optional)
plt.show()
