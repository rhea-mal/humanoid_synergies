import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# === Load CSV ===
csv_path = './recordings/ballet1.csv'
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip().str.replace('"', '')

# === Remapping body parts ===
remap = {
    'head': 'hip',
    'torso': 'left_foot',
    'hip': 'right_foot',
    'left_foot': 'torso',
    'right_foot': 'right_hand',
    'right_hand': 'head',
    'left_elbow': 'right_elbow',
    'right_elbow': 'left_elbow'
}

# === Original parts (excluding knees) ===
original_parts = [
    'head', 'torso', 'hip',
    'left_foot', 'right_foot',
    'left_hand', 'right_hand',
    'left_elbow', 'right_elbow'
]

# Apply remapping
body_parts = [remap.get(part, part) for part in original_parts]
body_parts = original_parts

# === Column mappings ===
positions = {part: [f'{part}_pos__{i}' for i in range(3)] for part in original_parts}
orientations = {part: [f'{part}_ori__{i}' for i in range(9)] for part in original_parts}

# === Matplotlib 3D Plot ===
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# === Iterate through frames at 100Hz ===
for i in range(len(df)):
    ax.clear()
    ax.set_title(f'Frame {i} @ 100Hz')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-1, 3)

    for idx, part in enumerate(original_parts):
        label = body_parts[idx]  # Remapped label
        if all(col in df.columns for col in positions[part]):
            pos = df.loc[i, positions[part]].values.astype(float)
            ax.scatter(*pos, label=label)

            if all(col in df.columns for col in orientations[part]):
                ori_vals = df.loc[i, orientations[part]].values.astype(float)
                ori = ori_vals.reshape(3, 3)
                ax.quiver(*pos, *ori[:, 0], length=0.05, color='r')  # x-axis
                ax.quiver(*pos, *ori[:, 1], length=0.05, color='g')  # y-axis
                ax.quiver(*pos, *ori[:, 2], length=0.05, color='b')  # z-axis

    ax.legend(loc='upper left', fontsize='x-small')
    plt.pause(0.01)

plt.show()
