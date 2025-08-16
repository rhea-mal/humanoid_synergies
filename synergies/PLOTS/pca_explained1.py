import matplotlib.pyplot as plt
import numpy as np

# Use Arial font and larger size for all text
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 23

# Raw segment variance data for each activity (PC1, PC2, PC3 percentages)
data = {
    'STEPS': np.array([
        [99, 1, 0], [80, 16, 2], [90, 10, 0], [97, 2, 1],
        [100, 0, 0], [93, 6, 1], [93, 6, 1], [96, 4, 0],
        [95, 4, 1], [99, 1, 0], [91, 9, 0], [90, 9, 1],
        [96, 3, 1], [100, 0, 0], [94, 6, 0], [95, 3, 1]
    ]),
    'JUMPING_JACKS': np.array([
        [79, 13, 5], [81, 11, 5], [57, 32, 6],
        [60, 16, 13], [70, 22, 3], [98, 2, 0]
    ]),
    'WALK_CIRCLE': np.array([
        [66, 14, 12], [66, 26, 4],
        [72, 16, 11], [83, 12, 3]
    ]),
    'SQUAT': np.array([
        [81, 15, 3], [70, 24, 3], [91, 4, 3]
    ])
}

# Compute means and standard deviations across segments
activities = list(data.keys())
means = np.array([data[act].mean(axis=0) for act in activities])
stds  = np.array([data[act].std(axis=0)  for act in activities])

# Custom labels
labels = ['Steps', 'Jumping Jacks', 'Walk in Circle', 'Squats']

# Plot setup
x = np.arange(len(activities))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
# colors = ['#3CAEA3', '#20639B', '#173F5F']
colors = ['#3CAEA3', '#20639B', '#F25F5C']

ax.bar(x - width, means[:, 0], width, yerr=stds[:, 0], label='PC1', color=colors[0], capsize=5)
ax.bar(x,         means[:, 1], width, yerr=stds[:, 1], label='PC2', color=colors[1], capsize=5)
ax.bar(x + width, means[:, 2], width, yerr=stds[:, 2], label='PC3', color=colors[2], capsize=5)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontweight='bold')
ax.set_ylabel('Mean Variance Explained (%)')
ax.legend()

plt.tight_layout()
plt.show()
