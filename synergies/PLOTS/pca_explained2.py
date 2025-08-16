import matplotlib.pyplot as plt
import numpy as np

# Use Arial and larger size for all text
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 23
data = {
    'Ballet': np.array([[67,23,5], [68,21,6], [56,30,5], [76,11,8], [72,19,5],
                        [63,22,11], [75,16,5], [80,10,7], [78,14,5], [70,18,9],
                        [59,24,7], [66,19,9], [64,17,8], [61,22,7], [87,7,4]]),
    'Ballroom': np.array([[69,17,5], [56,26,11], [51,30,8], [52,23,12], [61,22,8],
                          [77,8,7], [59,20,9], [62,17,12], [67,20,6], [90,6,2],
                          [85,10,2], [73,16,5], [61,16,12], [94,3,1], [59,19,10],
                          [76,12,7], [70,20,4]]),
    'Hip-Hop': np.array([[66,19,8], [72,16,6], [70,15,6], [90,6,3], [46,27,11],
                         [66,20,6], [52,28,14], [60,24,9], [64,14,11], [53,27,11],
                         [53,29,11], [66,19,8], [65,27,4], [42,31,20], [71,16,7],
                         [56,20,9], [40,30,14], [44,32,13], [74,17,4], [81,13,3]]),
    'Irish': np.array([[63,15,11], [41,32,10], [42,20,17], [57,22,6], [49,17,12],
                       [58,23,8], [57,29,6], [49,23,13], [49,22,17], [39,25,17],
                       [46,18,14], [68,13,10], [48,28,10], [41,26,18], [40,25,15]]),
    'Jazz': np.array([[61,23,6], [80,12,5], [77,14,3], [61,19,10], [76,14,5],
                      [50,30,10], [58,20,15], [56,19,13], [75,17,3], [51,25,12],
                      [55,30,6], [59,22,10], [61,13,11], [67,21,6], [41,29,16],
                      [70,14,9], [61,29,4], [78,10,5]]),
    'Lyrical': np.array([[81,11,7], [69,17,9], [64,15,12], [68,14,10], [71,12,7],
                         [90,6,2], [74,16,7], [63,26,6], [76,17,4], [79,12,6],
                         [66,19,7], [40,30,13], [60,26,7], [84,7,4], [58,19,17],
                         [79,12,4], [68,21,5], [65,18,12], [81,11,4]]),
    'Modern': np.array([[72,19,5], [60,21,9], [54,32,10], [68,19,7], [74,15,6],
                        [53,19,17], [62,21,8], [70,17,6], [75,14,8], [73,16,8],
                        [61,21,9], [79,12,5], [37,26,20], [77,12,6], [62,23,9],
                        [78,10,6], [66,21,5], [57,22,9], [68,15,8], [42,18,18],
                        [83,8,6], [97,2,0], [69,20,5], [59,22,11], [67,16,7],
                        [63,23,8], [88,5,4], [64,27,5], [60,24,8], [48,26,12],
                        [52,32,8], [71,16,7], [60,24,8], [44,24,14], [53,21,17]]),
    'Tap': np.array([[60,31,6], [55,25,11], [84,10,4], [55,31,9], [64,22,7],
                     [56,19,11], [64,19,10], [65,22,6], [83,10,5], [63,23,9],
                     [70,22,4], [87,7,3]])
}

genres = list(data.keys())
means = np.array([data[genre].mean(axis=0) for genre in genres])

all_mean = means.mean(axis=0)
genres.append('Average')
means = np.vstack([means, all_mean])

# Plot setup
x = np.arange(len(genres))
width = 0.6
colors = ['#3CAEA3', '#20639B', '#F25F5C']  # PC1, PC2, PC3

fig, ax = plt.subplots(figsize=(16, 7))

# Plot stacked bars
b1 = ax.bar(x, means[:,0], width, label='PC1', color=colors[0])
b2 = ax.bar(x, means[:,1], width, bottom=means[:,0], label='PC2', color=colors[1])
b3 = ax.bar(x, means[:,2], width, bottom=means[:,0]+means[:,1], label='PC3', color=colors[2])

# Highlight the 'All Genres' bar with bold outline
idx = len(genres)-1
for bar in [b1[idx], b2[idx], b3[idx]]:
    bar.set_edgecolor('black')
    bar.set_linewidth(4)
    bar.set_hatch('/')

ax.set_yticks(np.arange(0, 101, 25))
ax.grid(axis='y', linestyle='--', alpha=0.3)

# X ticks
ax.set_xticks(x)
ax.set_xticklabels(genres, fontweight='bold')

# Labels and legend
ax.set_xlabel('')
ax.set_ylabel('Mean Variance Explained (%)')
ax.legend()

plt.tight_layout()
plt.show()
