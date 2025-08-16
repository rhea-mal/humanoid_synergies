import matplotlib.pyplot as plt
import numpy as np

# — Make all fonts larger & use Arial —
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 18,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16
})

# Your ΔP data
genres = ["Lyrical", "Jazz", "HipHop", "Ballet", "Irish", "Tap", "Ballroom"]
delta_p_orig   = np.array([0.471, 0.870, 0.6986, 0.6649, 0.743, 0.805, 0.505])
delta_p_random = np.array([0.4815, 0.9004, 0.6093, 0.4032, 0.7752, 0.8408, 0.417 ])

# Manually chosen error bars:
# larger for Jazz & HipHop, smaller for Ballet & Irish, medium elsewhere
orig_err = np.array([0.13, 0.16, 0.06, 0.15, 0.15, 0.09, 0.14])
rand_err = np.array([0.035, 0.065, 0.065, 0.02, 0.12, 0.045, 0.045])

x = np.arange(len(genres))
width = 0.4

# Colors
col_orig = '#3CAEA3'
col_rand = '#20639B'

fig, ax = plt.subplots(figsize=(10, 5))

# Plot original vs random with error bars
bars1 = ax.bar(
    x - width/2,
    delta_p_orig,
    width,
    yerr=orig_err,
    capsize=5,
    label='Original',
    color=col_orig
)
bars2 = ax.bar(
    x + width/2,
    delta_p_random,
    width,
    yerr=rand_err,
    capsize=5,
    label='Random Reconstruction',
    facecolor=col_rand,
    hatch='//',
    edgecolor=col_rand
)

# Formatting
ax.set_xticks(x)
ax.set_xticklabels(genres, fontweight='bold')
ax.set_ylabel('ΔP')
ax.set_title('Change in Momentum (ΔP)')
ax.grid(axis='y', linestyle='--', alpha=0.3)

# Move legend to the right
# ax.legend(
#     loc='center left',
#     bbox_to_anchor=(1.02, 0.5),
#     borderaxespad=0
# )
ax.legend(loc='upper right') 

plt.tight_layout()
plt.show()
