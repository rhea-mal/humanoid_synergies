import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 20,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 15,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'legend.title_fontsize': 18
})

# original data
genres = ["Squats", "Jumping Jacks", "Walk in Circle", "Steps"]
delta_p_original  = np.array([0.10022, 0.301,   0.143,  0.179 ])
delta_p_random    = np.array([0.1266,  0.293,   0.1341, 0.1974])
delta_ke_original = np.array([0.766,   0.33,    0.447,  0.56  ])
delta_ke_random   = np.array([0.486,   0.32,    0.208,  0.2208])

p_orig_err  = np.array([0.010, 0.020, 0.015, 0.018])
p_rand_err  = np.array([0.015, 0.018, 0.022, 0.025])
ke_orig_err = np.array([0.050, 0.040, 0.030, 0.060])
ke_rand_err = np.array([0.055, 0.045, 0.035, 0.065])

# re‐order to swap positions 2 and 3
order = [0, 1, 3, 2]

genres        = [genres[i] for i in order]
delta_p_original  = delta_p_original[order]
delta_p_random    = delta_p_random[order]
delta_ke_original = delta_ke_original[order]
delta_ke_random   = delta_ke_random[order]

p_orig_err  = p_orig_err[order]
p_rand_err  = p_rand_err[order]
ke_orig_err = ke_orig_err[order]
ke_rand_err = ke_rand_err[order]

# plot
x = np.arange(len(genres))
width = 0.35
col_orig = '#BFA053'
col_orig = '#d0941c'
col_rand = '#20639B'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

# ΔP plot
ax1.bar(x - width/2, delta_p_original, width,
        yerr=p_orig_err, capsize=4,
        label='Original', color=col_orig)
ax1.bar(x + width/2, delta_p_random, width,
        yerr=p_rand_err, capsize=4,
        label='Synergy Reconstruction',
        facecolor=col_rand, hatch='//', edgecolor=col_rand)
ax1.set_xticks(x)
ax1.set_xticklabels(genres, fontweight='bold')
ax1.set_ylabel('ΔP')
ax1.set_title('Change in Momentum')
ax1.grid(axis='y', linestyle='--', alpha=0.3)

# ΔKE plot
ax2.bar(x - width/2, delta_ke_original, width,
        yerr=ke_orig_err, capsize=4,
        label='Original', color=col_orig)
ax2.bar(x + width/2, delta_ke_random, width,
        yerr=ke_rand_err, capsize=4,
        label='Synergy Reconstruction',
        facecolor=col_rand, hatch='//', edgecolor=col_rand)
ax2.set_xticks(x)
ax2.set_xticklabels(genres, fontweight='bold')
ax2.set_ylabel('ΔKE')
ax2.set_title('Change in Kinetic Energy')
ax2.legend()
ax2.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()
