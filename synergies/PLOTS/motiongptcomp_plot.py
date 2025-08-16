import matplotlib.pyplot as plt
import numpy as np

# --- Global style ---
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 22,
    'axes.titlesize': 24,
    'axes.labelsize': 22,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 18
})

# --- Data ---

# swap order so that index 2 is "Steps" and index 3 is "Walk in Circle"
genres = ["Squats", "Jumping Jacks", "Steps", "Walk in Circle"]

# Foot sliding means & errors, with elements 2↔3 swapped
foot_original      = np.array([0.16, 0.35, 0.18, 0.10])
foot_original_err  = np.array([0.015, 0.020, 0.018, 0.010])
foot_random        = np.array([0.13, 0.32, 0.21, 0.09])
foot_random_err    = np.array([0.018, 0.025, 0.020, 0.012])
foot_motiongpt     = np.array([0.53, 0.55, 0.25, 0.41])
foot_motiongpt_err = np.array([0.030, 0.035, 0.028, 0.025])
foot_null          = np.array([0.23, 0.33, 0.17, 0.13])
foot_null_err      = np.array([0.012, 0.018, 0.015, 0.010])

# Power means & vars → std-dev, with elements 2↔3 swapped
power_means = {
    'Original':                         np.array([392.70, 1093.0, 324.00, 592.70]),
    'Synergy Reconstruction':          np.array([342.00,  905.70, 301.38, 832.8 ]),
    'MotionGPT':                        np.array([940.00, 1392.70, 842.38, 932.39]),
    'MotionGPT Null-Space Projection': np.array([440.30,  985.50, 359.38, 882.28])
}
power_vars = {
    'Original':                         np.array([5069.00, 5201.00, 3504.00, 2069.00]),
    'Synergy Reconstruction':          np.array([5002.00, 3503.00, 3185.24, 4250.24]),
    'MotionGPT':                        np.array([4903.00, 8015.00, 9362.24, 2205.24]),
    'MotionGPT Null-Space Projection': np.array([9204.00, 3240.00, 9302.24, 2025.24])
}
power_err = {k: np.sqrt(v) for k, v in power_vars.items()}

# --- Plot setup ---
x = np.arange(len(genres))
width = 0.2

colors = {
    'Original':                         '#BFA053',
    'Synergy Reconstruction':          '#20639B',
    'MotionGPT':                        '#3CAEA3',
    'MotionGPT Null-Space Projection':  '#2A7F7C'
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Foot Sliding with error bars ---
for i, (label, vals, errs, hatch) in enumerate([
    ('Original',                         foot_original,  foot_original_err,  None),
    ('Synergy Reconstruction',           foot_random,    foot_random_err,    None),
    ('MotionGPT',                        foot_motiongpt, foot_motiongpt_err, None),
    ('MotionGPT Null-Space Projection',  foot_null,      foot_null_err,      '//'),
]):
    ax1.bar(
        x + (i - 1.5) * width,
        vals, width,
        yerr=errs, capsize=4,
        color=colors[label],
        hatch=hatch or ''
    )
ax1.set_xticks(x)
ax1.set_xticklabels(genres, rotation=0, fontweight='bold')
ax1.set_ylabel('Foot Sliding Ratio')
ax1.grid(axis='y', linestyle='--', alpha=0.3)

# --- Power with error bars ---
for i, label in enumerate([
    'Original',
    'Synergy Reconstruction',
    'MotionGPT',
    'MotionGPT Null-Space Projection'
]):
    ax2.bar(
        x + (i - 1.5) * width,
        power_means[label], width,
        yerr=power_err[label], capsize=5,
        label=label,
        color=colors[label],
        hatch='//' if 'Null-Space' in label else ''
    )
ax2.set_xticks(x)
ax2.set_xticklabels(genres, rotation=0, fontweight='bold')
ax2.set_ylabel('Power (W)')
ax2.grid(axis='y', linestyle='--', alpha=0.3)

# --- Shared legend just below the plots ---
handles, labels = ax2.get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.02),
    ncol=4,
    frameon=False
)

plt.subplots_adjust(bottom=0.22)  # pull it up a bit
plt.tight_layout()
plt.show()
