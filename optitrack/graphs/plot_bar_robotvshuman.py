import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Setup seaborn clean scientific style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)

# Activity labels
activities = ["Jump", "Walk in Place", "Squat", "Clap Hands", "Raise Arms"]

# Power data (Watts)
human_power = [383, 110, 179.4, 167.44, 89]
robot_power = [980, 417, 348.8, 381.60, 282]

# Error bars (hard-coded, you can adjust later)
human_errors = [54.1, 24.3, 40, 38, 29]  # adjust these manually
robot_errors = [68, 65.9, 42.2, 35, 23]  # adjust these manually

# Combine into a tidy DataFrame for seaborn
data = pd.DataFrame({
    "Activity": activities * 2,
    "Power Output (W)": human_power + robot_power,
    "Agent": ["Human"] * len(activities) + ["Robot"] * len(activities),
    "Error": human_errors + robot_errors
})

# Plot
plt.figure(figsize=(8, 6))
barplot = sns.barplot(
    data=data,
    x="Activity",
    y="Power Output (W)",
    hue="Agent",
    palette={"Human": "royalblue", "Robot": "#E69F00"},
    errorbar=None  # turn off seaborn's automatic error bars
)

# Add manual error bars
for i, (bar, err) in enumerate(zip(barplot.patches, data['Error'])):
    bar_center = bar.get_x() + bar.get_width() / 2
    bar_height = bar.get_height()
    plt.errorbar(
        bar_center, bar_height, 
        yerr=err, 
        ecolor='black', 
        capsize=5, 
        fmt='none', 
        elinewidth=1.2
    )

# Final formatting
plt.xlabel('')
plt.ylabel('Average Power Output (W)', fontsize=20)
# plt.title('Human vs Robot Power Output Across Activities', pad=15)
plt.legend(frameon=False, loc="upper right", fontsize=16)
plt.xticks(rotation=0, fontsize=16)
plt.tight_layout()

# Save high-res (optional)
# plt.savefig('power_bar_graph.png', dpi=600, bbox_inches='tight')

plt.show()
