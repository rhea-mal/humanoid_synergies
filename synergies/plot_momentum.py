import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# --- Style Settings ---
plt.rcParams.update({
    "font.family": "Helvetica",  # Use Helvetica
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.5,
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
    "legend.frameon": False,
    "figure.dpi": 200
})

# --- Load Data ---
file_path = "momentum_changes.txt"
sampling_rate_hz = 100  # 100 Hz => 0.01s per frame
momentum_values = []

with open(file_path, "r") as file:
    for line in file:
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                momentum = float(parts[1])
                momentum_values.append(momentum)
            except ValueError:
                continue

# --- Plot ---
if momentum_values:
    num_points = len(momentum_values)
    time_values = np.arange(num_points) * (1 / 20)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(time_values, momentum_values, color="#005C99", marker='o', linestyle='-', label="Momentum Change (ΔP)")

    # Axis labels
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Momentum ΔP")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.show()
else:
    print("No valid momentum data found.")
