import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path

# --- Setup ---
ROBOT_DQ_PATH = "robot_dq.csv"
ROBOT_Q_PATH = "robot_q.csv"
TXT = "momentum_changes.txt"
PCA_FOLDER = Path(f"../optitrack/recordings/dq_synergies/tap/")

def pca_per_segment(ROBOT_DQ_PATH, ROBOT_Q_PATH, txt_path, n_components=3):
    # Load cleaned CSVs
    df_dq = pd.read_csv(ROBOT_DQ_PATH, header=None)
    df_q = pd.read_csv(ROBOT_Q_PATH, header=None)

    pose_cols = list(range(1, df_dq.shape[1]))  # assume first column is time
    pose_data_dq = df_dq[pose_cols].to_numpy()
    pose_data_q = df_q[pose_cols].to_numpy()

    # Load momentum change times
    with open(txt_path, 'r') as f:
        valid_times = [float(line.strip()) for line in f if line.strip()]
    valid_times.sort()

    timestamps = df_dq[0].to_numpy()

    os.makedirs(PCA_FOLDER, exist_ok=True)

    variance_info = []
    reconstruction_errors = []

    # Iterate over segment pairs
    for j in range(len(valid_times) - 1):
        start_time, end_time = valid_times[j], valid_times[j + 1]
        mask = (timestamps >= start_time) & (timestamps <= end_time)
        segment = pose_data_dq[mask]
        
        if len(segment) < 2:
            print(f"⚠️ Skipping segment {j} (too few frames)")
            continue

        # Get initial and final q
        start_indices = np.where(timestamps >= start_time)[0]
        if len(start_indices) == 0:
            idx_start = len(timestamps) - 1  # fallback to last frame
        else:
            idx_start = start_indices[0]

        end_indices = np.where(timestamps >= end_time)[0]
        if len(end_indices) == 0:
            idx_end = len(timestamps) - 1  # fallback to last frame
        else:
            idx_end = end_indices[0]

        qi = pose_data_q[idx_start]
        qf = pose_data_q[idx_end]

        # PCA
        pca = PCA(n_components=n_components)
        pca.fit(segment)

        ratios = pca.explained_variance_ratio_
        percents = [int(round(r * 100)) for r in ratios]
        print(f"Segment {j}: {percents[0]}% PC1, {percents[1]}% PC2, and {percents[2]}% PC3")

        # store overall variance
        explained_var = ratios.sum()
        variance_info.append((j, explained_var))

        # Reconstruction error
        reduced = pca.transform(segment)
        reconstructed = pca.inverse_transform(reduced)
        error = np.linalg.norm(segment - reconstructed, axis=1).mean()
        reconstruction_errors.append((j, error))

        # Save scaled PCs
        scaled_components = pca.components_ * pca.singular_values_[:, np.newaxis]

        # Assemble rows
        rows = []
        rows.append(["robot_qi"] + qi.tolist())
        rows.append(["robot_qf"] + qf.tolist())
        for i in range(min(3, n_components)):
            row = [f"PC_{i}"] + scaled_components[i].tolist()
            rows.append(row)

        headers = ["timestamp"] + [f"joint_{k}" for k in range(len(qi))]
        pc_df = pd.DataFrame(rows, columns=headers)

        segment_path = PCA_FOLDER / f"segment_{j}.csv"
        pc_df.to_csv(segment_path, index=False)

    print("\n📊 Variance explained per segment:")
    for seg, var in variance_info:
        print(f"  Segment {seg}: {var:.4f} variance retained")

    print("\n📉 Mean reconstruction error per segment:")
    for seg, err in reconstruction_errors:
        print(f"  Segment {seg}: {err:.4f} average L2 error")
pca_per_segment(ROBOT_DQ_PATH, ROBOT_Q_PATH, TXT, n_components=3)
