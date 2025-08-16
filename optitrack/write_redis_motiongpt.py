import redis
import time
import argparse
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import numpy as np

# === Redis Configuration ===
REDIS_HOST = '127.0.0.1'
REDIS_PORT = 6379
USER_READY_KEY = [
    "sai2::optitrack::user_ready", 
    "sai2::optitrack::user_1_ready", 
    "sai2::optitrack::user_2_ready"
]

# === Connect to Redis ===
redis_client = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# === CSV Configuration ===
# CSV_PATH = './recordings/hiphop2.csv'
CSV_PATH="/Users/rheamalhotra/Desktop/robotics/optitrack_dance_demo/optitrack/recordings/motiongpt.csv"
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip().str.replace('"', '')

# === Mapping from CSV body part names → OptiTrack indices from updated YAML ===
csv_to_optitrack = {
    'head':         ('neck_link2', 4),
    'torso':        ('trunk_rz', 2),
    'hip':          ('hip_base', 1),
    'left_foot':    ('LL_foot', 46),
    'right_foot':   ('RL_foot', 50),
    'left_hand':    ('la_end_effector', 9),
    'right_hand':   ('ra_end_effector', 27),
    'left_elbow':   ('la_link4', 7),
    'right_elbow':  ('ra_link4', 26),
    'left_knee':    ('LL_KOSY_L56', 48),
    'right_knee':   ('RL_KOSY_L56', 44),
}

# === Format Redis Data ===
def format_redis_data(df):
    frames = []

    for i in range(0, len(df)):
        frame = {}
        # frame["timestamp"] = df["time"].iloc[i]
        for csv_name, (redis_name, idx) in csv_to_optitrack.items():
            try:
                # Position
                pos = [
                    df[f"{csv_name}_pos__0"].iloc[i],
                    df[f"{csv_name}_pos__1"].iloc[i],
                    df[f"{csv_name}_pos__2"].iloc[i]
                ]
                pos_key = f"motiongpt::0::{idx}::pos"
                frame[pos_key] = f"[{pos[0]}, {pos[1]}, {pos[2]}]"

                # Apply the Z-rotation: new matrix = Rz_90 * original
                rot_matrix = np.array([
                    [df[f"{csv_name}_ori__0"].iloc[i], df[f"{csv_name}_ori__1"].iloc[i], df[f"{csv_name}_ori__2"].iloc[i]],
                    [df[f"{csv_name}_ori__3"].iloc[i], df[f"{csv_name}_ori__4"].iloc[i], df[f"{csv_name}_ori__5"].iloc[i]],
                    [df[f"{csv_name}_ori__6"].iloc[i], df[f"{csv_name}_ori__7"].iloc[i], df[f"{csv_name}_ori__8"].iloc[i]]
                ])
                # quat = R.from_matrix(rot_matrix).as_quat()  # [x, y, z, w]
                quat_xyzw = R.from_matrix(rot_matrix).as_quat()  # [x, y, z, w]
                # quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
                ori_key = f"motiongpt::0::{idx}::ori"
                frame[ori_key] = f"[{quat_xyzw[0]}, {quat_xyzw[1]}, {quat_xyzw[2]}, {quat_xyzw[3]}]"

            except KeyError as e:
                print(f"[Warning] Missing data for {csv_name} at frame {i}: {e}")
                continue
            except ValueError as ve:
                print(f"[Warning] Invalid rotation matrix for {csv_name} at frame {i}: {ve}")
                continue

        frames.append(frame)

    return frames

# === Replay to Redis ===
def publish_to_redis(data, rate_hz=100, loop=False):
    interval = 1.0 / rate_hz
    idx = 0
    total_frames = len(data)

    while True:
        frame = data[idx]
        for key, value in frame.items():
            # redis_client.set(key, value)
            if key == "timestamp":
                pass
                # redis_client.set("timestamp", str(value))  # <-- Set timestamp
            else:
                redis_client.set(key, str(value))  # Ensure all values are strings


        time.sleep(interval)
        idx += 1

        if idx >= total_frames:
            if loop:
                idx = 0
            else:
                break

# === Main Execution ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Replay OptiTrack CSV to Redis')
    parser.add_argument('--loop', action='store_true', help='Loop the playback')
    args = parser.parse_args()

    data = format_redis_data(df)
    # Publish the first frame
    for key, value in data[0].items():
        redis_client.set(key, value)

    # Signal readiness
    for key in USER_READY_KEY:
        redis_client.set(key, 1)

    # Start replay
    publish_to_redis(data, rate_hz=10, loop=args.loop)
