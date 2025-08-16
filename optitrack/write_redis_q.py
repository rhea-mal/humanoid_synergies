import redis
import time
import argparse
import pandas as pd
import numpy as np

# === Redis Configuration ===
REDIS_HOST = '127.0.0.1'
REDIS_PORT = 6379
USER_READY_KEY = [
    "sai2::optitrack::user_ready",
    "sai2::optitrack::user_1_ready",
    "sai2::optitrack::user_2_ready"
]

ROBOT_QI_KEY = "robot_qi"
ROBOT_QF_KEY = "robot_qf"
ROBOT_DQ_KEY = "robot_dq"

SYNERGY_PATH = "/Users/rheamalhotra/Desktop/robotics/optitrack_dance_demo/optitrack/recordings/dq_synergies/segment_0.csv"

# === Connect to Redis ===
redis_client = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# === Load CSV ===
df = pd.read_csv(SYNERGY_PATH)
df.columns = df.columns.str.strip().str.replace('"', '')

# Extract keys (robot_qi, robot_qf, PC_0)
row_keys = df.iloc[:, 0].tolist()
data = df.iloc[:, 1:].to_numpy()

synergy_data = {key: data[i] for i, key in enumerate(row_keys)}

robot_qi = np.array(synergy_data['robot_qi'])
robot_qf = np.array(synergy_data['robot_qf'])
robot_pc0 = np.array(synergy_data['PC_0'])
robot_pc1 = np.array(synergy_data['PC_1'])
robot_pc2 = np.array(synergy_data['PC_2'])

# === Format for Redis ===
def array_to_redis_string(array):
    return '[' + ','.join(f"{x:.6f}" for x in array) + ']'

# === Main Execution ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Publish Synergy to Redis with incremental dq')
    parser.add_argument('--rate', type=float, default=100.0, help='Publish rate in Hz')
    parser.add_argument('--stepsize', type=float, default=0.0001, help='Stepsize multiplier for dq')
    parser.add_argument('--max_steps', type=int, default=100, help='Maximum number of increments')
    args = parser.parse_args()

    # publish qi and qf once
    redis_client.set(ROBOT_QI_KEY, array_to_redis_string(robot_qi))
    redis_client.set(ROBOT_QF_KEY, array_to_redis_string(robot_qf))
    for key in USER_READY_KEY:
        redis_client.set(key, 1)

    print("✅ Published robot_qi and robot_qf. Starting robot_dq publishing...")

    interval = 1.0 / args.rate
    a=0.0
    b=0.3
    c=3.2
    robot_dq = a*robot_pc0 + b*robot_pc1 + c*robot_pc2
    dq_accumulator = np.zeros_like(robot_dq)

    for step in range(args.max_steps):
        dq_accumulator += args.stepsize * robot_dq
        redis_client.set(ROBOT_DQ_KEY, array_to_redis_string(dq_accumulator))
        time.sleep(interval)

    print("✅ Finished publishing all steps.")
