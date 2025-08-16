import redis
import time
import csv
import argparse
from datetime import datetime

# Redis configuration
REDIS_HOST = '127.0.0.1'
REDIS_PORT = 6379
# INPUT_FILE = '/Users/rheamalhotra/Desktop/robotics/optitrack-robot-dance/optitrack/recordings/jump.txt'  # Input file with the data
# INPUT_FILE = '/Users/rheamalhotra/Desktop/robotics/optitrack-robot-dance/optitrack/recordings/jump.txt'  # Input file with the data
INPUT_FILE = './recordings/tracy_oct_6_24_conservative.txt'  # Input file with the data
# INPUT_FILE = './recordings/tracy_oct_6_24_in_place.txt'  # Input file with the data
# INPUT_FILE = './recordings/dual_oct_15_slow_dance_rev1.txt'  # Input file with the data
USER_READY_KEY = ["sai2::optitrack::user_ready", "sai2::optitrack::user_1_ready", "sai2::optitrack::user_2_ready"]

# Connect to Redis
redis_client = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

def read_data(file_path):
    """
    Read data from the input file and return it as a list of dictionaries.
    Each dictionary represents a row with key-value pairs.
    """
    data = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for row in reader:
            data.append(row)
    return data

def publish_to_redis(data, rate_hz, loop):
    """
    Publish each row of data to Redis at the specified rate.
    """
    interval = 1.0 / rate_hz
    while True:
        for row in data:
            # Remove timestamp from row if it exists
            row.pop('timestamp', None)  
            for key, value in row.items():
                redis_client.set(key, value)
            time.sleep(interval)
        if not loop:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Redis Data Publisher')
    parser.add_argument('--loop', action='store_true', help='Run the script in a loop')
    args = parser.parse_args()
    
    # reset user ready key 
    for key in USER_READY_KEY:
        redis_client.set(key, 0)
    
    # parse data 
    data = read_data(INPUT_FILE)
    
    # publish first frame 
    for key, value in data[0].items():
        redis_client.set(key, value)
    
    # set user ready key 
    for key in USER_READY_KEY:
        redis_client.set(key, 1)
    
    # publish rest of data when user is ready 
    publish_to_redis(data, rate_hz=100, loop=args.loop)
