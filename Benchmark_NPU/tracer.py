import time
import os

OUTPUT_FILE = "rknpu_trace3.csv"
DURATION = 30  # secondi
INTERVAL = 1  # secondi

RKNPU_PATH = "/proc/rknpu"
FIELDS = ["load", "freq", "power", "volt"]

def read_value(field):
    try:
        with open(os.path.join(RKNPU_PATH, field), 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return "NA"



with open(OUTPUT_FILE, 'w') as f:
    f.write("timestamp," + ",".join(FIELDS) + "\n")

    start_time = time.time()
    while (time.time() - start_time) < DURATION:
        now = time.strftime("%H:%M:%S")
        values = [read_value(field) for field in FIELDS]
        f.write(f"{now}," + ",".join(values) + "\n")
        time.sleep(INTERVAL)


