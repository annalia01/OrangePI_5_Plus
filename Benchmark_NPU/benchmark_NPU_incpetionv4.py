import time
import numpy as np
import cv2
import psutil
import os
from rknnlite.api import RKNNLite

# === CONFIG ===
RKNN_MODEL = './inception_v4.rknn'
IMAGE_PATH = './immagine.jpg'  
INPUT_SIZE = (299, 299)
N_RUNS = 15
t1=time.perf_counter()
# === INIZIALIZZA RKNN ===
rknn_lite = RKNNLite()
ret = rknn_lite.load_rknn(RKNN_MODEL)
if ret != 0:
    print("❌ ERRORE: caricamento modello fallito")
    exit(1)

ret = rknn_lite.init_runtime()
if ret != 0:
    print("❌ ERRORE: init runtime fallita")
    exit(1)

# === CARICA IMMAGINE ===
img = cv2.imread(IMAGE_PATH)
if img is None:
    print(f"❌ Immagine non trovata: {IMAGE_PATH}")
    exit(1)

img = cv2.resize(img, INPUT_SIZE)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.uint8)
img = np.expand_dims(img, axis=0)  # [1, H, W, C]

# === MONITOR ===
pid = os.getpid()
proc = psutil.Process(pid)

# === BENCHMARK ===
times = []
print(f"\n🚀 Benchmarking InceptionV4 per {N_RUNS} iterazioni...\n")
t2=time.perf_counter()
t3=time.perf_counter()
# Warm-up
for _ in range(5):
    rknn_lite.inference(inputs=[img])
t4=time.perf_counter()
t5=time.perf_counter()
for i in range(N_RUNS):
    cpu = proc.cpu_percent()
    ram = proc.memory_info().rss / (1024 * 1024)

    start = time.time()
    outputs = rknn_lite.inference(inputs=[img])
    end = time.time()

    elapsed = (end - start) * 1000
    times.append(elapsed)

    print(f"[{i+1:02d}] Time: {elapsed:.2f} ms ")



rknn_lite.release()
t6=time.perf_counter()

print(t2-t1)
print(t4-t3)
print(t6-t5)