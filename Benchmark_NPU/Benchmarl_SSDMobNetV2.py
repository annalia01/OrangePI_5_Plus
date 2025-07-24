import time
import numpy as np
import cv2
from rknnlite.api import RKNNLite

# === CONFIG ===
RKNN_MODEL = 'ssd_mobilenet_v2_int8.rknn'
IMAGE_PATH = 'immagine.jpg'  # Cambia con il nome reale dell'immagine
INPUT_SIZE = (300, 300)      # SSD-MobileNet usa tipicamente 300x300
N_RUNS = 15

# === LOAD RKNN MODEL ===
rknn_lite = RKNNLite()
print("🔄 Caricamento modello RKNN...")
ret = rknn_lite.load_rknn(RKNN_MODEL)
if ret != 0:
    print("❌ Errore nel caricamento del modello RKNN")
    exit(1)

# === INIT RUNTIME ===
print("⚙️ Inizializzazione runtime...")
ret = rknn_lite.init_runtime()
if ret != 0:
    print("❌ Errore durante init_runtime")
    exit(1)

# === PREPARE IMAGE ===
print("🖼️ Caricamento immagine...")
img = cv2.imread(IMAGE_PATH)
if img is None:
    print("❌ Immagine non trovata:", IMAGE_PATH)
    exit(1)

img = cv2.resize(img, INPUT_SIZE)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.uint8)
img = np.expand_dims(img, axis=0)

# === WARM-UP ===
print("🔥 Warm-up (5 inferenze)...")
for _ in range(5):
    rknn_lite.inference(inputs=[img])

# === BENCHMARK ===
print(f"⏱️ Inizio benchmark con {N_RUNS} iterazioni...")
times = []
for i in range(N_RUNS):
    start = time.time()
    outputs = rknn_lite.inference(inputs=[img])
    end = time.time()
    elapsed_ms = (end - start) * 1000
    times.append(elapsed_ms)
    print(f"[{i+1}/{N_RUNS}] Tempo inferenza: {elapsed_ms:.2f} ms")

# === RISULTATI ===
avg_time = sum(times) / len(times)
print("\n📊 RISULTATI:")
print(f"➤ Tempo medio: {avg_time:.2f} ms")
print(f"➤ Tempo minimo: {min(times):.2f} ms")
print(f"➤ Tempo massimo: {max(times):.2f} ms")

# === CLEANUP ===
rknn_lite.release()
print("✅ Benchmark completato e risorse rilasciate.")
