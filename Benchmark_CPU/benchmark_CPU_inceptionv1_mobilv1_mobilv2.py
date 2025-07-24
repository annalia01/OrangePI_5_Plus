import time
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

print("🔄 Caricamento modello...")
model = hub.load("https://www.kaggle.com/models/google/inception-v1/TensorFlow2/classification/2")
print("✅ Modello caricato.")
t1=time.perf_counter()

def load_image(path):
    img = Image.open(path).resize((224, 224)).convert("RGB")
    img = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)  

# Carica immagine
img = load_image("immagine.jpg")
t2=time.perf_counter()
t3=time.perf_counter()
# Numero di inferenze da eseguire
N = 15
times = []

print(f"🚀 Avvio benchmark su {N} inferenze...\n")

for i in range(N):
    start = time.time()
    pred = model(img)
    end = time.time()

    elapsed = end - start
    times.append(elapsed)

    top_class = tf.argmax(pred[0]).numpy()
    print(f"[{i+1}] Tempo: {elapsed:.4f}s - Classe predetta: {top_class}")
t4=time.perf_counter()
t5=time.perf_counter()
print(f"\n⏱️ Tempo medio inferenza: {np.mean(times):.4f} secondi")
t6=time.perf_counter()
print(t2-t1)
print(t4-t3)
print(t6-t5)