import time
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = '/home/orangepi/Downloads/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8/saved_model'
IMAGE_PATH = 'immagine.jpg'
N = 15


t1=time.perf_counter()
print("🔄 Caricamento modello SSD MobileNet v1...")
detect_fn = tf.saved_model.load(MODEL_PATH)
print("✅ Modello caricato.")


def load_image(path):
    img = Image.open(path).convert('RGB').resize((640, 640))
    img = np.array(img).astype(np.uint8) 
    return tf.convert_to_tensor([img], dtype=tf.uint8)  

image_tensor = load_image(IMAGE_PATH)


times = []

print(f"🚀 Avvio benchmark su {N} inferenze...\n")
t2=time.perf_counter()
t3=time.perf_counter()
for i in range(N):
    start = time.time()
    detections = detect_fn(image_tensor)
    end = time.time()

    elapsed = end - start
    times.append(elapsed)

    classes = detections['detection_classes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()

    top_class = int(classes[0])
    top_score = scores[0]

    print(f"[{i+1}] Tempo: {elapsed:.4f}s - Classe: {top_class} - Score: {top_score:.2f}")
t4=time.perf_counter()
t5=time.perf_counter()
print(f"\n⏱️ Tempo medio inferenza SSD MobileNet v2: {np.mean(times)*1000:.2f} ms")
t6=time.perf_counter()
print(t2-t1)
print(t4-t3)
print(t6-t5)