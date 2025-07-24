import time
import numpy as np
from PIL import Image
from inception_v4 import create_inception_v4

t1=time.perf_counter()

model = create_inception_v4(load_weights=False)


def load_image(path):
    img = Image.open(path).resize((299, 299)).convert("RGB")
    img = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)  

img = load_image("immagine.jpg")
t2=time.perf_counter()

N = 15
times = []

print("▶ Inizio benchmark su CPU")
t3=time.perf_counter()
for i in range(N):
    start = time.time()
    pred = model.predict(img)
    end = time.time()
    times.append(end - start)
    print(f"[{i+1}] Tempo: {times[-1]:.4f} sec - Classe predetta: {np.argmax(pred)}")
t4=time.perf_counter()
t5=time.perf_counter()
print(f"\n⏱️ Tempo medio: {np.mean(times):.4f} sec")
t6=time.perf_counter()
print(t2-t1)
print(t4-t3)
print(t6-t5)
