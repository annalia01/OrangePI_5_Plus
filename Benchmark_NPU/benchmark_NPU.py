import time
import numpy as np
import cv2
from rknnlite.api import RKNNLite


RKNN_MODEL = 'mobilenet_v2.rknn'
IMAGE_PATH = 'immagine.jpg'  
INPUT_SIZE = (224, 224)    

t1=time.perf_counter()
rknn_lite = RKNNLite()
ret = rknn_lite.load_rknn(RKNN_MODEL)
if ret != 0:
    print("❌ Failed to load RKNN model")
    exit(1)


ret = rknn_lite.init_runtime()



img = cv2.imread(IMAGE_PATH)
img = cv2.resize(img, INPUT_SIZE)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.uint8)  
img = np.expand_dims(img, axis=0) 
t2=time.perf_counter()




for _ in range(5):
    rknn_lite.inference(inputs=[img])


N_RUNS = 15
times = []
t3=time.perf_counter()
for i in range(N_RUNS):
    start = time.time()
    outputs = rknn_lite.inference(inputs=[img])
    end = time.time()

    elapsed = (end - start) * 1000  # in ms
    times.append(elapsed)
    print(f"[{i+1}] Inference time: {elapsed:.2f} ms")
t4=time.perf_counter()
t5=time.perf_counter()
rknn_lite.release()
t6=time.perf_counter()

print(t2-t1)
print(t4-t3)
print(t6-t5)
