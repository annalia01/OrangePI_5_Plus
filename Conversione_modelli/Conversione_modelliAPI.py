import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from rknn.api import RKNN
import os
import tf_keras 

# Per inception_v1 cambiare 'model_url' mettendo questo link: "https://www.kaggle.com/models/google/inception-v1/tensorFlow2/classification/2"
# Per MobileNet_v1 cambiare 'model_url' mettendo questo link: "https://www.kaggle.com/models/google/mobilenet-v1/tensorFlow2/100-224-classification/2"

# ===========================
# STEP 1: Scarica modello da API e salva come TFLite
# ===========================
model_url = "https://www.kaggle.com/models/google/mobilenet-v2/TensorFlow2/100-224-classification/2"
model = tf_keras.Sequential([hub.KerasLayer(model_url, input_shape=(224, 224, 3))])

# Salva il modello come SavedModel
saved_model_dir = './mobilenet_v2_savedmodel'
model.save(saved_model_dir)
print("✅ SavedModel salvato.")

# Converti SavedModel in TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

with open('mobilenet_v2.tflite', 'wb') as f:
    f.write(tflite_model)
print("✅ Modello TFLite salvato.")

# ===========================
# STEP 2: Conversione in RKNN
# ===========================

rknn = RKNN()

# Configuriamo mean e std e la piattaforma target
rknn.config(mean_values=[[127.5, 127.5, 127.5]], std_values=[[127.5, 127.5, 127.5]], target_platform='rk3588')

# Carica TFLite
print("🔄 Caricamento TFLite...")
ret = rknn.load_tflite('./mobilenet_v2.tflite')
if ret != 0:
    print("❌ Errore caricamento TFLite")
    exit(1)

# Costruisci RKNN
print("⚙️ Build RKNN in corso...")
ret = rknn.build(do_quantization=True, dataset='./dataset.txt') # <--- MODIFICA QUI
if ret != 0:
    print("❌ Errore build RKNN")
    exit(1)

# Salva RKNN
rknn.export_rknn('./mobilenet_v2.rknn')
print("✅ Modello RKNN salvato.")
