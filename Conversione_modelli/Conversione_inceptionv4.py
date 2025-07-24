import tensorflow as tf
import numpy as np 
from rknn.api import RKNN
import os
import tf_keras as keras 
from tensorflow.keras.utils import get_file

# --- Definizione Architettura Inception-v4 ---
TF_BACKEND_TF_DIM_ORDERING = "https://github.com/titu1994/Inception-v4/releases/download/v1.2/inception_v4_weights_tf_dim_ordering_tf_kernels.h5"

def conv_block(x, filters, kernel_size, strides=(1, 1), padding='same', use_bias=False):
    x = keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias)(x)
    x = keras.layers.BatchNormalization(axis=-1)(x)
    x = keras.layers.Activation('relu')(x)
    return x

def inception_stem(input_tensor):
    channel_axis = -1

    x = conv_block(input_tensor, 32, (3, 3), strides=(2, 2), padding='valid')
    x = conv_block(x, 32, (3, 3), padding='valid')
    x = conv_block(x, 64, (3, 3))

    x1 = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)
    x2 = conv_block(x, 96, (3, 3), strides=(2, 2), padding='valid')
    x = keras.layers.Concatenate(axis=channel_axis)([x1, x2])

    x1 = conv_block(x, 64, (1, 1))
    x1 = conv_block(x1, 96, (3, 3), padding='valid')

    x2 = conv_block(x, 64, (1, 1))
    x2 = conv_block(x2, 64, (1, 7))
    x2 = conv_block(x2, 64, (7, 1))
    x2 = conv_block(x2, 96, (3, 3), padding='valid')

    x = keras.layers.Concatenate(axis=channel_axis)([x1, x2])

    x1 = conv_block(x, 192, (3, 3), strides=(2, 2), padding='valid')
    x2 = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)

    x = keras.layers.Concatenate(axis=channel_axis)([x1, x2])
    return x

def inception_A(input_tensor):
    channel_axis = -1

    a1 = conv_block(input_tensor, 96, (1, 1))

    a2 = conv_block(input_tensor, 64, (1, 1))
    a2 = conv_block(a2, 96, (3, 3))

    a3 = conv_block(input_tensor, 64, (1, 1))
    a3 = conv_block(a3, 96, (3, 3))
    a3 = conv_block(a3, 96, (3, 3))

    a4 = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input_tensor)
    a4 = conv_block(a4, 96, (1, 1))

    return keras.layers.Concatenate(axis=channel_axis)([a1, a2, a3, a4])

def inception_B(input_tensor):
    channel_axis = -1

    b1 = conv_block(input_tensor, 384, (1, 1))

    b2 = conv_block(input_tensor, 192, (1, 1))
    b2 = conv_block(b2, 224, (1, 7))
    b2 = conv_block(b2, 256, (7, 1))

    b3 = conv_block(input_tensor, 192, (1, 1))
    b3 = conv_block(b3, 192, (7, 1))
    b3 = conv_block(b3, 224, (1, 7))
    b3 = conv_block(b3, 224, (7, 1))
    b3 = conv_block(b3, 256, (1, 7))

    b4 = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input_tensor)
    b4 = conv_block(b4, 128, (1, 1))

    return keras.layers.Concatenate(axis=channel_axis)([b1, b2, b3, b4])

def inception_C(input_tensor):
    channel_axis = -1

    c1 = conv_block(input_tensor, 256, (1, 1))

    c2 = conv_block(input_tensor, 384, (1, 1))
    c2_1 = conv_block(c2, 256, (1, 3))
    c2_2 = conv_block(c2, 256, (3, 1))
    c2 = keras.layers.Concatenate(axis=channel_axis)([c2_1, c2_2])

    c3 = conv_block(input_tensor, 384, (1, 1))
    c3 = conv_block(c3, 448, (3, 1))
    c3 = conv_block(c3, 512, (1, 3))
    c3_1 = conv_block(c3, 256, (1, 3))
    c3_2 = conv_block(c3, 256, (3, 1))
    c3 = keras.layers.Concatenate(axis=channel_axis)([c3_1, c3_2])

    c4 = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input_tensor)
    c4 = conv_block(c4, 256, (1, 1))

    return keras.layers.Concatenate(axis=channel_axis)([c1, c2, c3, c4])

def reduction_A(input_tensor):
    channel_axis = -1

    r1 = conv_block(input_tensor, 384, (3, 3), strides=(2, 2), padding='valid')

    r2 = conv_block(input_tensor, 192, (1, 1))
    r2 = conv_block(r2, 224, (3, 3))
    r2 = conv_block(r2, 256, (3, 3), strides=(2, 2), padding='valid')

    r3 = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(input_tensor)

    return keras.layers.Concatenate(axis=channel_axis)([r1, r2, r3])

def reduction_B(input_tensor):
    channel_axis = -1

    r1 = conv_block(input_tensor, 192, (1, 1))
    r1 = conv_block(r1, 192, (3, 3), strides=(2, 2), padding='valid')

    r2 = conv_block(input_tensor, 256, (1, 1))
    r2 = conv_block(r2, 256, (1, 7))
    r2 = conv_block(r2, 320, (7, 1))
    r2 = conv_block(r2, 320, (3, 3), strides=(2, 2), padding='valid')

    r3 = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(input_tensor)

    return keras.layers.Concatenate(axis=channel_axis)([r1, r2, r3])

def create_inception_v4_model(nb_classes=1001, load_weights=True):
    input_layer = keras.layers.Input((299, 299, 3)) 
    x = inception_stem(input_layer)

    for _ in range(4):
        x = inception_A(x)

    x = reduction_A(x)

    for _ in range(7):
        x = inception_B(x)

    x = reduction_B(x)

    for _ in range(3):
        x = inception_C(x)

    x = keras.layers.AveragePooling2D((8, 8))(x)
    x = keras.layers.Dropout(0.8)(x)
    x = keras.layers.Flatten()(x)
    out = keras.layers.Dense(units=nb_classes, activation='softmax')(x)

    model = keras.models.Model(input_layer, out, name='Inception-v4')

    if load_weights:
        weights_path = get_file('inception_v4_weights_tf_dim_ordering_tf_kernels.h5',
                                TF_BACKEND_TF_DIM_ORDERING, cache_subdir='models')
        try:
            model.load_weights(weights_path, by_name=True)
            print("Model weights loaded.")
        except Exception as e:
            print(f"❌ Errore durante il caricamento dei pesi (anche con by_name=True): {e}")
            print("Potrebbe esserci un mismatch di architettura o un file .h5 corrotto.")
            raise # Rilancia l'eccezione per fermare l'esecuzione


    return model

# --- Configurazione Modello Specifico (Inception-v4) ---
output_model_name = "inception_v4"
input_shape = (299, 299, 3)

# --- Gestione Dataset per Calibrazione ---

dataset_file = "dataset.txt"
calibration_image_name = "immagine.jpg"

# Verifica l'esistenza del file immagine e del file dataset.txt
if not os.path.exists(calibration_image_name):
    print(f"❌ Errore: L'immagine di calibrazione '{calibration_image_name}' non trovata nella stessa directory.")
    print("Assicurati che 'immagine.jpg' sia presente.")
    exit(1)
if not os.path.exists(dataset_file):
    print(f"❌ Errore: Il file dataset '{dataset_file}' non trovato nella stessa directory.")
    print(f"Creane uno che contenga solo la riga: {calibration_image_name}")
    exit(1)
else:
    # Piccolo controllo per assicurarsi che dataset.txt punti all'immagine giusta
    with open(dataset_file, 'r') as f:
        content = f.read().strip()
    if content != calibration_image_name:
        print(f"⚠️ Attenzione: Il file '{dataset_file}' non contiene '{calibration_image_name}'.")
        print(f"Dovrebbe contenere esattamente la riga '{calibration_image_name}'.")


print(f"✔️ Utilizzo di '{calibration_image_name}' come immagine di calibrazione e '{dataset_file}' per il dataset.")


# ===========================
# STEP 1: Crea/Carica modello e salva come TFLite
# ===========================

tflite_model_path = f'./{output_model_name}.tflite'

if not os.path.exists(tflite_model_path):
    print(f"🔄 Creazione e caricamento pesi per '{output_model_name}'...")
    try:
        # Chiama la funzione per creare il modello Inception-v4 e caricare i pesi
        model = create_inception_v4_model(load_weights=True)
        print(f"✅ Modello '{output_model_name}' creato e pesi caricati.")
    except Exception as e:
        print(f"❌ Errore durante la creazione o caricamento pesi di {output_model_name}: {e}")
        print("Potrebbe essere un problema di rete durante lo scaricamento del file .h5")
        exit(1)

    print(f"🔄 Conversione modello in formato TFLite in '{tflite_model_path}'...")
    try:
        # Converti direttamente dal modello Keras
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        # Assicurati che i tipi di input/output siano float32
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32

        tflite_model = converter.convert()

        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)
        print(f"✅ Modello TFLite salvato in '{tflite_model_path}'.")
    except Exception as e:
        print(f"❌ Errore durante la conversione o salvataggio TFLite: {e}")
        exit(1)
else:
    print(f"✔️ Modello TFLite '{tflite_model_path}' già presente.")


# ===========================
# STEP 2: Conversione in RKNN
# ===========================

rknn = RKNN()

rknn.config(mean_values=[[127.5, 127.5, 127.5]], std_values=[[127.5, 127.5, 127.5]], target_platform='rk3588')

# Carica TFLite
print(f"🔄 Caricamento TFLite da '{tflite_model_path}'...")
ret = rknn.load_tflite(tflite_model_path)
if ret != 0:
    print("❌ Errore caricamento TFLite")
    rknn.release() # Rilascia le risorse in caso di errore
    exit(1)

# Costruisci RKNN con il dataset per la quantizzazione
print("⚙️ Build RKNN in corso (con quantizzazione)...")
ret = rknn.build(do_quantization=True, dataset=dataset_file)
if ret != 0:
    print("❌ Errore build RKNN. Controlla i log sopra per i dettagli.")
    rknn.release() # Rilascia le risorse in caso di errore
    exit(1)

# Salva RKNN
rknn_output_path = f'./{output_model_name}.rknn'
rknn.export_rknn(rknn_output_path)
print(f"✅ Modello RKNN salvato in '{rknn_output_path}'.")

# Rilascia le risorse RKNN
rknn.release()
print("✅ Risorse RKNN rilasciate.")