from rknn.api import RKNN
import os

# --- Configurazione dei percorsi e parametri del modello ---
# PERCORSO: Sostituisci questo con il percorso esatto del file .tflite che scarichi
TFLITE_MODEL_PATH = '/path/to/your/downloaded_model.tflite' 

# Il percorso dove salviamo il modello .rknn convertito
RKNN_MODEL_PATH = './ssd_mobilenet_v2_int8.rknn'

# Dimensioni di input del modello: [Larghezza, Altezza]
INPUT_SHAPE = [320, 320] 

# Valori di media (mean_values) e deviazione standard (std_values) per la normalizzazione [-1, 1]
MEAN_VALUES = [[127.5, 127.5, 127.5]] 
STD_VALUES = [[127.5, 127.5, 127.5]]  

# Piattaforma target del chip Rockchip 
TARGET_PLATFORM = 'rk3588'

# Tipo di quantizzazione. Il modello è già INT8.
QUANTIZED_DTYPE = 'w8a8'

# --- Inizializzazione e Conversione ---
rknn = RKNN(verbose=True)

# Configura i parametri di conversione
print(f"--> Configuring RKNN model for {TARGET_PLATFORM}...")
ret = rknn.config(
    mean_values=MEAN_VALUES,
    std_values=STD_VALUES,
    target_platform=TARGET_PLATFORM,
    quantized_dtype=QUANTIZED_DTYPE,
    optimization_level=3
)
if ret != 0:
    print('RKNN config failed!')
    exit(ret)
print('Done.')

# Carica il modello TFLite
print(f'--> Loading TFLite model from {TFLITE_MODEL_PATH}...')
ret = rknn.load_tflite(model=TFLITE_MODEL_PATH, inputs_size_list=[INPUT_SHAPE])
if ret != 0:
    print('RKNN load TFLite model failed!')
    exit(ret)
print('Done.')

# Costruisci il modello RKNN
print('--> Building RKNN model...')
ret = rknn.build(do_quantization=False)
if ret != 0:
    print('RKNN build failed!')
    exit(ret)
print('Done.')

# Esporta il modello RKNN
print(f'--> Exporting RKNN model to {RKNN_MODEL_PATH}...')
ret = rknn.export_rknn(RKNN_MODEL_PATH)
if ret != 0:
    print('RKNN export failed!')
    exit(ret)
print('Done.')

# Libera le risorse RKNN
rknn.release()
print('RKNN resources released.')

print(f"Modello {RKNN_MODEL_PATH} creato con successo!")