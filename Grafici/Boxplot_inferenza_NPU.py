import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Dati originali in millisecondi
dati_ms = {
    'MobileNet_v2': [4.00, 3.82, 3.68, 3.22, 3.15, 3.10, 3.54, 3.67, 3.97, 4.37, 3.90, 3.83, 3.64, 3.61, 3.70],
    'MobileNet_v1': [3.35, 3.44, 3.32, 2.92, 2.80, 2.57, 2.59, 2.57, 2.57, 2.57, 2.58, 2.59, 2.60, 2.69, 2.59],
    'SSD MobileNet_v2': [2.14, 2.65, 2.38, 2.50, 2.67, 2.53, 2.64, 2.82, 2.55, 2.84, 3.09, 2.71, 2.55, 2.54, 3.15],
    'Inception_v1': [5.08, 5.07, 5.11, 5.19, 5.19, 5.18, 5.17, 5.14, 5.23, 5.65, 5.38, 5.28, 5.32, 5.66, 9.09],
    'Inception_v4': [67.86, 69.25, 76.29, 76.79, 97.31, 94.95, 97.08, 91.44, 92.64, 93.24, 91.12, 90.68, 92.38, 83.14, 75.46],
}

# Conversione ms → s e rimozione primo tempo (cold start)
dati_s = {modello: [round(x / 1000, 6) for x in tempi[1:]] for modello, tempi in dati_ms.items()}

# Preparazione DataFrame lungo
df_npu = pd.DataFrame([(modello, tempo) for modello, tempi in dati_s.items() for tempo in tempi], columns=['Modello', 'Tempo (s)'])

# Calcolo statistiche
statistiche = []
for modello, tempi in dati_s.items():
    media = sum(tempi) / len(tempi)
    mediana = sorted(tempi)[len(tempi)//2]
    minimo = min(tempi)
    massimo = max(tempi)
    std = pd.Series(tempi).std()
    statistiche.append({
        'Modello': modello,
        'Media (s)': round(media, 6),
        'Mediana (s)': round(mediana, 6),
        'Min (s)': round(minimo, 6),
        'Max (s)': round(massimo, 6),
        'Std Dev': round(std, 6)
    })

df_stats = pd.DataFrame(statistiche)

# Mostra le statistiche
print("📊 Statistiche NPU (senza cold start):\n")
print(df_stats.to_string(index=False))

# Box plot con font ingranditi
plt.figure(figsize=(12, 7))
sns.boxplot(x='Modello', y='Tempo (s)', data=df_npu, palette='pastel', showfliers=True)
plt.title('Tempi di Inferenza su NPU', fontsize=20)
plt.xlabel('')
plt.ylabel('Tempo di Inferenza (secondi)', fontsize=18)
plt.xticks(rotation=45, fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.show()
