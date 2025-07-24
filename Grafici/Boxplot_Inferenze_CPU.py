import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Dati reali raccolti (15 misurazioni per modello)
dati = {
    'MobileNet_v2': [0.6296, 0.0908, 0.0626, 0.0695, 0.0628, 0.0580, 0.0600, 0.0917, 0.0926, 0.0781, 0.1051, 0.0761, 0.0764, 0.0601, 0.0569],
    'MobileNet_v1': [0.6562, 0.0732, 0.0487, 0.0788, 0.0602, 0.0677, 0.0783, 0.0732, 0.0610, 0.0519, 0.0593, 0.0657, 0.0912, 0.0805, 0.0895],
    'SSD MobileNet_v1': [6.9905, 1.2085, 1.3504, 1.1924, 1.1763, 1.1970, 1.2161, 1.1732, 1.2691, 1.2091, 1.2873, 1.2473, 1.2165, 1.2066, 1.2687],
    'SSD MobileNet_v2': [6.1576, 0.3790, 0.3826, 0.3677, 0.4011, 0.4001, 0.3773, 0.3766, 0.4082, 0.3632, 0.4048, 0.3684, 0.4144, 0.4273, 0.3984],
    'Inception_v1': [0.6678, 0.1089, 0.0932, 0.0697, 0.0741, 0.0853, 0.0905, 0.1289, 0.1188, 0.1345, 0.1050, 0.0967, 0.1032, 0.0831, 0.0811],
    'Inception_v4': [4.7589, 0.5749, 0.5599, 0.5458, 0.5734, 0.5296, 0.5278, 0.7161, 0.5449, 0.5449, 0.5279, 0.5334, 0.5573, 0.5293, 0.5264],
}

# Rimuoviamo il primo valore (warm-up)
dati_warm = {modello: tempi[1:] for modello, tempi in dati.items()}

# Costruiamo DataFrame
df = pd.DataFrame([
    (modello, tempo) for modello, tempi in dati_warm.items() for tempo in tempi
], columns=['Modello', 'Tempo (s)'])

# Statistiche
statistiche = []
for modello, tempi in dati_warm.items():
    media = sum(tempi) / len(tempi)
    mediana = sorted(tempi)[len(tempi) // 2]
    minimo = min(tempi)
    massimo = max(tempi)
    std = pd.Series(tempi).std()
    statistiche.append({
        'Modello': modello,
        'Media (s)': round(media, 4),
        'Mediana (s)': round(mediana, 4),
        'Min (s)': round(minimo, 4),
        'Max (s)': round(massimo, 4),
        'Std Dev': round(std, 4)
    })

df_stats = pd.DataFrame(statistiche)
print("📊 Statistiche senza cold start:\n")
print(df_stats.to_string(index=False))

# Boxplot con font ingranditi
plt.figure(figsize=(12, 7))
sns.boxplot(x='Modello', y='Tempo (s)', data=df, palette='pastel', showfliers=True)
plt.title('Tempi di Inferenza su CPU', fontsize=20)
plt.xlabel('')
plt.ylabel('Tempo di Inferenza (secondi)', fontsize=18)
plt.xticks(rotation=45, fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.show()
