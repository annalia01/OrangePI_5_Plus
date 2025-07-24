import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Creazione DataFrame
records = []
for scenario, models in data.items():
    for model, times in models.items():
        for t in times:
            records.append({'Scenario': scenario, 'Modello': model, 'Tempo (s)': t / 1000})

df = pd.DataFrame(records)

# Ordine dei gruppi sull'asse X
scenario_order = ['cpu', 'vm', 'memcpy', 'cache', 'fork']

# Funzione per tracciare linee verticali tra i gruppi
def add_vertical_lines(ax, categories):
    positions = ax.get_xticks()
    for i in range(len(categories) - 1):
        x = (positions[i] + positions[i + 1]) / 2
        ax.axvline(x=x, color='grey', linestyle='--', linewidth=1)

# Parametri font
title_font = 20
label_font = 18
tick_font = 16
legend_font = 16

# Primo grafico
df1 = df[df['Modello'].isin(['inception_v1', 'inception_v4', 'mobilenet_v1'])]
plt.figure(figsize=(14,6))
ax1 = sns.boxplot(
    x='Scenario', y='Tempo (s)', hue='Modello', data=df1, fliersize=3, order=scenario_order
)
plt.yscale('log')
add_vertical_lines(ax1, scenario_order)
plt.title('(a) Inception v1, Inception v4, MobileNet v1 (CPU Stress Scenarios)', fontsize=title_font)
plt.xlabel('', fontsize=label_font)
plt.ylabel('Tempo (s)', fontsize=label_font)
plt.xticks(fontsize=tick_font)
plt.yticks(fontsize=tick_font)
plt.legend(title='Modello', fontsize=legend_font, title_fontsize=legend_font)
plt.tight_layout()
plt.show()

# Secondo grafico
df2 = df[df['Modello'].isin(['mobilenet_v2', 'SSD MobileNet_v2'])]
plt.figure(figsize=(14,6))
ax2 = sns.boxplot(
    x='Scenario', y='Tempo (s)', hue='Modello', data=df2, fliersize=3, order=scenario_order
)
plt.yscale('log')
add_vertical_lines(ax2, scenario_order)
plt.title('(b) MobileNet v2, SSD MobileNet v2 (CPU Stress Scenarios)', fontsize=title_font)
plt.xlabel('', fontsize=label_font)
plt.ylabel('Tempo (s)', fontsize=label_font)
plt.xticks(fontsize=tick_font)
plt.yticks(fontsize=tick_font)
plt.legend(title='Modello', fontsize=legend_font, title_fontsize=legend_font)
plt.tight_layout()
plt.show()
