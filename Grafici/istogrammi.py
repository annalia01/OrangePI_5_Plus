import matplotlib.pyplot as plt
import numpy as np


fasi = ['preprocessing', 'inferenza', 'postprocessing']


tempi_modello_A =  [0.28, 0.17, 0.002]
tempi_modello_B = [0.18, 0.05, 0.0026]



x = np.arange(len(fasi))
width = 0.25  


fig, ax = plt.subplots()
bar1 = ax.bar(x - width, tempi_modello_A, width, label='MobilenetV2')
bar2 = ax.bar(x, tempi_modello_B, width, label='SSDMobilenetV2')



ax.set_ylabel('Tempo (s)')
ax.set_title('Tempi di esecuzione per fase e modello')
ax.set_xticks(x)
ax.set_xticklabels(fasi)
ax.legend()


plt.tight_layout()
plt.show()
