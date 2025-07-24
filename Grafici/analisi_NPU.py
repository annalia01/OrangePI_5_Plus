import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


file_path = 'rknpu_trace.csv'


data = []
with open(file_path, 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('[') or line == ';;;':
            continue
            
        
        parts = line.split(';')
        if len(parts) < 4:
            continue
            
        timestamp_str = parts[0]
        core0 = parts[1].replace('Core0:', '').replace('%', '').strip()
        core1 = parts[2].replace('Core1:', '').replace('%', '').strip()
        core2 = parts[3].replace('Core2:', '').replace('%', '').strip()
        
        try:
            timestamp = datetime.strptime(timestamp_str, '%H:%M:%S')
            data.append({
                'timestamp': timestamp,
                'Core0': float(core0),
                'Core1': float(core1),
                'Core2': float(core2)
            })
        except ValueError:
            continue


df = pd.DataFrame(data)

plt.figure(figsize=(12, 6))


line_width = 1.0  

plt.plot(df['timestamp'], df['Core0'], label='Core0', color='#FFD700', linewidth=line_width)  
plt.plot(df['timestamp'], df['Core1'], label='Core1', color='#ff7f0e', linewidth=line_width)  
plt.plot(df['timestamp'], df['Core2'], label='Core2', color='red', linewidth=line_width)  


plt.title('Utilizzo Core NPU nel Tempo', pad=20, fontsize=14)
plt.xlabel('Tempo', fontsize=12)
plt.ylabel('Utilizzo (%)', fontsize=12)
plt.ylim(0, 4.0)
plt.legend(loc='upper right', framealpha=1)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=0)
plt.tight_layout()


plt.savefig('npu_core_usage_thin_lines.png', dpi=300, bbox_inches='tight')
plt.show()