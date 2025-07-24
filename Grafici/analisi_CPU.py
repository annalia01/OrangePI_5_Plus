import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

def parse_mpstat_time_series(filepath):
    pattern = re.compile(
        r"^(\d{2}:\d{2}:\d{2})\s+\S+\s+(\S+)\s+([\d.]+)\s+[\d.]+\s+([\d.]+)\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([\d.]+)"
    )
    records = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.match(line)
            if match:
                timestamp, cpu, usr, sys, idle = match.groups()
                if cpu != "all":
                    records.append({
                        "time": timestamp,
                        "cpu": int(cpu),
                        "%usr": float(usr),
                        "%sys": float(sys),
                        "%idle": float(idle)
                    })

    df = pd.DataFrame(records)
    df["second"] = df.groupby("cpu").cumcount()
    return df


log_path = "inceptionv1_1.log"
df = parse_mpstat_time_series(log_path)


plt.rcParams.update({
    'font.size': 16,         
    'axes.titlesize': 20,    
    'axes.labelsize': 18,    
    'xtick.labelsize': 16,   
    'ytick.labelsize': 16,   
    'legend.fontsize': 18,   
    'legend.title_fontsize': 18
})

# === GRAFICO %USR ===
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="second", y="%usr", hue="cpu", palette="tab10")
plt.title("Utilizzo %usr per CPU nel tempo")
plt.xlabel("Tempo (secondi)")
plt.ylabel("%usr")
plt.legend(title="CPU Core")
plt.tight_layout()
plt.show()

# === GRAFICO %sys ===
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="second", y="%sys", hue="cpu", palette="tab10")
plt.title("Utilizzo %sys per CPU nel tempo")
plt.xlabel("Tempo (secondi)")
plt.ylabel("%sys")
plt.legend(title="CPU Core")
plt.tight_layout()
plt.show()


