import matplotlib.pyplot as plt
import re

def plot_cyclictest_summary_latencies_with_fontsize(filename="rt_nomodels.txt"):
    """
    Parses a cyclictest output file and plots:
    1. Actual Latencies of all threads over time.
    2. Cumulative Max Latencies of all threads over time.
    with specified font sizes.
    """
    thread_data = {}
    time_step = 0

    with open(filename, 'r') as f:
        lines = f.readlines()

    latency_pattern = re.compile(
        r"T:\s*(\d+)\s+\(\s*\d+\)\s+P:\d+\s+I:\d+\s+C:\s*\d+\s+"
        r"Min:\s*(\d+)\s+Act:\s*(\d+)\s+Avg:\s*(\d+)\s+Max:\s*(\d+)"
    )
    
    for line in lines:
        if "policy: fifo:" in line or "policy: other:" in line:
            time_step += 1
            continue

        match = latency_pattern.search(line)
        if match:
            thread_id = int(match.group(1))
            act_latency = int(match.group(3))
            max_cumulative_latency = int(match.group(5))

            if thread_id not in thread_data:
                thread_data[thread_id] = {
                    'time': [], 'act': [], 'max_cumulative': []
                }
            
            # Ci assicuriamo che time_step è positivo per evitare i data garbage iniziali se ci sono
            if time_step > 0:
                thread_data[thread_id]['time'].append(time_step)
                thread_data[thread_id]['act'].append(act_latency)
                thread_data[thread_id]['max_cumulative'].append(max_cumulative_latency)

    if not thread_data:
        print("Nessun dato di latenza trovato nel file. Assicurati che il formato sia corretto.")
        return

    # --- Plottiamo le latenze attuali ---
    fig_act, ax_act = plt.subplots(figsize=(14, 7))
    for thread_id in sorted(thread_data.keys()):
        data = thread_data[thread_id]
        if data['time']:
            ax_act.plot(data['time'], data['act'], label=f'Thread {thread_id}', marker='.', markersize=2, linestyle='-', alpha=0.7)

    ax_act.set_xlabel('Passo Temporale (Blocchi di aggiornamento)', fontsize=18)
    ax_act.set_ylabel('Latenza Attuale (us)', fontsize=18)
    ax_act.set_title('Andamento Latenze Attuali di Tutti i Thread', fontsize=20)
    ax_act.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=16)
    ax_act.grid(True)
    ax_act.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout(rect=[0, 0.03, 0.85, 0.96])
    
    # --- Plottiamo le latenze massime ---
    fig_max, ax_max = plt.subplots(figsize=(14, 7))
    for thread_id in sorted(thread_data.keys()):
        data = thread_data[thread_id]
        if data['time']:
            ax_max.plot(data['time'], data['max_cumulative'], label=f'Thread {thread_id}', marker='.', markersize=2, linestyle='-', alpha=0.8)

    ax_max.set_xlabel('Passo Temporale (Blocchi di aggiornamento)', fontsize=18)
    ax_max.set_ylabel('Latenza Massima Cumulativa (us)', fontsize=18)
    ax_max.set_title('Andamento Latenze Massime Cumulative di Tutti i Thread', fontsize=20)
    ax_max.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=16)
    ax_max.grid(True)
    ax_max.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout(rect=[0, 0.03, 0.85, 0.96])

    plt.show()

plot_cyclictest_summary_latencies_with_fontsize("rt_nomodels.txt") # Assicuriamoci che il nome del file sia corretto