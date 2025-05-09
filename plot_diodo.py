import matplotlib.pyplot as plt

# Dati: [(tensione sperimentale (V), corrente simulata (mA))]
data = [
    (0.349, 0.0010),
    (0.465, 0.0264),
    (0.559, 0.3114),
    (0.566, 0.3399),
    (0.582, 0.5170),
    (0.600, 0.8130),
    (0.612, 1.0978),
    (0.639, 2.1394),
    (0.652, 2.97),
    (0.689, 7.533),
    (0.706, 11.60),
    (0.725, 19.44),
    (0.745, 31.7),
    (0.752, 41.88),
    (0.765, 56.9),
    (0.769, 65.9),
    (0.772, 73.88),
    (0.776, 81.65),
    (0.778, 88.1),
    (0.779, 92.5),
    (0.781, 98.8),
    (0.782, 104.63),
]

# Separazione in due liste
x_tensione_exp, y_corrente_sim = zip(*data)

# --- Plot scala normale ---
plt.figure(figsize=(10, 6))
plt.plot(x_tensione_exp, y_corrente_sim, 'o-', color='tab:green', label='Simulazione vs Tensione sperimentale')
plt.xlabel('Tensione sperimentale (V)')
plt.ylabel('Corrente simulata (mA)')
plt.title('Corrente simulata vs Tensione sperimentale (scala normale)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot scala semi-log (log in y) ---
plt.figure(figsize=(10, 6))
plt.plot(x_tensione_exp, y_corrente_sim, 'o-', color='tab:green', label='Simulazione vs Tensione sperimentale')
plt.yscale('log')
plt.xlabel('Tensione sperimentale (V)')
plt.ylabel('Corrente simulata (mA) [scala log]')
plt.title('Corrente simulata vs Tensione sperimentale (semi-log)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt

# Dati: [(tensione (V), corrente (mA))]
data = [
    (0.349, 0.0010),
    (0.465, 0.0264),
    (0.560, 0.3114),
    (0.569, 0.3399),
    (0.586, 0.5170),
    (0.605, 0.8130),
    (0.619, 1.0978),
    (0.651, 2.1394),
    (0.669, 2.97),
    (0.732, 7.533),
    (0.774, 11.60),
    (0.837, 19.44),
    (0.994, 41.88),
    (0.931, 31.7),
    (1.097, 56.9),
    (1.154, 65.9),
    (1.204, 73.88),
    (1.251, 81.65),
    (1.290, 88.1),
    (1.317, 92.5),
    (1.355, 98.8),
    (1.391, 104.63),
]

# Separazione in due liste
tensione_V, corrente_mA = zip(*data)

# Creazione del grafico semi-logaritmico (log in y)
plt.figure(figsize=(10, 6))
plt.plot(tensione_V, corrente_mA, 'o-', label='Corrente vs Tensione (semi-log)')
plt.yscale('log')
plt.xlabel('Tensione (V)')
plt.ylabel('Corrente (mA) [scala log]')
plt.title('Caratteristica I-V in scala semi-logaritmica')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# Dati: [(tensione (V), corrente (mA))]
data = [
    (0.349, 0.0010),
    (0.465, 0.0264),
    (0.560, 0.3114),
    (0.569, 0.3399),
    (0.586, 0.5170),
    (0.605, 0.8130),
    (0.619, 1.0978),
    (0.651, 2.1394),
    (0.669, 2.97),
    (0.732, 7.533),
    (0.774, 11.60),
    (0.837, 19.44),
    (0.994, 41.88),
    (0.931, 31.7),
    (1.097, 56.9),
    (1.154, 65.9),
    (1.204, 73.88),
    (1.251, 81.65),
    (1.290, 88.1),
    (1.317, 92.5),
    (1.355, 98.8),
    (1.391, 104.63),
]

# Separazione in due liste
x_tensione_simulata, y_corrente_simulata = zip(*data)

# --- Plot scala normale ---
plt.figure(figsize=(10, 6))
plt.plot(x_tensione_simulata, y_corrente_simulata, 'o-', label='Simulazione')
plt.xlabel('Tensione simulata (V)')
plt.ylabel('Corrente simulata (mA)')
plt.title('Corrente simulata vs Tensione simulata (scala normale)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot scala semi-log (logaritmica su y) ---
plt.figure(figsize=(10, 6))
plt.plot(x_tensione_simulata, y_corrente_simulata, 'o-', label='Simulazione')
plt.yscale('log')
plt.xlabel('Tensione simulata (V)')
plt.ylabel('Corrente simulata (mA) [scala log]')
plt.title('Corrente simulata vs Tensione simulata (semi-log)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# Dati: [(tensione sperimentale (V), corrente sperimentale (mA))]
data = [
    (0.349, 0.0010),
    (0.465, 0.0267),
    (0.559, 0.3106),
    (0.566, 0.3406),
    (0.582, 0.5177),
    (0.600, 0.8128),
    (0.612, 1.0910),
    (0.639, 2.1335),
    (0.652, 2.948),
    (0.689, 7.537),
    (0.706, 11.585),
    (0.725, 19.463),
    (0.745, 32.01),
    (0.752, 41.88),
    (0.765, 57.03),
    (0.769, 65.80),
    (0.772, 74.25),
    (0.776, 81.63),
    (0.778, 88.10),
    (0.779, 92.34),
    (0.781, 98.6),
    (0.782, 104.6),
]

# Separazione in due liste
x_tensione_exp, y_corrente_exp = zip(*data)

# --- Plot scala normale ---
plt.figure(figsize=(10, 6))
plt.plot(x_tensione_exp, y_corrente_exp, 'o-', color='tab:orange', label='Dati sperimentali')
plt.xlabel('Tensione sperimentale (V)')
plt.ylabel('Corrente sperimentale (mA)')
plt.title('Corrente sperimentale vs Tensione sperimentale (scala normale)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot scala semi-logaritmica ---
plt.figure(figsize=(10, 6))
plt.plot(x_tensione_exp, y_corrente_exp, 'o-', color='tab:orange', label='Dati sperimentali')
plt.yscale('log')
plt.xlabel('Tensione sperimentale (V)')
plt.ylabel('Corrente sperimentale (mA) [scala log]')
plt.title('Corrente sperimentale vs Tensione sperimentale (semi-log)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()
