import numpy as np
import matplotlib.pyplot as plt

# Dati forniti
Rload = np.array([10.01, 70.4, 130.2, 190.5, 251.1, 310.5, 370.8, 437.7, 498.0, 557.7, 
                  618, 679, 737, 798, 857, 918, 978, 1008])  # Rload in kOhm
Vout = np.array([0.156] * len(Rload))  # Vout in V

# Calcolo degli errori su Rload
errori_Rload = np.array([0.1 if r < 610 else 1 for r in Rload])  # Sensibilità: 0.1 kOhm sotto 610 kOhm, 1 kOhm sopra

# Creazione del grafico
plt.figure(figsize=(10, 6))
plt.errorbar(Rload, Vout, xerr=errori_Rload, yerr=0, fmt='o', ecolor='red', capsize=5, label='Dati misurati')

# Etichette e titolo
plt.xlabel('$V_{out}$ (V)')
plt.ylabel('$R_{load}$ (k$\Omega$)')
plt.title('Grafico di $R_{load}$ in funzione di $V_{out}$')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
