import matplotlib.pyplot as plt
import numpy as np
from iminuit.cost import LeastSquares
import iminuit
from scipy.stats import chi2
from scipy.optimize import curve_fit
def Model (x,k,i_0):
    """
    Modello esponenziale per il diodo, in scala logaritmica:
    """
    return (i_0 * (np.exp(k * x) - 1))

# Dati: [(tensione sperimentale (V), corrente sperimentale (mA))], contatto 1:
import matplotlib.pyplot as plt
import numpy as np

# Dati: (voltaggio, corrente, errore_corrente)
data_raw = [
    (0.349, 0.0010, 0.0001),
    (0.465, 0.0264, 0.0001),
    (0.560, 0.3114, 0.0001),
    (0.569, 0.3399, 0.0001),
    (0.586, 0.5170, 0.0003),
    (0.605, 0.8130, 0.0003),
    (0.619, 1.0978, 0.0004),
    (0.651, 2.1394, 0.0001),
    (0.669, 2.97, 0.02),
    (0.732, 7.533, 0.003),
    (0.774, 11.60, 0.02),
    (0.837, 19.44, 0.01),
    (0.994, 41.88, 0.01),
    (0.931, 31.7, 0.1),
    (1.097, 56.9, 0.1),
    (1.154, 65.9, 0.1),
    (1.204, 73.88, 0.05),
    (1.251, 81.65, 0.01),
    (1.290, 88.1, 0.1),
    (1.317, 92.5, 0.1),
    (1.355, 98.8, 0.01),
    (1.391, 104.63, 0.01),
]

# Estrazione dei dati
voltage = np.array([v for v, _, _ in data_raw])
current = np.array([i for _, i, _ in data_raw])
# Se l'errore non è specificato, si assume 1 sull'ultima cifra decimale
err_current = np.array([
    e for _, _, e in data_raw
])
err_voltage = (0.5/100)*voltage + 0.003  # errore costante sul voltaggio
print(err_current)
print(len(err_current))
print(len(current))

# ...existing code...

tol = 1e-4
max_iter = 50
params_old = np.array([0, 0])
params_new = np.array([0.1, 0.1])  # valori iniziali: k, i_0

for iteration in range(max_iter):
    # Calcolo errori sulle y considerando propagazione errori su x
    err_current_iter = []
    for i in range(len(current)):
        err_y = np.sqrt(
            (err_current[i] ** 2) +
            (params_new[1]*params_new[0]*np.exp(params_new[0]*voltage[i]) * err_voltage[i]) ** 2
        )
        err_current_iter.append(err_y)
    err_current_iter = np.array(err_current_iter)

    # Definizione funzione chi2 per questa iterazione
    def chi2_func_iter(k, i_0):
        residuals = current - Model(voltage, k, i_0)
        return np.sum((residuals / err_current_iter) ** 2)

    m_iter = iminuit.Minuit(chi2_func_iter, k=params_new[0], i_0=params_new[1])
    m_iter.migrad()
    params_old = params_new
    params_new = np.array([m_iter.values['k'], m_iter.values['i_0']])

    # Controllo stabilizzazione
    if np.all(np.abs((params_new - params_old) / (params_old + 1e-12)) < tol):
        print(f"Fit contatto 1 stabilizzato dopo {iteration+1} iterazioni.")
        break
else:
    print("Attenzione: fit contatto 1 non stabilizzato dopo il numero massimo di iterazioni.")

# Estrazione parametri finali
k_fit_1, i_0_fit_1 = params_new
err_k_fit_1 = m_iter.errors['k']
err_i_0_fit_1 = m_iter.errors['i_0']

# Calcolo chi2 finale
residuals_1 = current - Model(voltage, k_fit_1, i_0_fit_1)
chisquared_1 = np.sum((residuals_1 / err_current_iter) ** 2)
ndof_1 = len(current) - 2
chi2_reduced_1 = chisquared_1 / ndof_1
p_value_1 = 1 - chi2.cdf(chisquared_1, ndof_1)

print("Risultati del fit iterativo con Minuit (contatto 1):")
print(f"k = {k_fit_1:.4f} ± {err_k_fit_1:.4f}")
print(f"i_0 = {i_0_fit_1:.4f} ± {err_i_0_fit_1:.4f}")
print(f"Chi² = {chisquared_1:.2f}")
print(f"Chi² ridotto = {chi2_reduced_1:.2f}")
print(f"p-value = {p_value_1:.4f}")
print(m_iter.valid)

# Grafico finale
plt.figure(figsize=(10, 6))
plt.errorbar(voltage, current, yerr=err_current_iter, fmt='o', label='Dati sperimentali', color='blue', ecolor='gray', capsize=3)
voltage_fit_1 = np.linspace(min(voltage), max(voltage), 1000)
plt.plot(voltage_fit_1, Model(voltage_fit_1, k_fit_1, i_0_fit_1), color='red', label = 'legge di Schockley')

plt.xlabel('Tensione (V)')
plt.ylabel('Corrente (mA)')
plt.title('Fit esponenziale della corrente in funzione della tensione (contatto 1)')
plt.legend()
plt.grid(True)
plt.tight_layout()
textstr = (
    f"Chi² ridotto = {chi2_reduced_1:.2f}\n"
    f"p-value = {p_value_1:.4f}\n"
    f"q/gkT = {k_fit_1:.4f} ± {err_k_fit_1:.4f}\n"
    f"i_0 = {i_0_fit_1:.4f} ± {err_i_0_fit_1:.4f}\n"
)
plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
plt.show()



#interpolazione per la resistenza interna dell'amperometro
def Model2(x, R, c):
    """
    Modello lineare per la resistenza interna dell'amperometro:
    """
    return ((x /R)+c)
dati_2 = [
    (0.837, 19.44, 0.01),
    (0.994, 41.88, 0.01),
    (0.931, 31.7, 0.1),
    (1.097, 56.9, 0.1),
    (1.154, 65.9, 0.1),
    (1.204, 73.88, 0.05),
    (1.251, 81.65, 0.01),
    (1.290, 88.1, 0.1),
    (1.317, 92.5, 0.1),
    (1.355, 98.8, 0.01),
    (1.391, 104.63, 0.01),
]
voltage_2 = np.array([v for v, _, _ in dati_2])*np.ones(len(dati_2))  # conversione in V
current_2 = np.array([i for _, i, _ in dati_2])*pow(10,-3)  # conversione in A
err_current_2 = np.array([e for _, _, e in dati_2])*pow(10,-3)  # conversione in A
err_voltage_2 = (0.5/100)*voltage_2+0.003  # errore costante sul voltaggio
# ...existing code...

tol = 1e-4
max_iter = 50
params_old = np.array([0, 0])
params_new = np.array([100, 0])  # valori iniziali: R, c

for iteration in range(max_iter):
    # Calcolo errori sulle y considerando propagazione errori su x
    err_current_iter = []
    for i in range(len(current_2)):
        err_y = np.sqrt(
            (err_current_2[i] ** 2) +
            ((1/params_new[0]) * err_voltage_2[i]) ** 2
        )
        err_current_iter.append(err_y)
    err_current_iter = np.array(err_current_iter)

    # Definizione funzione chi2 per questa iterazione
    def chi2_func_iter(R, c):
        residuals = current_2 - Model2(voltage_2, R, c)
        return np.sum((residuals / err_current_iter) ** 2)

    m_iter = iminuit.Minuit(chi2_func_iter, R=params_new[0], c=params_new[1])
    m_iter.migrad()
    params_old = params_new
    params_new = np.array([m_iter.values['R'], m_iter.values['c']])

    # Controllo stabilizzazione
    if np.all(np.abs((params_new - params_old) / (params_old + 1e-12)) < tol):
        print(f"Fit resistenza interna stabilizzato dopo {iteration+1} iterazioni.")
        break
else:
    print("Attenzione: fit resistenza interna non stabilizzato dopo il numero massimo di iterazioni.")

# Estrazione parametri finali
R_fit_final, c_fit_final = params_new
err_R_fit_final = m_iter.errors['R']
err_c_fit_final = m_iter.errors['c']

# Calcolo chi2 finale
residuals_R = current_2 - Model2(voltage_2, R_fit_final, c_fit_final)
chisquared_R = np.sum((residuals_R / err_current_iter) ** 2)
ndof_R = len(current_2) - 2
chi2_reduced_R = chisquared_R / ndof_R
p_value_R = 1 - chi2.cdf(chisquared_R, ndof_R)

print("Risultati del fit iterativo con Minuit (resistenza interna):")
print(f"R = {R_fit_final:.4f} ± {err_R_fit_final:.4f}")
print(f"c = {c_fit_final:.4f} ± {err_c_fit_final:.4f}")
print(f"Chi² = {chisquared_R:.2f}")
print(f"Chi² ridotto = {chi2_reduced_R:.2f}")
print(f"p-value = {p_value_R:.4f}")
print(m_iter.valid)

# Grafico finale
plt.figure(figsize=(10, 6))
plt.errorbar(voltage_2, current_2, yerr=err_current_iter, fmt='o', label='Dati sperimentali', color='blue', ecolor='gray', capsize=3)
voltage_fit_2 = np.linspace(min(voltage_2), max(voltage_2), 1000)
plt.plot(voltage_fit_2, Model2(voltage_fit_2, R_fit_final, c_fit_final),
        color='red', label = 'V/R +c')
plt.xlabel('Tensione (V)')
plt.ylabel('Corrente (A)')
plt.title('Fit lineare della corrente in funzione della tensione per il calcolo della resistenza interna dell\'amperometro')
plt.legend()
plt.grid(True)
plt.tight_layout()

textstr = (
    f"Chi² ridotto = {chi2_reduced_R:.2f}\n"
    f"p-value = {p_value_R:.4f}\n"
    f"R_interna = {R_fit_final:.4f} ± {err_R_fit_final:.4f}\n"
    f"c = {c_fit_final:.4f} ± {err_c_fit_final:.4f}\n"
)
plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
plt.show()



# Dati: [(tensione sperimentale (V), corrente sperimentale (mA))], secondo contatto:
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
voltage_contatto_2 = np.array([v for v, _ in data])
current_contatto_2 = np.array([i for _, i in data])  # conversione in A
err_current_contatto_2_raw = [0.0001,
0.0001,
0.0001,
0.0001,
0.0001,
0.0001,
0.0001,
0.0001,
0.002,
0.02,
0.005,
0.001,
0.01,
0.01,
0.01,
0.01,
0.01,
0.01,
0.01,
0.01,
0.1,
0.1] 
err_current_contatto_2 = np.array(err_current_contatto_2_raw)
err_voltage_contatto_2 = (0.5/100)*voltage_contatto_2+0.003 # errore costante sul voltaggio


def Model_real(x,k,i_0):
    return (i_0 * (np.exp(k * x) - 1))



# Iterative fit con Minuit fino a stabilizzazione dei parametri
tol = 1e-4  # tolleranza relativa
max_iter = 50
params_old = np.array([0, 0, 0])
params_new = np.array([0.1, 0.1])  # valori iniziali: k, i_0

for iteration in range(max_iter):
    # Calcolo errori sulle y considerando propagazione errori su x
    second_error_current = []
    for i in range(len(current_contatto_2)):
        err_y = np.sqrt(
            (err_current_contatto_2[i] ** 2) +
            (( params_new[1]*params_new[0]*np.exp(params_new[0]*voltage_contatto_2[i])) * err_voltage_contatto_2[i]) ** 2
        )
        second_error_current.append(err_y)
    second_error_current = np.array(second_error_current)

    # Definizione funzione chi2 per questa iterazione
    def chi2_func_iter(k, i_0):
        residuals = current_contatto_2 - Model_real(voltage_contatto_2, k, i_0)
        return np.sum((residuals / second_error_current) ** 2)

    m_iter = iminuit.Minuit(chi2_func_iter, k=params_new[0], i_0=params_new[1])
    m_iter.migrad()
    params_old = params_new
    params_new = np.array([m_iter.values['k'], m_iter.values['i_0']])

    # Controllo stabilizzazione
    if np.all(np.abs((params_new - params_old) / (params_old + 1e-12)) < tol):
        print(f"Fit stabilizzato dopo {iteration+1} iterazioni.")
        break
else:
    print("Attenzione: fit non stabilizzato dopo il numero massimo di iterazioni.")

# Estrazione parametri finali
k_fit_final, i_0_fit_final = params_new
err_k_fit_final = m_iter.errors['k']
err_i_0_fit_final = m_iter.errors['i_0']


# Calcolo chi2 finale
residuals_final = current_contatto_2 - Model_real(voltage_contatto_2, k_fit_final, i_0_fit_final)
chisquared_final = np.sum((residuals_final / second_error_current) ** 2)
ndof_final = len(current_contatto_2) - 3
chi2_reduced_final = chisquared_final / ndof_final
p_value_final = 1 - chi2.cdf(chisquared_final, ndof_final)

print("Risultati del fit iterativo con Minuit (parametri stabilizzati):")
print(f"k = {k_fit_final:.4f} ± {err_k_fit_final:.4f}")
print(f"i_0 = {i_0_fit_final:.15f} ± {err_i_0_fit_final:.15f}")
print(f"Chi² = {chisquared_final:.2f}")
print(f"Chi² ridotto = {chi2_reduced_final:.2f}")
print(f"p-value = {p_value_final:.4f}")
print(m_iter.valid)

# Grafico finale
plt.figure(figsize=(10, 6))
plt.errorbar(voltage_contatto_2, current_contatto_2, yerr=second_error_current, fmt='o', label='Dati sperimentali', color='blue', ecolor='gray', capsize=3)
voltage_fit_contatto_2 = np.linspace(min(voltage_contatto_2), max(voltage_contatto_2), 1000)
plt.plot(voltage_fit_contatto_2, Model_real(voltage_fit_contatto_2, k_fit_final, i_0_fit_final),
        label = "legge di Schokley", color = 'red')
plt.xlabel('Tensione (V)')
plt.ylabel('Corrente (mA)')
plt.title('Fit esponenziale della corrente in funzione della tensione (secondo contatto)')
plt.legend()
plt.grid(True)
plt.tight_layout()
textstr = (
    f"Chi² ridotto = {chi2_reduced_final:.2f}\n"
    f"p-value = {p_value_final:.4f}\n"
    f"q/gkT = {k_fit_final:.4f} ± {err_k_fit_final:.4f}\n"
    f"i_0 = {i_0_fit_final:.4f} ± {err_i_0_fit_final:.4f}\n"
)

plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
plt.show()

g = 38.6 / k_fit_final
err_g = (38.6/(k_fit_final**2))*err_k_fit_final
print(g,err_g)
print(np.exp(-1.154*k_fit_final)/(k_fit_final*i_0_fit_final))
