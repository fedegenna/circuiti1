import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from scipy.stats import chi2
from iminuit.cost import LeastSquares
# -------------------
# Dati, set 2
# -------------------
corrente = np.array([
    0.00, 0.07, 0.14, 0.19, 0.24, 0.30, 0.35, 0.40, 0.47, 0.52,
    0.58, 0.62, 0.73, 0.83, 0.93, 1.07, 1.24, 1.56, 2.00, 2.08
])

angoli_gradi = np.array([
    0, 18, 38, 46, 52, 58, 62, 65, 68, 70,
    72, 72, 75, 76, 78, 80, 81, 82, 83, 83
])

# -------------------
# Conversione in radianti e propagazione errore
# -------------------
angoli_rad = angoli_gradi * np.pi / 180
errore_angolo_rad = np.pi / 180  # 1 grado in radianti

# Tangente e propagazione dell’errore
tangente = np.tan(angoli_rad)
errore_tangente = (errore_angolo_rad) / (np.cos(angoli_rad) ** 2)

# -------------------
# Fit con Minuit: y = a * x
# -------------------
def retta(corrente,a):
    return ( a * corrente)
my_cost_func = LeastSquares(corrente,tangente, errore_tangente, retta)
m = Minuit(my_cost_func, a=1.0)
m.migrad()

# Estrazione risultati
a_fit = m.values['a']
a_err = m.errors['a']
chi2_val = m.fval
ndof = len(corrente) - 1  # un solo parametro
chi2_red = chi2_val / ndof
p_value = 1 - chi2.cdf(chi2_val, ndof)

# -------------------
# Plot
# -------------------
plt.figure(figsize=(10, 6))
plt.errorbar(corrente, tangente, yerr=errore_tangente, fmt='o', label='Dati')
plt.plot(corrente, a_fit * corrente, 'r-', label=f'Fit: y = a·x\n a = {a_fit:.3f} ± {a_err:.3f}')
plt.xlabel('Corrente (A)')
plt.ylabel('tan(θ)')
plt.title('Fit: y = a·x')
plt.grid(True)
plt.legend()

# Annotazioni
textstr = f"$\chi^2_{{rid}}$ = {chi2_red:.2f}\n$p$-value = {p_value:.3f}"
plt.text(0.05, max(tangente)*0.7, textstr, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
plt.show()
a_fit = m.values['a']
a_err = m.errors['a']
#estrapolazione campo magnetico terrestre:
mu_0 = 4 * np.pi * 1e-7  # T*m/A
raggio = 12.25
err_raggio = 0.1
N = 31  # numero di spire
def campo_magnetico(a_fit, raggio):
    return (N*mu_0)/(2*raggio*a_fit)
def errore_campo_magnetico(a_err, raggio, err_raggio):
    return (N*mu_0)/(2*raggio*a_fit**2) * np.sqrt((err_raggio)**2 + (a_err)**2)
B = campo_magnetico(a_fit, raggio)
err_B = errore_campo_magnetico(a_err, raggio, err_raggio)
print(f"Campo magnetico terrestre: {B:.2e} ± {err_B:.2e} T")

#set 2:
corrente2 = np.array([
    0.08, 0.15, 0.22, 0.30, 0.41, 0.50, 0.57, 0.70, 0.80, 0.90,
    0.99, 1.06, 1.17, 1.37, 1.60
])

angoli_gradi2 = np.array([
    24, 45, 56, 65, 72, 76, 79, 81, 83, 84,
    85, 86, 87, 88, 89
])

# -------------------
# Conversione in radianti e propagazione errore

# -------------------
angoli_rad2 = angoli_gradi2 * np.pi / 180
errore_angolo_rad2 = np.pi / 180  # 1 grado in radianti

# Tangente e propagazione dell’errore
tangente2 = np.tan(angoli_rad2)
errore_tangente2 = (errore_angolo_rad2) / (np.cos(angoli_rad2) ** 2)

# -------------------
# Fit con Minuit: y = a * x
# -------------------
def retta2(corrente2,a):
    return ( a * corrente2)
my_cost_func2 = LeastSquares(corrente2,tangente2, errore_tangente2, retta2)
m2 = Minuit(my_cost_func2, a=1.0)
m2.migrad()

# Estrazione risultati
a_fit2 = m2.values['a']
a_err2 = m2.errors['a']
chi2_val2 = m2.fval
ndof2 = len(corrente2) - 1  # un solo parametro

chi2_red2 = chi2_val2 / ndof2

p_value2 = 1 - chi2.cdf(chi2_val2, ndof2)

# -------------------
# Plot
# -------------------
plt.figure(figsize=(10, 6))
plt.errorbar(corrente2, tangente2, yerr=errore_tangente2, fmt='o', label='Dati')
plt.plot(corrente2, a_fit2 * corrente2, 'r-', label=f'Fit: y = a·x\n a = {a_fit2:.3f} ± {a_err2:.3f}')
plt.xlabel('Corrente (A)')
plt.ylabel('tan(θ)')
plt.title('Fit: y = a·x')
plt.grid(True)
plt.legend()

# Annotazioni
textstr2 = f"$\chi^2_{{rid}}$ = {chi2_red2:.2f}\n$p$-value = {p_value2:.3f}"
plt.text(0.05, max(tangente2)*0.7, textstr2, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
plt.tight_layout()
plt.show()
a_fit2 = m2.values['a']
a_err2 = m2.errors['a']
#estrapolazione campo magnetico terrestre:

B2 = campo_magnetico(a_fit2, raggio)
err_B2 = errore_campo_magnetico(a_err2, raggio, err_raggio)
print(f"Campo magnetico terrestre: {B2:.2e} ± {err_B2:.2e} T")
