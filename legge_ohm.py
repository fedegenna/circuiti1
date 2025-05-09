import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2

def interpolazione_lineare(correnti, voltaggi, errori_correnti=None, errori_voltaggi=None):
    """
    Interpola i dati con una retta generica V = m * I + q.
    Calcola i coefficienti m e q con i relativi errori e stampa sul grafico i risultati.

    :param correnti: Lista o array delle correnti (ascisse).
    :param voltaggi: Lista o array dei voltaggi (ordinate).
    :param errori_correnti: Lista o array degli errori sulle correnti (opzionale).
    :param errori_voltaggi: Lista o array degli errori sui voltaggi (opzionale).
    """
    # Converti le liste in array NumPy
    correnti = np.array(correnti)
    voltaggi = np.array(voltaggi)

    # Funzione per il fit lineare
    def retta(I, m, q):
        return m * I + q

    # Propagazione degli errori (se forniti)
    if errori_correnti is not None and errori_voltaggi is not None:
        errori_correnti = np.array(errori_correnti)
        errori_voltaggi = np.array(errori_voltaggi)
        errori_totali = np.sqrt(errori_voltaggi**2 + (errori_correnti * correnti)**2)
    else:
        errori_totali = errori_voltaggi

    # Fit con curve_fit
    popt, pcov = curve_fit(retta, correnti, voltaggi, sigma=errori_totali, absolute_sigma=True)
    m, q = popt
    errore_m, errore_q = np.sqrt(np.diag(pcov))

    # Calcolo del chi-quadro
    voltaggi_fit = retta(correnti, m, q)
    chi_quadro = np.sum(((voltaggi - voltaggi_fit) / errori_totali) ** 2)
    dof = len(correnti) - len(popt)  # Gradi di libertà
    p_value = 1 - chi2.cdf(chi_quadro, dof)

    # Disegna il grafico
    plt.errorbar(correnti, voltaggi, xerr=errori_correnti, yerr=errori_voltaggi, fmt='o', ecolor='red', capsize=5, label="Dati sperimentali")
    x_fit = np.linspace(min(correnti), max(correnti), 500)
    y_fit = retta(x_fit, m, q)
    plt.plot(x_fit, y_fit, label=f"Fit: V = m * I + q\nm = {m:.2e} ± {errore_m:.2e}\nq = {q:.2e} ± {errore_q:.2e}")
    plt.title("Interpolazione Lineare")
    plt.xlabel("Corrente (I)")
    plt.ylabel("Voltaggio (V)")
    plt.legend(loc='upper left')
    plt.grid(True)

    # Stampa chi-quadro e p-value sul grafico
    plt.text(0.05, 0.15, f"$\chi^2$ = {chi_quadro:.2f}\nDOF = {dof}\np-value = {p_value:.4f}",
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    plt.show()

    # Stampa i risultati in console
    print(f"Coefficiente angolare (m): {m:.2e} ± {errore_m:.2e}")
    print(f"Intercetta (q): {q:.2e} ± {errore_q:.2e}")
    print(f"Chi-quadro: {chi_quadro:.2f}")
    print(f"Gradi di libertà: {dof}")
    print(f"P-value: {p_value:.4f}")

def main():
    # Dati di esempio
    correnti = [2.171, 6.068, 10.778, 18.032, 24.923]  # Correnti in ampere
    voltaggi = [0.102, 0.283, 0.501, 0.838, 1.156]  # Voltaggi in volt
    errori_correnti = [0.001, 0.001, 0.001, 0.001, 0.001]  # Errori sulle correnti
    errori_voltaggi = [0.001, 0.001, 0.001, 0.001, 0.001]  # Errori sui voltaggi

    # Analizza il segnale
    interpolazione_lineare(correnti, voltaggi, errori_correnti=errori_correnti, errori_voltaggi=errori_voltaggi)

if __name__ == "__main__":
    main()
