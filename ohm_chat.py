import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2

def interpolazione_lineare(correnti, voltaggi, errori_correnti=None, errori_voltaggi=None):
    """
    Interpola i dati con una retta passante per l'origine V = m * I usando Minuit.
    Calcola il coefficiente m con il relativo errore e stampa sul grafico i risultati.
    Esegue 5 ripetizioni aggiornando la propagazione degli errori sulle y.
    """
    correnti = np.array(correnti)
    voltaggi = np.array(voltaggi)
    errori_correnti = np.array(errori_correnti)
    errori_voltaggi = np.array(errori_voltaggi)

    def retta(I, m):
        return m * I

    # Primo fit solo con errori sui voltaggi
    cost = LeastSquares(correnti, voltaggi, errori_voltaggi, retta)
    m_fit = Minuit(cost, m=1.0)
    m_fit.migrad()
    m = m_fit.values['m']

    for i in range(5):
        # Propagazione errori: errore_y^2 + (m*errore_x)^2
        errori_totali = np.sqrt(errori_voltaggi**2 + (m * errori_correnti)**2)
        cost = LeastSquares(correnti, voltaggi, errori_totali, retta)
        m_fit = Minuit(cost, m=m)
        m_fit.migrad()
        m = m_fit.values['m']
        errore_m = m_fit.errors['m']
        print(f"Iterazione {i+1}: m = {m:.4f} ± {errore_m:.4f}")

    # Calcolo del chi-quadro finale
    voltaggi_fit = retta(correnti, m)
    chi_quadro = np.sum(((voltaggi - voltaggi_fit) / errori_totali) ** 2)
    dof = len(correnti) - 1
    p_value = 1 - chi2.cdf(chi_quadro, dof)

    # Disegna il grafico
    plt.errorbar(correnti, voltaggi, xerr=errori_correnti, yerr=errori_voltaggi, fmt='o', ecolor='red', capsize=5, label="Dati sperimentali")
    x_fit = np.linspace(min(correnti), max(correnti), 500)
    y_fit = retta(x_fit, m)
    plt.plot(x_fit, y_fit, label=f"Fit: V = m * I\nm = {m:.2e} ± {errore_m:.2e}")
    plt.title("Interpolazione Lineare (Passante per l'origine, Minuit)")
    plt.xlabel("Corrente (I)")
    plt.ylabel("Voltaggio (V)")
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.text(0.05, 0.15, f"$\chi^2$ = {chi_quadro:.2f}\nDOF = {dof}\np-value = {p_value:.4f}",
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    plt.show()

    # Stampa i risultati finali in console
    print(f"Coefficiente angolare (m): {m:.6e} ± {errore_m:.6e}")
    print(f"Chi-quadro: {chi_quadro:.2f}")
    print(f"Gradi di libertà: {dof}")
    print(f"P-value: {p_value:.4f}")

def resistenza_amperometro (m, R) :
    return (m-R)

def errore_resistenza_amperometro (m, R, errore_m, errore_R) :
    return np.sqrt(errore_m**2 + errore_R**2)

def resistenza_voltmetro (m, R) :
    return (1/(1/m - 1/R))

def errore_resistenza_voltmetro (m, R, errore_m, errore_R) :
    return np.sqrt((errore_m/(m**2))**2 + (errore_R/(R**2))**2)

def main():

    errori_correnti_a = [0.001, 0.001, 0.001, 0.001, 0.001]  # Errori sulle correnti
    errori_correnti_else = [0.0001, 0.0001, 0.0001, 0.0001, 0.0001]  # Errori sulle correnti
    errori_voltaggi = [0.001, 0.001, 0.001, 0.001, 0.001]  # Errori sui voltaggi

    voltaggio_1_a = [0.102, 0.283, 0.501, 0.838, 1.156]
    corrente_1_a = [2.171, 6.068, 10.778, 18.032, 24.923]
    voltaggio_2_a = [0.088, 0.244, 0.432, 0.724, 0.999]
    corrente_2_a = [2.170, 6.063, 10.771, 18.037, 24.920]

    voltaggio_1_b = [0.135, 3.621, 7.13, 11.95, 17.06]
    corrente_1_b = [0.0669, 1.8048, 3.5484, 5.9513, 8.5055]
    voltaggio_2_b = [0.135, 3.609, 7.110, 11.910, 17.02]
    corrente_2_b = [0.0669, 1.8051, 3.5492, 5.9526, 8.5072]

    voltaggio_1_c = [0.655, 4.583, 8.53, 12.68, 17.16]
    corrente_1_c = [0.0011, 0.0076, 0.0140, 0.0209, 0.0282]
    voltaggio_2_c = [0.654, 4.583, 8.53, 12.68, 17.16]
    corrente_2_c = [0.0012, 0.0080, 0.0149, 0.0221, 0.0299]

    # Analizza il segnale
    interpolazione_lineare(corrente_1_a, voltaggio_1_a, errori_correnti=errori_correnti_a, errori_voltaggi=errori_voltaggi)
    interpolazione_lineare(corrente_2_a, voltaggio_2_a, errori_correnti=errori_correnti_a, errori_voltaggi=errori_voltaggi)
    interpolazione_lineare(corrente_1_b, voltaggio_1_b, errori_correnti=errori_correnti_else, errori_voltaggi=errori_voltaggi)
    interpolazione_lineare(corrente_2_b, voltaggio_2_b, errori_correnti=errori_correnti_else, errori_voltaggi=errori_voltaggi)
    interpolazione_lineare(corrente_1_c, voltaggio_1_c, errori_correnti=errori_correnti_else, errori_voltaggi=errori_voltaggi)
    interpolazione_lineare(corrente_2_c, voltaggio_2_c, errori_correnti=errori_correnti_else, errori_voltaggi=errori_voltaggi)
    
    m_a = [46.3, 2006.1, 608994.5]
    errore_m_a = [0.1, 0.2, 2852.4]
    m_v = [40.1, 2000.6, 574947.5]
    errore_m_v = [0.1, 0.2, 2852.3]
    R = [40.3, 2002, 608000]

    # Calcolo della resistenza dell'amperometro
    for i in range(len(m_a)):
        R_a = resistenza_amperometro(m_a[i], R[i])
        errore_R_a = errore_resistenza_amperometro(m_a[i], R[i], errore_m_a[i], errore_m_v[i])
        print(f"Resistenza dell'amperometro {i+1}: {R_a:.4f} ± {errore_R_a:.4f} Ohm")

    # Calcolo della resistenza del voltmetro
    for i in range(len(m_v)):
        R_v = resistenza_voltmetro(m_v[i], R[i])
        errore_R_v = errore_resistenza_voltmetro(m_v[i], R[i], errore_m_v[i], errore_m_a[i])
        print(f"Resistenza del voltmetro {i+1}: {R_v:.4f} ± {errore_R_v:.4f} Ohm")

if __name__ == "__main__":
    main()
