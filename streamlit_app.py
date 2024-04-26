import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2_contingency

# Función para cargar datos desde un archivo CSV
def load_data(filename):
    data = pd.read_csv(filename)
    return data

# Función para mostrar el histograma de datos
def plot_histogram(data):
    plt.hist(data, bins=20, alpha=0.75, edgecolor='black')
    plt.xlabel('Valores')
    plt.ylabel('Frecuencia')
    st.pyplot()

# Función para ajustar una distribución normal y mostrarla
def fit_and_plot_normal(data):
    # Ajustar la distribución normal a los datos
    mu, std = norm.fit(data)
    x = np.linspace(min(data), max(data), 100)
    y = norm.pdf(x, mu, std)

    # Calcular la frecuencia observada y esperada
    observed_values, _ = np.histogram(data, bins=20)
    expected_values = len(data) * norm.pdf((bins[1:] + bins[:-1]) / 2, mu, std) * (bins[1] - bins[0])

    # Realizar la prueba de chi-cuadrado
    chi2_stat, p_value = chi2_contingency([observed_values, expected_values])

    # Graficar la distribución normal ajustada
    plt.plot(x, y, 'r-', label='Normal')
    plt.hist(data, bins=20, density=True, alpha=0.75, edgecolor='black', label='Datos')
    plt.xlabel('Valores')
    plt.ylabel('Densidad de probabilidad')
    plt.legend()
    st.pyplot()

    # Mostrar los resultados de la prueba de chi-cuadrado
    st.write(f'Estadístico Chi-cuadrado: {chi2_stat:.4f}')
    st.write(f'Valor p: {p_value:.4f}')

    # Evaluar el resultado de la prueba de chi-cuadrado
    if p_value < 0.05:
        st.write('La distribución normal no es un buen ajuste para los datos.')
    else:
        st.write('La distribución normal es un buen ajuste para los datos.')

def main():
    st.title('Distribución Normal con prueba de Chi-cuadrado')

    # Cargar datos desde archivo CSV
    filename = st.file_uploader("Cargar archivo CSV", type=['csv'])
    if filename is not None:
        data = load_data(filename)
        st.subheader('Visualización de datos')
        plot_histogram(data)

        st.subheader('Ajuste de Distribución Normal y prueba de Chi-cuadrado')
        fit_and_plot_normal(data)

if __name__ == "__main__":
    main()

