import numpy as np
from scipy.stats import poisson, norm
import matplotlib.pyplot as plt
import streamlit as st

# Cargar los datos
data = np.genfromtxt('data1.csv', delimiter=',', skip_header=1, usecols=1)

# Función para el análisis de distribución de Poisson
def poisson_analysis(data):
    # Calcular la media de los datos
    mu = np.mean(data)

    # Generar la distribución de Poisson con la media calculada
    poisson_pmf = poisson.pmf(np.arange(np.max(data) + 1), mu)

    # Calcular la prueba de chi-cuadrado
    chi_square_poisson, p_value_poisson = poisson.chisquare(data, mu)

    # Crear una figura para graficar los datos y el ajuste de Poisson
    fig, ax = plt.subplots()
    ax.hist(data, bins=np.arange(np.max(data) + 2) - 0.5, density=True, label='Datos')
    ax.plot(np.arange(np.max(data) + 1), poisson_pmf, 'r-', lw=2, label='Ajuste Poisson')
    ax.set_title('Distribución de Poisson')
    ax.set_xlabel('Valor')
    ax.set_ylabel('Densidad')
    ax.legend()

    return fig, chi_square_poisson, p_value_poisson

# Función para el análisis de distribución Gaussiana
def gaussian_analysis(data):
    # Calcular la media y la desviación estándar de los datos
    mean = np.mean(data)
    std = np.std(data)

    # Ajustar la distribución Gaussiana
    gaussian_params = norm.fit(data)
    gaussian_fitted = norm.pdf(np.linspace(np.min(data), np.max(data), 100), *gaussian_params)

    # Calcular la prueba de chi-cuadrado
    chi_square_gaussian, p_value_gaussian = norm.chisquare(data, gaussian_params)

    # Crear una figura para graficar los datos y el ajuste Gaussiano
    fig, ax = plt.subplots()
    ax.hist(data, bins=20, density=True, label='Datos')
    ax.plot(np.linspace(np.min(data), np.max(data), 100), gaussian_fitted, 'r-', lw=2, label='Ajuste Gaussiano')
    ax.set_title('Distribución Gaussiana')
    ax.set_xlabel('Valor')
    ax.set_ylabel('Densidad')
    ax.legend()

    return fig, chi_square_gaussian, p_value_gaussian

# Aplicación Streamlit
st.title("Análisis de distribución de datos")

# Radio button para seleccionar la distribución
distribution = st.sidebar.radio("Seleccionar distribución", ("Poisson", "Gaussiana"))

# Sección de análisis
if distribution == "Poisson":
    st.header("Análisis de distribución de Poisson")
    fig, chi_square, p_value = poisson_analysis(data)
    st.pyplot(fig)
    st.write(f"Valor de chi-cuadrado: {chi_square:.2f}")
    st.write(f"Valor p: {p_value:.4f}")
else:
    st.header("Análisis de distribución Gaussiana")
    fig, chi_square, p_value = gaussian_analysis(data)
    st.pyplot(fig)
    st.write(f"Valor de chi-cuadrado: {chi_square:.2f}")
    st.write(f"Valor p: {p_value:.4f}")
