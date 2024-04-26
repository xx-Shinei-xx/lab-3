import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson

# Función para graficar la distribución gaussiana
def plot_gaussian_distribution(data):
    mu, std = np.mean(data), np.std(data)
    x = np.linspace(mu - 3*std, mu + 3*std, 100)
    plt.plot(x, norm.pdf(x, mu, std))
    plt.title('Distribución Gaussiana')
    plt.xlabel('Valor')
    plt.ylabel('Densidad de probabilidad')
    st.pyplot()

# Función para graficar la distribución de Poisson
def plot_poisson_distribution(data):
    mu = np.mean(data)
    x = np.arange(0, max(data) + 1)
    plt.plot(x, poisson.pmf(x, mu), 'bo', ms=8)
    plt.vlines(x, 0, poisson.pmf(x, mu), colors='b', lw=5, alpha=0.5)
    plt.title('Distribución de Poisson')
    plt.xlabel('Valor')
    plt.ylabel('Probabilidad')
    st.pyplot()

# Cargar los datos desde el archivo CSV
data = np.genfromtxt('data1.csv', delimiter=',', skip_header=1, usecols=1)

# Crear la aplicación Streamlit
st.title('Distribuciones Estadísticas')

# Agregar un botón para cambiar entre las distribuciones
distribution = st.radio('Seleccionar distribución:', ('Gaussiana', 'Poisson'))

# Mostrar la distribución seleccionada
if distribution == 'Gaussiana':
    plot_gaussian_distribution(data)
elif distribution == 'Poisson':
    plot_poisson_distribution(data)
    
