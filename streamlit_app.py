import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm, chisquare, poisson

# Cargar los datos 1 y 2
data1 = np.genfromtxt('data1.csv', delimiter=',', skip_header=1, usecols=1)
data2 = np.genfromtxt('data2.csv', delimiter=',', skip_header=1, usecols=1)

# distribución gaussiana
def plot_gaussian_distribution(data):
    mu, sigma = np.mean(data), np.std(data)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    y = norm.pdf(x, mu, sigma)
    st.line_chart(list(zip(x, y)))

# función para calcular el estadístico de chi-cuadrado para la distribución gaussiana
def calcular_chi_cuadrado_gaussiano(data):
    mu, sigma = np.mean(data), np.std(data)
    valores_unicos, frecuencia_observada = np.unique(data, return_counts=True)
    frecuencia_esperada = [len(data) * norm.pdf(valor, mu, sigma) for valor in valores_unicos]
    chi_square_statistic, _ = chisquare(frecuencia_observada, frecuencia_esperada)
    return chi_square_statistic

# función para calcular la distribución de Poisson
def plot_poisson_distribution(data):
    mu = np.mean(data)
    x = np.arange(0, max(data) + 1)
    y = poisson.pmf(x, mu)
    st.bar_chart(list(zip(x, y)))

# Streamlit
st.title('Análisis de Datos')

# Botón para seleccionar el conjunto de datos
selected_data = st.radio('Seleccionar conjunto de datos:', ('data1.csv', 'data2.csv'))

if selected_data == 'data1.csv':
    st.subheader('Distribuciones en el decaimiento solo con el aire')
    st.subheader('Distribución de Gauss:')
    plot_gaussian_distribution(data1)
    if st.button('Calcular Chi Cuadrado para Gaussiana'):
        chi_square_gaussian = calcular_chi_cuadrado_gaussiano(data1)
        st.write(f"Valor del estadístico Chi-Cuadrado para la distribución gaussiana: {chi_square_gaussian}")
    st.subheader('Distribución de Poisson:')
    plot_poisson_distribution(data1)

elif selected_data == 'data2.csv':
    st.subheader('Distribuciones en el decaimiento del cesio-137')
    st.subheader('Distribución de Gauss:')
    plot_gaussian_distribution(data2)
    if st.button('Calcular Chi Cuadrado para Gaussiana'):
        chi_square_gaussian = calcular_chi_cuadrado_gaussiano(data2)
        st.write(f"Valor del estadístico Chi-Cuadrado para la distribución gaussiana: {chi_square_gaussian}")
    st.subheader('Distribución de Poisson:')
    plot_poisson_distribution(data2)
