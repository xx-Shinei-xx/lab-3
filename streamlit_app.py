import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import poisson, chisquare

# Función para calcular la prueba de chi-cuadrado
def chi_square_test(data, distribution):
    if distribution == 'gaussian':
        mu = np.mean(data)
        sigma = np.std(data)
        expected_counts, _ = np.histogram(np.random.normal(mu, sigma, len(data)), bins=10)
    elif distribution == 'poisson':
        mu = np.mean(data)
        expected_counts = poisson.pmf(np.arange(10), mu) * len(data)
    
    observed_counts, _ = np.histogram(data, bins=10)
    
    _, p_value = chisquare(observed_counts, expected_counts)
    
    return p_value, observed_counts, expected_counts

# Cargar los datos 1 y 2
data1 = np.genfromtxt('data1.csv', delimiter=',', skip_header=1, usecols=1)
data2 = np.genfromtxt('data2.csv', delimiter=',', skip_header=1, usecols=1)

#  Streamlit
st.title('Análisis de Datos')

# Botón para seleccionar el conjunto de datos
selected_data = st.radio('Seleccionar conjunto de datos:', ('data1.csv', 'data2.csv'))

if selected_data == 'data1.csv':
    st.subheader('Distribuciones en el decaimiento solo con el aire')
    
    if st.button('Realizar ajuste de chi-cuadrado para distribución Gaussiana'):
        p_value_gaussian_data1, _, _ = chi_square_test(data1, 'gaussian')
        st.write(f"Valor p para distribución Gaussiana en el decaimiento solo con el aire: {p_value_gaussian_data1}")

    if st.button('Realizar ajuste de chi-cuadrado para distribución de Poisson'):
        p_value_poisson_data1, _, _ = chi_square_test(data1, 'poisson')
        st.write(f"Valor p para distribución de Poisson en el decaimiento solo con el aire: {p_value_poisson_data1}")

elif selected_data == 'data2.csv':
    st.subheader('Distribuciones en el decaimiento del cesio-137')

    if st.button('Realizar ajuste de chi-cuadrado para distribución Gaussiana'):
        p_value_gaussian_data2, _, _ = chi_square_test(data2, 'gaussian')
        st.write(f"Valor p para distribución Gaussiana en el decaimiento del cesio-137: {p_value_gaussian_data2}")

    if st.button('Realizar ajuste de chi-cuadrado para distribución de Poisson'):
        p_value_poisson_data2, _, _ = chi_square_test(data2, 'poisson')
        st.write(f"Valor p para distribución de Poisson en el decaimiento del cesio-137: {p_value_poisson_data2}")
