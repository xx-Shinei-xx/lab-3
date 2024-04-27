import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm, chisquare, poisson
from scipy.optimize import curve_fit

# Función para graficar la distribución gaussiana
def plot_gaussian_distribution(data):
    mu, sigma = np.mean(data), np.std(data)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    y = norm.pdf(x, mu, sigma)
    fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines', name='Distribución Gaussiana'))
    fig.add_trace(go.Histogram(x=data, histnorm='probability density', name='Histograma Gaussiano'))
    fig.update_layout(title='Distribución Gaussiana', xaxis_title='Valor', yaxis_title='Densidad de probabilidad')
    st.plotly_chart(fig)

# Función para ajustar la distribución gaussiana
def fit_gaussian(data):
    def gaussian_function(x, mu, sigma):
        return norm.pdf(x, mu, sigma)

    popt, _ = curve_fit(gaussian_function, data, np.ones_like(data))
    
    return popt

# Función para graficar la distribución de Poisson
def plot_poisson_distribution(data):
    lambda_poisson = np.mean(data)
    x = np.arange(0, np.max(data) + 1)
    y = poisson.pmf(x, lambda_poisson)
    fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines', name='Distribución de Poisson'))
    fig.add_trace(go.Histogram(x=data, histnorm='probability density', name='Histograma de Poisson'))
    fig.update_layout(title='Distribución de Poisson', xaxis_title='Valor', yaxis_title='Densidad de probabilidad')
    st.plotly_chart(fig)

# Función para realizar la prueba de chi-cuadrado y determinar el ajuste que se adapta mejor
def chi_square_test(data, distribution='gaussian'):
    if distribution == 'gaussian':
        mu = np.mean(data)
        sigma = np.std(data)
        expected_counts, _ = np.histogram(np.random.normal(mu, sigma, len(data)), bins=10)
    elif distribution == 'poisson':
        lambda_poisson = np.mean(data)
        expected_counts = poisson.pmf(np.arange(0, np.max(data) + 1), lambda_poisson) * len(data)
    
    observed_counts, _ = np.histogram(data, bins=10)
    
    _, chi_value = chisquare(observed_counts, expected_counts)
    
    return chi_value

# Cargar los datos desde los archivos CSV
data1 = np.genfromtxt('data1.csv', delimiter=',', skip_header=1, usecols=1)
data2 = np.genfromtxt('data2.csv', delimiter=',', skip_header=1, usecols=1)

# Crear la aplicación Streamlit
st.title('Análisis de Datos')

# Botón para seleccionar el conjunto de datos
selected_data = st.radio('Seleccionar conjunto de datos:', ('data1.csv', 'data2.csv'))

if selected_data == 'data1.csv':
    st.subheader('Distribuciones de data1.csv')
    
    st.subheader('Distribución de Gauss:')
    plot_gaussian_distribution(data1)
    mu_gaussian, sigma_gaussian = fit_gaussian(data1)
    
    if st.button('Realizar ajuste de chi-cuadrado para distribución Gaussiana'):
        chi_value_gaussian_data1 = chi_square_test(data1, distribution='gaussian')
        st.write(f"Valor de la prueba de chi-cuadrado para distribución Gaussiana: {chi_value_gaussian_data1}")
    
    st.subheader('Distribución de Poisson:')
    plot_poisson_distribution(data1)
    
    if st.button('Realizar ajuste de chi-cuadrado para distribución de Poisson'):
        chi_value_poisson_data1 = chi_square_test(data1, distribution='poisson')
        st.write(f"Valor de la prueba de chi-cuadrado para distribución de Poisson: {chi_value_poisson_data1}")

elif selected_data == 'data2.csv':
    st.subheader('Distribuciones de data2.csv')

    st.subheader('Distribución de Gauss:')
    plot_gaussian_distribution(data2)
    mu_gaussian, sigma_gaussian = fit_gaussian(data2)
  
    if st.button('Realizar ajuste de chi-cuadrado para distribución Gaussiana'):
        chi_value_gaussian_data2 = chi_square_test(data2, distribution='gaussian')
        st.write(f"Valor de la prueba de chi-cuadrado para distribución Gaussiana: {chi_value_gaussian_data2}")
    
    st.subheader('Distribución de Poisson:')
    plot_poisson_distribution(data2)
    
    if st.button('Realizar ajuste de chi-cuadrado para distribución de Poisson'):
        chi_value_poisson_data2 = chi_square_test(data2, distribution='poisson')
        st.write(f"Valor de la prueba de chi-cuadrado para distribución de Poisson: {chi_value_poisson_data2}")
