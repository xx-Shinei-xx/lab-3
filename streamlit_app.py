import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm, poisson, chisquare

# distribución gaussiana
def plot_gaussian_distribution(data):
    mu, sigma = np.mean(data), np.std(data)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    y = norm.pdf(x, mu, sigma)
    fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines', name='Distribución Gaussiana'))
    fig.add_trace(go.Histogram(x=np.random.normal(mu, sigma, 1000), histnorm='probability density', name='Histograma Gaussiano'))
    fig.update_layout(title='Distribución Gaussiana', xaxis_title='Valor', yaxis_title='Densidad de probabilidad')
    st.plotly_chart(fig)

#distribución de Poisson
def fit_poisson_distribution(data):
    mu = np.mean(data)
    return mu

#prueba de chi-cuadrado
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

# Función para graficar la prueba de chi-cuadrado
def plot_chi_square_test(p_value, observed_counts, expected_counts, distribution):
    # No mostramos las gráficas de la prueba de chi-cuadrado
    pass

#fit de la distribución de poisson 
def plot_poisson_distribution(data):
    mu = np.mean(data)
    x = np.arange(0, max(data) + 1)
    y = poisson.pmf(x, mu)
    fit_y = poisson.pmf(x, mu)

    fig = go.Figure(data=[go.Bar(x=x, y=y, name='Distribución de Poisson'),
                          go.Scatter(x=x, y=fit_y, mode='lines', name='Ajuste de Poisson', line=dict(color='red', width=2))])
    fig.update_layout(title='Distribución de Poisson', xaxis_title='Valor', yaxis_title='Probabilidad')
    st.plotly_chart(fig)

# Cargar los datos 1 y 2
data1 = np.genfromtxt('data1.csv', delimiter=',', skip_header=1, usecols=1)
data2 = np.genfromtxt('data2.csv', delimiter=',', skip_header=1, usecols=1)

#  Streamlit
st.title('Análisis de Datos')

# Botón para seleccionar el conjunto de datos
selected_data = st.radio('Seleccionar conjunto de datos:', ('data1.csv', 'data2.csv'))

if selected_data == 'data1.csv':
    st.subheader('Distribuciones en el decaimiento solo con el aire')
    
    st.subheader('Distribución de Gauss:')
    plot_gaussian_distribution(data1)

    st.subheader('Distribución de Poisson:')
    plot_poisson_distribution(data1)

    if st.button('Realizar ajuste de chi-cuadrado para distribución Gaussiana'):
        p_value_gaussian_data1, _, _ = chi_square_test(data1, 'gaussian')
        st.write(f"Valor p para distribución Gaussiana en el decaimiento solo con el aire: {p_value_gaussian_data1}")

    if st.button('Realizar ajuste de chi-cuadrado para distribución de Poisson'):
        p_value_poisson_data1, _, _ = chi_square_test(data1, 'poisson')
        st.write(f"Valor p para distribución de Poisson en el decaimiento solo con el aire: {p_value_poisson_data1}")

elif selected_data == 'data2.csv':
    st.subheader('Distribuciones en el decaimiento del cesio-137')

    st.subheader('Distribución de Gauss:')
    plot_gaussian_distribution(data2)

    st.subheader('Distribución de Poisson:')
    plot_poisson_distribution(data2)

    if st.button('Realizar ajuste de chi-cuadrado para distribución Gaussiana'):
        p_value_gaussian_data2, _, _ = chi_square_test(data2, 'gaussian')
        st.write(f"Valor p para distribución Gaussiana en el decaimiento del cesio-137: {p_value_gaussian_data2}")

    if st.button('Realizar ajuste de chi-cuadrado para distribución de Poisson'):
        p_value_poisson_data2, _, _ = chi_square_test(data2, 'poisson')
        st.write(f"Valor p para distribución de Poisson en el decaimiento del cesio-137: {p_value_poisson_data2}")
