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
    fig.add_trace(go.Histogram(x=data, histnorm='probability density', name='Histograma Gaussiano'))
    fig.update_layout(title='Distribución Gaussiana', xaxis_title='Valor', yaxis_title='Densidad de probabilidad')
    st.plotly_chart(fig)

    if st.button('Realizar ajuste de chi-cuadrado para distribución Gaussiana'):
        p_value_gaussian_data, observed_counts, expected_counts = chi_square_test(data, 'gaussian')
        st.write(f"Valor p para distribución Gaussiana: {p_value_gaussian_data}")
        plot_chi_square_test(p_value_gaussian_data, observed_counts, expected_counts, 'gaussian')

# distribución de Poisson
def fit_poisson_distribution(data):
    mu = np.mean(data)
    return mu

# prueba de chi-cuadrado
def chi_square_test(data, distribution):
    if distribution == 'gaussian':
        mu = np.mean(data)
        sigma = np.std(data)
        # Calcular las frecuencias esperadas utilizando la función de densidad de probabilidad de la distribución normal
        expected_counts, bins = np.histogram(data, bins=10, density=True)
        expected_counts *= len(data)
        observed_counts, _ = np.histogram(data, bins=bins)
    elif distribution == 'poisson':
        mu = np.mean(data)
        expected_counts = poisson.pmf(np.arange(max(data) + 1), mu) * len(data)
        observed_counts, _ = np.histogram(data, bins=max(data) + 1)

    _, p_value = chisquare(observed_counts, expected_counts)
    return p_value, observed_counts, expected_counts

# Función para graficar la prueba de chi-cuadrado
def plot_chi_square_test(p_value, observed_counts, expected_counts, distribution):
    bins = np.arange(len(observed_counts) + 1)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=bins[:-1], y=observed_counts, name='Observados'))
    fig.add_trace(go.Scatter(x=bins[:-1], y=expected_counts, mode='lines', name='Esperados'))
    fig.update_layout(title=f'Prueba de chi-cuadrado para distribución {distribution}',
                      xaxis_title='Valor', yaxis_title='Frecuencia')
    st.plotly_chart(fig)
    st.write(f"Valor p: {p_value}")

# fit de la distribución de poisson
def plot_poisson_distribution(data):
    mu = np.mean(data)
    x = np.arange(0, max(data) + 1)
    y = poisson.pmf(x, mu)
    fit_y = poisson.pmf(x, mu)
    fig = go.Figure(data=[go.Bar(x=x, y=y, name='Distribución de Poisson'),
                          go.Scatter(x=x, y=fit_y, mode='lines', name='Ajuste de Poisson', line=dict(color='red', width=2))])
    fig.update_layout(title='Distribución de Poisson', xaxis_title='Valor', yaxis_title='Probabilidad')
    st.plotly_chart(fig)

   # if st.button('Realizar ajuste de chi-cuadrado para distribución de Poisson'):
     #   p_value_poisson_data, observed_counts, expected_counts = chi_square_test(data, 'poisson')
     #   st.write(f"Valor p para distribución de Poisson: {p_value_poisson_data}")
      #  plot_chi_square_test(p_value_poisson_data, observed_counts, expected_counts, 'poisson')

# Cargar los datos 1 y 2
data1 = np.genfromtxt('data1.csv', delimiter=',', skip_header=1, usecols=1)
data2 = np.genfromtxt('data2.csv', delimiter=',', skip_header=1, usecols=1)

# Streamlit
st.title('Análisis de Datos')

# Botón para seleccionar el conjunto de datos
selected_data = st.radio('Seleccionar conjunto de datos:', ('data1.csv', 'data2.csv'))

if selected_data == 'data1.csv':
    st.subheader('Distribuciones en el decaimiento solo con el aire')
    st.subheader('Distribución de Gauss:')
    plot_gaussian_distribution(data1)
    st.subheader('Distribución de Poisson:')
    plot_poisson_distribution(data1)

elif selected_data == 'data2.csv':
    st.subheader('Distribuciones en el decaimiento del cesio-137')
    st.subheader('Distribución de Gauss:')
    plot_gaussian_distribution(data2)
    st.subheader('Distribución de Poisson:')
    plot_poisson_distribution(data2)
