import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm, chisquare, poisson
import plotly.graph_objects as go

# Cargar los datos 1 y 2
data1 = np.genfromtxt('data1.csv', delimiter=',', skip_header=1, usecols=1)
data2 = np.genfromtxt('data2.csv', delimiter=',', skip_header=1, usecols=1)

# Estilo de página
st.markdown("""
<style>
.big-font {
    font-size:24px !important;
}
</style>
""", unsafe_allow_html=True)

# distribución gaussiana
def plot_gaussian_distribution(data):
    mu, sigma = np.mean(data), np.std(data)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    y = norm.pdf(x, mu, sigma)
    fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines', name='Distribución Gaussiana'))
    fig.add_trace(go.Histogram(x=data, histnorm='probability density', name='Histograma Gaussiano'))
    fig.update_layout(title='Distribución Gaussiana', xaxis_title='Valor', yaxis_title='Densidad de probabilidad')
    st.plotly_chart(fig)

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
    fig = go.Figure(data=go.Bar(x=x, y=y, name='Distribución de Poisson'))
    fig.update_layout(title='Distribución de Poisson', xaxis_title='Valor', yaxis_title='Probabilidad')
    st.plotly_chart(fig)

# Streamlit
st.title('Análisis de Datos')

# Botón para seleccionar el conjunto de datos
selected_data = st.radio('Seleccionar conjunto de datos:', ('data1.csv', 'data2.csv'))

if selected_data == 'data1.csv':
    st.markdown("<h2 class='big-font'>Distribuciones en el decaimiento solo con el aire</h2>", unsafe_allow_html=True)
    st.subheader('Distribución de Gauss:')
    plot_gaussian_distribution(data1)
    if st.button('Calcular Chi Cuadrado para Gaussiana'):
        chi_square_gaussian = calcular_chi_cuadrado_gaussiano(data1)
        st.write(f"Valor del estadístico Chi-Cuadrado para la distribución gaussiana: {chi_square_gaussian}")
    st.subheader('Distribución de Poisson:')
    plot_poisson_distribution(data1)

elif selected_data == 'data2.csv':
    st.markdown("<h2 class='big-font'>Distribuciones en el decaimiento del cesio-137</h2>", unsafe_allow_html=True)
    st.subheader('Distribución de Gauss:')
    plot_gaussian_distribution(data2)
    if st.button('Calcular Chi Cuadrado para Gaussiana'):
        chi_square_gaussian = calcular_chi_cuadrado_gaussiano(data2)
        st.write(f"Valor del estadístico Chi-Cuadrado para la distribución gaussiana: {chi_square_gaussian}")
    st.subheader('Distribución de Poisson:')
    plot_poisson_distribution(data2)
