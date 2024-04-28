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

# distribución de Poisson
def plot_poisson_distribution(data):
    mu = np.mean(data)
    x = np.arange(0, max(data) + 1)
    y = poisson.pmf(x, mu)
    fit_y = poisson.pmf(x, mu)
    fig = go.Figure(data=[go.Bar(x=x, y=y, name='Distribución de Poisson'),
                          go.Scatter(x=x, y=fit_y, mode='lines', name='Ajuste de Poisson', line=dict(color='red', width=2))])
    fig.update_layout(title='Distribución de Poisson', xaxis_title='Valor', yaxis_title='Probabilidad')
    st.plotly_chart(fig)

# función para calcular las frecuencias observadas y esperadas
def calcular_frecuencias(data):
    valores_unicos, frecuencia_observada = np.unique(data, return_counts=True)
    tasa_promedio = np.mean(data)
    frecuencia_esperada = [poisson.pmf(valor, tasa_promedio) * len(data) for valor in valores_unicos]
    return valores_unicos, frecuencia_observada, frecuencia_esperada

# función para mostrar la tabla de frecuencias
def mostrar_tabla(data):
    valores_unicos, frecuencia_observada, frecuencia_esperada = calcular_frecuencias(data)
    tabla_data = {"Valor": valores_unicos, "Frecuencia Observada": frecuencia_observada, "Frecuencia Esperada": frecuencia_esperada}
    tabla = st.table(tabla_data)

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
    if st.button('Mostrar Tabla'):
        mostrar_tabla(data1)

elif selected_data == 'data2.csv':
    st.subheader('Distribuciones en el decaimiento del cesio-137')
    st.subheader('Distribución de Gauss:')
    plot_gaussian_distribution(data2)
    st.subheader('Distribución de Poisson:')
    plot_poisson_distribution(data2)
    if st.button('Mostrar Tabla'):
        mostrar_tabla(data2)
