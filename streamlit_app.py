import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm, poisson, chisquare

# Función para calcular chi cuadrado
def calcular_chi_cuadrado(f_obs, f_exp):
    return np.sum((f_obs - f_exp)**2 / f_exp)

# Función para calcular frecuencias observadas y esperadas
def calcular_frecuencias(data):
    valores_unicos, counts = np.unique(data, return_counts=True)
    frecuencia_observada = counts
    mu = np.mean(data)
    frecuencia_esperada = poisson.pmf(valores_unicos, mu) * len(data)
    return valores_unicos, frecuencia_observada, frecuencia_esperada

# Función para mostrar la tabla de frecuencias y el valor de chi cuadrado
def mostrar_tabla_y_chi(data):
    valores_unicos, frecuencia_observada, frecuencia_esperada = calcular_frecuencias(data)
    tabla_data = {"Valor": valores_unicos, "Frecuencia Observada": frecuencia_observada, "Frecuencia Esperada": frecuencia_esperada}
    tabla = st.table(tabla_data)
    chi_cuadrado = calcular_chi_cuadrado(frecuencia_observada, frecuencia_esperada)
    st.write(f"Valor de chi cuadrado: {chi_cuadrado}")

# Streamlit
st.title('Análisis de Datos')
st.markdown('---')

# Botón para seleccionar el conjunto de datos
selected_data = st.radio('Seleccionar conjunto de datos:', ('data1', 'data2'))
st.markdown('---')

if selected_data == 'data1':
    st.subheader('Distribuciones en el decaimiento solo con el aire')
    st.markdown('---')
    
    st.subheader('Distribución de Gauss:')
    # Cargar datos
    data1 = np.genfromtxt('data1.csv', delimiter=',', skip_header=1, usecols=1)
    mu, sigma = np.mean(data1), np.std(data1)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    y = norm.pdf(x, mu, sigma)
    fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines', name='Distribución Gaussiana'))
    fig.add_trace(go.Histogram(x=data1, histnorm='probability density', name='Histograma Gaussiano'))
    fig.update_layout(title='Distribución Gaussiana', xaxis_title='Valor', yaxis_title='Densidad de probabilidad')
    st.plotly_chart(fig)
    st.markdown('---')

    if st.button('Mostrar Tabla y Valor de Chi Cuadrado (Gaussiana)'):
        mostrar_tabla_y_chi(data1)
        st.markdown('---')

    st.subheader('Distribución de Poisson:')
    mu = np.mean(data1)
    x = np.arange(0, max(data1) + 1)
    y = poisson.pmf(x, mu)
    fit_y = poisson.pmf(x, mu)
    fig = go.Figure(data=[go.Bar(x=x, y=y, name='Distribución de Poisson'),
                          go.Scatter(x=x, y=fit_y, mode='lines', name='Ajuste de Poisson', line=dict(color='red', width=2))])
    fig.update_layout(title='Distribución de Poisson', xaxis_title='Valor', yaxis_title='Probabilidad')
    st.plotly_chart(fig)
    st.markdown('---')

    if st.button('Mostrar Tabla y Valor de Chi Cuadrado (Poisson)'):
        mostrar_tabla_y_chi(data1)
        st.markdown('---')

elif selected_data == 'data2':
    st.subheader('Distribuciones en el decaimiento del cesio-137')
    st.markdown('---')
    
    st.subheader('Distribución de Gauss:')
    # Cargar datos
    data2 = np.genfromtxt('data2.csv', delimiter=',', skip_header=1, usecols=1)
    mu, sigma = np.mean(data2), np.std(data2)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    y = norm.pdf(x, mu, sigma)
    fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines', name='Distribución Gaussiana'))
    fig.add_trace(go.Histogram(x=data2, histnorm='probability density', name='Histograma Gaussiano'))
    fig.update_layout(title='Distribución Gaussiana', xaxis_title='Valor', yaxis_title='Densidad de probabilidad')
    st.plotly_chart(fig)
    st.markdown('---')

    if st.button('Mostrar Tabla y Valor de Chi Cuadrado (Gaussiana)'):
        mostrar_tabla_y_chi(data2)
        st.markdown('---')

    st.subheader('Distribución de Poisson:')
    mu = np.mean(data2)
    x = np.arange(0, max(data2) + 1)
    y = poisson.pmf(x, mu)
    fit_y = poisson.pmf(x, mu)
    fig = go.Figure(data=[go.Bar(x=x, y=y, name='Distribución de Poisson'),
                          go.Scatter(x=x, y=fit_y, mode='lines', name='Ajuste de Poisson', line=dict(color='red', width=2))])
    fig.update_layout(title='Distribución de Poisson', xaxis_title='Valor', yaxis_title='Probabilidad')
    st.plotly_chart(fig)
    st.markdown('---')

    if st.button('Mostrar Tabla y Valor de Chi Cuadrado (Poisson)'):
        mostrar_tabla_y_chi(data2)
        st.markdown('---')
