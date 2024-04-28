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
    st.write(f"Valor de chi cuadrado: {chi_cuadrado:.2f}")

# Función para plotear la distribución de Gauss
def plot_gaussian_distribution(data, title):
    mu, sigma = np.mean(data), np.std(data)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    y = norm.pdf(x, mu, sigma)
    fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines', name='Distribución Gaussiana'))
    fig.add_trace(go.Histogram(x=data, histnorm='probability density', name='Histograma Gaussiano'))
    fig.update_layout(title=title, xaxis_title='Valor', yaxis_title='Densidad de probabilidad')
    return fig

# Función para plotear la distribución de Poisson
def plot_poisson_distribution(data, title):
    mu = np.mean(data)
    x = np.arange(0, max(data) + 1)
    y = poisson.pmf(x, mu)
    fit_y = poisson.pmf(x, mu)
    fig = go.Figure(data=[go.Bar(x=x, y=y, name='Distribución de Poisson'),
                          go.Scatter(x=x, y=fit_y, mode='lines', name='Ajuste de Poisson', line=dict(color='red', width=2))])
    fig.update_layout(title=title, xaxis_title='Valor', yaxis_title='Probabilidad')
    return fig

# Streamlit
st.set_page_config(page_title="Análisis de Datos", page_icon=":bar_chart:", layout="wide")

# Título y subtítulo
st.title('Análisis de Datos')
st.write("Bienvenido al análisis estadístico de los conjuntos de datos.")
st.markdown("---")

# Sidebar
st.sidebar.title("Navegación")
selected_option = st.sidebar.radio('Seleccionar opción:', ('Marco Teórico', 'Data1', 'Data2'))

# Contenido principal
if selected_option == 'Marco Teórico':
    st.write("  Marco Teórico.")
elif selected_option == 'Data1':
    st.subheader('Decaimiento solo con el aire')
    st.markdown("---")

    st.subheader('Distribución de Gauss:')
    st.write("Distribución de Gauss para el conjunto de datos 'data1.csv'.")
    data1 = np.genfromtxt('data1.csv', delimiter=',', skip_header=1, usecols=1)
    fig = plot_gaussian_distribution(data1, 'Distribución de Gauss - Data1')
    st.plotly_chart(fig)
    if st.button('Mostrar Tabla y Valor de Chi Cuadrado (Gaussiana)'):
        mostrar_tabla_y_chi(data1)
        st.markdown("---")

    st.subheader('Distribución de Poisson:')
    st.write("Distribución de Poisson para el conjunto de datos 'data1.csv'.")
    data1 = np.genfromtxt('data1.csv', delimiter=',', skip_header=1, usecols=1)
    fig = plot_poisson_distribution(data1, 'Distribución de Poisson - Data1')
    st.plotly_chart(fig)
    if st.button('Mostrar Tabla y Valor de Chi Cuadrado (Poisson)'):
        mostrar_tabla_y_chi(data1)
        st.markdown("---")

elif selected_option == 'Data2':
    st.subheader('Decaimiento del cesio-137')
    st.markdown("---")

    st.subheader('Distribución de Gauss:')
    st.write("Distribución de Gauss para el conjunto de datos 'data2.csv'.")
    data2 = np.genfromtxt('data2.csv', delimiter=',', skip_header=1, usecols=1)
    fig = plot_gaussian_distribution(data2, 'Distribución de Gauss - Data2')
    st.plotly_chart(fig)
    if st.button('Mostrar Tabla y Valor de Chi Cuadrado (Gaussiana)'):
        mostrar_tabla_y_chi(data2)
        st.markdown("---")

    st.subheader('Distribución de Poisson:')
    st.write("Distribución de Poisson para el conjunto de datos 'data2.csv'.")
    data2 = np.genfromtxt('data2.csv', delimiter=',', skip_header=1, usecols=1)
    fig = plot_poisson_distribution(data2, 'Distribución de Poisson - Data2')
    st.plotly_chart(fig)
    if st.button('Mostrar Tabla y Valor de Chi Cuadrado (Poisson)'):
        mostrar_tabla_y_chi(data2)
        st.markdown("---")
