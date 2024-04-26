import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm, poisson

# Función para graficar la distribución gaussiana
def plot_gaussian_distribution(mu, sigma):
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    y = norm.pdf(x, mu, sigma)
    fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines', name='Distribución Gaussiana'))
    fig.update_layout(title='Distribución Gaussiana', xaxis_title='Valor', yaxis_title='Densidad de probabilidad')
    st.plotly_chart(fig)

# Función para graficar la distribución de Poisson con barras
def plot_poisson_distribution(data):
    mu = np.mean(data)
    x = np.arange(0, max(data) + 1)
    y = poisson.pmf(x, mu)
    fig = go.Figure(data=go.Bar(x=x, y=y))
    fig.update_layout(title='Distribución de Poisson', xaxis_title='Valor', yaxis_title='Probabilidad')
    st.plotly_chart(fig)

# Función para mostrar histograma y distribuciones
def show_histogram_and_distributions(df):
    st.subheader('Histograma de datos')
    st.bar_chart(df)

    st.subheader('Distribuciones Estadísticas')
    plot_gaussian_distribution(df['Value'].mean(), df['Value'].std())
    plot_poisson_distribution(df['Value'])

# Cargar los datos desde el archivo CSV
data1 = np.genfromtxt('data1.csv', delimiter=',', skip_header=1, usecols=1)
data2 = np.genfromtxt('data2.csv', delimiter=',', skip_header=1, usecols=1)

# Crear la aplicación Streamlit
st.title('Análisis de Datos')

# Agregar un botón para cambiar entre las distribuciones de data1.csv y data2.csv
selected_data = st.radio('Seleccionar conjunto de datos:', ('data1.csv', 'data2.csv'))

if selected_data == 'data1.csv':
    st.subheader('Distribuciones de data1.csv')
    show_histogram_and_distributions(pd.DataFrame({'Value': data1}))
elif selected_data == 'data2.csv':
    st.subheader('Distribuciones de data2.csv')
    show_histogram_and_distributions(pd.DataFrame({'Value': data2}))
    
