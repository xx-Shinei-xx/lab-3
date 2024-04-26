import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm, poisson, chisquare

# Función para graficar la distribución gaussiana con un fit visible y realizar la prueba de chi cuadrado
def plot_gaussian_distribution(data):
    mu, std = np.mean(data), np.std(data)
    x = np.linspace(mu - 3*std, mu + 3*std, 100)
    y = norm.pdf(x, mu, std)
    # Fit visible
    fit_x = np.linspace(min(data), max(data), 100)
    fit_y = norm.pdf(fit_x, mu, std)
    fig = go.Figure(data=[go.Scatter(x=x, y=y, mode='lines', name='Distribución Gaussiana'),
                          go.Scatter(x=fit_x, y=fit_y, mode='lines', name='Fit Gaussiano', line=dict(color='red', width=2))])
    fig.update_layout(title='Distribución Gaussiana', xaxis_title='Valor', yaxis_title='Densidad de probabilidad')
    st.plotly_chart(fig)
    # Realizar prueba de chi cuadrado
    expected_freq = norm.pdf(data, mu, std) * len(data)
    _, p_value = chisquare(data, f_exp=expected_freq)
    st.write(f'Valor p para la prueba de chi cuadrado: {p_value}')

# Función para graficar la distribución de Poisson con barras y realizar la prueba de chi cuadrado
def plot_poisson_distribution(data):
    mu = np.mean(data)
    x = np.arange(0, max(data) + 1)
    y = poisson.pmf(x, mu)
    fig = go.Figure(data=go.Bar(x=x, y=y))
    fig.update_layout(title='Distribución de Poisson', xaxis_title='Valor', yaxis_title='Probabilidad')
    st.plotly_chart(fig)
    # Realizar prueba de chi cuadrado
    expected_freq = poisson.pmf(x, mu) * len(data)
    _, p_value = chisquare(data, f_exp=expected_freq)
    st.write(f'Valor p para la prueba de chi cuadrado: {p_value}')

# Cargar los datos desde el archivo CSV
data = np.genfromtxt('data1.csv', delimiter=',', skip_header=1, usecols=1)

# Crear la aplicación Streamlit
st.title('Distribuciones Estadísticas')

# Agregar un botón para cambiar entre las distribuciones
distribution = st.radio('Seleccionar distribución:', ('Gaussiana', 'Poisson'))

# Mostrar la distribución seleccionada y realizar prueba de chi cuadrado
if distribution == 'Gaussiana':
    plot_gaussian_distribution(data)
elif distribution == 'Poisson':
    plot_poisson_distribution(data)
    
