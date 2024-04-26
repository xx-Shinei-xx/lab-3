import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm, poisson

# Función para graficar la distribución gaussiana
def plot_gaussian_distribution(data):
    mu, std = np.mean(data), np.std(data)
    x = np.linspace(mu - 3*std, mu + 3*std, 100)
    y = norm.pdf(x, mu, std)
    fig = go.Figure(data=go.Scatter(x=x, y=y))
    fig.update_layout(title='Distribución Gaussiana', xaxis_title='Valor', yaxis_title='Densidad de probabilidad')
    st.plotly_chart(fig)

# Función para graficar la distribución de Poisson
def plot_poisson_distribution(data):
    mu = np.mean(data)
    x = np.arange(0, max(data) + 1)
    y = poisson.pmf(x, mu)
    fig = go.Figure(data=go.Bar(x=x, y=y))
    fig.update_layout(title='Distribución de Poisson', xaxis_title='Valor', yaxis_title='Probabilidad')
    st.plotly_chart(fig)

# Cargar los datos desde el archivo CSV
data = np.genfromtxt('data1.csv', delimiter=',', skip_header=1, usecols=1)

# Crear la aplicación Streamlit
st.title('Distribuciones Estadísticas')

# Agregar un botón para cambiar entre las distribuciones
distribution = st.radio('Seleccionar distribución:', ('Gaussiana', 'Poisson'))

# Mostrar la distribución seleccionada
if distribution == 'Gaussiana':
    plot_gaussian_distribution(data)
elif distribution == 'Poisson':
    plot_poisson_distribution(data)
    
