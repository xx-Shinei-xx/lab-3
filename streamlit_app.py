import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm, poisson, chisquare

# Función para graficar la distribución gaussiana
def plot_gaussian_distribution(mu, sigma):
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    y = norm.pdf(x, mu, sigma)
    fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines', name='Distribución Gaussiana'))
    fig.add_trace(go.Histogram(x=np.random.normal(mu, sigma, 1000), histnorm='probability density', name='Histograma Gaussiano'))
    fig.update_layout(title='Distribución Gaussiana', xaxis_title='Valor', yaxis_title='Densidad de probabilidad')
    st.plotly_chart(fig)

# Función para ajustar la distribución de Poisson
def fit_poisson_distribution(data):
    mu = np.mean(data)
    return mu

# Función para realizar la prueba de chi-cuadrado y determinar el ajuste que se adapta mejor
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
    
    return p_value

# Función para graficar la distribución de Poisson con el ajuste
def plot_poisson_distribution(data):
    mu = np.mean(data)
    x = np.arange(0, max(data) + 1)
    y = poisson.pmf(x, mu)
    
    # Ajustar distribución de Poisson
    fit_y = poisson.pmf(x, mu)

    fig = go.Figure(data=[go.Bar(x=x, y=y, name='Distribución de Poisson'),
                          go.Scatter(x=x, y=fit_y, mode='lines', name='Ajuste de Poisson', line=dict(color='red', width=2))])
    fig.update_layout(title='Distribución de Poisson', xaxis_title='Valor', yaxis_title='Probabilidad')
    st.plotly_chart(fig)

# Función para mostrar histograma y distribuciones
def show_histogram_and_distributions(df, selected_data):
    st.subheader('Histograma de datos')
    st.bar_chart(df)

    st.subheader('Distribuciones Estadísticas')
    if selected_data == 'data1.csv':
        mu_gaussian = df['Value'].mean()
        sigma_gaussian = df['Value'].std()
        plot_gaussian_distribution(mu_gaussian, sigma_gaussian)
    elif selected_data == 'data2.csv':
        mu_poisson = df['Value'].mean()
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
    show_histogram_and_distributions(pd.DataFrame({'Value': data1}), selected_data)
    if st.button('Realizar ajuste de chi-cuadrado para distribución gaussiana'):
        p_value_gaussian = chi_square_test(data1, 'gaussian')
        st.write(f"Valor p para distribución gaussiana: {p_value_gaussian}")
elif selected_data == 'data2.csv':
    st.subheader('Distribuciones de data2.csv')
    show_histogram_and_distributions(pd.DataFrame({'Value': data2}), selected_data)
    if st.button('Ver distribución de Poisson'):
        plot_poisson_distribution(data2)
        
