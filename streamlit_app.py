import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm, poisson, chisquare

# Cargar los datos 1 y 2
data1 = np.genfromtxt('data1.csv', delimiter=',', skip_header=1, usecols=1)
data2 = np.genfromtxt('data2.csv', delimiter=',', skip_header=1, usecols=1)

# distribución gaussiana
def plot_gaussian_distribution(data):
    mu, sigma = np.mean(data), np.std(data)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    y = norm.pdf(x, mu, sigma)
    fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines', name='Distribución Gaussiana'))
    fig.add_trace(go.Histogram(x=data, histnorm='probability density', name='Histograma Gaussiano'))
    fig.update_layout(title='Distribución Gaussiana', xaxis_title='Valor', yaxis_title='Densidad de probabilidad')
    st.plotly_chart(fig)


def chi_square_test(data):
    num_bins = 10
    hist, bins = np.histogram(data, bins=num_bins)

    mu, std = norm.fit(data)
    expected = len(data) * np.diff(bins) * norm.pdf(bins[:-1], mu, std)
    chi2 = np.sum((hist - expected)**2 / expected)
    df = num_bins - 1
    p_value = 1 - stats.chi2.cdf(chi2, df)
    return chi2, df, p_value






























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

#-------------------


# Función para calcular frecuencias observadas y esperadas
def calcular_frecuencias(data):
    valores_unicos, counts = np.unique(data, return_counts=True)
    frecuencia_observada = counts
    mu = np.mean(data)
    frecuencia_esperada = poisson.pmf(valores_unicos, mu) * len(data)
    return valores_unicos, frecuencia_observada, frecuencia_esperada

# Función para mostrar la tabla de frecuencias
def mostrar_tabla(data):
    valores_unicos, frecuencia_observada, frecuencia_esperada = calcular_frecuencias(data)
    tabla_data = {"Valor": valores_unicos, "Frecuencia Observada": frecuencia_observada, "Frecuencia Esperada": frecuencia_esperada}
    tabla = st.table(tabla_data)
# Función para calcular y mostrar la distribución de Poisson
def plot_poisson_distribution(data):
    mu = np.mean(data)
    x = np.arange(0, max(data) + 1)
    y = poisson.pmf(x, mu)
    fit_y = poisson.pmf(x, mu)
    fig = go.Figure(data=[go.Bar(x=x, y=y, name='Distribución de Poisson'),
                          go.Scatter(x=x, y=fit_y, mode='lines', name='Ajuste de Poisson', line=dict(color='red', width=2))])
    fig.update_layout(title='Distribución de Poisson', xaxis_title='Valor', yaxis_title='Probabilidad')
    st.plotly_chart(fig)

# Función para calcular chi cuadrado
def calcular_chi_cuadrado(f_obs, f_exp):
    return np.sum((f_obs - f_exp)**2 / f_exp)



#-----------------------------------

# Streamlit
st.title('Análisis de Datos')

# Botón para seleccionar el conjunto de datos
selected_data = st.radio('Seleccionar conjunto de datos:', ('data1.csv', 'data2.csv'))

if selected_data == 'data1.csv':
    st.subheader('Distribuciones en el decaimiento solo con el aire')
    st.subheader('Distribución de Gauss:')
    plot_gaussian_distribution(data1)
    if st.button('Mostrar Tabla'):
        chi2, df, p_value = chi_square_test(data1)
         st.write("Estadístico de chi-cuadrado:", chi2)
         st.write("Grados de libertad:", df)
         st.write("Valor p:", p_value)

    
        
    st.subheader('Distribución de Poisson:')
    plot_poisson_distribution(data1)
    if st.button('Mostrar Tabla'):
        mostrar_tabla(data1)
        valores_unicos, frecuencia_observada, frecuencia_esperada = calcular_frecuencias(data1)
        chi_cuadrado = calcular_chi_cuadrado(frecuencia_observada, frecuencia_esperada)
        st.write(f"Valor de chi cuadrado: {chi_cuadrado}")




elif selected_data == 'data2.csv':
    st.subheader('Distribuciones en el decaimiento del cesio-137')
    st.subheader('Distribución de Gauss:')
    plot_gaussian_distribution(data2)
     if st.button('Mostrar Tabla'):
        chi2, df, p_value = chi_square_test(data1)
         st.write("Estadístico de chi-cuadrado:", chi2)
         st.write("Grados de libertad:", df)
         st.write("Valor p:", p_value)

    
    st.subheader('Distribución de Poisson:')
    plot_poisson_distribution(data2)
    if st.button('Mostrar Tabla'):
        mostrar_tabla(data2)
        valores_unicos, frecuencia_observada, frecuencia_esperada = calcular_frecuencias(data2)
        chi_cuadrado = calcular_chi_cuadrado(frecuencia_observada, frecuencia_esperada)
        st.write(f"Valor de chi cuadrado: {chi_cuadrado}")





        
        # Realizar la prueba de chi-cuadrado
        

       




