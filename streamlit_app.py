import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm

# Cargar datos
data1 = pd.read_csv('data1.csv')

# Definir funciones para ajustar distribuciones
def ajustar_poisson(data):
    # Ajustar distribución de Poisson
    params = poisson.fit(data)
    return params

def ajustar_gaussiana(data):
    # Ajustar distribución gaussiana
    media, desviacion = norm.fit(data)
    return media, desviacion

# Definir funciones para trazar histogramas
def trazar_histograma(data, bins, titulo):
    plt.hist(data, bins=bins, density=True, alpha=0.6, color='g')
    plt.title(titulo)
    plt.xlabel('Valor')
    plt.ylabel('Probabilidad')
    plt.grid(True)

def trazar_ajuste_poisson(data, params):
    x = np.arange(0, max(data) + 1)
    y = poisson.pmf(x, *params)
    plt.plot(x, y, 'r-', linewidth=2)

def trazar_ajuste_gaussiana(media, desviacion):
    x = np.linspace(-5, 5, 100)
    y = norm.pdf(x, media, desviacion)
    plt.plot(x, y, 'b-', linewidth=2)

# Aplicación Streamlit
st.title('Histogramas de Distribuciones')

# Controles del panel lateral
distribucion = st.sidebar.selectbox(
    'Seleccionar Distribución',
    ('Poisson', 'Gaussiana')
)

# Trazar histogramas y ajustes
if distribucion == 'Poisson':
    st.header('Distribución de Poisson')
    # Ajustar distribución de Poisson usando los datos de "Decaimiento"
    params = ajustar_poisson(data1['Decaimiento solo con el aire'])
    # Trazar histograma
    trazar_histograma(data1['Decaimiento solo con el aire'], bins=20, titulo='Distribución de Poisson')
    trazar_ajuste_poisson(data1['Decaimiento solo con el aire'], params)
    st.pyplot()

elif distribucion == 'Gaussiana':
    st.header('Distribución Gaussiana')
    # Ajustar distribución gaussiana usando los datos de "Decaimiento"
    media, desviacion = ajustar_gaussiana(data1['Decaimiento solo con el aire'])
    # Trazar histograma
    trazar_histograma(data1['Decaimiento solo con el aire'], bins=20, titulo='Distribución Gaussiana')
    trazar_ajuste_gaussiana(media, desviacion)
    st.pyplot()
