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

# Streamlit
st.set_page_config(page_title="Análisis de Datos", page_icon=":bar_chart:", layout="wide")

# Título y subtítulo
st.title('Análisis de Datos')
st.write("Bienvenido al análisis estadístico de los conjuntos de datos.")
st.markdown("---")

# Sidebar
st.sidebar.title("Opciones")
st.sidebar.markdown("Utiliza los botones a continuación para navegar por la página:")
selected_data = st.sidebar.radio('Seleccionar conjunto de datos:', ('data1', 'data2'))

# Contenido principal
col1, col2 = st.columns(2)

with col1:
    st.subheader('Distribución de Gauss:')
    st.write("En esta sección se muestra la distribución de Gauss para el conjunto de datos seleccionado.")
    st.write("Utiliza el botón para mostrar la tabla de frecuencias y el valor de chi cuadrado.")

with col2:
    st.subheader('Distribución de Poisson:')
    st.write("Aquí se muestra la distribución de Poisson correspondiente al conjunto de datos seleccionado.")
    st.write("Utiliza el botón para mostrar la tabla de frecuencias y el valor de chi cuadrado.")

# Línea divisoria
st.markdown("---")

# Contenido principal
if selected_data == 'data1':
    st.subheader('Decaimiento solo con el aire')
    st.markdown("---")
    
    st.subheader('Distribución de Gauss:')
    st.write("Aquí se muestra la distribución de Gauss para el conjunto de datos 'data1.csv'.")
    st.markdown("---")

    st.subheader('Distribución de Poisson:')
    st.write("Aquí se muestra la distribución de Poisson para el conjunto de datos 'data1.csv'.")
    st.markdown("---")

    st.subheader('Opciones:')
    if st.button('Mostrar Tabla y Valor de Chi Cuadrado (Gaussiana)'):
        data1 = np.genfromtxt('data1.csv', delimiter=',', skip_header=1, usecols=1)
        mostrar_tabla_y_chi(data1)
        st.markdown("---")

    if st.button('Mostrar Tabla y Valor de Chi Cuadrado (Poisson)'):
        data1 = np.genfromtxt('data1.csv', delimiter=',', skip_header=1, usecols=1)
        mostrar_tabla_y_chi(data1)
        st.markdown("---")

elif selected_data == 'data2':
    st.subheader('Decaimiento del cesio-137')
    st.markdown("---")
    
    st.subheader('Distribución de Gauss:')
    st.write("Aquí se muestra la distribución de Gauss para el conjunto de datos 'data2.csv'.")
    st.markdown("---")

    st.subheader('Distribución de Poisson:')
    st.write("Aquí se muestra la distribución de Poisson para el conjunto de datos 'data2.csv'.")
    st.markdown("---")

    st.subheader('Opciones:')
    if st.button('Mostrar Tabla y Valor de Chi Cuadrado (Gaussiana)'):
        data2 = np.genfromtxt('data2.csv', delimiter=',', skip_header=1, usecols=1)
        mostrar_tabla_y_chi(data2)
        st.markdown("---")

    if st.button('Mostrar Tabla y Valor de Chi Cuadrado (Poisson)'):
        data2 = np.genfromtxt('data2.csv', delimiter=',', skip_header=1, usecols=1)
        mostrar_tabla_y_chi(data2)
        st.markdown("---")

# Botones de navegación
st.sidebar.markdown("---")
st.sidebar.markdown("### Navegación")
if st.sidebar.button("Ir al Marco Teórico"):
    st.write("Aquí va el contenido del Marco Teórico.")
if st.sidebar.button("Ver Gráficas y Tablas de Data1"):
    st.write("Aquí van las gráficas y tablas del conjunto de datos 1.")
if st.sidebar.button("Ver Gráficas y Tablas de Data2"):
    st.write("Aquí van las gráficas y tablas del conjunto de datos 2.")
