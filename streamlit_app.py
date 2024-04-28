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
    fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines', name='Ajuste de Gauss'))
    fig.add_trace(go.Histogram(x=data, histnorm='probability density', name='Distribución Gaussiana', marker=dict(color='green')))
    fig.update_layout(title=title, xaxis_title='Valor', yaxis_title='Densidad de probabilidad')
    return fig

# Función para plotear la distribución de Poisson
def plot_poisson_distribution(data, title):
    mu = np.mean(data)
    x = np.arange(0, max(data) + 1)
    y = poisson.pmf(x, mu)
    fit_y = poisson.pmf(x, mu)
    fig = go.Figure(data=[ go.Scatter(x=x, y=fit_y, mode='lines', name='Ajuste de Poisson', line=dict(color='red', width=2)),
    go.Bar(x=x, y=y, name='Distribución de Poisson', marker=dict(color='orange')) ])
    fig.update_layout(title=title, xaxis_title='Valor', yaxis_title='Probabilidad')
    return fig




# Streamlit
#st.set_page_config(page_title="Análisis de Datos", page_icon=":bar_chart:", layout="wide")

# Título y subtítulo
st.markdown("<h1 style='text-align: center;'>Laboratorio 3</h1>", unsafe_allow_html=True)
st.write("<p style='text-align: center;'>José Guillermo Monterroso Marroquín, 202005689 y Shawn César Alejandro García Fernández, 201906567.</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.title("Navegación")
selected_option = st.sidebar.radio('Seleccionar opción:', ('Reporte', 'Data1', 'Data2'))

# Contenido principal
if selected_option == 'Reporte':
    col1, col2 = st.columns(2)

    with col1:
        st.write("<div class='big-title'>Resumen</div>", unsafe_allow_html=True)
        st.markdown(
            """
            En los procesos atómicos, se generan partículas con altas velocidades. El Cesio-137, un isótopo radioactivo, emite una cantidad de estas partículas de alta velocidad y energía, conocidas como 'partículas beta'. La medición de estas partículas es factible mediante un instrumento llamado contador Geiger, el cual cuantifica la cantidad de partículas que atraviesan el detector. Al realizar mediciones experimentales de la desintegración de un número determinado de partículas, podemos prever la cantidad de partículas que se desintegran mediante un ajuste de nuestros datos experimentales.
            """
        )

        st.write("<div class='big-title'></div>", unsafe_allow_html=True)
        st.subheader('Objetivos')
        st.subheader('Generales')
        st.markdown(
            """
            - Llevar a cabo el análisis para cuantificar las partículas de alta velocidad generadas por el Cesio-137, así como realizar mediciones en condiciones ambientales normales.
            """
        )
        st.subheader('Específicos')
        st.markdown(
            """
            - Verificar qué tipo de distribución se ajusta de mejor manera a los datos tomados.
            - Comprobar que existe una diferencia entre las mediciones usando el Cesio-137 y las mediciones en un ambiente normal.
            """
        )
       
        st.write("<div class='big-title'>Diseño Experimental</div>", unsafe_allow_html=True)
        st.markdown(
            """
            Para el experimento, se utilizó cesio-137, un contador Geiger y papel para registrar la cantidad de partículas que medía nuestra herramienta. El proceso es el siguiente:
            """
        )
        st.markdown(
            """
            - Se conectó el contador Geiger a una fuente de poder. Esta fuente de poder tiene que calibrarse para poder cotabilizar de manera correcta el decaimiento.
            - Se registró el número de partículas detectadas por el contador Geiger en dos escenarios: utilizando cesio-137 y en un entorno natural (el aire).
            """
        )

    with col2:
        st.write("<div class='big-title'>Discusión de Resultados</div>", unsafe_allow_html=True)
        st.markdown(
            """
            - La distribución Gaussiana parece ajustarse de mejor manera a los datos tomados en un medio natural.
            - La distribución de Poisson se ajusta de manera correcta a los datos medidos con el Cesio-137.
            """
        )

        st.write("<div class='big-title'>Conclusiones</div>", unsafe_allow_html=True)
        st.markdown(
            """
            - Existe una notable diferencia entre las mediciones de decaimiento entre los casos del Cesio-137 y el medio natural. Además, las elecciones de los ajustes fueron acertadas para el comportamiento que tenían las mediciones de datos.
            """
        )

        st.write("<div class='big-title'>Referencias</div>", unsafe_allow_html=True)
        st.markdown(
            """
            - Taylor, John R. “An introduction to error analysis, The study of uncertainties in physical measurements”. Second edition. University science books. 1982.
            - Bevington, P. R. (2003). "Data Reduction and Error Analysis for the Physical Sciences". McGraw-Hill Education.
            - Taylor, J. R. (1997). "An Introduction to Error Analysis: The Study of Uncertainties in Physical Measurements" (2nd ed.). University Science Books.
            - [Chi-squared test](https://en.wikipedia.org/wiki/Chi-squared_test)
            - [Poisson distribution](https://en.wikipedia.org/wiki/Poisson_distribution)
            - [Normal distribution](https://en.wikipedia.org/wiki/Normal_distribution)
            """
        )



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
