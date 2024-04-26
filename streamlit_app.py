import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import poisson, norm
from scipy.optimize import curve_fit

# Cargar datos
data1 = pd.read_csv('data1.csv')
data2 = pd.read_csv('data2.csv')

# Definir funciones para ajustar distribuciones
def ajustar_poisson(data_column):
    # Ajustar distribución de Poisson
    params = poisson.fit(data_column)
    return params

def ajustar_gaussiana(data):
    # Ajustar distribución gaussiana
    media, desviacion = norm.fit(data)
    return media, desviacion

# Aplicación Streamlit
st.title('Histogramas de Distribuciones')

# Controles del panel lateral
distribucion = st.sidebar.selectbox(
    'Seleccionar Distribución',
    ('Poisson', 'Gaussiana')
)

# Botón para cambiar entre data1 y data2
dataset = st.sidebar.radio("Seleccione el conjunto de datos:", ('data1', 'data2'))

# Seleccionar el conjunto de datos apropiado
if dataset == 'data1':
    data = data1
else:
    data = data2

# Trazar histogramas y ajustes
if distribucion == 'Poisson':
    st.header('Distribución de Poisson')

    # Ajustar distribución de Poisson usando los datos seleccionados
    params = ajustar_poisson(data['Decaimiento solo con el aire'])

    # Crear histograma con Plotly
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data['Decaimiento solo con el aire'], marker=dict(color=data['Decaimiento solo con el aire'], colorscale='blue'), opacity=0.6))
    fig.update_layout(title_text='Distribución de Poisson', xaxis_title='Valor', yaxis_title='Probabilidad')
    st.plotly_chart(fig)

elif distribucion == 'Gaussiana':
    st.header('Distribución Gaussiana')

    # Ajustar distribución gaussiana usando los datos seleccionados
    media, desviacion = ajustar_gaussiana(data['Decaimiento solo con el aire'].values)

    # Crear histograma con Plotly
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data['Decaimiento solo con el aire'], marker=dict(color=data['Decaimiento solo con el aire'], colorscale='Reds'), opacity=0.6))
    fig.update_layout(title_text='Distribución Gaussiana', xaxis_title='Valor', yaxis_title='Probabilidad')
    st.plotly_chart(fig)
    
