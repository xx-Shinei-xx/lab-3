# Importa las bibliotecas necesarias
import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats

# Carga tus datos (reemplaza 'data1.csv' con el nombre de tu archivo)
data1 = pd.read_csv('data1.csv')

# Agrega una barra lateral para seleccionar la distribución
distribution_choice = st.sidebar.selectbox("Selecciona una distribución:", ["Gaussiana", "Poisson"])

# Genera datos de ejemplo (puedes reemplazar esto con tus datos reales)
if distribution_choice == "Gaussiana":
    mu = st.sidebar.slider("Media (μ)", min_value=0.0, max_value=10.0, value=5.0)
    sigma = st.sidebar.slider("Desviación estándar (σ)", min_value=0.1, max_value=5.0, value=1.0)
    sample_size = st.sidebar.slider("Tamaño de la muestra", min_value=10, max_value=1000, value=50)
    sample_data = np.random.normal(mu, sigma, sample_size)
else:
    lambda_val = st.sidebar.slider("Tasa (λ)", min_value=0.1, max_value=10.0, value=1.0)
    sample_size = st.sidebar.slider("Tamaño de la muestra", min_value=10, max_value=1000, value=50)
    sample_data = np.random.poisson(lambda_val, sample_size)

# Visualiza los datos generados
st.write("Datos generados:")
st.write(sample_data)

# Realiza la prueba de chi-cuadrado
observed_counts, bin_edges = np.histogram(sample_data, bins='auto')
bin_middles = 0.5 * (bin_edges[1:] + bin_edges[:-1])

# Calcula los valores esperados para cada distribución
if distribution_choice == "Gaussiana":
    expected_counts = stats.norm.pdf(bin_middles, mu, sigma) * sample_size * np.diff(bin_edges)
else:
    expected_counts = stats.poisson.pmf(bin_middles, lambda_val) * sample_size * np.diff(bin_edges)

# Asegúrate de que la suma de los valores esperados sea igual a la suma de los valores observados
expected_counts *= np.sum(observed_counts) / np.sum(expected_counts)

# Realiza la prueba de chi-cuadrado
chi2_statistic, p_value = stats.chisquare(observed_counts, expected_counts)

# Muestra los resultados de la prueba de chi-cuadrado
st.write(f"Estadístico de chi-cuadrado: {chi2_statistic:.4f}")
st.write(f"Valor p: {p_value:.4f}")

# Compara el valor p con un umbral (por ejemplo, 0.05) para determinar la significancia
if p_value < 0.05:
    st.write("Hay una diferencia significativa entre los datos observados y los esperados.")
else:
    st.write("No hay una diferencia significativa entre los datos observados y los esperados.")

# Puedes agregar más visualizaciones y análisis según tus necesidades

# Ejecuta la aplicación con 'streamlit run nombre_del_archivo.py'
