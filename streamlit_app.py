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
    mu = st.slider("Media (μ)", min_value=0.0, max_value=10.0, value=5.0)
    sigma = st.slider("Desviación estándar (σ)", min_value=0.1, max_value=5.0, value=1.0)
    sample_size = st.slider("Tamaño de la muestra", min_value=10, max_value=100, value=50)
    sample_data = np.random.normal(mu, sigma, sample_size)
else:
    lambda_val = st.slider("Tasa (λ)", min_value=0.1, max_value=5.0, value=1.0)
    sample_size = st.slider("Tamaño de la muestra", min_value=10, max_value=100, value=50)
    sample_data = np.random.poisson(lambda_val, sample_size)

# Visualiza los datos generados
st.write("Datos generados:")
st.write(sample_data)

# Realiza la prueba de chi-cuadrado
observed_counts, bin_edges = np.histogram(sample_data, bins="auto")
bin_middles = 0.5 * (bin_edges[1:] + bin_edges[:-1])

# Calcula los valores esperados para cada distribución
if distribution_choice == "Gaussiana":
    expected_counts = stats.norm.pdf(bin_middles, mu, sigma) * sample_size
else:
    expected_counts = stats.poisson.pmf(np.arange(len(observed_counts)), lambda_val) * sample_size

# Asegúrate de que la suma de los valores esperados sea igual a la suma de los valores observados
expected_counts *= np.sum(observed_counts) / np.sum(expected_counts)

# Realiza la prueba de chi-cuadrado
chi2_statistic, p_value = stats.chisquare(observed_counts, expected_counts)

# Muestra los resultados de la prueba de chi-cuadrado
st.write(f"Estadístico de chi-cuadrado: {chi2_statistic:.4f}")
st.write(f"Valor
